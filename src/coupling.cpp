#include "coupling.h"
#include "boundary.h"
#include <cstdio>
#include <cmath>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <sys/stat.h>

std::string CoupledSolver::make_filename(const Config& cfg, const std::string& prefix, int frame) {
    std::ostringstream ss;
    ss << cfg.output_dir << "/" << prefix << "_"
       << std::setw(6) << std::setfill('0') << frame << ".vti";
    return ss.str();
}

void CoupledSolver::write_diagnostics(const Grid& grid, const Fields& fields,
                                       double t_corr, const Config& cfg) {
    int solid_count = 0;
    int N = grid.N_total;
    double C_total = 0.0;

    for (int i = 0; i < N; ++i) {
        if (grid.node_type[i] == SOLID_MG) solid_count++;
        C_total += fields.C[i];
    }

    // Node-based volume loss: fraction of dissolved nodes
    double VL_node = 1.0 - static_cast<double>(solid_count) /
                     static_cast<double>(initial_solid_count_ + 1);

    // Mass-based volume loss: 1 - (sum C(t)) / (sum C(0))
    // This tracks the integral of concentration over the entire domain
    double VL_mass = 1.0 - C_total / (initial_C_total_ + 1e-30);
    if (VL_mass < 0.0) VL_mass = 0.0; // can happen if concentration is produced

    // Max velocity in fluid
    double v_max = 0.0;
    double C_max = 0.0;
    for (int i = 0; i < N; ++i) {
        if (grid.node_type[i] == FLUID) {
            double v = norm(fields.vel[i]);
            if (v > v_max) v_max = v;
            if (fields.C[i] > C_max) C_max = fields.C[i];
        }
    }

    std::printf("  t=%.1f s (%.2f h)  VL_node=%.2f%%  VL_mass=%.2f%%  solid=%d  v_max=%.3e  C_max=%.4f\n",
                t_corr, t_corr / 3600.0, VL_node * 100.0, VL_mass * 100.0,
                solid_count, v_max, C_max);

    // Append to diagnostics CSV
    std::string csv_path = cfg.output_dir + "/diagnostics.csv";
    static bool header_written = false;
    std::ofstream csv(csv_path, std::ios::app);
    if (!header_written) {
        csv << "time_s,time_h,VL_node_pct,VL_mass_pct,solid_nodes,v_max,C_max\n";
        header_written = true;
    }
    csv << std::scientific << std::setprecision(6)
        << t_corr << "," << t_corr / 3600.0 << ","
        << VL_node * 100.0 << "," << VL_mass * 100.0 << ","
        << solid_count << "," << v_max << "," << C_max << "\n";

    // Also append to a simple volume_loss.csv for easy plotting
    std::string vl_path = cfg.output_dir + "/volume_loss.csv";
    static bool vl_header_written = false;
    std::ofstream vl(vl_path, std::ios::app);
    if (!vl_header_written) {
        vl << "time_h,VL_node_pct,VL_mass_pct\n";
        vl_header_written = true;
    }
    vl << std::fixed << std::setprecision(6)
       << t_corr / 3600.0 << "," << VL_node * 100.0 << ","
       << VL_mass * 100.0 << "\n";
}

void CoupledSolver::run(Grid& grid, Fields& fields, const Config& cfg) {
    Timer t_total("total_simulation");

    // Create output directory
    mkdir(cfg.output_dir.c_str(), 0755);

    // Count initial solid nodes and total initial concentration
    initial_solid_count_ = 0;
    initial_C_total_ = 0.0;
    for (int i = 0; i < grid.N_total; ++i) {
        if (grid.node_type[i] == SOLID_MG) initial_solid_count_++;
        initial_C_total_ += fields.C[i];
    }
    std::printf("Initial solid nodes: %d\n", initial_solid_count_);
    std::printf("Initial total concentration: %.4f\n", initial_C_total_);

    // Initialize solvers
    flow_solver_.init(grid, cfg);
    ard_solver_.init(grid, cfg);

    // Write initial state
    std::string fname = make_filename(cfg, "state", frame_count_);
    writer_.write(fname, grid, fields, cfg);
    writer_.add_timestep(0.0, fname);
    frame_count_++;

    double t_corr = 0.0;
    int cycle = 0;

    while (t_corr < cfg.T_final) {
        cycle++;
        std::printf("\n=== Coupling cycle %d, t=%.1f s (%.2f h) ===\n",
                    cycle, t_corr, t_corr / 3600.0);

        // Phase 1: Solve flow to steady state
        int flow_iters = flow_solver_.solve_steady(fields, grid, cfg);

        // Write flow solution
        fname = make_filename(cfg, "flow", frame_count_);
        writer_.write(fname, grid, fields, cfg);
        writer_.add_timestep(t_corr, fname);
        frame_count_++;

        // Phase 2: Corrosion with frozen velocity field
        double dt_corr = ard_solver_.compute_dt(fields, grid, cfg);
        std::printf("  Corrosion dt = %.4e s\n", dt_corr);

        for (int step = 1; step <= cfg.corrosion_steps_per_check; ++step) {
            // Apply concentration BCs
            apply_inlet_bc(fields, grid, cfg);
            apply_outlet_bc(fields, grid, cfg);

            ard_solver_.step(fields, grid, cfg, dt_corr);

            // Swap C buffer
            std::swap(fields.C, fields.C_new);

            t_corr += dt_corr;

            if (step % cfg.output_every_corr == 0) {
                fname = make_filename(cfg, "corr", frame_count_);
                writer_.write(fname, grid, fields, cfg);
                writer_.add_timestep(t_corr, fname);
                frame_count_++;
                write_diagnostics(grid, fields, t_corr, cfg);
            }

            if (t_corr >= cfg.T_final) break;
        }

        // Phase 3: Check dissolution
        int n_dissolved = ard_solver_.apply_phase_change(fields, grid, cfg);
        if (n_dissolved > 0) {
            std::printf("  Phase change: %d nodes dissolved\n", n_dissolved);
            update_node_types_after_dissolution(grid, fields);

            // Rebuild neighbor list if significant changes
            if (n_dissolved > 10) {
                std::printf("  Rebuilding neighbor list...\n");
                grid.build_neighbors();
            }
        }

        write_diagnostics(grid, fields, t_corr, cfg);

        // Check if all solid has dissolved
        int solid_remaining = 0;
        for (int i = 0; i < grid.N_total; ++i) {
            if (grid.node_type[i] == SOLID_MG) solid_remaining++;
        }
        if (solid_remaining == 0) {
            std::printf("\n=== All solid nodes dissolved at t=%.1f s (%.2f h) ===\n",
                        t_corr, t_corr / 3600.0);
            break;
        }
    }

    // Write final state
    fname = make_filename(cfg, "final", frame_count_);
    writer_.write(fname, grid, fields, cfg);
    writer_.add_timestep(t_corr, fname);

    // Write PVD time series file
    writer_.write_pvd(cfg.output_dir + "/simulation.pvd");

    std::printf("\n=== Simulation complete ===\n");
    std::printf("  Final time: %.1f s (%.2f h)\n", t_corr, t_corr / 3600.0);
    t_total.report();
}
