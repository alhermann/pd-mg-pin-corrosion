#include "coupling.h"
#include "boundary.h"
#include <cstdio>
#include <cmath>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <sys/stat.h>

std::string CoupledSolver::make_filename(const Config& cfg, const std::string& prefix,
                                          double time_s, int frame) {
    std::ostringstream ss;
    ss << cfg.output_dir << "/" << prefix
       << "_" << std::setw(6) << std::setfill('0') << frame
       << "_t" << std::fixed << std::setprecision(1) << time_s << "s.vti";
    return ss.str();
}

void CoupledSolver::write_diagnostics(const Grid& grid, const Fields& fields,
                                       double t_corr, const Config& cfg) {
    int solid_count = 0;
    double C_solid_sum = 0.0;
    int N = grid.N_total;

    for (int i = 0; i < N; ++i) {
        if (grid.node_type[i] == SOLID_MG) {
            solid_count++;
            C_solid_sum += fields.C[i];
        }
    }

    // Pin mass loss: fraction of initial solid mass that has been lost
    // Initial mass ~ initial_solid_count * C_solid_init (= 1.0)
    // Current mass ~ sum(C[solid_nodes])  +  dissolved_nodes*0
    double pin_mass_loss = (1.0 - C_solid_sum / (initial_solid_count_ + 1e-30)) * 100.0;
    if (pin_mass_loss < 0.0) pin_mass_loss = 0.0;

    // Max velocity and concentration in fluid
    double v_max = 0.0;
    double C_max = 0.0;
    for (int i = 0; i < N; ++i) {
        if (grid.node_type[i] == FLUID) {
            double v = norm(fields.vel[i]);
            if (v > v_max) v_max = v;
            if (fields.C[i] > C_max) C_max = fields.C[i];
        }
    }

    std::printf("  t=%.1f s (%.2f h)  pin_mass_loss=%.2f%%  solid=%d  v_max=%.3e  C_max_fluid=%.4f\n",
                t_corr, t_corr / 3600.0, pin_mass_loss,
                solid_count, v_max, C_max);

    // Append to diagnostics CSV (opened in append mode; header written by init_csv_files)
    std::string csv_path = cfg.output_dir + "/diagnostics.csv";
    std::ofstream csv(csv_path, std::ios::app);
    csv << std::scientific << std::setprecision(6)
        << t_corr << "," << t_corr / 3600.0 << ","
        << pin_mass_loss << ","
        << solid_count << "," << v_max << "," << C_max << "\n";

    // Simple mass loss CSV for easy plotting
    std::string ml_path = cfg.output_dir + "/mass_loss.csv";
    std::ofstream ml(ml_path, std::ios::app);
    ml << std::fixed << std::setprecision(6)
       << t_corr / 3600.0 << "," << pin_mass_loss << "\n";
}

// Initialize CSV files with headers (truncates any stale data from previous runs)
static void init_csv_files(const Config& cfg) {
    {
        std::ofstream csv(cfg.output_dir + "/diagnostics.csv", std::ios::trunc);
        csv << "time_s,time_h,pin_mass_loss_pct,solid_nodes,v_max,C_max_fluid\n";
    }
    {
        std::ofstream ml(cfg.output_dir + "/mass_loss.csv", std::ios::trunc);
        ml << "time_h,pin_mass_loss_pct\n";
    }
}

void CoupledSolver::run(Grid& grid, Fields& fields, const Config& cfg) {
    Timer t_total("total_simulation");

    // Create output directory
    mkdir(cfg.output_dir.c_str(), 0755);

    // Initialize CSV files (truncates old data)
    init_csv_files(cfg);

    // Count initial solid nodes
    initial_solid_count_ = 0;
    for (int i = 0; i < grid.N_total; ++i) {
        if (grid.node_type[i] == SOLID_MG) initial_solid_count_++;
    }
    std::printf("Initial solid nodes: %d\n", initial_solid_count_);

    // Initialize solvers
    flow_solver_.init(grid, cfg);
    ard_solver_.init(grid, cfg);

    // Write initial state (this goes into PVD)
    std::string fname = make_filename(cfg, "state", 0.0, frame_count_);
    writer_.write(fname, grid, fields, cfg);
    writer_.add_timestep(0.0, fname);
    frame_count_++;

    double t_corr = 0.0;
    int cycle = 0;
    bool need_flow_solve = true; // always solve flow on first cycle

    while (t_corr < cfg.T_final) {
        cycle++;
        std::printf("\n=== Coupling cycle %d, t=%.1f s (%.2f h) ===\n",
                    cycle, t_corr, t_corr / 3600.0);

        // Phase 1: Solve flow to steady state (only when geometry changed)
        if (need_flow_solve) {
            flow_solver_.solve_steady(fields, grid, cfg);

            // Write flow solution (for debugging only — NOT added to PVD)
            fname = make_filename(cfg, "flow", t_corr, frame_count_);
            writer_.write(fname, grid, fields, cfg);
            frame_count_++;
        } else {
            std::printf("  Skipping flow solve (no dissolution in previous cycle)\n");
        }

        // Phase 2: Corrosion with frozen velocity field
        double dt_corr = ard_solver_.compute_dt(fields, grid, cfg);
        std::printf("  Corrosion dt = %.4e s\n", dt_corr);

        for (int step = 1; step <= cfg.corrosion_steps_per_check; ++step) {
            // Apply concentration BCs
            apply_inlet_bc(fields, grid, cfg);
            apply_outlet_bc(fields, grid, cfg);
            apply_wall_concentration_bc(fields, grid);

            ard_solver_.step(fields, grid, cfg, dt_corr);

            // Swap C buffer
            std::swap(fields.C, fields.C_new);

            // Smooth concentration near inlet/outlet to fix PD truncation artifacts
            smooth_boundary_concentration(fields, grid, cfg);

            t_corr += dt_corr;

            if (step % cfg.output_every_corr == 0) {
                fname = make_filename(cfg, "corr", t_corr, frame_count_);
                writer_.write(fname, grid, fields, cfg);
                writer_.add_timestep(t_corr, fname);
                frame_count_++;
                write_diagnostics(grid, fields, t_corr, cfg);
            }

            if (t_corr >= cfg.T_final) break;
        }

        // Phase 3: Check dissolution
        int n_dissolved = ard_solver_.apply_phase_change(fields, grid, cfg);
        need_flow_solve = (n_dissolved > 0);
        if (n_dissolved > 0) {
            std::printf("  Phase change: %d nodes dissolved\n", n_dissolved);
            update_node_types_after_dissolution(grid, fields);

            if (n_dissolved > 10) {
                std::printf("  Rebuilding neighbor list...\n");
                grid.build_neighbors();
            }
        } else {
            std::printf("  No phase changes this cycle\n");
        }

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
    fname = make_filename(cfg, "final", t_corr, frame_count_);
    writer_.write(fname, grid, fields, cfg);
    writer_.add_timestep(t_corr, fname);

    // Write PVD time series file
    writer_.write_pvd(cfg.output_dir + "/simulation.pvd");

    std::printf("\n=== Simulation complete ===\n");
    std::printf("  Final time: %.1f s (%.2f h)\n", t_corr, t_corr / 3600.0);
    t_total.report();
}
