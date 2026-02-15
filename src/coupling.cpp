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
    std::string ext = cfg.use_amr ? ".vtu" : ".vti";
    ss << cfg.output_dir << "/" << prefix
       << "_" << std::setw(6) << std::setfill('0') << frame
       << "_t" << std::fixed << std::setprecision(1) << time_s << "s" << ext;
    return ss.str();
}

void CoupledSolver::write_diagnostics(const Grid& grid, const Fields& fields,
                                       double t_corr, const Config& cfg) {
    int solid_count = 0;
    int N = grid.N_total;
    for (int i = 0; i < N; ++i) {
        if (grid.node_type[i] == SOLID_MG) solid_count++;
    }

    // Pin mass/volume loss: sum C for ALL initially-solid nodes (regardless
    // of current type). This gives smooth behavior across dissolution events:
    // when a node dissolves, its C is set to C_thresh and then decays via
    // fluid-phase transport — no discrete jump in the loss metric.
    double C_solid_sum = 0.0;
    for (int idx : initial_solid_indices_) {
        C_solid_sum += fields.C[idx];
    }

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

    // Set PVD paths so they are rewritten after each timestep (crash-safe)
    writer_.set_pvd_path(cfg.output_dir + "/simulation.pvd");
    flow_writer_.set_pvd_path(cfg.output_dir + "/flow.pvd");

    // Initialize CSV files (truncates old data)
    init_csv_files(cfg);

    // Record initial solid nodes (for smooth volume loss tracking)
    initial_solid_count_ = 0;
    initial_solid_indices_.clear();
    for (int i = 0; i < grid.N_total; ++i) {
        if (grid.node_type[i] == SOLID_MG) {
            initial_solid_count_++;
            initial_solid_indices_.push_back(i);
        }
    }
    std::printf("Initial solid nodes: %d\n", initial_solid_count_);

    // Initialize solvers
    flow_solver_.init(grid, cfg);
    ard_solver_.init(grid, cfg);
    if (cfg.use_implicit) {
        ard_implicit_solver_.init(grid, cfg);
        std::printf("Using IMPLICIT ARD solver (dt_max=%.1f s, fraction=%.2f)\n",
                    cfg.implicit_dt_max, cfg.implicit_dt_fraction);
    } else {
        std::printf("Using EXPLICIT ARD solver\n");
    }

    // Write initial state (this goes into PVD)
    std::string fname = make_filename(cfg, "state", 0.0, frame_count_);
    if (cfg.use_amr) writer_.write_vtu(fname, grid, fields, cfg);
    else             writer_.write(fname, grid, fields, cfg);
    writer_.add_timestep(0.0, fname);
    frame_count_++;

    double t_corr = 0.0;
    int cycle = 0;
    bool need_flow_solve = true; // always solve flow on first cycle
    dissolved_since_flow_ = 0;

    while (t_corr < cfg.T_final) {
        cycle++;
        std::printf("\n=== Coupling cycle %d, t=%.1f s (%.2f h) ===\n",
                    cycle, t_corr, t_corr / 3600.0);

        // Phase 1: Solve flow to steady state (only when geometry changed)
        if (need_flow_solve) {
            std::printf("  Flow re-solve triggered (%d nodes dissolved since last flow solve)\n",
                        dissolved_since_flow_);
            flow_solver_.solve_steady(fields, grid, cfg);
            if (cfg.use_amr) grid.update_fictitious(fields);
            dissolved_since_flow_ = 0;
            need_flow_solve = false;

            // Write flow solution snapshot (added to flow PVD)
            fname = make_filename(cfg, "flow", t_corr, frame_count_);
            if (cfg.use_amr) writer_.write_vtu(fname, grid, fields, cfg);
            else             writer_.write(fname, grid, fields, cfg);
            flow_writer_.add_timestep(t_corr, fname);
            frame_count_++;
        } else {
            std::printf("  Skipping flow solve (no dissolution since last flow solve)\n");
        }

        // Phase 2: Corrosion with frozen velocity field
        if (cfg.use_implicit) {
            // --- IMPLICIT PATH ---
            // Compute current volume loss fraction for decay model
            double C_solid_sum = 0.0;
            for (int idx : initial_solid_indices_) {
                C_solid_sum += fields.C[idx];
            }
            double vol_loss_frac = 1.0 - C_solid_sum / (initial_solid_count_ + 1e-30);
            if (vol_loss_frac < 0.0) vol_loss_frac = 0.0;
            ard_implicit_solver_.set_volume_loss(vol_loss_frac);

            // Assemble PD operator matrix once per coupling cycle
            ard_implicit_solver_.assemble(fields, grid, cfg);

            int implicit_step = 0;  // local counter for this cycle
            double t_cycle_start = t_corr;
            bool dissolution_occurred = false;

            // Run corrosion_steps_per_check implicit steps per cycle,
            // or until a node actually dissolves (C < C_thresh).
            while (implicit_step < cfg.corrosion_steps_per_check &&
                   t_corr < cfg.T_final && !dissolution_occurred) {
                double dt_impl = ard_implicit_solver_.compute_adaptive_dt(fields, grid, cfg);

                // Apply concentration BCs
                apply_inlet_bc(fields, grid, cfg);
                apply_outlet_bc(fields, grid, cfg);
                apply_wall_concentration_bc(fields, grid, cfg);

                ard_implicit_solver_.step(fields, grid, cfg, dt_impl);

                // Smooth truncated PD neighborhoods near inlet/outlet
                smooth_boundary_concentration(fields, grid, cfg);

                t_corr += dt_impl;
                implicit_step++;
                total_implicit_steps_++;

                if (total_implicit_steps_ % cfg.diagnostic_every == 0) {
                    write_diagnostics(grid, fields, t_corr, cfg);
                }
                if (total_implicit_steps_ % cfg.implicit_output_every == 0) {
                    fname = make_filename(cfg, "corr", t_corr, frame_count_);
                    if (cfg.use_amr) writer_.write_vtu(fname, grid, fields, cfg);
                    else             writer_.write(fname, grid, fields, cfg);
                    writer_.add_timestep(t_corr, fname);
                    frame_count_++;
                }

                // Check if any solid node has actually reached dissolution threshold
                for (int i = 0; i < grid.N_total; ++i) {
                    if (grid.node_type[i] == SOLID_MG && fields.C[i] < cfg.C_thresh) {
                        dissolution_occurred = true;
                        break;
                    }
                }
            }

            std::printf("  Implicit cycle: %d steps, t=%.2f to %.2f s (%.4f h)\n",
                        implicit_step, t_cycle_start, t_corr, t_corr / 3600.0);
        } else {
            // --- EXPLICIT PATH (fallback) ---
            // Compute current volume loss fraction for decay model
            {
                double C_solid_sum = 0.0;
                for (int idx : initial_solid_indices_) {
                    C_solid_sum += fields.C[idx];
                }
                double vol_loss_frac = 1.0 - C_solid_sum / (initial_solid_count_ + 1e-30);
                if (vol_loss_frac < 0.0) vol_loss_frac = 0.0;
                ard_solver_.set_volume_loss(vol_loss_frac);
            }
            double dt_corr = ard_solver_.compute_dt(fields, grid, cfg);
            std::printf("  Corrosion dt = %.4e s\n", dt_corr);

            for (int step = 1; step <= cfg.corrosion_steps_per_check; ++step) {
                apply_inlet_bc(fields, grid, cfg);
                apply_outlet_bc(fields, grid, cfg);
                apply_wall_concentration_bc(fields, grid, cfg);

                ard_solver_.step(fields, grid, cfg, dt_corr);
                std::swap(fields.C, fields.C_new);

                t_corr += dt_corr;

                if (step % cfg.output_every_corr == 0) {
                    fname = make_filename(cfg, "corr", t_corr, frame_count_);
                    if (cfg.use_amr) writer_.write_vtu(fname, grid, fields, cfg);
                    else             writer_.write(fname, grid, fields, cfg);
                    writer_.add_timestep(t_corr, fname);
                    frame_count_++;
                    write_diagnostics(grid, fields, t_corr, cfg);
                }

                if (t_corr >= cfg.T_final) break;
            }
        }

        // Phase 3: Check dissolution
        int n_dissolved = cfg.use_implicit
            ? ard_implicit_solver_.apply_phase_change(fields, grid, cfg)
            : ard_solver_.apply_phase_change(fields, grid, cfg);
        total_dissolved_ += n_dissolved;
        dissolved_since_flow_ += n_dissolved;
        if (n_dissolved > 0) {
            std::printf("  Phase change: %d nodes dissolved (total: %d, since flow: %d)\n",
                        n_dissolved, total_dissolved_, dissolved_since_flow_);
            update_node_types_after_dissolution(grid, fields);

            // Rebuild neighbor list so newly-FLUID nodes have correct bond structure
            std::printf("  Rebuilding neighbor list...\n");
            if (cfg.use_amr)
                grid.build_neighbors_celllist(cfg);
            else
                grid.build_neighbors();

            // Re-solve flow next cycle: dissolved nodes are now interior fluid
            // and need velocity/pressure computed by PD-NS
            need_flow_solve = true;
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
    if (cfg.use_amr) writer_.write_vtu(fname, grid, fields, cfg);
    else             writer_.write(fname, grid, fields, cfg);
    writer_.add_timestep(t_corr, fname);

    // PVD files are already up-to-date (written incrementally after each add_timestep)
    std::printf("\n=== Simulation complete ===\n");
    std::printf("  Final time: %.1f s (%.2f h)\n", t_corr, t_corr / 3600.0);
    t_total.report();
}
