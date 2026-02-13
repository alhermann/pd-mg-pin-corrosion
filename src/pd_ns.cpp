#include "pd_ns.h"
#include "boundary.h"
#include <cmath>
#include <cstdio>
#include <omp.h>

void PD_NS_Solver::init(const Grid& grid, const Config& cfg) {
    alpha_ = static_cast<double>(DIM);

    if constexpr (DIM == 2) {
        V_H_ = PI * cfg.delta * cfg.delta;
        beta_lap_ = 4.0 / (PI * cfg.delta * cfg.delta);
    } else {
        V_H_ = (4.0 / 3.0) * PI * cfg.delta * cfg.delta * cfg.delta;
        beta_lap_ = 12.0 / (PI * cfg.delta * cfg.delta);
    }
}

void PD_NS_Solver::compute_pressure(Fields& f, const Config& cfg) {
    double rho0 = cfg.rho_f;
    double c0 = cfg.c0;
    double gamma = cfg.gamma_eos;
    double B = rho0 * c0 * c0 / gamma;

    int N = static_cast<int>(f.rho.size());
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < N; ++i) {
        double ratio = f.rho[i] / rho0;
        if (ratio < 0.5) ratio = 0.5;
        if (ratio > 2.0) ratio = 2.0;
        f.pressure[i] = B * (std::pow(ratio, gamma) - 1.0);
    }
}

double PD_NS_Solver::compute_dt(const Fields& fields, const Grid& grid, const Config& cfg) {
    double v_max = 0.0;
    int N = grid.N_total;

    #pragma omp parallel for reduction(max:v_max) schedule(static)
    for (int i = 0; i < N; ++i) {
        if (grid.node_type[i] == FLUID) {
            double v = norm(fields.vel[i]);
            if (v > v_max) v_max = v;
        }
    }

    double dt_cfl = cfg.dx / (cfg.c0 + v_max + 1e-30);
    double nu = cfg.mu_f / cfg.rho_f;
    double dt_visc = 0.25 * cfg.dx * cfg.dx / (nu + 1e-30);
    double D_v = cfg.eta_density * cfg.c0 * cfg.delta;
    double dt_dens = 0.25 * cfg.dx * cfg.dx / (D_v + 1e-30);

    double dt = cfg.cfl_factor * std::min({dt_cfl, dt_visc, dt_dens});
    return dt;
}

void PD_NS_Solver::step(Fields& fields, const Grid& grid, const Config& cfg, double dt) {
    compute_pressure(fields, cfg);

    double mu = cfg.mu_f;
    double delta = cfg.delta;
    double c0_local = cfg.c0;
    double D_v = cfg.eta_density * c0_local * delta;

    double inv_VH = 1.0 / V_H_;
    int N = grid.N_total;

    // Density diffusion uses the same PD Laplacian coefficient
    double dens_diff_coeff = beta_lap_ * D_v;

    #pragma omp parallel for schedule(dynamic, 256)
    for (int i = 0; i < N; ++i) {
        NodeType nt = grid.node_type[i];

        // Skip OUTSIDE and prescribed nodes
        if (nt == OUTSIDE || nt == INLET || nt == OUTLET) {
            fields.rho_new[i] = fields.rho[i];
            fields.vel_new[i] = fields.vel[i];
            continue;
        }

        double rho_i = fields.rho[i];
        Vec vel_i = fields.vel[i];
        double p_i = fields.pressure[i];

        double mass_conv = 0.0;
        double mass_diff = 0.0;
        Vec mom_conv = vec_zero();
        Vec mom_pres = vec_zero();
        Vec mom_visc = vec_zero();

        for (int jj = grid.nbr_offset[i]; jj < grid.nbr_offset[i + 1]; ++jj) {
            int j = grid.nbr_index[jj];
            double xi = grid.nbr_dist[jj];
            Vec e_ij = grid.nbr_evec[jj];
            double V_j = grid.nbr_vol[jj];

            if (V_j < 1e-30) continue;

            double rho_j = fields.rho[j];
            Vec vel_j = fields.vel[j];
            double p_j = fields.pressure[j];

            double inv_xi = 1.0 / xi;
            double inv_xi2 = inv_xi * inv_xi;

            // --- Mass: PD divergence of (rho*v) ---
            Vec flux_j = rho_j * vel_j;
            Vec flux_i = rho_i * vel_i;
            mass_conv += dot(flux_j - flux_i, e_ij) * inv_xi * V_j;

            // --- Density diffusion: PD Laplacian of rho ---
            mass_diff += dens_diff_coeff * (rho_j - rho_i) * inv_xi2 * V_j;

            // --- Momentum convection: PD divergence of (rho*v*v) ---
            for (int d = 0; d < DIM; ++d) {
                double conv_d = 0.0;
                for (int dp = 0; dp < DIM; ++dp) {
                    conv_d += (rho_j * vel_j[d] * vel_j[dp] -
                               rho_i * vel_i[d] * vel_i[dp]) * e_ij[dp];
                }
                mom_conv[d] += conv_d * inv_xi * V_j;
            }

            // --- Pressure gradient: PD gradient of p ---
            for (int d = 0; d < DIM; ++d) {
                mom_pres[d] += (p_j - p_i) * e_ij[d] * inv_xi * V_j;
            }

            // --- Viscous term: FULL PD Laplacian of velocity ---
            // mu * nabla^2(v) = mu * beta_lap * sum_j [(v_j - v_i)/|xi|^2] V_j
            // Uses the scalar Laplacian coefficient applied component-wise.
            for (int d = 0; d < DIM; ++d) {
                mom_visc[d] += (vel_j[d] - vel_i[d]) * inv_xi2 * V_j;
            }
        }

        // Update density (FLUID, WALL, SOLID_MG all evolve)
        double rho_new = rho_i + dt * (
            -(alpha_ * inv_VH) * mass_conv
            + mass_diff
        );

        // Clamp density
        if (rho_new < 0.5 * cfg.rho_f) rho_new = 0.5 * cfg.rho_f;
        if (rho_new > 2.0 * cfg.rho_f) rho_new = 2.0 * cfg.rho_f;
        fields.rho_new[i] = rho_new;

        // Update velocity only for FLUID nodes
        if (nt == FLUID) {
            double inv_rho = 1.0 / rho_i;
            for (int d = 0; d < DIM; ++d) {
                fields.vel_new[i][d] = vel_i[d] + dt * inv_rho * (
                    -(alpha_ * inv_VH) * mom_conv[d]
                    -(alpha_ * inv_VH) * mom_pres[d]
                    + mu * beta_lap_ * mom_visc[d]
                );
            }
        } else {
            // WALL and SOLID_MG: velocity enforced by BCs
            fields.vel_new[i] = fields.vel[i];
        }
    }
}

int PD_NS_Solver::solve_steady(Fields& fields, const Grid& grid, const Config& cfg) {
    std::printf("\n--- Flow solver: solving to steady state ---\n");
    Timer t("flow_solve");

    double dt = compute_dt(fields, grid, cfg);
    std::printf("  Initial dt = %.4e s\n", dt);

    int N = grid.N_total;
    double epsilon = 1.0;
    int iter = 0;
    bool diverged = false;

    for (iter = 1; iter <= cfg.flow_max_iters; ++iter) {
        // Apply boundary conditions before step
        apply_inlet_bc(fields, grid, cfg);
        apply_outlet_bc(fields, grid, cfg);
        apply_wall_bc(fields, grid);
        apply_solid_surface_bc(fields, grid);

        // One timestep
        step(fields, grid, cfg, dt);

        // Apply BCs to new fields (wall mirroring)
        apply_wall_bc_new(fields, grid);

        // Convergence check
        if (iter <= 10 || iter % 100 == 0) {
            double num = 0.0, den = 0.0;
            double v_max_new = 0.0;
            double rho_min = 1e30, rho_max = -1e30;
            bool has_nan = false;

            for (int i = 0; i < N; ++i) {
                if (grid.node_type[i] == FLUID) {
                    if (std::isnan(fields.vel_new[i][0]) || std::isnan(fields.rho_new[i])) {
                        has_nan = true;
                        break;
                    }
                    Vec dv = fields.vel_new[i] - fields.vel[i];
                    num += dot(dv, dv);
                    den += dot(fields.vel[i], fields.vel[i]);
                    double vn = norm(fields.vel_new[i]);
                    if (vn > v_max_new) v_max_new = vn;
                    if (fields.rho_new[i] < rho_min) rho_min = fields.rho_new[i];
                    if (fields.rho_new[i] > rho_max) rho_max = fields.rho_new[i];
                }
            }

            if (has_nan) {
                std::printf("  Flow DIVERGED (NaN) at iter %d\n", iter);
                diverged = true;
                break;
            }

            epsilon = (den > 1e-30) ? std::sqrt(num / den) : std::sqrt(num);

            bool print_it = (iter <= 10) || (iter % cfg.output_every_flow == 0);
            if (print_it) {
                std::printf("  Flow iter %6d: eps=%.3e  v_max=%.4e  rho=[%.2f,%.2f]  dt=%.3e\n",
                            iter, epsilon, v_max_new, rho_min, rho_max, dt);
            }

            // Check for velocity blowup
            if (v_max_new > 100.0 * cfg.U_in) {
                std::printf("  Flow DIVERGED (v_max=%.2e >> U_in=%.2e) at iter %d\n",
                            v_max_new, cfg.U_in, iter);
                diverged = true;
                break;
            }
        }

        // Swap buffers
        fields.swap_buffers();

        // Recompute dt periodically
        if (iter % 200 == 0) {
            dt = compute_dt(fields, grid, cfg);
        }

        if (epsilon < cfg.flow_conv_tol && iter > 100) {
            std::printf("  Flow converged at iter %d, eps=%.3e\n", iter, epsilon);
            break;
        }
    }

    if (!diverged && iter > cfg.flow_max_iters) {
        std::printf("  Flow did NOT converge after %d iters, eps=%.3e\n",
                    cfg.flow_max_iters, epsilon);
    }

    // Poiseuille validation: compare with analytical profile at mid-section
    if (!diverged) {
        // Find a cross-section at y = L_wire/2 (mid pin) or y = -L_upstream/2 (upstream)
        // For validation, use the upstream section where there's no pin
        double y_check = -cfg.L_upstream / 2.0;
        double err_sum = 0.0, norm_sum = 0.0;
        int n_check = 0;

        // For 2D channel: v_analytical(x) = (3/2)*U_in*(1 - (x/R_tube)^2)
        for (int i = 0; i < N; ++i) {
            if (grid.node_type[i] != FLUID) continue;
            double py = (DIM == 2) ? grid.pos[i][1] : grid.pos[i][2];
            if (std::abs(py - y_check) > 0.6 * cfg.dx) continue;

            double px = grid.pos[i][0];
            if constexpr (DIM == 2) {
                double r_norm = px / cfg.R_tube;
                if (std::abs(r_norm) > 1.0) continue;
                double v_analytical = 1.5 * cfg.U_in * (1.0 - r_norm * r_norm);
                double v_numerical = fields.vel[i][1]; // axial velocity
                err_sum += (v_numerical - v_analytical) * (v_numerical - v_analytical);
                norm_sum += v_analytical * v_analytical;
                n_check++;
            }
        }
        if (n_check > 0 && norm_sum > 1e-30) {
            double rel_err = std::sqrt(err_sum / norm_sum);
            std::printf("  Poiseuille validation (upstream, %d nodes): L2 rel error = %.3e\n",
                        n_check, rel_err);
        }
    }

    t.report();
    return iter;
}
