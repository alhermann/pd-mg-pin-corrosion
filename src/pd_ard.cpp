#include "pd_ard.h"
#include <cmath>
#include <cstdio>
#include <omp.h>

void PD_ARD_Solver::init(const Grid& grid, const Config& cfg) {
    alpha_p_ = static_cast<double>(DIM);

    if constexpr (DIM == 2) {
        V_H_ = PI * cfg.delta * cfg.delta;
        beta_coeff_ = 4.0 / (PI * cfg.delta * cfg.delta);
    } else {
        V_H_ = (4.0 / 3.0) * PI * cfg.delta * cfg.delta * cfg.delta;
        beta_coeff_ = 12.0 / (PI * cfg.delta * cfg.delta);
    }
}

double PD_ARD_Solver::compute_dt(const Fields& fields, const Grid& grid, const Config& cfg) {
    double D_max = std::max({cfg.D_liquid, cfg.D_grain, cfg.D_gb});
    double v_max = 0.0;
    int N = grid.N_total;

    #pragma omp parallel for reduction(max:v_max) schedule(static)
    for (int i = 0; i < N; ++i) {
        if (grid.node_type[i] == FLUID) {
            double v = norm(fields.vel[i]);
            if (v > v_max) v_max = v;
        }
    }

    double dt_diff = 0.25 * cfg.dx * cfg.dx / (D_max + 1e-30);
    double dt_adv = cfg.dx / (v_max + 1e-30);
    // Also consider k_corr stability: dt * k_corr < 0.5
    double dt_corr = 0.5 / (cfg.k_corr * cfg.gb_corr_factor + 1e-30);
    double dt = cfg.cfl_factor * std::min({dt_diff, dt_adv, dt_corr});
    return dt;
}

void PD_ARD_Solver::step(Fields& fields, const Grid& grid, const Config& cfg, double dt) {
    int N = grid.N_total;
    double w = cfg.w_advect;
    double div_coeff = alpha_p_ / V_H_;

    #pragma omp parallel for schedule(dynamic, 256)
    for (int i = 0; i < N; ++i) {
        NodeType nt = grid.node_type[i];

        // Skip wall, inlet, outlet, outside
        if (nt == WALL || nt == INLET || nt == OUTLET || nt == OUTSIDE) {
            fields.C_new[i] = fields.C[i];
            continue;
        }

        double C_i = fields.C[i];
        double D_i = fields.D_map[i];

        if (nt == SOLID_MG) {
            // ============================================
            // SOLID NODE: internal diffusion + surface dissolution
            // ============================================
            // 1) PD diffusion ONLY with other solid neighbors (within-solid transport)
            // 2) Surface dissolution: k_corr source term proportional to fluid exposure
            double diff_sum = 0.0;
            int n_fluid_nbr = 0;
            int n_total_nbr = 0;

            for (int jj = grid.nbr_offset[i]; jj < grid.nbr_offset[i + 1]; ++jj) {
                int j = grid.nbr_index[jj];
                double V_j = grid.nbr_vol[jj];
                if (V_j < 1e-30) continue;

                n_total_nbr++;

                if (grid.node_type[j] == SOLID_MG) {
                    // Solid-solid diffusion (within-grain transport)
                    double xi = grid.nbr_dist[jj];
                    double inv_xi2 = 1.0 / (xi * xi);
                    double D_j = fields.D_map[j];
                    // Harmonic mean for diffusivity (rate-limited by slower side)
                    double D_avg = 2.0 * D_i * D_j / (D_i + D_j + 1e-30);
                    double C_j = fields.C[j];
                    diff_sum += beta_coeff_ * D_avg * (C_j - C_i) * inv_xi2 * V_j;
                } else if (grid.node_type[j] == FLUID) {
                    n_fluid_nbr++;
                }
            }

            // Surface dissolution: concentration decreases proportional to fluid exposure
            double f_exposure = (n_total_nbr > 0) ?
                static_cast<double>(n_fluid_nbr) / static_cast<double>(n_total_nbr) : 0.0;

            // Corrosion rate: higher at grain boundaries
            double k_eff = cfg.k_corr;
            if (fields.is_gb[i]) {
                k_eff *= cfg.gb_corr_factor;
            }

            fields.C_new[i] = C_i + dt * (diff_sum - k_eff * f_exposure);

        } else {
            // ============================================
            // FLUID NODE: full PD advection-diffusion
            // ============================================
            Vec vel_i = fields.vel[i];

            double diff_sum = 0.0;
            double adv_down = 0.0;
            double adv_up = 0.0;

            for (int jj = grid.nbr_offset[i]; jj < grid.nbr_offset[i + 1]; ++jj) {
                int j = grid.nbr_index[jj];
                double xi = grid.nbr_dist[jj];
                Vec e_ij = grid.nbr_evec[jj];
                double V_j = grid.nbr_vol[jj];

                if (V_j < 1e-30) continue;

                double C_j = fields.C[j];
                double D_j = fields.D_map[j];
                Vec vel_j = fields.vel[j];

                double inv_xi = 1.0 / xi;
                double inv_xi2 = inv_xi * inv_xi;

                // --- Diffusion (PD Laplacian) ---
                // Use harmonic mean for D across different materials
                double D_avg = 2.0 * D_i * D_j / (D_i + D_j + 1e-30);
                diff_sum += beta_coeff_ * D_avg * (C_j - C_i) * inv_xi2 * V_j;

                // --- Advection (PD divergence of C*v) ---
                double vi_dot_e = dot(vel_i, e_ij);
                double vj_dot_e = dot(vel_j, e_ij);

                // Downwind
                adv_down += (C_j * vj_dot_e - C_i * vi_dot_e) * inv_xi * V_j;

                // Upwind (reverse direction)
                adv_up += (C_j * (-vj_dot_e) - C_i * (-vi_dot_e)) * inv_xi * V_j;
            }

            double adv_sum = div_coeff * (w * adv_down + (1.0 - w) * adv_up);

            fields.C_new[i] = C_i + dt * (diff_sum - adv_sum);
        }

        // Clamp concentration
        if (fields.C_new[i] < 0.0) fields.C_new[i] = 0.0;
        if (fields.C_new[i] > 1.5) fields.C_new[i] = 1.5;
    }
}

int PD_ARD_Solver::apply_phase_change(Fields& fields, Grid& grid, const Config& cfg) {
    int n_dissolved = 0;
    int N = grid.N_total;

    for (int i = 0; i < N; ++i) {
        if (fields.phase[i] == 0 && grid.node_type[i] == SOLID_MG) {
            if (fields.C[i] < cfg.C_thresh) {
                fields.phase[i] = 1; // become liquid
                grid.node_type[i] = FLUID;
                fields.D_map[i] = cfg.D_liquid;
                fields.rho[i] = cfg.rho_f;
                fields.vel[i] = vec_zero();
                n_dissolved++;
            }
        }
    }

    return n_dissolved;
}
