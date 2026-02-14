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

    // Include artificial diffusion in stability limit
    double D_eff_max = D_max + cfg.alpha_art_diff * v_max * cfg.dx;
    double dt_diff = 0.25 * cfg.dx * cfg.dx / (D_eff_max + 1e-30);
    double dt_adv = cfg.dx / (v_max + 1e-30);
    double dt = cfg.cfl_factor_corr * std::min(dt_diff, dt_adv);
    return dt;
}

void PD_ARD_Solver::step(Fields& fields, const Grid& grid, const Config& cfg, double dt) {
    int N = grid.N_total;
    double div_coeff = alpha_p_ / V_H_;

    // Precompute salt-layer blocking flags for solid nodes
    // (Jafarzadeh et al. 2018): if a solid node has ANY liquid neighbor
    // with C >= C_sat, all its interface bonds are disabled.
    std::vector<bool> salt_blocked(N, false);
    #pragma omp parallel for schedule(dynamic, 256)
    for (int i = 0; i < N; ++i) {
        if (grid.node_type[i] != SOLID_MG) continue;
        for (int jj = grid.nbr_offset[i]; jj < grid.nbr_offset[i + 1]; ++jj) {
            int j = grid.nbr_index[jj];
            if (grid.nbr_vol[jj] < 1e-30) continue;
            if (grid.node_type[j] == FLUID && fields.C[j] >= cfg.C_sat) {
                salt_blocked[i] = true;
                break;
            }
        }
    }

    #pragma omp parallel for schedule(dynamic, 256)
    for (int i = 0; i < N; ++i) {
        NodeType nt_i = grid.node_type[i];

        // Skip wall, inlet, outlet, outside — values set by BCs
        if (nt_i == WALL || nt_i == INLET || nt_i == OUTLET || nt_i == OUTSIDE) {
            fields.C_new[i] = fields.C[i];
            continue;
        }

        double C_i = fields.C[i];
        bool i_is_fluid = (nt_i == FLUID);
        bool i_is_solid = (nt_i == SOLID_MG);
        bool i_is_gb = fields.is_gb[i];
        bool i_salt_blocked = i_is_solid && salt_blocked[i];

        // Velocity (only fluid nodes)
        Vec vel_i = i_is_fluid ? fields.vel[i] : vec_zero();
        double vi_mag = i_is_fluid ? norm(vel_i) : 0.0;

        double diff_sum = 0.0;
        double adv_sum = 0.0;

        for (int jj = grid.nbr_offset[i]; jj < grid.nbr_offset[i + 1]; ++jj) {
            int j = grid.nbr_index[jj];
            double xi = grid.nbr_dist[jj];
            Vec e_ij = grid.nbr_evec[jj];
            double V_j = grid.nbr_vol[jj];

            if (V_j < 1e-30) continue;

            NodeType nt_j = grid.node_type[j];

            // Skip WALL and OUTSIDE for transport
            if (nt_j == WALL || nt_j == OUTSIDE) continue;

            double C_j = fields.C[j];
            double inv_xi = 1.0 / xi;
            double inv_xi2 = inv_xi * inv_xi;

            // Bond classification:
            //   Liquid-Liquid:  D_liquid + advection
            //   Interface:      harmonic_mean(D_liquid, D_solid) + advection on fluid side
            //   Solid-Solid:    skip (no transport within solid bulk)
            bool j_is_fluid = (nt_j == FLUID || nt_j == INLET || nt_j == OUTLET);
            bool j_is_solid = (nt_j == SOLID_MG);

            // Skip solid-solid bonds (no diffusion within bulk solid)
            if (i_is_solid && j_is_solid) continue;

            double D_avg = 0.0;
            if (i_is_fluid && j_is_fluid) {
                // Liquid-Liquid bond
                D_avg = cfg.D_liquid;
            } else {
                // Interface bond (one fluid, one solid):
                // harmonic mean of D_liquid and D_solid
                bool solid_is_gb;
                bool solid_salt_blocked;
                if (i_is_solid) {
                    solid_is_gb = i_is_gb;
                    solid_salt_blocked = i_salt_blocked;
                } else {
                    solid_is_gb = fields.is_gb[j];
                    solid_salt_blocked = salt_blocked[j];
                }

                if (solid_salt_blocked) {
                    D_avg = 0.0;
                } else {
                    double D_s = solid_is_gb ? cfg.D_gb : cfg.D_grain;
                    D_avg = 2.0 * cfg.D_liquid * D_s / (cfg.D_liquid + D_s + 1e-30);
                }
            }

            // Artificial diffusion (only for liquid-liquid bonds with velocity)
            double D_art = 0.0;
            if (i_is_fluid && j_is_fluid) {
                double vj_mag = norm(fields.vel[j]);
                D_art = cfg.alpha_art_diff * std::max(vi_mag, vj_mag) * cfg.dx;
            }

            // PD diffusion
            diff_sum += beta_coeff_ * (D_avg + D_art) * (C_j - C_i) * inv_xi2 * V_j;

            // Advection: non-conservative form v_i . e_ij (fluid-fluid bonds only).
            // Interface bonds (fluid-solid) carry ONLY diffusion —
            // advection does not apply across the solid-liquid interface.
            if (i_is_fluid && j_is_fluid) {
                double vi_dot_e = dot(vel_i, e_ij);
                adv_sum += (C_j - C_i) * vi_dot_e * inv_xi * V_j;
            }
        }

        adv_sum *= div_coeff;

        fields.C_new[i] = C_i + dt * (diff_sum - adv_sum);

        // Clamp concentration to physical range
        if (fields.C_new[i] < 0.0) fields.C_new[i] = 0.0;
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
                fields.C[i] = cfg.C_thresh;
                n_dissolved++;
            }
        }
    }

    return n_dissolved;
}
