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
    double dt_corr = 0.5 / (cfg.k_corr * cfg.gb_corr_factor + 1e-30);
    double dt = cfg.cfl_factor_corr * std::min({dt_diff, dt_adv, dt_corr});
    return dt;
}

void PD_ARD_Solver::step(Fields& fields, const Grid& grid, const Config& cfg, double dt) {
    int N = grid.N_total;
    double div_coeff = alpha_p_ / V_H_;

    #pragma omp parallel for schedule(dynamic, 256)
    for (int i = 0; i < N; ++i) {
        NodeType nt = grid.node_type[i];

        // Skip wall, inlet, outlet, outside — values set by BCs
        if (nt == WALL || nt == INLET || nt == OUTLET || nt == OUTSIDE) {
            fields.C_new[i] = fields.C[i];
            continue;
        }

        double C_i = fields.C[i];
        double D_i = fields.D_map[i];

        if (nt == SOLID_MG) {
            // ============================================
            // SOLID NODE: surface dissolution only (no PD)
            // ============================================
            // Solid C represents structural integrity (1=intact, 0=corroded).
            // Only surface nodes (those exposed to fluid) dissolve.
            int n_fluid_nbr = 0;
            int n_total_nbr = 0;
            bool salt_layer_blocked = false;

            for (int jj = grid.nbr_offset[i]; jj < grid.nbr_offset[i + 1]; ++jj) {
                int j = grid.nbr_index[jj];
                if (grid.nbr_vol[jj] < 1e-30) continue;
                n_total_nbr++;
                if (grid.node_type[j] == FLUID) {
                    n_fluid_nbr++;
                    // Salt layer model (Jafarzadeh et al. 2018): if any
                    // adjacent fluid node is at saturation, the local
                    // electrochemical driving force vanishes and
                    // dissolution is temporarily blocked.
                    if (fields.C[j] >= cfg.C_sat) salt_layer_blocked = true;
                }
            }

            double f_exposure = (n_total_nbr > 0) ?
                static_cast<double>(n_fluid_nbr) / static_cast<double>(n_total_nbr) : 0.0;

            double k_eff = cfg.k_corr;
            if (fields.is_gb[i]) {
                k_eff *= cfg.gb_corr_factor;
            }

            // Block dissolution when salt layer forms
            if (salt_layer_blocked) k_eff = 0.0;

            fields.C_new[i] = C_i - dt * k_eff * f_exposure;
            if (fields.C_new[i] < 0.0) fields.C_new[i] = 0.0;
            continue;

        } else {
            // ============================================
            // FLUID NODE: PD advection-diffusion
            // ============================================
            // PD transport is computed ONLY over fluid-type bonds
            // (FLUID, INLET, OUTLET). Bonds to SOLID_MG and WALL
            // are skipped for transport; instead, a dissolution
            // source term accounts for mass entering from the pin.
            Vec vel_i = fields.vel[i];
            double vi_mag = norm(vel_i);

            double diff_sum = 0.0;
            double adv_sum = 0.0;
            double diss_k_sum = 0.0;
            int n_total_nbr = 0;
            int n_solid_nbr = 0;

            for (int jj = grid.nbr_offset[i]; jj < grid.nbr_offset[i + 1]; ++jj) {
                int j = grid.nbr_index[jj];
                double xi = grid.nbr_dist[jj];
                Vec e_ij = grid.nbr_evec[jj];
                double V_j = grid.nbr_vol[jj];

                if (V_j < 1e-30) continue;
                n_total_nbr++;

                NodeType nt_j = grid.node_type[j];

                // --- Dissolution source: count solid neighbours ---
                if (nt_j == SOLID_MG) {
                    n_solid_nbr++;
                    double k_j = cfg.k_corr;
                    if (fields.is_gb[j]) k_j *= cfg.gb_corr_factor;
                    diss_k_sum += k_j;
                    continue; // no PD bond to solid
                }

                // Skip WALL and OUTSIDE for transport
                if (nt_j == WALL || nt_j == OUTSIDE) continue;

                // --- PD bonds: FLUID, INLET, OUTLET ---
                double C_j = fields.C[j];
                double D_j = fields.D_map[j];
                Vec vel_j = fields.vel[j];

                double inv_xi = 1.0 / xi;
                double inv_xi2 = inv_xi * inv_xi;

                // Diffusion (PD Laplacian) with artificial diffusion for stability
                // D_art = alpha * max(|v_i|, |v_j|) * dx  (streamline upwind stabilization)
                double D_avg = 2.0 * D_i * D_j / (D_i + D_j + 1e-30);
                double vj_mag = norm(vel_j);
                double D_art = cfg.alpha_art_diff * std::max(vi_mag, vj_mag) * cfg.dx;
                diff_sum += beta_coeff_ * (D_avg + D_art) * (C_j - C_i) * inv_xi2 * V_j;

                // Advection: non-conservative form v·∇C via PD gradient
                //
                // We use v_i · ∇C rather than ∇·(Cv) because the weakly
                // compressible flow has ∇·v ≠ 0 (order Ma²). The conservative
                // divergence form ∇·(Cv) = C∇·v + v·∇C introduces a spurious
                // C·∇·v source term that causes concentration drift at flow
                // convergence/divergence points.
                //
                // PD gradient: ∇C ≈ (d/V_H) Σ_j (C_j - C_i) e_ij / ξ V_j
                // Then: v·∇C = (d/V_H) Σ_j (C_j - C_i)(v_i · e_ij) / ξ V_j
                double vi_dot_e = dot(vel_i, e_ij);
                adv_sum += (C_j - C_i) * vi_dot_e * inv_xi * V_j;
            }

            adv_sum *= div_coeff;

            // Dissolution source: dissolved Mg entering fluid from solid surface
            // Salt layer model: if this fluid node is already at saturation,
            // no additional dissolved species can enter — source is blocked.
            double source = 0.0;
            if (n_solid_nbr > 0 && n_total_nbr > 0 && C_i < cfg.C_sat) {
                double avg_k = diss_k_sum / n_solid_nbr;
                double interface_frac = static_cast<double>(n_solid_nbr) /
                                        static_cast<double>(n_total_nbr);
                source = avg_k * interface_frac * (cfg.rho_m / cfg.rho_f);
            }

            fields.C_new[i] = C_i + dt * (diff_sum - adv_sum + source);
        }

        // Clamp concentration to physical range [0, C_sat]
        // C_sat represents the solubility limit: above this, salt precipitates
        // form (Jafarzadeh et al. 2018), removing dissolved species from solution.
        if (fields.C_new[i] < 1e-30) fields.C_new[i] = 0.0;
        if (fields.C_new[i] > cfg.C_sat) fields.C_new[i] = cfg.C_sat;
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
                // Newly dissolved node starts with some dissolved Mg concentration
                fields.C[i] = cfg.C_thresh;
                n_dissolved++;
            }
        }
    }

    return n_dissolved;
}
