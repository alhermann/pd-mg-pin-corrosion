#include "pd_ard_implicit.h"
#include <cmath>
#include <cstdio>
#include <algorithm>
#include <omp.h>

// ============================================================================
// Initialization
// ============================================================================

void PD_ARD_ImplicitSolver::init(const Grid& grid, const Config& cfg) {
    alpha_p_ = static_cast<double>(DIM);

    if constexpr (DIM == 2) {
        V_H_ = PI * cfg.delta * cfg.delta;
        beta_coeff_ = 4.0 / (PI * cfg.delta * cfg.delta);
    } else {
        V_H_ = (4.0 / 3.0) * PI * cfg.delta * cfg.delta * cfg.delta;
        beta_coeff_ = 12.0 / (PI * cfg.delta * cfg.delta);
    }

    // Smoothing width for salt-layer source cutoff.
    // The dissolution source ramps linearly to zero over [C_sat - eps, C_sat].
    // 2% of C_sat gives a narrow ramp that's smooth enough for Newton convergence
    // without significantly altering the physical behavior.
    eps_sat_ = 0.02 * cfg.C_sat;
}

// ============================================================================
// Index map: fluid nodes <-> local unknowns
// ============================================================================

void PD_ARD_ImplicitSolver::build_index_map(const Grid& grid) {
    int N = grid.N_total;
    global_to_local_.assign(N, -1);
    local_to_global_.clear();
    local_to_global_.reserve(N / 4);

    int idx = 0;
    for (int i = 0; i < N; ++i) {
        if (grid.node_type[i] == FLUID) {
            global_to_local_[i] = idx;
            local_to_global_.push_back(i);
            idx++;
        }
    }
    n_unknowns_ = idx;
}

// ============================================================================
// Assemble linear PD operator matrix M (once per coupling cycle)
// ============================================================================

void PD_ARD_ImplicitSolver::assemble(const Fields& fields, const Grid& grid,
                                      const Config& cfg) {
    Timer t_asm("implicit_assemble");

    build_index_map(grid);
    std::printf("  Implicit: %d fluid unknowns\n", n_unknowns_);

    double div_coeff = alpha_p_ / V_H_;

    using Triplet = Eigen::Triplet<double>;
    std::vector<Triplet> triplets;
    triplets.reserve(n_unknowns_ * 30);

    // Thread-local storage to avoid contention
    int n_threads = 1;
    #pragma omp parallel
    { n_threads = omp_get_num_threads(); }

    std::vector<std::vector<Triplet>> thread_triplets(n_threads);
    for (auto& v : thread_triplets) v.reserve(n_unknowns_ * 30 / n_threads);
    struct BCInfo {
        int local_k;
        int global_j;
        double weight;
    };
    std::vector<std::vector<BCInfo>> thread_bc_info(n_threads);

    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        auto& my_triplets = thread_triplets[tid];
        auto& my_bc = thread_bc_info[tid];

        #pragma omp for schedule(dynamic, 256)
        for (int k = 0; k < n_unknowns_; ++k) {
            int i = local_to_global_[k];
            double D_i = fields.D_map[i];
            Vec vel_i = fields.vel[i];
            double vi_mag = norm(vel_i);

            double diag = 0.0;

            for (int jj = grid.nbr_offset[i]; jj < grid.nbr_offset[i + 1]; ++jj) {
                int j = grid.nbr_index[jj];
                double xi = grid.nbr_dist[jj];
                Vec e_ij = grid.nbr_evec[jj];
                double V_j = grid.nbr_vol[jj];

                if (V_j < 1e-30) continue;

                NodeType nt_j = grid.node_type[j];

                // Skip solid, wall, outside for transport (same as explicit)
                if (nt_j == SOLID_MG || nt_j == WALL || nt_j == OUTSIDE) continue;

                double inv_xi = 1.0 / xi;
                double inv_xi2 = inv_xi * inv_xi;

                // Diffusion weight
                double D_j = fields.D_map[j];
                double D_avg = 2.0 * D_i * D_j / (D_i + D_j + 1e-30);
                double vj_mag = norm(fields.vel[j]);
                double D_art = cfg.alpha_art_diff * std::max(vi_mag, vj_mag) * cfg.dx;
                double w_diff = beta_coeff_ * (D_avg + D_art) * inv_xi2 * V_j;

                // Advection weight (non-conservative: v_i . e_ij)
                double vi_dot_e = dot(vel_i, e_ij);
                double w_adv = div_coeff * vi_dot_e * inv_xi * V_j;

                // Total: dC_i/dt += w_ij*(C_j - C_i) where w_ij = w_diff - w_adv
                // M[k, col_j] = w_ij,  M[k, k] -= w_ij
                double w_ij = w_diff - w_adv;

                diag -= w_ij;

                if (nt_j == FLUID) {
                    int col_j = global_to_local_[j];
                    if (col_j >= 0) {
                        my_triplets.emplace_back(k, col_j, w_ij);
                    }
                } else {
                    // INLET or OUTLET: C_j is known (set by BCs), goes to RHS
                    my_bc.push_back({k, j, w_ij});
                }
            }

            // Diagonal entry
            my_triplets.emplace_back(k, k, diag);
        }
    }

    // Merge thread triplets
    size_t total_triplets = 0;
    for (auto& v : thread_triplets) total_triplets += v.size();
    triplets.reserve(total_triplets);
    for (auto& v : thread_triplets) {
        triplets.insert(triplets.end(), v.begin(), v.end());
    }

    M_.resize(n_unknowns_, n_unknowns_);
    M_.setFromTriplets(triplets.begin(), triplets.end());

    // Build BC neighbor CSR structure
    std::vector<BCInfo> all_bc;
    for (auto& v : thread_bc_info) {
        all_bc.insert(all_bc.end(), v.begin(), v.end());
    }
    std::sort(all_bc.begin(), all_bc.end(),
              [](const BCInfo& a, const BCInfo& b) { return a.local_k < b.local_k; });

    bc_nbr_offset_.assign(n_unknowns_ + 1, 0);
    bc_nbr_global_.clear();
    bc_nbr_weight_.clear();
    bc_nbr_global_.reserve(all_bc.size());
    bc_nbr_weight_.reserve(all_bc.size());

    for (const auto& e : all_bc) {
        bc_nbr_offset_[e.local_k + 1]++;
    }
    for (int k = 0; k < n_unknowns_; ++k) {
        bc_nbr_offset_[k + 1] += bc_nbr_offset_[k];
    }
    for (const auto& e : all_bc) {
        bc_nbr_global_.push_back(e.global_j);
        bc_nbr_weight_.push_back(e.weight);
    }

    t_asm.report();
}

// ============================================================================
// Smooth dissolution source and derivative
// ============================================================================

double PD_ARD_ImplicitSolver::source_val(double C, double base_rate, double C_sat) const {
    if (base_rate < 1e-30) return 0.0;
    // Linear ramp: 1 when C <= C_sat - eps, 0 when C >= C_sat
    double x = (C_sat - C) / eps_sat_;
    double factor = std::max(0.0, std::min(1.0, x));
    return base_rate * factor;
}

double PD_ARD_ImplicitSolver::source_deriv(double C, double base_rate, double C_sat) const {
    if (base_rate < 1e-30) return 0.0;
    double gap = C_sat - C;
    // Derivative is -base_rate/eps inside the ramp, zero outside
    if (gap > 0.0 && gap < eps_sat_) {
        return -base_rate / eps_sat_;
    }
    return 0.0;
}

// ============================================================================
// Precompute dissolution base rates (geometry-dependent, constant during Newton)
// ============================================================================

void PD_ARD_ImplicitSolver::compute_diss_base_rates(const Fields& fields,
                                                     const Grid& grid,
                                                     const Config& cfg) {
    diss_base_rate_.assign(n_unknowns_, 0.0);

    #pragma omp parallel for schedule(dynamic, 256)
    for (int k = 0; k < n_unknowns_; ++k) {
        int i = local_to_global_[k];

        int n_solid_nbr = 0;
        int n_total_nbr = 0;
        double diss_k_sum = 0.0;

        for (int jj = grid.nbr_offset[i]; jj < grid.nbr_offset[i + 1]; ++jj) {
            int j = grid.nbr_index[jj];
            if (grid.nbr_vol[jj] < 1e-30) continue;
            n_total_nbr++;

            if (grid.node_type[j] == SOLID_MG) {
                n_solid_nbr++;
                double k_j = cfg.k_corr;
                if (fields.is_gb[j]) k_j *= cfg.gb_corr_factor;
                diss_k_sum += k_j;
            }
        }

        if (n_solid_nbr > 0 && n_total_nbr > 0) {
            double avg_k = diss_k_sum / n_solid_nbr;
            double interface_frac = static_cast<double>(n_solid_nbr) /
                                    static_cast<double>(n_total_nbr);
            diss_base_rate_[k] = avg_k * interface_frac * (cfg.rho_m / cfg.rho_f);
        }
    }
}

// ============================================================================
// BC contribution (constant during Newton — inlet/outlet C values don't change)
// ============================================================================

void PD_ARD_ImplicitSolver::compute_bc_rhs(Eigen::VectorXd& bc_rhs,
                                            const Fields& fields) const {
    bc_rhs.setZero(n_unknowns_);
    for (int k = 0; k < n_unknowns_; ++k) {
        double bc_sum = 0.0;
        for (int p = bc_nbr_offset_[k]; p < bc_nbr_offset_[k + 1]; ++p) {
            bc_sum += bc_nbr_weight_[p] * fields.C[bc_nbr_global_[p]];
        }
        bc_rhs[k] = bc_sum;
    }
}

// ============================================================================
// Residual:  R(C) = C - C_old - dt * [ M*C + bc + source(C) ]
// ============================================================================

void PD_ARD_ImplicitSolver::compute_residual(Eigen::VectorXd& R,
                                              const Eigen::VectorXd& C,
                                              const Eigen::VectorXd& C_old,
                                              const Eigen::VectorXd& bc_rhs,
                                              double dt, const Config& cfg) const {
    Eigen::VectorXd MC = M_ * C;
    R.resize(n_unknowns_);

    for (int k = 0; k < n_unknowns_; ++k) {
        double src = source_val(C[k], diss_base_rate_[k], cfg.C_sat);
        R[k] = C[k] - C_old[k] - dt * (MC[k] + bc_rhs[k] + src);
    }
}

// ============================================================================
// Jacobian:  J = I - dt * ( M + diag(dsource/dC) )
// ============================================================================

void PD_ARD_ImplicitSolver::assemble_jacobian(Eigen::SparseMatrix<double>& J,
                                               const Eigen::VectorXd& C,
                                               double dt, const Config& cfg) const {
    // Start from the linear part: J = -dt * M
    J = M_;
    J *= -dt;

    // Add identity and source derivative to diagonal:
    // J_kk += 1.0 - dt * dsource_k/dC_k
    for (int k = 0; k < n_unknowns_; ++k) {
        double ds = source_deriv(C[k], diss_base_rate_[k], cfg.C_sat);
        J.coeffRef(k, k) += 1.0 - dt * ds;
    }
}

// ============================================================================
// Newton-Raphson step with GMRES inner solve
// ============================================================================

int PD_ARD_ImplicitSolver::step(Fields& fields, const Grid& grid,
                                 const Config& cfg, double dt) {
    // Precompute dissolution base rates (depends on solid geometry, not on C)
    compute_diss_base_rates(fields, grid, cfg);

    // Precompute BC contribution (constant during Newton)
    Eigen::VectorXd bc_rhs;
    compute_bc_rhs(bc_rhs, fields);

    // Extract current concentration vector
    Eigen::VectorXd C_old(n_unknowns_);
    for (int k = 0; k < n_unknowns_; ++k) {
        C_old[k] = fields.C[local_to_global_[k]];
    }

    // Initial guess: C = C_old
    Eigen::VectorXd C = C_old;

    // Compute initial residual
    Eigen::VectorXd R(n_unknowns_);
    compute_residual(R, C, C_old, bc_rhs, dt, cfg);
    double R0_norm = R.norm();

    // If already converged (e.g. steady state, or no source and zero BC)
    if (R0_norm < 1e-14) {
        std::printf("    Newton: converged at initial guess (|R|=%.2e)\n", R0_norm);
        // Still need to update solid nodes below
    }

    // GMRES solver with ILU preconditioner
    Eigen::GMRES<Eigen::SparseMatrix<double>, Eigen::IncompleteLUT<double>> gmres;
    gmres.setMaxIterations(200);
    gmres.setTolerance(1e-10);
    gmres.set_restart(50);

    Eigen::SparseMatrix<double> J;
    int newton_iter = 0;
    int total_gmres_iter = 0;

    for (; newton_iter < cfg.newton_max_iter && R0_norm > 1e-14; ++newton_iter) {
        double R_norm = R.norm();

        // Convergence check (relative + absolute floor)
        if (R_norm < cfg.newton_tol * R0_norm + 1e-14) {
            break;
        }

        // Assemble Jacobian J = I - dt*(M + diag(dsource/dC))
        assemble_jacobian(J, C, dt, cfg);

        // Solve J * delta = -R with GMRES + ILU preconditioner
        gmres.compute(J);
        if (gmres.info() != Eigen::Success) {
            std::fprintf(stderr, "WARNING: GMRES preconditioner setup failed at Newton iter %d\n",
                         newton_iter);
            break;
        }

        Eigen::VectorXd delta = gmres.solve(-R);
        total_gmres_iter += gmres.iterations();

        if (gmres.info() != Eigen::Success) {
            std::fprintf(stderr, "WARNING: GMRES did not converge at Newton iter %d "
                         "(%d iters, |res|=%.2e)\n",
                         newton_iter, (int)gmres.iterations(), gmres.error());
        }

        // Newton update
        C += delta;

        // Recompute residual (no clamping during Newton — the smooth source
        // function handles C near C_sat continuously, and clamping would
        // break Newton convergence by modifying the iterate inconsistently)
        compute_residual(R, C, C_old, bc_rhs, dt, cfg);
    }

    std::printf("    Newton: %d iter, |R|=%.2e -> %.2e, GMRES total=%d\n",
                newton_iter, R0_norm, R.norm(), total_gmres_iter);

    // Write solution back to fields
    for (int k = 0; k < n_unknowns_; ++k) {
        int i = local_to_global_[k];
        double val = C[k];
        if (val < 1e-30) val = 0.0;
        if (val > cfg.C_sat) val = cfg.C_sat;
        fields.C[i] = val;
    }

    // Advance solid nodes analytically: C -= k_eff * f_exposure * dt
    // (uses updated fluid C for salt-layer blocking check)
    int N = grid.N_total;
    #pragma omp parallel for schedule(dynamic, 256)
    for (int i = 0; i < N; ++i) {
        if (grid.node_type[i] != SOLID_MG) continue;

        int n_fluid_nbr = 0;
        int n_total_nbr = 0;
        bool salt_layer_blocked = false;

        for (int jj = grid.nbr_offset[i]; jj < grid.nbr_offset[i + 1]; ++jj) {
            int j = grid.nbr_index[jj];
            if (grid.nbr_vol[jj] < 1e-30) continue;
            n_total_nbr++;
            if (grid.node_type[j] == FLUID) {
                n_fluid_nbr++;
                if (fields.C[j] >= cfg.C_sat) salt_layer_blocked = true;
            }
        }

        double f_exposure = (n_total_nbr > 0) ?
            static_cast<double>(n_fluid_nbr) / static_cast<double>(n_total_nbr) : 0.0;

        double k_eff = cfg.k_corr;
        if (fields.is_gb[i]) k_eff *= cfg.gb_corr_factor;
        if (salt_layer_blocked) k_eff = 0.0;

        double C_new_val = fields.C[i] - dt * k_eff * f_exposure;
        if (C_new_val < 0.0) C_new_val = 0.0;
        fields.C[i] = C_new_val;
    }

    return newton_iter;
}

// ============================================================================
// Adaptive dt: fraction of time until fastest surface node reaches C_thresh
// ============================================================================

double PD_ARD_ImplicitSolver::compute_adaptive_dt(const Fields& fields,
                                                   const Grid& grid,
                                                   const Config& cfg) const {
    double min_t_phase = cfg.implicit_dt_max;
    int N = grid.N_total;

    #pragma omp parallel for reduction(min:min_t_phase) schedule(dynamic, 256)
    for (int i = 0; i < N; ++i) {
        if (grid.node_type[i] != SOLID_MG) continue;

        int n_fluid_nbr = 0;
        int n_total_nbr = 0;
        bool salt_layer_blocked = false;

        for (int jj = grid.nbr_offset[i]; jj < grid.nbr_offset[i + 1]; ++jj) {
            int j = grid.nbr_index[jj];
            if (grid.nbr_vol[jj] < 1e-30) continue;
            n_total_nbr++;
            if (grid.node_type[j] == FLUID) {
                n_fluid_nbr++;
                if (fields.C[j] >= cfg.C_sat) salt_layer_blocked = true;
            }
        }

        if (n_fluid_nbr == 0 || salt_layer_blocked) continue;

        double f_exposure = static_cast<double>(n_fluid_nbr) /
                            static_cast<double>(n_total_nbr);
        double k_eff = cfg.k_corr;
        if (fields.is_gb[i]) k_eff *= cfg.gb_corr_factor;

        double rate = k_eff * f_exposure;
        if (rate < 1e-30) continue;

        double t_phase = (fields.C[i] - cfg.C_thresh) / rate;
        if (t_phase > 0.0 && t_phase < min_t_phase) {
            min_t_phase = t_phase;
        }
    }

    double dt;
    if (min_t_phase < 2.0) {
        // Close to dissolution: step directly to the event instead of
        // fractioning (avoids Zeno's paradox of never reaching C_thresh)
        dt = min_t_phase;
    } else {
        dt = cfg.implicit_dt_fraction * min_t_phase;
    }
    dt = std::min(dt, cfg.implicit_dt_max);
    dt = std::max(dt, 1e-6);

    return dt;
}

// ============================================================================
// Phase change: dissolve solid nodes below C_thresh
// ============================================================================

int PD_ARD_ImplicitSolver::apply_phase_change(Fields& fields, Grid& grid,
                                               const Config& cfg) {
    int n_dissolved = 0;
    int N = grid.N_total;

    for (int i = 0; i < N; ++i) {
        if (fields.phase[i] == 0 && grid.node_type[i] == SOLID_MG) {
            if (fields.C[i] < cfg.C_thresh) {
                fields.phase[i] = 1;
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
