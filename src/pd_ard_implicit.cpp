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

    // AMR: precompute per-node PD constants
    if (cfg.use_amr) {
        use_amr_ = true;
        V_H_node_.resize(grid.N_total);
        beta_node_.resize(grid.N_total);
        for (int i = 0; i < grid.N_total; ++i) {
            double d = grid.delta_local[i];
            if constexpr (DIM == 2) {
                V_H_node_[i] = PI * d * d;
                beta_node_[i] = 4.0 / (PI * d * d);
            } else {
                V_H_node_[i] = (4.0 / 3.0) * PI * d * d * d;
                beta_node_[i] = 12.0 / (PI * d * d);
            }
        }
    }
}

// ============================================================================
// Index map: FLUID and SOLID_MG nodes <-> local unknowns
// ============================================================================

void PD_ARD_ImplicitSolver::build_index_map(const Grid& grid) {
    int N = grid.N_total;
    global_to_local_.assign(N, -1);
    local_to_global_.clear();
    local_to_global_.reserve(N / 2);

    int idx = 0;
    for (int i = 0; i < N; ++i) {
        if (grid.node_type[i] == FLUID || grid.node_type[i] == SOLID_MG ||
            grid.node_type[i] == FICTITIOUS) {
            global_to_local_[i] = idx;
            local_to_global_.push_back(i);
            idx++;
        }
    }
    n_unknowns_ = idx;
}

// ============================================================================
// Precompute salt-layer blocking flags for solid nodes
// (Jafarzadeh et al. 2018): if a solid node has ANY liquid neighbor with
// C >= C_sat, all its interface bonds are disabled (dissolution blocked).
// ============================================================================

void PD_ARD_ImplicitSolver::compute_salt_blocked(const Fields& fields,
                                                   const Grid& grid,
                                                   const Config& cfg) {
    int N = grid.N_total;
    salt_blocked_.assign(N, false);

    #pragma omp parallel for schedule(dynamic, 256)
    for (int i = 0; i < N; ++i) {
        if (grid.node_type[i] != SOLID_MG) continue;

        for (int jj = grid.nbr_offset[i]; jj < grid.nbr_offset[i + 1]; ++jj) {
            int j = grid.nbr_index[jj];
            if (grid.nbr_vol[jj] < 1e-30) continue;
            if (grid.node_type[j] == FLUID && fields.C[j] >= cfg.C_sat) {
                salt_blocked_[i] = true;
                break;
            }
        }
    }
}

// ============================================================================
// Assemble bi-material PD operator matrix M (once per coupling cycle)
//
// Bond micro-diffusivities (Jafarzadeh, Chen & Bobaru 2018):
//   Liquid-Liquid:  D_avg = D_liquid
//   Solid-Solid:    skipped (no diffusion within bulk solid)
//   Interface:      D_avg = harmonic_mean(D_liquid, D_solid)
//                   where D_solid = D_gb if solid node is GB, else D_grain
//
// Salt-layer blocking: if a solid node is salt-blocked, all its bonds
// get D_avg = 0 (dissolution disabled for that node).
//
// Advection: only applied when node i is FLUID (solid nodes have v=0).
// ============================================================================

void PD_ARD_ImplicitSolver::assemble(const Fields& fields, const Grid& grid,
                                      const Config& cfg) {
    Timer t_asm("implicit_assemble");

    build_index_map(grid);

    // Count unknowns by type for logging
    int n_fluid = 0, n_solid = 0, n_fict = 0;
    for (int k = 0; k < n_unknowns_; ++k) {
        NodeType nt = grid.node_type[local_to_global_[k]];
        if (nt == FLUID) n_fluid++;
        else if (nt == SOLID_MG) n_solid++;
        else if (nt == FICTITIOUS) n_fict++;
    }
    std::printf("  Implicit: %d unknowns (%d fluid + %d solid + %d fictitious)\n",
                n_unknowns_, n_fluid, n_solid, n_fict);

    // Precompute salt-layer blocking
    compute_salt_blocked(fields, grid, cfg);

    // Volume-loss-dependent decay factor (Hermann et al. 2022, Eq. 42):
    // decay = 10^(-V_L / l), reduces interface micro-diffusivity as corrosion progresses
    double decay_factor = 1.0;
    if (cfg.corrosion_decay_l > 0.0) {
        decay_factor = std::pow(10.0, -volume_loss_fraction_ / cfg.corrosion_decay_l);
        std::printf("  Interface decay: V_L=%.4f, l=%.3f, factor=%.6f\n",
                    volume_loss_fraction_, cfg.corrosion_decay_l, decay_factor);
    }

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
            NodeType nt_i = grid.node_type[i];

            // FICTITIOUS nodes: zero M row (coupling added after A = I - dt*M)
            // Pre-allocate sparsity pattern for IDW coupling entries
            if (nt_i == FICTITIOUS) {
                my_triplets.emplace_back(k, k, 0.0);
                for (int p = grid.fict_offset[i]; p < grid.fict_offset[i + 1]; ++p) {
                    int j = grid.fict_source[p];
                    int col = global_to_local_[j];
                    if (col >= 0) {
                        my_triplets.emplace_back(k, col, 0.0);
                    }
                }
                continue;
            }

            bool i_is_fluid = (nt_i == FLUID);
            bool i_is_solid = (nt_i == SOLID_MG);
            bool i_is_gb = fields.is_gb[i];
            bool i_salt_blocked = i_is_solid && salt_blocked_[i];

            // Solid micro-diffusivity for node i: GB > precipitate > grain
            double D_solid_i = i_is_gb ? cfg.D_gb
                             : (fields.is_precip[i] ? cfg.D_precip : cfg.D_grain);
            (void)D_solid_i;

            // Per-node PD constants
            double beta_i = use_amr_ ? beta_node_[i] : beta_coeff_;
            double V_H_i = use_amr_ ? V_H_node_[i] : V_H_;
            double div_coeff = alpha_p_ / V_H_i;

            // Velocity (only fluid nodes have nonzero velocity)
            Vec vel_i = i_is_fluid ? fields.vel[i] : vec_zero();

            double diag = 0.0;

            for (int jj = grid.nbr_offset[i]; jj < grid.nbr_offset[i + 1]; ++jj) {
                int j = grid.nbr_index[jj];
                double xi = grid.nbr_dist[jj];
                Vec e_ij = grid.nbr_evec[jj];
                double V_j = grid.nbr_vol[jj];

                if (V_j < 1e-30) continue;

                NodeType nt_j = grid.node_type[j];

                // Skip WALL and OUTSIDE — no transport bonds
                if (nt_j == WALL || nt_j == OUTSIDE) continue;

                double inv_xi = 1.0 / xi;
                double inv_xi2 = inv_xi * inv_xi;

                // Bond classification:
                //   Liquid-Liquid:  D_liquid + advection
                //   Interface:      harmonic_mean(D_liquid, D_solid) + advection on fluid side
                //   Solid-Solid:    skip (no transport within solid bulk)
                //   FICTITIOUS neighbors: treated like FLUID for bond classification
                //   (valid when amr_buffer > delta_c, ensuring all aux nodes are in fluid region)
                bool j_is_fluid = (nt_j == FLUID || nt_j == INLET || nt_j == OUTLET || nt_j == FICTITIOUS);
                bool j_is_solid = (nt_j == SOLID_MG);

                // Skip solid-solid bonds (no diffusion within bulk solid)
                if (i_is_solid && j_is_solid) continue;

                double D_avg = 0.0;
                if (i_is_fluid && j_is_fluid) {
                    // Liquid-Liquid bond (including FICTITIOUS)
                    D_avg = cfg.D_liquid;
                } else if ((i_is_fluid || nt_i == FICTITIOUS) && j_is_solid) {
                    // Interface bond: fluid/fictitious i, solid j
                    bool solid_is_gb = fields.is_gb[j];
                    bool solid_salt_blocked = salt_blocked_[j];
                    if (solid_salt_blocked) {
                        D_avg = 0.0;
                    } else {
                        double D_s = solid_is_gb ? cfg.D_gb
                                   : (fields.is_precip[j] ? cfg.D_precip : cfg.D_grain);
                        D_s *= decay_factor;  // volume-loss decay (Eq. 42)
                        D_avg = 2.0 * cfg.D_liquid * D_s / (cfg.D_liquid + D_s + 1e-30);
                    }
                } else if (i_is_solid && j_is_fluid) {
                    // Interface bond: solid i, fluid/fictitious j
                    bool solid_is_gb = i_is_gb;
                    bool solid_salt_blocked = i_salt_blocked;
                    if (solid_salt_blocked) {
                        D_avg = 0.0;
                    } else {
                        double D_s = solid_is_gb ? cfg.D_gb
                                   : (fields.is_precip[i] ? cfg.D_precip : cfg.D_grain);
                        D_s *= decay_factor;  // volume-loss decay (Eq. 42)
                        D_avg = 2.0 * cfg.D_liquid * D_s / (cfg.D_liquid + D_s + 1e-30);
                    }
                }

                // ---------------------------------------------------------
                // Diffusion + non-conservative advection with M-matrix
                // upwind stabilization (gradient form)
                //
                // Diffusion:  β * D / ξ² * (C_j - C_i) * V_j
                //
                // Advection (gradient/non-conservative form):
                //   (α/V_H) * (v_i · e_ij) * (C_j - C_i) / ξ * V_j
                //
                // Upwind stabilization (per-bond, anisotropic):
                //   For downwind bonds (v·e > 0) where the advection coeff
                //   exceeds diffusion, add minimum artificial diffusion in
                //   the bond direction to keep off-diagonal ≥ 0.
                //   - Upwind bonds (v·e < 0): untouched (already positive)
                //   - Cross-flow bonds (v·e ≈ 0): untouched (pure diffusion)
                //   This gives monotone solutions at high Pe with artificial
                //   diffusion ONLY in the flow direction.
                // ---------------------------------------------------------

                // Diffusion contribution (all bond types)
                double w_diff = beta_i * D_avg * inv_xi2 * V_j;
                double w_ij      = w_diff;   // off-diagonal (C_j coefficient)
                double w_ii_bond = w_diff;   // diagonal contribution (C_i)

                // Advection on liquid-liquid bonds only
                if (i_is_fluid && j_is_fluid) {
                    double v_dot_e = dot(vel_i, e_ij);
                    double w_adv = div_coeff * v_dot_e * inv_xi * V_j;

                    // Per-bond upwind stabilization: clamp off-diagonal ≥ 0
                    double w_stab = std::max(0.0, w_adv - w_diff);

                    w_ij      = (w_diff + w_stab) - w_adv;  // ≥ 0 guaranteed
                    w_ii_bond = (w_diff + w_stab) - w_adv;  // symmetric per bond
                }

                diag -= w_ii_bond;

                // Off-diagonal entry
                if (nt_j == FLUID || nt_j == SOLID_MG || nt_j == FICTITIOUS) {
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
// BC contribution (constant during the step — inlet/outlet C values don't change)
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
// Implicit step: solve (I - dt*M) * C_new = C_old + dt * bc_rhs
//
// The bi-material PD system is purely linear (no source term).
// A single GMRES solve replaces the old Newton iteration.
// ============================================================================

int PD_ARD_ImplicitSolver::step(Fields& fields, const Grid& grid,
                                 const Config& cfg, double dt) {
    // Precompute BC contribution
    Eigen::VectorXd bc_rhs;
    compute_bc_rhs(bc_rhs, fields);

    // Extract current concentration vector
    Eigen::VectorXd C_old(n_unknowns_);
    for (int k = 0; k < n_unknowns_; ++k) {
        C_old[k] = fields.C[local_to_global_[k]];
    }

    // Build system matrix: A = I - dt * M
    Eigen::SparseMatrix<double> A = M_;
    A *= -dt;
    for (int k = 0; k < n_unknowns_; ++k) {
        A.coeffRef(k, k) += 1.0;
    }

    // RHS: b = C_old + dt * bc_rhs
    Eigen::VectorXd b = C_old + dt * bc_rhs;

    // Apply fictitious node coupling (overwrite FICTITIOUS rows in A and b)
    if (use_amr_) {
        apply_fictitious_coupling(A, b, grid, fields);
    }

    // Solve A * C_new = b with GMRES + ILU preconditioner
    Eigen::GMRES<Eigen::SparseMatrix<double>, Eigen::IncompleteLUT<double>> gmres;
    gmres.setMaxIterations(200);
    gmres.setTolerance(1e-10);
    gmres.set_restart(50);

    gmres.compute(A);
    if (gmres.info() != Eigen::Success) {
        std::fprintf(stderr, "WARNING: GMRES preconditioner setup failed\n");
    }

    Eigen::VectorXd C_new = gmres.solve(b);

    if (gmres.info() != Eigen::Success) {
        std::fprintf(stderr, "WARNING: GMRES did not converge (%d iters, |res|=%.2e)\n",
                     (int)gmres.iterations(), gmres.error());
    }

    std::printf("    Linear solve: GMRES %d iters, |res|=%.2e\n",
                (int)gmres.iterations(), gmres.error());

    // Write solution back to fields with physical clamping [0, C_solid_init]
    for (int k = 0; k < n_unknowns_; ++k) {
        int i = local_to_global_[k];
        double val = C_new[k];
        if (val < 0.0) val = 0.0;
        if (val > cfg.C_solid_init) val = cfg.C_solid_init;
        fields.C[i] = val;
    }

    return 1; // single linear solve (replaces Newton iteration count)
}

// ============================================================================
// Adaptive dt: based on PD diffusion flux at solid interface nodes
//
// For each solid node at the interface, estimate the dissolution rate from
// the assembled M matrix row, then compute time to reach C_thresh.
// ============================================================================

double PD_ARD_ImplicitSolver::compute_adaptive_dt(const Fields& fields,
                                                   const Grid& grid,
                                                   const Config& cfg) const {
    double min_t_phase = cfg.implicit_dt_max;
    int N = grid.N_total;

    #pragma omp parallel for reduction(min:min_t_phase) schedule(dynamic, 256)
    for (int i = 0; i < N; ++i) {
        if (grid.node_type[i] != SOLID_MG) continue;
        if (fields.C[i] <= cfg.C_thresh) continue;

        int k = global_to_local_[i];
        if (k < 0) continue;

        // Compute dC_i/dt from the M matrix row: dC/dt = sum_j M[k,j]*C[j]
        // This includes all bonds (solid-solid and interface)
        double dCdt = 0.0;

        // Sparse row iteration
        for (Eigen::SparseMatrix<double>::InnerIterator it(M_, k); it; ++it) {
            int col = it.col();
            int g = local_to_global_[col];
            dCdt += it.value() * fields.C[g];
        }

        // Also add BC contribution
        for (int p = bc_nbr_offset_[k]; p < bc_nbr_offset_[k + 1]; ++p) {
            dCdt += bc_nbr_weight_[p] * fields.C[bc_nbr_global_[p]];
        }

        // dCdt should be negative for dissolving nodes (C flowing out to liquid)
        // rate = |dC/dt| when dC/dt < 0
        if (dCdt >= 0.0) continue;

        double rate = -dCdt;
        if (rate < 1e-30) continue;

        double t_phase = (fields.C[i] - cfg.C_thresh) / rate;
        if (t_phase > 0.0 && t_phase < min_t_phase) {
            min_t_phase = t_phase;
        }
    }

    double dt = cfg.implicit_dt_fraction * min_t_phase;
    dt = std::min(dt, cfg.implicit_dt_max);
    // Floor: at least 1% of dt_max so the simulation makes progress.
    // Backward Euler is unconditionally stable, so larger dt is safe.
    dt = std::max(dt, cfg.implicit_dt_max * 0.01);

    return dt;
}

// ============================================================================
// Phase change: dissolve solid nodes below C_thresh
// ============================================================================

// ============================================================================
// Fictitious node coupling: overwrite FICTITIOUS rows with IDW constraint
// C_fict = sum(w_j * C_source_j) => row: A[k,k]=1, A[k,source_j]=-w_j, b[k]=0
// ============================================================================

void PD_ARD_ImplicitSolver::apply_fictitious_coupling(
    Eigen::SparseMatrix<double>& A, Eigen::VectorXd& b,
    const Grid& grid, const Fields& fields)
{
    if (grid.fict_offset.empty()) return;

    for (int i = 0; i < grid.N_total; ++i) {
        if (grid.node_type[i] != FICTITIOUS) continue;
        int k = global_to_local_[i];
        if (k < 0) continue;

        // Row k of A is already [0,...,0,1,0,...,0] from A = I - dt*M
        // (FICTITIOUS rows in M are zero). The sparsity pattern was pre-allocated
        // in assemble() with zero-valued IDW entries. Just set the values.
        //
        // Constraint: C_fict = sum(w_j * C_source_j)
        // => A[k,k]*C_k + sum_j A[k,col_j]*C_j = b[k]
        // => 1*C_k - sum(w_j * C_j for unknowns) = sum(w_j * C_j for BCs)

        A.coeffRef(k, k) = 1.0;
        double bc_sum = 0.0;

        for (int p = grid.fict_offset[i]; p < grid.fict_offset[i + 1]; ++p) {
            int j = grid.fict_source[p];
            int col = global_to_local_[j];
            if (col >= 0) {
                A.coeffRef(k, col) = -grid.fict_weight[p];
            } else {
                // BC node (INLET/OUTLET/WALL): contribution goes to RHS
                bc_sum += grid.fict_weight[p] * fields.C[j];
            }
        }

        b[k] = bc_sum;
    }
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
