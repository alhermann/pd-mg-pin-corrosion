#pragma once
#include "grid.h"
#include "fields.h"
#include "config.h"

#include <Eigen/Sparse>
#include <Eigen/IterativeLinearSolvers>
#include <unsupported/Eigen/IterativeSolvers>
#include <vector>

class PD_ARD_ImplicitSolver {
public:
    void init(const Grid& grid, const Config& cfg);

    // Build sparse matrix M from PD transport operators (call once per coupling cycle)
    void assemble(const Fields& fields, const Grid& grid, const Config& cfg);

    // Solve one implicit step: (I - dt*M)*C_new = C_old + dt*bc_rhs.
    // The system is linear (no source term), so GMRES solves directly.
    // Returns 1 (single linear solve).
    int step(Fields& fields, const Grid& grid, const Config& cfg, double dt);

    // Compute adaptive dt from fastest-dissolving solid node (PD flux based)
    double compute_adaptive_dt(const Fields& fields, const Grid& grid,
                               const Config& cfg) const;

    // Check and apply phase changes. Returns number of newly dissolved nodes.
    int apply_phase_change(Fields& fields, Grid& grid, const Config& cfg);

private:
    // PD coefficients
    double alpha_p_;
    double V_H_;
    double beta_coeff_;

    // Global-to-local index map: global node index -> local unknown index
    // (-1 if not an unknown, i.e. not FLUID or SOLID_MG)
    std::vector<int> global_to_local_;
    // Local-to-global: local unknown index -> global node index
    std::vector<int> local_to_global_;
    int n_unknowns_ = 0;

    // Sparse PD operator matrix M (diffusion + advection, linear in C, frozen per cycle)
    Eigen::SparseMatrix<double> M_;

    // BC neighbor CSR: for each unknown, boundary neighbors and their weights
    std::vector<int> bc_nbr_offset_;
    std::vector<int> bc_nbr_global_;
    std::vector<double> bc_nbr_weight_;

    // Per-solid-node salt-layer blocking flag (indexed by global node index)
    // True if a solid node has ANY liquid neighbor with C >= C_sat
    std::vector<bool> salt_blocked_;

    void build_index_map(const Grid& grid);

    // Precompute constant BC contribution: sum_j(w_kj * C_j) for INLET/OUTLET neighbors
    void compute_bc_rhs(Eigen::VectorXd& bc_rhs, const Fields& fields) const;

    // Precompute salt-layer blocking flags for solid nodes
    void compute_salt_blocked(const Fields& fields, const Grid& grid, const Config& cfg);
};
