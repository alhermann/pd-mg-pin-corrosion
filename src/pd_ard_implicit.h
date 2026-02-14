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

    // Solve one implicit step with Newton-Raphson + GMRES. Returns Newton iterations used.
    int step(Fields& fields, const Grid& grid, const Config& cfg, double dt);

    // Compute adaptive dt from fastest-dissolving solid node
    double compute_adaptive_dt(const Fields& fields, const Grid& grid,
                               const Config& cfg) const;

    // Check and apply phase changes. Returns number of newly dissolved nodes.
    int apply_phase_change(Fields& fields, Grid& grid, const Config& cfg);

private:
    // PD coefficients
    double alpha_p_;
    double V_H_;
    double beta_coeff_;

    // Global-to-local index map: global node index -> local unknown index (-1 if not fluid)
    std::vector<int> global_to_local_;
    // Local-to-global: local unknown index -> global node index
    std::vector<int> local_to_global_;
    int n_unknowns_ = 0;

    // Sparse PD operator matrix M (diffusion + advection, linear in C, frozen per cycle)
    Eigen::SparseMatrix<double> M_;

    // BC neighbor CSR: for each fluid unknown, boundary neighbors and their weights
    std::vector<int> bc_nbr_offset_;
    std::vector<int> bc_nbr_global_;
    std::vector<double> bc_nbr_weight_;

    // Per-unknown dissolution base rate (geometry-dependent, constant during Newton)
    // base_rate_k = avg_k * interface_frac * (rho_m / rho_f) for nodes at solid interface,
    // zero otherwise.
    std::vector<double> diss_base_rate_;

    // Smoothing width for salt-layer cutoff: source ramps from base_rate to 0
    // over C in [C_sat - eps_sat_, C_sat] instead of a hard Heaviside step.
    double eps_sat_;

    void build_index_map(const Grid& grid);

    // Precompute dissolution base rates from solid neighbor geometry
    void compute_diss_base_rates(const Fields& fields, const Grid& grid,
                                 const Config& cfg);

    // Smooth dissolution source and its C-derivative (for Newton Jacobian)
    //   source(C) = base_rate * clamp((C_sat - C) / eps, 0, 1)
    //   dsource/dC = -base_rate / eps   when C in (C_sat - eps, C_sat)
    double source_val(double C, double base_rate, double C_sat) const;
    double source_deriv(double C, double base_rate, double C_sat) const;

    // Precompute constant BC contribution: sum_j(w_kj * C_j) for INLET/OUTLET neighbors
    void compute_bc_rhs(Eigen::VectorXd& bc_rhs, const Fields& fields) const;

    // Compute residual R(C) = C - C_old - dt * (M*C + bc_rhs + source(C))
    void compute_residual(Eigen::VectorXd& R, const Eigen::VectorXd& C,
                          const Eigen::VectorXd& C_old,
                          const Eigen::VectorXd& bc_rhs,
                          double dt, const Config& cfg) const;

    // Assemble Jacobian J = I - dt * (M + diag(dsource/dC))
    void assemble_jacobian(Eigen::SparseMatrix<double>& J, const Eigen::VectorXd& C,
                           double dt, const Config& cfg) const;
};
