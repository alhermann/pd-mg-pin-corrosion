#pragma once
#include "grid.h"
#include "fields.h"
#include "config.h"
#include <vector>

class PD_NS_Solver {
public:
    void init(const Grid& grid, const Config& cfg);

    // One flow timestep
    void step(Fields& fields, const Grid& grid, const Config& cfg, double dt);

    // Solve to steady state, return number of iterations
    int solve_steady(Fields& fields, const Grid& grid, const Config& cfg);

    double compute_dt(const Fields& fields, const Grid& grid, const Config& cfg);

private:
    double alpha_;      // = DIM (for divergence/gradient)
    double V_H_;        // horizon volume (uniform grid)
    double beta_lap_;   // PD Laplacian coefficient (uniform grid)

    // Per-node PD constants (only for AMR)
    bool use_amr_ = false;
    std::vector<double> V_H_node_;
    std::vector<double> beta_lap_node_;

    void compute_pressure(Fields& f, const Config& cfg);
};
