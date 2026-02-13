#pragma once
#include "grid.h"
#include "fields.h"
#include "config.h"

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
    double V_H_;        // horizon volume
    double beta_lap_;   // PD Laplacian coefficient = 4/(pi*delta^2) [2D]

    void compute_pressure(Fields& f, const Config& cfg);
};
