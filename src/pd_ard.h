#pragma once
#include "grid.h"
#include "fields.h"
#include "config.h"

class PD_ARD_Solver {
public:
    void init(const Grid& grid, const Config& cfg);

    // One transport timestep
    void step(Fields& fields, const Grid& grid, const Config& cfg, double dt);

    double compute_dt(const Fields& fields, const Grid& grid, const Config& cfg);

    // Check and apply phase changes. Returns number of newly dissolved nodes.
    int apply_phase_change(Fields& fields, Grid& grid, const Config& cfg);

private:
    double alpha_p_;
    double V_H_;
    double beta_coeff_; // diffusion PD coefficient
};
