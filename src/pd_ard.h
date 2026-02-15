#pragma once
#include "grid.h"
#include "fields.h"
#include "config.h"
#include <vector>

class PD_ARD_Solver {
public:
    void init(const Grid& grid, const Config& cfg);

    // Set current volume loss fraction [0,1] for decay model
    void set_volume_loss(double vl) { volume_loss_fraction_ = vl; }

    // One transport timestep
    void step(Fields& fields, const Grid& grid, const Config& cfg, double dt);

    double compute_dt(const Fields& fields, const Grid& grid, const Config& cfg);

    // Check and apply phase changes. Returns number of newly dissolved nodes.
    int apply_phase_change(Fields& fields, Grid& grid, const Config& cfg);

private:
    double alpha_p_;
    double V_H_;
    double beta_coeff_; // diffusion PD coefficient
    double volume_loss_fraction_ = 0.0;

    // Per-node PD constants (only for AMR)
    bool use_amr_ = false;
    std::vector<double> V_H_node_;
    std::vector<double> beta_node_;
};
