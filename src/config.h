#pragma once
#include <string>

struct Config {
    // Grid
    double dx = 5.0e-6;
    int m_ratio = 3;

    // Geometry [m]
    double R_wire = 40.0e-6;
    double L_wire = 400.0e-6;
    double R_tube = 150.0e-6;
    double L_upstream = 80.0e-6;
    double L_downstream = 80.0e-6;

    // Fluid
    double rho_f = 1000.0;
    double mu_f = 1.0e-3;
    double gamma_eos = 7.0;
    double c0 = 0.5;
    double eta_density = 0.1;

    // Flow
    double Q_flow = 1.667e-8;

    // Mg solid
    double rho_m = 1738.0;

    // Transport
    double D_liquid = 1.0e-9;
    double D_grain = 5.0e-11;
    double D_gb = 5.0e-9;
    double C_solid_init = 1.0;
    double C_liquid_init = 0.0;
    double C_thresh = 0.2;
    double w_advect = 0.8;
    double alpha_art_diff = 0.1;  // artificial diffusion: D_art = alpha * |v| * dx

    // Grain structure
    double grain_size_mean = 40.0e-6;
    double grain_size_std = 5.0e-6;
    int gb_width_cells = 1;

    // Corrosion kinetics
    double k_corr = 1.0e-3;       // surface corrosion rate [1/s]
    double gb_corr_factor = 3.0;   // grain boundary corrosion enhancement factor

    // Time stepping
    double cfl_factor = 0.25;       // CFL for flow solver
    double cfl_factor_corr = 0.25;  // CFL for corrosion/ARD solver (can be larger)

    // Coupling
    int flow_max_iters = 50000;
    double flow_conv_tol = 5.0e-6;
    double T_final = 32400.0;
    int corrosion_steps_per_check = 200;
    int output_every_flow = 2000;
    int output_every_corr = 100;
    std::string output_dir = "output";

    // Derived (computed after load)
    double delta;       // horizon = m_ratio * dx
    double U_in;        // inlet velocity from Q_flow and geometry

    void load(const std::string& filename);
    void compute_derived();
    void print() const;
};
