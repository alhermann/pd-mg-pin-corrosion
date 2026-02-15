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

    // Transport — bi-material PD diffusion model (Jafarzadeh, Chen & Bobaru 2018)
    // D_liquid: diffusivity for liquid-liquid bonds (Mg²⁺ in water)
    // D_grain:  micro-diffusivity for grain interior interface bonds
    // D_gb:     micro-diffusivity for grain boundary interface bonds
    // Interface bonds use harmonic mean: d = 2·d_S·d_L / (d_S + d_L)
    // Solid-solid bonds are skipped (no diffusion within bulk solid)
    double D_liquid = 1.0e-9;
    double D_grain = 5.0e-11;
    double D_gb = 5.0e-9;
    double D_precip = 5.0e-15;      // precipitate micro-diffusivity
    double precip_fraction = 0.05;   // fraction of grain interior nodes that are precipitates
    double C_solid_init = 1.0;
    double C_liquid_init = 0.0;
    double C_thresh = 0.2;
    double C_sat = 0.9;           // salt layer threshold: block dissolution when fluid C >= C_sat
    double alpha_art_diff = 0.1;  // artificial diffusion: D_art = alpha * |v| * dx

    // Volume-loss-dependent micro-diffusivity (Hermann et al. 2022, Eq. 42):
    // D_interface = D_base * 10^(-V_L / corrosion_decay_l)
    // where V_L = normalized volume loss [0,1], l = corrosion_decay_l
    // Set to 0 to disable (constant micro-diffusivity).
    double corrosion_decay_l = 0.0;

    // Grain structure
    double grain_size_mean = 40.0e-6;
    double grain_size_std = 5.0e-6;
    int gb_width_cells = 1;
    int precip_cluster_cells = 0;  // precipitate cluster radius in cells (0=single node)

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

    // Implicit ARD solver
    int use_implicit = 1;               // 1=implicit, 0=explicit (fallback)
    double implicit_dt_fraction = 0.5;  // fraction of min(t_phase) for adaptive dt
    double implicit_dt_max = 60.0;      // maximum implicit dt [s]
    int implicit_output_every = 10;     // VTI output every N implicit steps
    int diagnostic_every = 1;            // diagnostics CSV every N implicit steps

    // Newton-Raphson (inside implicit solver)
    double newton_tol = 1.0e-8;         // relative residual convergence tolerance
    int newton_max_iter = 20;           // max Newton iterations per implicit step

    // AMR (Adaptive Mesh Refinement)
    int use_amr = 0;              // 0=uniform, 1=two-level AMR
    int amr_ratio = 3;            // dx_coarse = amr_ratio * dx (fine)
    double amr_buffer = 50.0e-6;  // fine zone extends this far beyond wire surface [m]

    // Derived (computed after load)
    double delta;       // horizon = m_ratio * dx
    double U_in;        // inlet velocity from Q_flow and geometry
    double dx_coarse;   // = amr_ratio * dx
    double delta_coarse; // = m_ratio * dx_coarse

    void load(const std::string& filename);
    void compute_derived();
    void print() const;
};
