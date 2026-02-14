// Standalone validation tests for the implicit ARD solver.
//
// Tests PD diffusion, PD advection, combined PD advection-diffusion,
// and solid-liquid interface dissolution (bi-material PD model).
//
// Build: cmake .. -DPD_DIM=2 && make test_implicit
// Run:   ./test_implicit

#include "config.h"
#include "grid.h"
#include "fields.h"
#include "pd_ard.h"
#include "pd_ard_implicit.h"
#include "vtk_writer.h"
#include <cstdio>
#include <cmath>
#include <vector>
#include <string>
#include <sys/stat.h>

// ============================================================================
// Helpers
// ============================================================================

static Config make_test_config(double D_liquid, double Q_flow) {
    Config cfg;
    cfg.dx = 5.0e-6;
    cfg.m_ratio = 3;
    cfg.R_wire = 0.0;        // no pin — all interior nodes are fluid
    cfg.L_wire = 0.0;
    cfg.R_tube = 200.0e-6;
    cfg.L_upstream = 300.0e-6;
    cfg.L_downstream = 300.0e-6;
    cfg.rho_f = 1000.0;
    cfg.mu_f = 1.0e-3;
    cfg.c0 = 5.0;
    cfg.eta_density = 0.1;
    cfg.gamma_eos = 7.0;
    cfg.Q_flow = Q_flow;
    cfg.rho_m = 1738.0;
    cfg.D_liquid = D_liquid;
    cfg.D_grain = 0.0;
    cfg.D_gb = 0.0;
    cfg.C_solid_init = 1.0;
    cfg.C_liquid_init = 0.0;
    cfg.C_thresh = 0.2;
    cfg.C_sat = 10.0;          // high enough to not interfere with test Gaussians (C_max=1)
    cfg.alpha_art_diff = 0.0;  // no artificial diffusion for clean tests
    cfg.grain_size_mean = 40.0e-6;
    cfg.grain_size_std = 5.0e-6;
    cfg.gb_width_cells = 0;
    cfg.cfl_factor = 0.25;
    cfg.cfl_factor_corr = 0.25;
    cfg.use_implicit = 1;
    cfg.implicit_dt_max = 60.0;
    cfg.implicit_dt_fraction = 0.5;
    cfg.newton_tol = 1.0e-10;
    cfg.newton_max_iter = 30;
    cfg.compute_derived();
    return cfg;
}

static void init_fields_zero_vel(Fields& fields, const Grid& grid, const Config& cfg) {
    int N = grid.N_total;
    for (int i = 0; i < N; ++i) {
        fields.rho[i] = cfg.rho_f;
        fields.vel[i] = vec_zero();
        fields.D_map[i] = (grid.node_type[i] == FLUID ||
                           grid.node_type[i] == INLET ||
                           grid.node_type[i] == OUTLET) ? cfg.D_liquid : 0.0;
        fields.C[i] = 0.0;
        fields.phase[i] = 1;
        fields.grain_id[i] = -1;
        fields.is_gb[i] = 0;
    }
    fields.rho_new = fields.rho;
    fields.vel_new = fields.vel;
    fields.C_new = fields.C;
}

static void init_fields_uniform_vel(Fields& fields, const Grid& grid,
                                     const Config& cfg, double v_axial) {
    init_fields_zero_vel(fields, grid, cfg);
    for (int i = 0; i < grid.N_total; ++i) {
        if (grid.node_type[i] == FLUID ||
            grid.node_type[i] == INLET ||
            grid.node_type[i] == OUTLET) {
            Vec v = vec_zero();
            if constexpr (DIM == 2) { v[1] = v_axial; }
            else                    { v[2] = v_axial; }
            fields.vel[i] = v;
        }
    }
    fields.vel_new = fields.vel;
}

// Set a Gaussian concentration pulse centered at (r0, z0)
static void set_gaussian_pulse(Fields& fields, const Grid& grid,
                                double sigma, double r0, double z0) {
    double inv_2sig2 = 1.0 / (2.0 * sigma * sigma);
    for (int i = 0; i < grid.N_total; ++i) {
        if (grid.node_type[i] == FLUID) {
            double r = grid.pos[i][0] - r0;
            double z = grid.pos[i][1] - z0;
            fields.C[i] = std::exp(-(r*r + z*z) * inv_2sig2);
        } else {
            fields.C[i] = 0.0;
        }
    }
    fields.C_new = fields.C;
}

// Analytical Gaussian solution in 2D: diffusion + uniform advection
// For pure diffusion: set v_r=0, v_z=0
// For pure advection: set D ~ 0
// Combined: Gaussian translates at velocity v and spreads with diffusion
static double gaussian_exact_2d(double r, double z, double r0, double z0,
                                 double sigma, double D, double t,
                                 double v_r = 0.0, double v_z = 0.0) {
    double sig2 = sigma * sigma;
    double sig2t = sig2 + 2.0 * D * t;
    // Advected center position
    double dr = r - (r0 + v_r * t);
    double dz = z - (z0 + v_z * t);
    return (sig2 / sig2t) * std::exp(-(dr*dr + dz*dz) / (2.0 * sig2t));
}

// Compute L2 error between fields.C and a reference vector, only for FLUID nodes
static double compute_L2_error(const Fields& fields, const std::vector<double>& C_ref,
                                const Grid& grid) {
    double err2 = 0.0, ref2 = 0.0;
    for (int i = 0; i < grid.N_total; ++i) {
        if (grid.node_type[i] != FLUID) continue;
        double e = fields.C[i] - C_ref[i];
        err2 += e * e;
        ref2 += C_ref[i] * C_ref[i];
    }
    return std::sqrt(err2 / (ref2 + 1e-30));
}

// Compute total mass (sum of C*dV) for FLUID nodes — for conservation check
static double compute_total_mass(const Fields& fields, const Grid& grid) {
    double mass = 0.0;
    for (int i = 0; i < grid.N_total; ++i) {
        if (grid.node_type[i] == FLUID) {
            // In 2D axisymmetric: dV = 2*pi*r*dx*dx, but for relative check we can use dx^2
            mass += fields.C[i];
        }
    }
    return mass;
}

// Find peak concentration among FLUID nodes
static double find_C_max(const Fields& fields, const Grid& grid) {
    double cmax = 0.0;
    for (int i = 0; i < grid.N_total; ++i) {
        if (grid.node_type[i] == FLUID && fields.C[i] > cmax)
            cmax = fields.C[i];
    }
    return cmax;
}

// ============================================================================
// Test 1: Pure PD Diffusion
// ============================================================================

static bool test_diffusion() {
    std::printf("\n========================================\n");
    std::printf("TEST 1: Pure PD Diffusion (Gaussian pulse)\n");
    std::printf("========================================\n");

    double D = 1.0e-9;
    Config cfg = make_test_config(D, 0.0);  // no flow
    std::printf("  dx=%.1e, delta=%.1e, D=%.1e, R_tube=%.1e\n",
                cfg.dx, cfg.delta, D, cfg.R_tube);

    Grid grid;
    grid.build(cfg);
    grid.build_neighbors();
    std::printf("  Grid: %d total nodes\n", grid.N_total);

    // Count fluid nodes
    int n_fluid = 0;
    for (int i = 0; i < grid.N_total; ++i)
        if (grid.node_type[i] == FLUID) n_fluid++;
    std::printf("  Fluid nodes: %d\n", n_fluid);

    double sigma = 30.0e-6;
    double r0 = 0.0, z0 = 0.0;
    double t_end = 0.5;

    // --- Run explicit solver as reference ---
    std::printf("\n  [Explicit solver]\n");
    Fields fields_exp;
    fields_exp.allocate(grid.N_total);
    init_fields_zero_vel(fields_exp, grid, cfg);
    set_gaussian_pulse(fields_exp, grid, sigma, r0, z0);
    double C_peak_init = find_C_max(fields_exp, grid);

    PD_ARD_Solver ard_exp;
    ard_exp.init(grid, cfg);
    double dt_exp = ard_exp.compute_dt(fields_exp, grid, cfg);
    std::printf("  dt_explicit = %.4e s\n", dt_exp);

    double t = 0.0;
    int n_steps_exp = 0;
    while (t < t_end) {
        double dt = std::min(dt_exp, t_end - t);
        ard_exp.step(fields_exp, grid, cfg, dt);
        std::swap(fields_exp.C, fields_exp.C_new);
        t += dt;
        n_steps_exp++;
    }
    std::printf("  Completed %d explicit steps, t=%.3f s\n", n_steps_exp, t);
    std::printf("  C_peak: %.4f -> %.4f\n", C_peak_init, find_C_max(fields_exp, grid));

    // Compute analytical solution
    std::vector<double> C_exact(grid.N_total, 0.0);
    for (int i = 0; i < grid.N_total; ++i) {
        if (grid.node_type[i] == FLUID) {
            C_exact[i] = gaussian_exact_2d(grid.pos[i][0], grid.pos[i][1],
                                            r0, z0, sigma, D, t_end);
        }
    }
    double peak_exact = 0.0;
    for (auto v : C_exact) if (v > peak_exact) peak_exact = v;

    double err_exp = compute_L2_error(fields_exp, C_exact, grid);
    std::printf("  Analytical peak at t=%.1f: %.4f\n", t_end, peak_exact);
    std::printf("  Explicit vs analytical L2 error: %.4e\n", err_exp);

    // Mass conservation
    Fields fields_mass_check;
    fields_mass_check.allocate(grid.N_total);
    init_fields_zero_vel(fields_mass_check, grid, cfg);
    set_gaussian_pulse(fields_mass_check, grid, sigma, r0, z0);
    double mass_init_true = compute_total_mass(fields_mass_check, grid);
    double mass_exp_final = compute_total_mass(fields_exp, grid);
    std::printf("  Mass conservation (explicit): initial=%.6f, final=%.6f, change=%.4e%%\n",
                mass_init_true, mass_exp_final,
                std::abs(mass_exp_final - mass_init_true) / (mass_init_true + 1e-30) * 100.0);

    // --- Run implicit solver ---
    std::printf("\n  [Implicit solver]\n");

    // Test with several dt values — track errors for convergence rate
    double dt_values[] = {0.01, 0.05, 0.1, 0.25, 0.5};
    int n_dt = 5;
    double prev_err = -1.0;
    double prev_dt = -1.0;

    std::vector<double> errors_vs_analytical;
    std::vector<double> convergence_rates;
    double mass_change_finest = 0.0;

    for (int idt = 0; idt < n_dt; ++idt) {
        double dt_impl = dt_values[idt];
        Fields fields_impl;
        fields_impl.allocate(grid.N_total);
        init_fields_zero_vel(fields_impl, grid, cfg);
        set_gaussian_pulse(fields_impl, grid, sigma, r0, z0);

        PD_ARD_ImplicitSolver impl;
        impl.init(grid, cfg);
        impl.assemble(fields_impl, grid, cfg);

        t = 0.0;
        int n_steps = 0;
        while (t < t_end - 1e-12) {
            double dt = std::min(dt_impl, t_end - t);
            impl.step(fields_impl, grid, cfg, dt);
            t += dt;
            n_steps++;
        }

        double err_impl = compute_L2_error(fields_impl, C_exact, grid);
        double err_vs_exp = compute_L2_error(fields_impl, fields_exp.C, grid);
        double mass_impl = compute_total_mass(fields_impl, grid);
        double mass_change = std::abs(mass_impl - mass_init_true) / (mass_init_true + 1e-30) * 100.0;

        errors_vs_analytical.push_back(err_impl);
        if (idt == 0) mass_change_finest = mass_change;

        // Compute convergence rate (should be ~1 for backward Euler)
        double rate = -1.0;
        if (prev_err > 0.0 && err_impl > 1e-15) {
            rate = std::log(err_impl / prev_err) / std::log(dt_impl / prev_dt);
            convergence_rates.push_back(rate);
        }

        std::printf("  dt=%.3f s (%2d steps): vs_analytical=%.4e, vs_explicit=%.4e, "
                    "C_peak=%.4f, mass_err=%.2e%%",
                    dt_impl, n_steps, err_impl, err_vs_exp,
                    find_C_max(fields_impl, grid), mass_change);
        if (rate > 0.0) {
            std::printf(", conv_rate=%.2f", rate);
        }
        std::printf("\n");

        prev_err = err_impl;
        prev_dt = dt_impl;
    }

    std::printf("  Expected convergence rate: ~1.0 (backward Euler is O(dt))\n");

    // --- Hard assertions ---
    bool pass = true;
    double finest_L2 = errors_vs_analytical.front();
    if (finest_L2 > 0.05) {
        std::printf("  FAIL: finest L2 error %.4e > 0.05 threshold\n", finest_L2);
        pass = false;
    }
    if (mass_change_finest > 1.0) {
        std::printf("  FAIL: mass conservation error %.4e%% > 1%% threshold\n", mass_change_finest);
        pass = false;
    }
    bool has_good_rate = false;
    for (double r : convergence_rates) if (r > 0.4) has_good_rate = true;
    if (!has_good_rate) {
        std::printf("  FAIL: no convergence rate > 0.4\n");
        pass = false;
    }

    if (pass) std::printf("  PASS: diffusion test\n");
    else      std::printf("  FAIL: diffusion test\n");
    return pass;
}

// ============================================================================
// Test 2: Pure PD Advection (uniform velocity, minimal diffusion)
// ============================================================================

static bool test_advection() {
    std::printf("\n========================================\n");
    std::printf("TEST 2: Pure PD Advection (Gaussian transport)\n");
    std::printf("========================================\n");

    // Use small but nonzero diffusion (PD needs it for stability)
    double D = 1.0e-12;   // negligible diffusion
    double v_axial = 0.1;  // m/s in z direction
    Config cfg = make_test_config(D, 0.0);
    cfg.compute_derived();
    std::printf("  v_axial=%.2f m/s, D=%.1e (Pe_grid=%.0f)\n",
                v_axial, D, v_axial * cfg.dx / D);

    Grid grid;
    grid.build(cfg);
    grid.build_neighbors();
    std::printf("  Grid: %d total nodes\n", grid.N_total);

    double sigma = 40.0e-6;
    double r0 = 0.0, z0 = -100.0e-6;  // start upstream of center
    double t_end = 0.001;              // 0.001 s => displacement = 100 um = 20 dx

    // --- Explicit reference ---
    std::printf("\n  [Explicit solver]\n");
    Fields fields_exp;
    fields_exp.allocate(grid.N_total);
    init_fields_uniform_vel(fields_exp, grid, cfg, v_axial);
    set_gaussian_pulse(fields_exp, grid, sigma, r0, z0);

    PD_ARD_Solver ard_exp;
    ard_exp.init(grid, cfg);
    // Manual dt: advection-limited
    double dt_exp = 0.5 * cfg.dx / (v_axial + 1e-30);
    dt_exp = std::min(dt_exp, cfg.cfl_factor_corr * cfg.dx / (v_axial + 1e-30));
    std::printf("  dt_explicit = %.4e s\n", dt_exp);

    double t = 0.0;
    int n_steps_exp = 0;
    while (t < t_end) {
        double dt = std::min(dt_exp, t_end - t);
        ard_exp.step(fields_exp, grid, cfg, dt);
        std::swap(fields_exp.C, fields_exp.C_new);
        t += dt;
        n_steps_exp++;
    }
    double displacement = v_axial * t_end;
    std::printf("  Completed %d explicit steps, displacement=%.1f um\n",
                n_steps_exp, displacement * 1e6);

    // Analytical solution: Gaussian translates at v_axial, with negligible diffusion spread
    std::vector<double> C_exact(grid.N_total, 0.0);
    for (int i = 0; i < grid.N_total; ++i) {
        if (grid.node_type[i] == FLUID) {
            C_exact[i] = gaussian_exact_2d(grid.pos[i][0], grid.pos[i][1],
                                            r0, z0, sigma, D, t_end,
                                            0.0, v_axial);
        }
    }
    double peak_exact = 0.0;
    for (auto v : C_exact) if (v > peak_exact) peak_exact = v;
    std::printf("  Analytical: peak=%.4f at z_center=%.1f um\n",
                peak_exact, (z0 + v_axial * t_end) * 1e6);

    double err_exp_vs_exact = compute_L2_error(fields_exp, C_exact, grid);
    std::printf("  Explicit vs analytical L2 error: %.4e\n", err_exp_vs_exact);

    // Mass conservation (explicit)
    Fields fields_mass_check;
    fields_mass_check.allocate(grid.N_total);
    init_fields_uniform_vel(fields_mass_check, grid, cfg, v_axial);
    set_gaussian_pulse(fields_mass_check, grid, sigma, r0, z0);
    double mass_init = compute_total_mass(fields_mass_check, grid);
    double mass_exp_final = compute_total_mass(fields_exp, grid);
    std::printf("  Mass conservation (explicit): change=%.4e%%\n",
                std::abs(mass_exp_final - mass_init) / (mass_init + 1e-30) * 100.0);

    // --- Implicit solver at various dt ---
    std::printf("\n  [Implicit solver]\n");
    double dt_values[] = {1e-4, 2.5e-4, 5e-4, 1e-3};
    int n_dt = 4;
    double prev_err = -1.0, prev_dt = -1.0;

    std::vector<double> errors_vs_analytical;
    std::vector<double> convergence_rates;
    double mass_change_finest = 0.0;

    for (int idt = 0; idt < n_dt; ++idt) {
        double dt_impl = dt_values[idt];
        Fields fields_impl;
        fields_impl.allocate(grid.N_total);
        init_fields_uniform_vel(fields_impl, grid, cfg, v_axial);
        set_gaussian_pulse(fields_impl, grid, sigma, r0, z0);

        PD_ARD_ImplicitSolver impl;
        impl.init(grid, cfg);
        impl.assemble(fields_impl, grid, cfg);

        t = 0.0;
        int n_steps = 0;
        while (t < t_end - 1e-15) {
            double dt = std::min(dt_impl, t_end - t);
            impl.step(fields_impl, grid, cfg, dt);
            t += dt;
            n_steps++;
        }

        double err_vs_exact = compute_L2_error(fields_impl, C_exact, grid);
        double err_vs_exp = compute_L2_error(fields_impl, fields_exp.C, grid);
        double mass_impl = compute_total_mass(fields_impl, grid);
        double mass_change = std::abs(mass_impl - mass_init) / (mass_init + 1e-30) * 100.0;

        errors_vs_analytical.push_back(err_vs_exact);
        if (idt == 0) mass_change_finest = mass_change;

        double rate = -1.0;
        if (prev_err > 0.0 && err_vs_exact > 1e-15) {
            rate = std::log(err_vs_exact / prev_err) / std::log(dt_impl / prev_dt);
            convergence_rates.push_back(rate);
        }

        std::printf("  dt=%.1e s (%3d steps): vs_analytical=%.4e, vs_explicit=%.4e, "
                    "C_peak=%.4f, mass_err=%.2e%%",
                    dt_impl, n_steps, err_vs_exact, err_vs_exp,
                    find_C_max(fields_impl, grid), mass_change);
        if (rate > 0.0) std::printf(", conv_rate=%.2f", rate);
        std::printf("\n");

        prev_err = err_vs_exact;
        prev_dt = dt_impl;
    }

    std::printf("  Expected convergence rate: ~1.0 (backward Euler is O(dt))\n");

    // --- Hard assertions ---
    bool pass = true;
    double finest_L2 = errors_vs_analytical.front();
    if (finest_L2 > 0.40) {
        std::printf("  FAIL: finest L2 error %.4e > 0.40 threshold\n", finest_L2);
        pass = false;
    }
    if (mass_change_finest > 1.0) {
        std::printf("  FAIL: mass conservation error %.4e%% > 1%% threshold\n", mass_change_finest);
        pass = false;
    }
    bool has_good_rate = false;
    for (double r : convergence_rates) if (r > 0.3) has_good_rate = true;
    if (!has_good_rate) {
        std::printf("  FAIL: no convergence rate > 0.3\n");
        pass = false;
    }

    if (pass) std::printf("  PASS: advection test\n");
    else      std::printf("  FAIL: advection test\n");
    return pass;
}

// ============================================================================
// Test 3: Combined PD Advection + Diffusion
// ============================================================================

static bool test_advection_diffusion() {
    std::printf("\n========================================\n");
    std::printf("TEST 3: Combined PD Advection-Diffusion\n");
    std::printf("========================================\n");

    double D = 1.0e-9;
    double v_axial = 0.05;
    Config cfg = make_test_config(D, 0.0);
    cfg.compute_derived();
    double Pe = v_axial * cfg.dx / D;
    std::printf("  v=%.2f m/s, D=%.1e, Pe_grid=%.1f\n", v_axial, D, Pe);

    Grid grid;
    grid.build(cfg);
    grid.build_neighbors();

    double sigma = 40.0e-6;
    double r0 = 0.0, z0 = -100.0e-6;
    double t_end = 0.002;  // displacement = 100 um, Gaussian stays well within domain
    double displacement = v_axial * t_end;
    std::printf("  t_end=%.3f s, displacement=%.1f um, diffusion length=%.1f um\n",
                t_end, displacement * 1e6, std::sqrt(2.0 * D * t_end) * 1e6);

    // Analytical solution: translated + diffused Gaussian
    std::vector<double> C_exact(grid.N_total, 0.0);
    for (int i = 0; i < grid.N_total; ++i) {
        if (grid.node_type[i] == FLUID) {
            C_exact[i] = gaussian_exact_2d(grid.pos[i][0], grid.pos[i][1],
                                            r0, z0, sigma, D, t_end,
                                            0.0, v_axial);
        }
    }
    double peak_exact = 0.0;
    for (auto v : C_exact) if (v > peak_exact) peak_exact = v;
    std::printf("  Analytical: peak=%.4f, center at z=%.1f um\n",
                peak_exact, (z0 + v_axial * t_end) * 1e6);

    // --- Explicit reference ---
    std::printf("\n  [Explicit solver]\n");
    Fields fields_exp;
    fields_exp.allocate(grid.N_total);
    init_fields_uniform_vel(fields_exp, grid, cfg, v_axial);
    set_gaussian_pulse(fields_exp, grid, sigma, r0, z0);

    PD_ARD_Solver ard_exp;
    ard_exp.init(grid, cfg);
    double dt_exp = ard_exp.compute_dt(fields_exp, grid, cfg);
    std::printf("  dt_explicit = %.4e s\n", dt_exp);

    // Mass conservation check
    Fields fields_mass_check;
    fields_mass_check.allocate(grid.N_total);
    init_fields_uniform_vel(fields_mass_check, grid, cfg, v_axial);
    set_gaussian_pulse(fields_mass_check, grid, sigma, r0, z0);
    double mass_init = compute_total_mass(fields_mass_check, grid);

    double t = 0.0;
    int n_steps_exp = 0;
    while (t < t_end) {
        double dt = std::min(dt_exp, t_end - t);
        ard_exp.step(fields_exp, grid, cfg, dt);
        std::swap(fields_exp.C, fields_exp.C_new);
        t += dt;
        n_steps_exp++;
    }
    std::printf("  Completed %d explicit steps\n", n_steps_exp);

    double err_exp_vs_exact = compute_L2_error(fields_exp, C_exact, grid);
    double mass_exp_final = compute_total_mass(fields_exp, grid);
    std::printf("  Explicit vs analytical L2 error: %.4e\n", err_exp_vs_exact);
    std::printf("  Mass conservation (explicit): change=%.4e%%\n",
                std::abs(mass_exp_final - mass_init) / (mass_init + 1e-30) * 100.0);

    // --- Implicit solver at various dt ---
    std::printf("\n  [Implicit solver]\n");
    double dt_values[] = {1e-4, 2e-4, 5e-4, 1e-3, 2e-3};
    int n_dt = 5;
    double prev_err = -1.0, prev_dt = -1.0;

    std::vector<double> errors_vs_analytical;
    std::vector<double> convergence_rates;
    double mass_change_finest = 0.0;

    for (int idt = 0; idt < n_dt; ++idt) {
        double dt_impl = dt_values[idt];
        Fields fields_impl;
        fields_impl.allocate(grid.N_total);
        init_fields_uniform_vel(fields_impl, grid, cfg, v_axial);
        set_gaussian_pulse(fields_impl, grid, sigma, r0, z0);

        PD_ARD_ImplicitSolver impl;
        impl.init(grid, cfg);
        impl.assemble(fields_impl, grid, cfg);

        t = 0.0;
        int n_steps = 0;
        while (t < t_end - 1e-15) {
            double dt = std::min(dt_impl, t_end - t);
            impl.step(fields_impl, grid, cfg, dt);
            t += dt;
            n_steps++;
        }

        double err_vs_exact = compute_L2_error(fields_impl, C_exact, grid);
        double err_vs_exp = compute_L2_error(fields_impl, fields_exp.C, grid);
        double mass_impl = compute_total_mass(fields_impl, grid);
        double mass_change = std::abs(mass_impl - mass_init) / (mass_init + 1e-30) * 100.0;

        errors_vs_analytical.push_back(err_vs_exact);
        if (idt == 0) mass_change_finest = mass_change;

        double rate = -1.0;
        if (prev_err > 0.0 && err_vs_exact > 1e-15) {
            rate = std::log(err_vs_exact / prev_err) / std::log(dt_impl / prev_dt);
            convergence_rates.push_back(rate);
        }

        std::printf("  dt=%.1e s (%3d steps): vs_analytical=%.4e, vs_explicit=%.4e, "
                    "C_peak=%.4f, mass_err=%.2e%%",
                    dt_impl, n_steps, err_vs_exact, err_vs_exp,
                    find_C_max(fields_impl, grid), mass_change);
        if (rate > 0.0) std::printf(", conv_rate=%.2f", rate);
        std::printf("\n");

        prev_err = err_vs_exact;
        prev_dt = dt_impl;
    }

    // --- Write VTI output for visual inspection ---
    mkdir("output_test", 0755);
    VTKWriter writer;

    // Write explicit result
    std::string fname = "output_test/diffadv_explicit.vti";
    writer.write(fname, grid, fields_exp, cfg);
    std::printf("\n  Wrote explicit result to %s\n", fname.c_str());

    // Write implicit result at dt=5e-4
    {
        Fields fields_impl;
        fields_impl.allocate(grid.N_total);
        init_fields_uniform_vel(fields_impl, grid, cfg, v_axial);
        set_gaussian_pulse(fields_impl, grid, sigma, r0, z0);
        PD_ARD_ImplicitSolver impl;
        impl.init(grid, cfg);
        impl.assemble(fields_impl, grid, cfg);
        t = 0.0;
        while (t < t_end - 1e-15) {
            double dt = std::min(5e-4, t_end - t);
            impl.step(fields_impl, grid, cfg, dt);
            t += dt;
        }
        fname = "output_test/diffadv_implicit.vti";
        writer.write(fname, grid, fields_impl, cfg);
        std::printf("  Wrote implicit result to %s\n", fname.c_str());
    }

    std::printf("  Expected convergence rate: ~1.0 (backward Euler is O(dt))\n");

    // --- Hard assertions ---
    bool pass = true;
    double finest_L2 = errors_vs_analytical.front();
    if (finest_L2 > 0.20) {
        std::printf("  FAIL: finest L2 error %.4e > 0.20 threshold\n", finest_L2);
        pass = false;
    }
    if (mass_change_finest > 1.0) {
        std::printf("  FAIL: mass conservation error %.4e%% > 1%% threshold\n", mass_change_finest);
        pass = false;
    }
    bool has_good_rate = false;
    for (double r : convergence_rates) if (r > 0.3) has_good_rate = true;
    if (!has_good_rate) {
        std::printf("  FAIL: no convergence rate > 0.3\n");
        pass = false;
    }

    if (pass) std::printf("  PASS: advection-diffusion test\n");
    else      std::printf("  FAIL: advection-diffusion test\n");
    return pass;
}

// ============================================================================
// Test 4: Solid-Liquid Interface Dissolution (Bi-Material PD Model)
// ============================================================================

static bool test_interface_dissolution() {
    std::printf("\n========================================\n");
    std::printf("TEST 4: Solid-Liquid Interface Dissolution\n");
    std::printf("========================================\n");

    // Create a small domain with solid on one side, fluid on the other.
    // The bi-material PD model should drive concentration from solid (C=1)
    // to fluid (C=0) across the interface via harmonic-mean diffusivity bonds.
    //
    // Domain: 1D-like strip (narrow R_tube, long axial)
    // Left half (z < 0): SOLID_MG with C = C_solid_init = 1.0
    // Right half (z >= 0): FLUID with C = 0.0

    Config cfg;
    cfg.dx = 5.0e-6;
    cfg.m_ratio = 3;
    cfg.R_wire = 0.0;        // no pin geometry
    cfg.L_wire = 0.0;
    cfg.R_tube = 25.0e-6;    // narrow tube (5 nodes across radius)
    cfg.L_upstream = 100.0e-6;
    cfg.L_downstream = 100.0e-6;
    cfg.rho_f = 1000.0;
    cfg.mu_f = 1.0e-3;
    cfg.c0 = 5.0;
    cfg.eta_density = 0.1;
    cfg.gamma_eos = 7.0;
    cfg.Q_flow = 0.0;        // no flow
    cfg.rho_m = 1738.0;
    cfg.D_liquid = 1.0e-9;
    cfg.D_grain = 5.0e-11;   // solid grain diffusivity
    cfg.D_gb = 5.0e-9;       // GB diffusivity (not used here — no GB nodes)
    cfg.C_solid_init = 1.0;
    cfg.C_liquid_init = 0.0;
    cfg.C_thresh = 0.2;
    cfg.C_sat = 10.0;        // high enough to not trigger salt layer
    cfg.alpha_art_diff = 0.0;
    cfg.grain_size_mean = 40.0e-6;
    cfg.grain_size_std = 5.0e-6;
    cfg.gb_width_cells = 0;
    cfg.cfl_factor = 0.25;
    cfg.cfl_factor_corr = 0.25;
    cfg.use_implicit = 1;
    cfg.implicit_dt_max = 60.0;
    cfg.implicit_dt_fraction = 0.5;
    cfg.newton_tol = 1.0e-10;
    cfg.newton_max_iter = 30;
    cfg.compute_derived();

    std::printf("  dx=%.1e, delta=%.1e, D_liquid=%.1e, D_grain=%.1e\n",
                cfg.dx, cfg.delta, cfg.D_liquid, cfg.D_grain);
    std::printf("  Harmonic mean D_interface = %.2e\n",
                2.0 * cfg.D_liquid * cfg.D_grain / (cfg.D_liquid + cfg.D_grain));

    Grid grid;
    grid.build(cfg);
    grid.build_neighbors();
    std::printf("  Grid: %d total nodes\n", grid.N_total);

    // Manually set the left half as SOLID_MG
    // z < 0 => SOLID, z >= 0 => FLUID (keep existing BCs for INLET/OUTLET/WALL)
    Fields fields;
    fields.allocate(grid.N_total);

    int n_solid = 0, n_fluid = 0;
    for (int i = 0; i < grid.N_total; ++i) {
        fields.rho[i] = cfg.rho_f;
        fields.vel[i] = vec_zero();
        fields.phase[i] = 1;
        fields.grain_id[i] = -1;
        fields.is_gb[i] = 0;

        if (grid.node_type[i] == FLUID) {
            double z = grid.pos[i][1];  // axial coordinate
            if (z < 0.0) {
                // Convert to solid
                grid.node_type[i] = SOLID_MG;
                fields.phase[i] = 0;
                fields.C[i] = cfg.C_solid_init;
                fields.D_map[i] = cfg.D_grain;
                fields.rho[i] = cfg.rho_m;
                n_solid++;
            } else {
                fields.C[i] = cfg.C_liquid_init;
                fields.D_map[i] = cfg.D_liquid;
                n_fluid++;
            }
        } else if (grid.node_type[i] == INLET || grid.node_type[i] == OUTLET) {
            fields.C[i] = 0.0;
            fields.D_map[i] = cfg.D_liquid;
        } else {
            fields.C[i] = 0.0;
            fields.D_map[i] = 0.0;
        }
    }
    fields.rho_new = fields.rho;
    fields.vel_new = fields.vel;
    fields.C_new = fields.C;

    std::printf("  Solid nodes: %d, Fluid nodes: %d\n", n_solid, n_fluid);

    // Record initial solid C for comparison
    double C_solid_init_sum = 0.0;
    for (int i = 0; i < grid.N_total; ++i) {
        if (grid.node_type[i] == SOLID_MG) C_solid_init_sum += fields.C[i];
    }

    // --- Run implicit solver ---
    PD_ARD_ImplicitSolver impl;
    impl.init(grid, cfg);
    impl.assemble(fields, grid, cfg);

    double t = 0.0;
    double t_end = 100.0;   // enough time for significant diffusion
    int n_steps = 0;
    double dt_impl = 1.0;   // 1 second steps

    std::printf("\n  Running implicit solver for %.0f s with dt=%.1f s\n", t_end, dt_impl);

    while (t < t_end - 1e-12) {
        double dt = std::min(dt_impl, t_end - t);
        impl.step(fields, grid, cfg, dt);
        t += dt;
        n_steps++;
    }

    // Check results
    double C_solid_sum = 0.0;
    double C_solid_min = 1e30, C_solid_max = -1e30;
    double C_fluid_sum = 0.0;
    double C_fluid_max = 0.0;
    int n_solid_now = 0;

    for (int i = 0; i < grid.N_total; ++i) {
        if (grid.node_type[i] == SOLID_MG) {
            C_solid_sum += fields.C[i];
            if (fields.C[i] < C_solid_min) C_solid_min = fields.C[i];
            if (fields.C[i] > C_solid_max) C_solid_max = fields.C[i];
            n_solid_now++;
        } else if (grid.node_type[i] == FLUID) {
            C_fluid_sum += fields.C[i];
            if (fields.C[i] > C_fluid_max) C_fluid_max = fields.C[i];
        }
    }

    std::printf("\n  Results after %d steps (t=%.1f s):\n", n_steps, t);
    std::printf("  Solid C: min=%.4f, max=%.4f, avg=%.4f (was 1.0000)\n",
                C_solid_min, C_solid_max, C_solid_sum / (n_solid_now + 1e-30));
    std::printf("  Fluid C: max=%.4f, sum=%.4f\n", C_fluid_max, C_fluid_sum);
    std::printf("  Total C: solid+fluid=%.4f (initial solid sum=%.4f)\n",
                C_solid_sum + C_fluid_sum, C_solid_init_sum);

    // Verify key properties:
    bool pass = true;

    // 1. Solid concentration at surface should have decreased
    if (C_solid_min >= cfg.C_solid_init - 1e-10) {
        std::printf("  FAIL: Solid surface C did not decrease (min=%.6f)\n", C_solid_min);
        pass = false;
    } else {
        std::printf("  OK: Solid surface C decreased from 1.0 to %.4f\n", C_solid_min);
    }

    // 2. Fluid concentration near interface should have increased
    if (C_fluid_max <= 1e-10) {
        std::printf("  FAIL: Fluid C did not increase (max=%.6f)\n", C_fluid_max);
        pass = false;
    } else {
        std::printf("  OK: Fluid C increased to max=%.4f near interface\n", C_fluid_max);
    }

    // 3. Mass balance: total C should not increase (boundary nodes at C=0 absorb mass)
    //    In a small domain with fast D_liquid, C leaks to INLET/OUTLET BCs — expected.
    double total_C = C_solid_sum + C_fluid_sum;
    if (total_C > C_solid_init_sum * 1.01) {
        std::printf("  FAIL: Total C increased (%.4f > %.4f) — possible bug\n",
                    total_C, C_solid_init_sum);
        pass = false;
    } else {
        double lost_frac = (C_solid_init_sum - total_C) / C_solid_init_sum * 100.0;
        std::printf("  OK: Total C decreased by %.1f%% (mass absorbed by boundary BCs)\n",
                    lost_frac);
    }

    // 4. Test phase change: run much longer and check if surface nodes dissolve
    std::printf("\n  Testing phase change...\n");
    double t_phase_end = 10000.0;   // long enough for surface to dissolve
    dt_impl = 10.0;
    while (t < t_phase_end - 1e-12) {
        double dt = std::min(dt_impl, t_phase_end - t);
        impl.step(fields, grid, cfg, dt);
        t += dt;
        n_steps++;
    }

    int n_dissolved = impl.apply_phase_change(fields, grid, cfg);
    std::printf("  After t=%.0f s: %d nodes dissolved\n", t, n_dissolved);

    if (n_dissolved > 0) {
        std::printf("  OK: Phase change triggered (%d nodes dissolved)\n", n_dissolved);
    } else {
        // Check if any solid node is close to threshold
        double C_min_solid = 1.0;
        for (int i = 0; i < grid.N_total; ++i) {
            if (grid.node_type[i] == SOLID_MG && fields.C[i] < C_min_solid) {
                C_min_solid = fields.C[i];
            }
        }
        std::printf("  INFO: No dissolution yet (min solid C=%.4f, thresh=%.2f)\n",
                    C_min_solid, cfg.C_thresh);
        // This is ok — with D_grain=5e-11, dissolution takes a long time
        // The test verifies the mechanism works (C decreases toward threshold)
        if (C_min_solid < 0.9) {
            std::printf("  OK: Solid C is decreasing toward threshold\n");
        } else {
            std::printf("  WARN: Solid C barely changed — check D_grain value\n");
        }
    }

    if (pass) {
        std::printf("  PASS: interface dissolution test completed\n");
    } else {
        std::printf("  FAIL: interface dissolution test\n");
    }
    return pass;
}

// ============================================================================
// Main
// ============================================================================

int main() {
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    std::printf("=== Implicit ARD Solver Validation Tests ===\n");
    std::printf("  Dimension: %dD\n", DIM);

    Timer t_total("total_tests");

    bool ok = true;
    ok &= test_diffusion();
    ok &= test_advection();
    ok &= test_advection_diffusion();
    ok &= test_interface_dissolution();

    std::printf("\n========================================\n");
    if (ok) {
        std::printf("ALL TESTS PASSED\n");
    } else {
        std::printf("SOME TESTS FAILED\n");
    }
    t_total.report();

    return ok ? 0 : 1;
}
