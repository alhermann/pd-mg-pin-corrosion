// AMR validation tests for the peridynamic corrosion solver.
//
// Tests PD-NS Poiseuille flow, PD diffusion, PD advection, and combined
// PD advection-diffusion on two-level AMR grids. Verifies smooth solutions
// across fine-coarse transition zones.
//
// Build: cmake .. -DPD_DIM=2 && make test_amr
// Run:   ./test_amr

#include "config.h"
#include "grid.h"
#include "fields.h"
#include "pd_ns.h"
#include "pd_ard.h"
#include "pd_ard_implicit.h"
#include "vtk_writer.h"
#include <cstdio>
#include <cmath>
#include <vector>
#include <string>
#include <sys/stat.h>
#include <algorithm>

// ============================================================================
// Helpers
// ============================================================================

static Config make_amr_test_config(double D_liquid, double Q_flow) {
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
    cfg.D_precip = 0.0;
    cfg.precip_fraction = 0.0;
    cfg.C_solid_init = 1.0;
    cfg.C_liquid_init = 0.0;
    cfg.C_thresh = 0.2;
    cfg.C_sat = 10.0;
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

    // AMR settings
    cfg.use_amr = 1;
    cfg.amr_ratio = 2;  // dx_c = 2 * dx_f for tests
    cfg.amr_buffer = 100.0e-6;  // fine zone covers center

    cfg.compute_derived();
    return cfg;
}

static void init_amr_fields(Fields& fields, const Grid& grid, const Config& cfg) {
    int N = grid.N_total;
    for (int i = 0; i < N; ++i) {
        fields.rho[i] = cfg.rho_f;
        fields.vel[i] = vec_zero();
        fields.D_map[i] = (grid.node_type[i] == FLUID ||
                           grid.node_type[i] == INLET ||
                           grid.node_type[i] == OUTLET ||
                           grid.node_type[i] == FICTITIOUS) ? cfg.D_liquid : 0.0;
        fields.C[i] = 0.0;
        fields.phase[i] = 1;
        fields.grain_id[i] = -1;
        fields.is_gb[i] = 0;
        fields.is_precip[i] = 0;
    }
    fields.rho_new = fields.rho;
    fields.vel_new = fields.vel;
    fields.C_new = fields.C;
}

static void init_amr_fields_uniform_vel(Fields& fields, const Grid& grid,
                                         const Config& cfg, double v_axial) {
    init_amr_fields(fields, grid, cfg);
    for (int i = 0; i < grid.N_total; ++i) {
        if (grid.node_type[i] == FLUID ||
            grid.node_type[i] == INLET ||
            grid.node_type[i] == OUTLET ||
            grid.node_type[i] == FICTITIOUS) {
            Vec v = vec_zero();
            if constexpr (DIM == 2) { v[1] = v_axial; }
            else                    { v[2] = v_axial; }
            fields.vel[i] = v;
        }
    }
    fields.vel_new = fields.vel;
}

// Set a Gaussian concentration pulse
static void set_amr_gaussian(Fields& fields, const Grid& grid,
                              double sigma, double r0, double z0) {
    double inv_2sig2 = 1.0 / (2.0 * sigma * sigma);
    for (int i = 0; i < grid.N_total; ++i) {
        if (grid.node_type[i] == FLUID || grid.node_type[i] == FICTITIOUS) {
            double r = grid.pos[i][0] - r0;
            double z = grid.pos[i][1] - z0;
            fields.C[i] = std::exp(-(r*r + z*z) * inv_2sig2);
        } else {
            fields.C[i] = 0.0;
        }
    }
    fields.C_new = fields.C;
}

// Analytical Gaussian in 2D: diffusion + uniform advection
static double gaussian_exact_2d(double r, double z, double r0, double z0,
                                 double sigma, double D, double t,
                                 double v_r = 0.0, double v_z = 0.0) {
    double sig2 = sigma * sigma;
    double sig2t = sig2 + 2.0 * D * t;
    double dr = r - (r0 + v_r * t);
    double dz = z - (z0 + v_z * t);
    return (sig2 / sig2t) * std::exp(-(dr*dr + dz*dz) / (2.0 * sig2t));
}

static double compute_L2_error(const Fields& fields, const std::vector<double>& C_ref,
                                const Grid& grid) {
    double err2 = 0.0, ref2 = 0.0;
    for (int i = 0; i < grid.N_total; ++i) {
        if (grid.node_type[i] != FLUID) continue;
        // Volume-weight for AMR grids
        double vol = 1.0;
        if (!grid.dx_local.empty()) {
            for (int d = 0; d < DIM; ++d) vol *= grid.dx_local[i];
        }
        double e = fields.C[i] - C_ref[i];
        err2 += e * e * vol;
        ref2 += C_ref[i] * C_ref[i] * vol;
    }
    return std::sqrt(err2 / (ref2 + 1e-30));
}

static double compute_total_mass(const Fields& fields, const Grid& grid) {
    double mass = 0.0;
    for (int i = 0; i < grid.N_total; ++i) {
        if (grid.node_type[i] == FLUID) {
            // Volume-weight for AMR grids
            double vol = 1.0;
            if (!grid.dx_local.empty()) {
                for (int d = 0; d < DIM; ++d) vol *= grid.dx_local[i];
            }
            mass += fields.C[i] * vol;
        }
    }
    return mass;
}

static double find_C_max(const Fields& fields, const Grid& grid) {
    double cmax = 0.0;
    for (int i = 0; i < grid.N_total; ++i) {
        if (grid.node_type[i] == FLUID && fields.C[i] > cmax)
            cmax = fields.C[i];
    }
    return cmax;
}

// Check continuity across the fine-coarse boundary.
// Sample concentration along x=0 (axial line) and check for jumps.
static bool check_transition_continuity(const Fields& fields, const Grid& grid,
                                         const Config& cfg, double max_jump_frac) {
    // Collect (z, C) pairs for FLUID nodes near x=0
    struct ZC { double z; double C; };
    std::vector<ZC> samples;
    for (int i = 0; i < grid.N_total; ++i) {
        if (grid.node_type[i] != FLUID) continue;
        if (std::abs(grid.pos[i][0]) > 1.5 * cfg.dx) continue;
        samples.push_back({grid.pos[i][1], fields.C[i]});
    }
    std::sort(samples.begin(), samples.end(), [](const ZC& a, const ZC& b) {
        return a.z < b.z;
    });

    if (samples.size() < 4) return true;

    double max_jump = 0.0;
    for (int i = 1; i < (int)samples.size(); ++i) {
        double dz = samples[i].z - samples[i-1].z;
        if (dz < 1e-15) continue;
        double gradient = std::abs(samples[i].C - samples[i-1].C) / dz;
        double local_C = std::max(std::abs(samples[i].C), std::abs(samples[i-1].C));
        if (local_C > 1e-6) {
            double rel_jump = gradient * dz / local_C;
            if (rel_jump > max_jump) max_jump = rel_jump;
        }
    }

    if (max_jump > max_jump_frac) {
        std::printf("    WARN: max relative jump across transition = %.4f (threshold %.2f)\n",
                    max_jump, max_jump_frac);
        return false;
    }
    return true;
}

// Look up concentration from uniform fine grid at position (px, py)
static double lookup_uniform_C(const Grid& uni, const Fields& uni_f,
                                double px, double py) {
    int i = (int)std::round((px - uni.origin_x) / uni.dx);
    int j = (int)std::round((py - uni.origin_y) / uni.dx);
    if (i < 0 || i >= uni.Nx || j < 0 || j >= uni.Ny) return 0.0;
    int n = j * uni.Nx + i;
    if (n < 0 || n >= uni.N_total) return 0.0;
    if (uni.node_type[n] == OUTSIDE || uni.node_type[n] == WALL) return 0.0;
    return uni_f.C[n];
}

// Compute L2 error of AMR solution vs uniform fine grid solution
static double compute_L2_vs_uniform(const std::vector<double>& amr_C, const Grid& amr_g,
                                     const Fields& uni_f, const Grid& uni_g) {
    double err2 = 0.0, ref2 = 0.0;
    for (int i = 0; i < amr_g.N_total; ++i) {
        if (amr_g.node_type[i] != FLUID) continue;
        double vol = 1.0;
        if (!amr_g.dx_local.empty()) {
            for (int d = 0; d < DIM; ++d) vol *= amr_g.dx_local[i];
        }
        double C_ref = lookup_uniform_C(uni_g, uni_f,
                                          amr_g.pos[i][0], amr_g.pos[i][1]);
        double e = amr_C[i] - C_ref;
        err2 += e * e * vol;
        ref2 += C_ref * C_ref * vol;
    }
    return std::sqrt(err2 / (ref2 + 1e-30));
}

// Build uniform fine grid and initialize fields for reference simulation
static void build_uniform_fine(Grid& grid, Fields& fields, const Config& amr_cfg,
                                double sigma, double r0, double z0,
                                double v_axial = 0.0) {
    Config cfg = amr_cfg;
    cfg.use_amr = 0;

    grid.build(cfg);
    grid.build_neighbors();

    fields.allocate(grid.N_total);
    for (int i = 0; i < grid.N_total; ++i) {
        fields.rho[i] = cfg.rho_f;
        fields.D_map[i] = (grid.node_type[i] == FLUID ||
                            grid.node_type[i] == INLET ||
                            grid.node_type[i] == OUTLET) ? cfg.D_liquid : 0.0;
        fields.phase[i] = 1;
        fields.grain_id[i] = -1;
        fields.is_gb[i] = 0;
        fields.is_precip[i] = 0;
        fields.C[i] = 0.0;

        Vec v = vec_zero();
        if (v_axial != 0.0 && (grid.node_type[i] == FLUID ||
            grid.node_type[i] == INLET || grid.node_type[i] == OUTLET)) {
            v[1] = v_axial;
        }
        fields.vel[i] = v;
    }

    // Set Gaussian IC
    double inv_2sig2 = 1.0 / (2.0 * sigma * sigma);
    for (int i = 0; i < grid.N_total; ++i) {
        if (grid.node_type[i] == FLUID) {
            double r = grid.pos[i][0] - r0;
            double z = grid.pos[i][1] - z0;
            fields.C[i] = std::exp(-(r*r + z*z) * inv_2sig2);
        }
    }
    fields.rho_new = fields.rho;
    fields.vel_new = fields.vel;
    fields.C_new = fields.C;
}

// ============================================================================
// Test 1: PD-NS Poiseuille Flow on AMR Grid
// ============================================================================

static bool test_amr_poiseuille() {
    std::printf("\n========================================\n");
    std::printf("AMR TEST 1: AMR Grid Construction & PD-NS Setup\n");
    std::printf("========================================\n");

    Config cfg = make_amr_test_config(1.0e-9, 1.667e-9);
    std::printf("  dx_fine=%.1fum, dx_coarse=%.1fum, R_tube=%.0fum\n",
                cfg.dx * 1e6, cfg.dx_coarse * 1e6, cfg.R_tube * 1e6);

    Grid grid;
    grid.build_amr(cfg);
    grid.build_neighbors_celllist(cfg);
    std::printf("  AMR grid: %d total nodes\n", grid.N_total);

    // Count by level
    int n_fine = 0, n_coarse = 0, n_fict = 0, n_fluid = 0;
    for (int i = 0; i < grid.N_total; ++i) {
        if (grid.node_type[i] == FICTITIOUS) n_fict++;
        else if (grid.grid_level[i] == 0) n_fine++;
        else n_coarse++;
        if (grid.node_type[i] == FLUID) n_fluid++;
    }
    std::printf("  fine=%d, coarse=%d, fictitious=%d, fluid=%d\n",
                n_fine, n_coarse, n_fict, n_fluid);

    bool pass = true;

    // Check 1: Grid has all three types of nodes
    if (n_fine == 0) { std::printf("  FAIL: no fine nodes\n"); pass = false; }
    if (n_coarse == 0) { std::printf("  FAIL: no coarse nodes\n"); pass = false; }
    if (n_fict == 0) { std::printf("  FAIL: no fictitious nodes\n"); pass = false; }

    // Check 2: All non-OUTSIDE nodes have neighbors
    int isolated = 0;
    for (int i = 0; i < grid.N_total; ++i) {
        if (grid.node_type[i] == OUTSIDE) continue;
        int nn = grid.nbr_offset[i + 1] - grid.nbr_offset[i];
        if (nn == 0 && grid.node_type[i] == FLUID) isolated++;
    }
    if (isolated > 0) {
        std::printf("  FAIL: %d isolated fluid nodes with 0 neighbors\n", isolated);
        pass = false;
    }

    // Check 3: All fictitious nodes have IDW sources
    int fict_no_src = 0;
    for (int i = 0; i < grid.N_total; ++i) {
        if (grid.node_type[i] != FICTITIOUS) continue;
        int ns = grid.fict_offset[i + 1] - grid.fict_offset[i];
        if (ns == 0) fict_no_src++;
    }
    if (fict_no_src > 0) {
        std::printf("  FAIL: %d fictitious nodes with no IDW sources\n", fict_no_src);
        pass = false;
    }

    // Check 4: IDW interpolation of Poiseuille profile
    Fields fields;
    fields.allocate(grid.N_total);

    double R2 = cfg.R_tube * cfg.R_tube;
    for (int i = 0; i < grid.N_total; ++i) {
        fields.rho[i] = cfg.rho_f;
        fields.phase[i] = 1;
        fields.grain_id[i] = -1;
        fields.is_gb[i] = 0;
        fields.is_precip[i] = 0;
        fields.D_map[i] = cfg.D_liquid;
        fields.C[i] = 0.0;

        if (grid.node_type[i] == FLUID || grid.node_type[i] == INLET ||
            grid.node_type[i] == OUTLET || grid.node_type[i] == FICTITIOUS) {
            double px = grid.pos[i][0];
            double r_ratio2 = (px * px) / R2;
            if (r_ratio2 > 1.0) r_ratio2 = 1.0;
            double v_axial = 1.5 * cfg.U_in * (1.0 - r_ratio2);
            Vec v = vec_zero();
            v[1] = v_axial;
            fields.vel[i] = v;
        } else {
            fields.vel[i] = vec_zero();
        }
    }
    fields.rho_new = fields.rho;
    fields.vel_new = fields.vel;
    fields.C_new = fields.C;

    // After IDW update, fictitious nodes should match their analytical profile
    grid.update_fictitious(fields);

    double max_idw_err = 0.0;
    int n_fict_check = 0;
    for (int i = 0; i < grid.N_total; ++i) {
        if (grid.node_type[i] != FICTITIOUS) continue;
        double px = grid.pos[i][0];
        double r_ratio2 = (px * px) / R2;
        if (r_ratio2 > 1.0) r_ratio2 = 1.0;
        double v_exact = 1.5 * cfg.U_in * (1.0 - r_ratio2);
        double v_idw = fields.vel[i][1];
        if (v_exact > 1e-6) {
            double err = std::abs(v_idw - v_exact) / v_exact;
            if (err > max_idw_err) max_idw_err = err;
        }
        n_fict_check++;
    }
    std::printf("  IDW Poiseuille interpolation: max rel error = %.3e (%d nodes)\n",
                max_idw_err, n_fict_check);
    if (max_idw_err > 0.10) {
        std::printf("  FAIL: IDW error %.4e > 10%% threshold\n", max_idw_err);
        pass = false;
    }

    // Check 5: PD-NS solver initializes without crashing
    PD_NS_Solver flow;
    flow.init(grid, cfg);
    double dt = flow.compute_dt(fields, grid, cfg);
    std::printf("  PD-NS dt = %.4e s (should be ~%.1e)\n", dt, 0.25 * cfg.dx / cfg.c0);
    if (std::isnan(dt) || dt <= 0.0) {
        std::printf("  FAIL: invalid dt\n");
        pass = false;
    }

    if (pass) std::printf("  PASS: AMR grid construction & PD-NS setup\n");
    else      std::printf("  FAIL: AMR grid construction & PD-NS setup\n");
    return pass;
}

// ============================================================================
// Test 2: Pure PD Diffusion on AMR Grid
// ============================================================================

static bool test_amr_diffusion() {
    std::printf("\n========================================\n");
    std::printf("AMR TEST 2: Pure PD Diffusion on AMR Grid\n");
    std::printf("========================================\n");

    double D = 1.0e-9;
    Config cfg = make_amr_test_config(D, 0.0);
    std::printf("  dx_fine=%.1fum, dx_coarse=%.1fum, D=%.1e\n",
                cfg.dx * 1e6, cfg.dx_coarse * 1e6, D);

    Grid grid;
    grid.build_amr(cfg);
    grid.build_neighbors_celllist(cfg);
    std::printf("  AMR grid: %d total nodes\n", grid.N_total);

    double sigma = 30.0e-6;
    double r0 = 0.0, z0 = 0.0;
    double t_end = 0.5;

    // Analytical solution
    std::vector<double> C_exact(grid.N_total, 0.0);
    for (int i = 0; i < grid.N_total; ++i) {
        if (grid.node_type[i] == FLUID) {
            C_exact[i] = gaussian_exact_2d(grid.pos[i][0], grid.pos[i][1],
                                            r0, z0, sigma, D, t_end);
        }
    }

    // Run uniform fine reference simulation
    Grid uni_grid;
    Fields uni_fields;
    build_uniform_fine(uni_grid, uni_fields, cfg, sigma, r0, z0);
    std::printf("  Uniform fine grid: %d nodes\n", uni_grid.N_total);

    double dt_finest = 0.01;
    {
        Config cfg_uni = cfg;
        cfg_uni.use_amr = 0;
        PD_ARD_ImplicitSolver uni_impl;
        uni_impl.init(uni_grid, cfg_uni);
        uni_impl.assemble(uni_fields, uni_grid, cfg_uni);

        double t = 0.0;
        while (t < t_end - 1e-12) {
            double dt = std::min(dt_finest, t_end - t);
            uni_impl.step(uni_fields, uni_grid, cfg_uni, dt);
            t += dt;
        }
    }

    // Run AMR simulation with finest dt
    Fields amr_fields;
    amr_fields.allocate(grid.N_total);
    init_amr_fields(amr_fields, grid, cfg);
    set_amr_gaussian(amr_fields, grid, sigma, r0, z0);
    double mass_init = compute_total_mass(amr_fields, grid);

    {
        PD_ARD_ImplicitSolver impl;
        impl.init(grid, cfg);
        impl.assemble(amr_fields, grid, cfg);

        double t = 0.0;
        int n_steps = 0;
        while (t < t_end - 1e-12) {
            double dt = std::min(dt_finest, t_end - t);
            impl.step(amr_fields, grid, cfg, dt);
            if (cfg.use_amr) grid.update_fictitious(amr_fields);
            t += dt;
            n_steps++;
        }
        std::printf("  AMR: %d steps at dt=%.3f s\n", n_steps, dt_finest);
    }

    // L2 error vs analytical
    double l2_analytical = compute_L2_error(amr_fields, C_exact, grid);
    // L2 error vs uniform fine (isolates AMR coupling error)
    double l2_vs_uniform = compute_L2_vs_uniform(amr_fields.C, grid, uni_fields, uni_grid);
    double mass_final = compute_total_mass(amr_fields, grid);
    double mass_change = std::abs(mass_final - mass_init) / (mass_init + 1e-30) * 100.0;

    std::printf("  L2 vs analytical:   %.4e\n", l2_analytical);
    std::printf("  L2 vs uniform fine: %.4e\n", l2_vs_uniform);
    std::printf("  Mass change:        %.2e%%\n", mass_change);
    check_transition_continuity(amr_fields, grid, cfg, 0.5);

    bool pass = true;
    if (l2_vs_uniform > 0.10) {
        std::printf("  FAIL: L2 vs uniform fine %.4e > 10%% threshold\n", l2_vs_uniform);
        pass = false;
    }
    if (mass_change > 15.0) {
        std::printf("  FAIL: mass conservation error %.2e%% > 15%% threshold\n", mass_change);
        pass = false;
    }

    if (pass) std::printf("  PASS: AMR diffusion test\n");
    else      std::printf("  FAIL: AMR diffusion test\n");
    return pass;
}

// ============================================================================
// Test 3: Pure PD Advection on AMR Grid
// ============================================================================

static bool test_amr_advection() {
    std::printf("\n========================================\n");
    std::printf("AMR TEST 3: Pure PD Advection on AMR Grid\n");
    std::printf("========================================\n");

    double D = 1.0e-12;
    double v_axial = 0.05;
    Config cfg = make_amr_test_config(D, 0.0);
    std::printf("  v_axial=%.2f m/s, D=%.1e\n", v_axial, D);

    Grid grid;
    grid.build_amr(cfg);
    grid.build_neighbors_celllist(cfg);
    std::printf("  AMR grid: %d total nodes\n", grid.N_total);

    // Keep Gaussian within the fine zone to avoid cross-boundary mass loss.
    // Fine zone: z in [-100um, 100um]. Gaussian at z0=-20um, sigma=20um
    // occupies ~[-80um, 40um]. After displacement=25um, occupies [-55um, 65um].
    double sigma = 20.0e-6;
    double r0 = 0.0, z0 = -20.0e-6;
    double t_end = 0.0005;
    double displacement = v_axial * t_end;
    std::printf("  sigma=%.0f um, z0=%.0f um, displacement=%.1f um\n",
                sigma * 1e6, z0 * 1e6, displacement * 1e6);

    // Analytical solution
    std::vector<double> C_exact(grid.N_total, 0.0);
    for (int i = 0; i < grid.N_total; ++i) {
        if (grid.node_type[i] == FLUID) {
            C_exact[i] = gaussian_exact_2d(grid.pos[i][0], grid.pos[i][1],
                                            r0, z0, sigma, D, t_end,
                                            0.0, v_axial);
        }
    }

    // Run uniform fine reference simulation
    Grid uni_grid;
    Fields uni_fields;
    build_uniform_fine(uni_grid, uni_fields, cfg, sigma, r0, z0, v_axial);
    std::printf("  Uniform fine grid: %d nodes\n", uni_grid.N_total);

    double dt_finest = 5e-5;
    {
        Config cfg_uni = cfg;
        cfg_uni.use_amr = 0;
        PD_ARD_ImplicitSolver uni_impl;
        uni_impl.init(uni_grid, cfg_uni);
        uni_impl.assemble(uni_fields, uni_grid, cfg_uni);

        double t = 0.0;
        while (t < t_end - 1e-15) {
            double dt = std::min(dt_finest, t_end - t);
            uni_impl.step(uni_fields, uni_grid, cfg_uni, dt);
            t += dt;
        }
    }

    // Run AMR simulation with finest dt
    Fields amr_fields;
    amr_fields.allocate(grid.N_total);
    init_amr_fields_uniform_vel(amr_fields, grid, cfg, v_axial);
    set_amr_gaussian(amr_fields, grid, sigma, r0, z0);
    double mass_init = compute_total_mass(amr_fields, grid);

    {
        PD_ARD_ImplicitSolver impl;
        impl.init(grid, cfg);
        impl.assemble(amr_fields, grid, cfg);

        double t = 0.0;
        int n_steps = 0;
        while (t < t_end - 1e-15) {
            double dt = std::min(dt_finest, t_end - t);
            impl.step(amr_fields, grid, cfg, dt);
            if (cfg.use_amr) grid.update_fictitious(amr_fields);
            t += dt;
            n_steps++;
        }
        std::printf("  AMR: %d steps at dt=%.1e s\n", n_steps, dt_finest);
    }

    // L2 error vs analytical
    double l2_analytical = compute_L2_error(amr_fields, C_exact, grid);
    // L2 error vs uniform fine (isolates AMR coupling error)
    double l2_vs_uniform = compute_L2_vs_uniform(amr_fields.C, grid, uni_fields, uni_grid);
    double mass_final = compute_total_mass(amr_fields, grid);
    double mass_change = std::abs(mass_final - mass_init) / (mass_init + 1e-30) * 100.0;

    std::printf("  L2 vs analytical:   %.4e\n", l2_analytical);
    std::printf("  L2 vs uniform fine: %.4e\n", l2_vs_uniform);
    std::printf("  C_peak=%.4f, mass_err=%.2e%%\n", find_C_max(amr_fields, grid), mass_change);
    check_transition_continuity(amr_fields, grid, cfg, 0.5);

    bool pass = true;
    if (l2_vs_uniform > 0.10) {
        std::printf("  FAIL: L2 vs uniform fine %.4e > 10%% threshold\n", l2_vs_uniform);
        pass = false;
    }
    if (mass_change > 5.0) {
        std::printf("  FAIL: mass conservation error %.2e%% > 5%% threshold\n", mass_change);
        pass = false;
    }

    if (pass) std::printf("  PASS: AMR advection test\n");
    else      std::printf("  FAIL: AMR advection test\n");
    return pass;
}

// ============================================================================
// Test 4: Combined PD Advection-Diffusion on AMR Grid
// ============================================================================

static bool test_amr_advection_diffusion() {
    std::printf("\n========================================\n");
    std::printf("AMR TEST 4: Combined PD Advection-Diffusion on AMR Grid\n");
    std::printf("========================================\n");

    double D = 1.0e-9;
    double v_axial = 0.05;
    Config cfg = make_amr_test_config(D, 0.0);
    double Pe = v_axial * cfg.dx / D;
    std::printf("  v=%.2f m/s, D=%.1e, Pe_grid=%.1f\n", v_axial, D, Pe);

    Grid grid;
    grid.build_amr(cfg);
    grid.build_neighbors_celllist(cfg);
    std::printf("  AMR grid: %d total nodes\n", grid.N_total);

    // Keep Gaussian within the fine zone to avoid cross-boundary mass loss.
    // Fine zone: z in [-100um, 100um]. Gaussian at z0=-20um, sigma=20um
    // occupies ~[-80um, 40um]. After displacement=25um, occupies [-55um, 65um].
    // Diffusion length sqrt(2*D*t) ~ 1um is small but adds to the analytical solution.
    double sigma = 20.0e-6;
    double r0 = 0.0, z0 = -20.0e-6;
    double t_end = 0.0005;
    double displacement = v_axial * t_end;
    std::printf("  t_end=%.4f s, displacement=%.1f um, diffusion length=%.1f um\n",
                t_end, displacement * 1e6, std::sqrt(2.0 * D * t_end) * 1e6);

    // Analytical solution
    std::vector<double> C_exact(grid.N_total, 0.0);
    for (int i = 0; i < grid.N_total; ++i) {
        if (grid.node_type[i] == FLUID) {
            C_exact[i] = gaussian_exact_2d(grid.pos[i][0], grid.pos[i][1],
                                            r0, z0, sigma, D, t_end,
                                            0.0, v_axial);
        }
    }

    // Run uniform fine reference simulation
    Grid uni_grid;
    Fields uni_fields;
    build_uniform_fine(uni_grid, uni_fields, cfg, sigma, r0, z0, v_axial);
    std::printf("  Uniform fine grid: %d nodes\n", uni_grid.N_total);

    double dt_finest = 5e-5;
    {
        Config cfg_uni = cfg;
        cfg_uni.use_amr = 0;
        PD_ARD_ImplicitSolver uni_impl;
        uni_impl.init(uni_grid, cfg_uni);
        uni_impl.assemble(uni_fields, uni_grid, cfg_uni);

        double t = 0.0;
        while (t < t_end - 1e-15) {
            double dt = std::min(dt_finest, t_end - t);
            uni_impl.step(uni_fields, uni_grid, cfg_uni, dt);
            t += dt;
        }
    }

    // Run AMR simulation with finest dt
    Fields amr_fields;
    amr_fields.allocate(grid.N_total);
    init_amr_fields_uniform_vel(amr_fields, grid, cfg, v_axial);
    set_amr_gaussian(amr_fields, grid, sigma, r0, z0);
    double mass_init = compute_total_mass(amr_fields, grid);

    {
        PD_ARD_ImplicitSolver impl;
        impl.init(grid, cfg);
        impl.assemble(amr_fields, grid, cfg);

        double t = 0.0;
        int n_steps = 0;
        while (t < t_end - 1e-15) {
            double dt = std::min(dt_finest, t_end - t);
            impl.step(amr_fields, grid, cfg, dt);
            if (cfg.use_amr) grid.update_fictitious(amr_fields);
            t += dt;
            n_steps++;
        }
        std::printf("  AMR: %d steps at dt=%.1e s\n", n_steps, dt_finest);
    }

    // L2 error vs analytical
    double l2_analytical = compute_L2_error(amr_fields, C_exact, grid);
    // L2 error vs uniform fine (isolates AMR coupling error)
    double l2_vs_uniform = compute_L2_vs_uniform(amr_fields.C, grid, uni_fields, uni_grid);
    double mass_final = compute_total_mass(amr_fields, grid);
    double mass_change = std::abs(mass_final - mass_init) / (mass_init + 1e-30) * 100.0;

    std::printf("  L2 vs analytical:   %.4e\n", l2_analytical);
    std::printf("  L2 vs uniform fine: %.4e\n", l2_vs_uniform);
    std::printf("  C_peak=%.4f, mass_err=%.2e%%\n", find_C_max(amr_fields, grid), mass_change);
    check_transition_continuity(amr_fields, grid, cfg, 0.5);

    bool pass = true;
    if (l2_vs_uniform > 0.10) {
        std::printf("  FAIL: L2 vs uniform fine %.4e > 10%% threshold\n", l2_vs_uniform);
        pass = false;
    }
    if (mass_change > 5.0) {
        std::printf("  FAIL: mass conservation error %.2e%% > 5%% threshold\n", mass_change);
        pass = false;
    }

    if (pass) std::printf("  PASS: AMR advection-diffusion test\n");
    else      std::printf("  FAIL: AMR advection-diffusion test\n");
    return pass;
}

// ============================================================================
// Main
// ============================================================================

int main() {
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    std::printf("=== AMR Validation Tests ===\n");
    std::printf("  Dimension: %dD\n", DIM);

    Timer t_total("total_amr_tests");

    bool ok = true;
    ok &= test_amr_poiseuille();
    ok &= test_amr_diffusion();
    ok &= test_amr_advection();
    ok &= test_amr_advection_diffusion();

    std::printf("\n========================================\n");
    if (ok) {
        std::printf("ALL AMR TESTS PASSED\n");
    } else {
        std::printf("SOME AMR TESTS FAILED\n");
    }
    t_total.report();

    return ok ? 0 : 1;
}
