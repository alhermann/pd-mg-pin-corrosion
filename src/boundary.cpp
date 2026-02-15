#include "boundary.h"
#include <cmath>
#include <omp.h>

// ---------------------------------------------------------------------------
// Helper: compute grid indices from flat node index
// ---------------------------------------------------------------------------
static inline void flat_to_ijk(int n, int Nx, int Ny,
                                int& i, int& j, int& k) {
    if constexpr (DIM == 2) {
        k = 0;
        j = n / Nx;
        i = n % Nx;
    } else {
        k = n / (Nx * Ny);
        int rem = n % (Nx * Ny);
        j = rem / Nx;
        i = rem % Nx;
    }
}

// ---------------------------------------------------------------------------
// Inlet BC: prescribed Poiseuille velocity, density extrapolated from interior
//
// Standard CFD approach for velocity inlets in weakly compressible methods:
//   - PRESCRIBE velocity (Poiseuille profile)
//   - EXTRAPOLATE density from interior fluid neighbors
//     (avoids over-constraining: pressure develops naturally)
//   - PRESCRIBE concentration (fresh SBF, C=0)
// ---------------------------------------------------------------------------
void apply_inlet_bc(Fields& f, const Grid& g, const Config& cfg) {
    int N = g.N_total;
    double R2 = cfg.R_tube * cfg.R_tube;

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < N; ++i) {
        if (g.node_type[i] != INLET) continue;

        // Prescribed Poiseuille velocity
        double px = g.pos[i][0];
        double v_axial;

        if constexpr (DIM == 2) {
            double r_ratio2 = (px * px) / R2;
            if (r_ratio2 > 1.0) r_ratio2 = 1.0;
            v_axial = 1.5 * cfg.U_in * (1.0 - r_ratio2);
        } else {
            double py = g.pos[i][1];
            double r_ratio2 = (px * px + py * py) / R2;
            if (r_ratio2 > 1.0) r_ratio2 = 1.0;
            v_axial = 2.0 * cfg.U_in * (1.0 - r_ratio2);
        }

        Vec v_in = vec_zero();
        if constexpr (DIM == 2) { v_in[1] = v_axial; }
        else                    { v_in[2] = v_axial; }

        f.vel[i] = v_in;

        // Extrapolate density from fluid neighbors (smooth pressure transition)
        double rho_avg = 0.0;
        int count = 0;
        for (int jj = g.nbr_offset[i]; jj < g.nbr_offset[i + 1]; ++jj) {
            int j = g.nbr_index[jj];
            if (g.node_type[j] == FLUID) {
                rho_avg += f.rho[j];
                count++;
            }
        }
        f.rho[i] = (count > 0) ? (rho_avg / count) : cfg.rho_f;

        // Fresh SBF
        f.C[i] = cfg.C_liquid_init;
    }
}

// ---------------------------------------------------------------------------
// Outlet BC: pressure outlet (p=0) with zero-gradient velocity/concentration
//
// Following Song, Chen & Bobaru (2025): the pressure of outlet fictitious
// nodes is set to zero, which means rho = rho_f via the Tait EOS. This
// anchors the pressure field and prevents outlet pressure artifacts.
//
// Velocity and concentration are extrapolated from interior fluid neighbors
// using zero-gradient (Neumann) BCs. The transverse velocity component is
// set to zero to suppress spurious cross-flow at the outlet.
// ---------------------------------------------------------------------------
void apply_outlet_bc(Fields& f, const Grid& g, const Config& cfg) {
    int N = g.N_total;
    int ax = (DIM == 2) ? 1 : 2; // axial direction index

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < N; ++i) {
        if (g.node_type[i] != OUTLET) continue;

        // Enforce p=0 via rho = rho_f (Tait EOS gives p=0 when rho=rho_0)
        f.rho[i] = cfg.rho_f;

        // Extrapolate velocity and concentration from fluid neighbors
        Vec v_avg = vec_zero();
        double C_avg = 0.0;
        int count = 0;

        for (int jj = g.nbr_offset[i]; jj < g.nbr_offset[i + 1]; ++jj) {
            int j = g.nbr_index[jj];
            if (g.node_type[j] == FLUID || g.node_type[j] == OUTLET) {
                for (int d = 0; d < DIM; ++d) v_avg[d] += f.vel[j][d];
                C_avg += f.C[j];
                count++;
            }
        }

        if (count > 0) {
            double inv_c = 1.0 / count;
            for (int d = 0; d < DIM; ++d) v_avg[d] *= inv_c;

            // Keep axial velocity from extrapolation, zero transverse
            Vec v_out = vec_zero();
            v_out[ax] = v_avg[ax];
            f.vel[i] = v_out;
        } else {
            Vec v_in = vec_zero();
            v_in[ax] = cfg.U_in;
            f.vel[i] = v_in;
        }

        // Convective outflow: zero-gradient extrapolation from interior
        // (prevents artificial reflection of concentration at outlet)
        f.C[i] = (count > 0) ? (C_avg / count) : 0.0;
    }
}

// ---------------------------------------------------------------------------
// Wall BC: proper FNM reflection across tube wall surface
//
// For each WALL node, compute its mirror point by reflecting across the
// physical tube boundary (|x| = R_tube in 2D).  The mirror point falls
// exactly on a grid node for a uniform grid when R_tube is grid-aligned.
//
// Velocity: antisymmetric (no-slip)  → vel_wall = -vel_mirror
// Density:  symmetric (smooth p)     → rho_wall = rho_mirror
// ---------------------------------------------------------------------------
static void apply_wall_mirror_proper(Fields& f, const Grid& g,
                                      const Config& cfg, bool use_new) {
    int N = g.N_total;

    #pragma omp parallel for schedule(static)
    for (int n = 0; n < N; ++n) {
        if (g.node_type[n] != WALL) continue;

        int i_grid, j_grid, k_grid;
        flat_to_ijk(n, g.Nx, g.Ny, i_grid, j_grid, k_grid);

        double x = g.origin_x + i_grid * g.dx;

        // --- Compute mirror x-coordinate across tube wall surface ---
        int mirror_idx = -1;

        if constexpr (DIM == 2) {
            double x_mirror;
            if (x > cfg.R_tube) {
                x_mirror = 2.0 * cfg.R_tube - x;
            } else if (x < -cfg.R_tube) {
                x_mirror = -2.0 * cfg.R_tube - x;
            } else {
                // Wall node inside tube radius — shouldn't normally happen.
                // Fall through to nearest-neighbour fallback.
                goto fallback;
            }

            int i_mirror = static_cast<int>(
                std::round((x_mirror - g.origin_x) / g.dx));
            int j_mirror = j_grid;  // same axial position

            if (i_mirror >= 0 && i_mirror < g.Nx &&
                j_mirror >= 0 && j_mirror < g.Ny) {
                int idx = g.idx(i_mirror, j_mirror, 0);
                NodeType mt = g.node_type[idx];
                if (mt == FLUID || mt == INLET || mt == OUTLET || mt == SOLID_MG)
                    mirror_idx = idx;
            }
        } else {
            // 3D: cylindrical wall — mirror across r = R_tube
            double y = g.origin_y + j_grid * g.dx;
            double r = std::sqrt(x * x + y * y);
            if (r > cfg.R_tube && r > 1e-30) {
                double r_mirror = 2.0 * cfg.R_tube - r;
                double x_mirror = x * r_mirror / r;
                double y_mirror = y * r_mirror / r;

                int i_mirror = static_cast<int>(
                    std::round((x_mirror - g.origin_x) / g.dx));
                int j_mirror = static_cast<int>(
                    std::round((y_mirror - g.origin_y) / g.dx));
                int k_mirror = k_grid;

                if (i_mirror >= 0 && i_mirror < g.Nx &&
                    j_mirror >= 0 && j_mirror < g.Ny &&
                    k_mirror >= 0 && k_mirror < g.Nz) {
                    int idx = g.idx(i_mirror, j_mirror, k_mirror);
                    NodeType mt = g.node_type[idx];
                    if (mt == FLUID || mt == INLET || mt == OUTLET || mt == SOLID_MG)
                        mirror_idx = idx;
                }
            }
        }

        // --- Fallback: nearest fluid neighbour (original approach) ---
        fallback:
        if (mirror_idx < 0) {
            double best_dist = 1e30;
            for (int jj = g.nbr_offset[n]; jj < g.nbr_offset[n + 1]; ++jj) {
                int j = g.nbr_index[jj];
                if (g.node_type[j] == FLUID && g.nbr_dist[jj] < best_dist) {
                    best_dist = g.nbr_dist[jj];
                    mirror_idx = j;
                }
            }
        }

        // --- Apply mirroring ---
        if (mirror_idx >= 0) {
            if (use_new) {
                for (int d = 0; d < DIM; ++d)
                    f.vel_new[n][d] = -f.vel_new[mirror_idx][d];
                f.rho_new[n] = f.rho_new[mirror_idx];
            } else {
                for (int d = 0; d < DIM; ++d)
                    f.vel[n][d] = -f.vel[mirror_idx][d];
                f.rho[n] = f.rho[mirror_idx];
            }
        } else {
            if (use_new) {
                f.vel_new[n] = vec_zero();
                f.rho_new[n] = cfg.rho_f;
            } else {
                f.vel[n] = vec_zero();
                f.rho[n] = cfg.rho_f;
            }
        }
    }
}

void apply_wall_bc(Fields& f, const Grid& g, const Config& cfg) {
    apply_wall_mirror_proper(f, g, cfg, false);
}

void apply_wall_bc_new(Fields& f, const Grid& g, const Config& cfg) {
    apply_wall_mirror_proper(f, g, cfg, true);
}

// ---------------------------------------------------------------------------
// Wall concentration BC: Neumann (zero-gradient) at tube walls.
//
// Walls are impermeable — no mass flux through the wall, so dC/dn = 0.
// Ghost nodes extrapolate concentration from nearest interior fluid neighbor.
// ---------------------------------------------------------------------------
void apply_wall_concentration_bc(Fields& f, const Grid& g, const Config& cfg) {
    int N = g.N_total;

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < N; ++i) {
        if (g.node_type[i] != WALL) continue;

        // Extrapolate from nearest fluid neighbor (zero-gradient)
        double c_avg = 0.0;
        int count = 0;
        for (int jj = g.nbr_offset[i]; jj < g.nbr_offset[i + 1]; ++jj) {
            int j = g.nbr_index[jj];
            if (g.node_type[j] == FLUID) {
                c_avg += f.C[j];
                count++;
            }
        }
        f.C[i] = (count > 0) ? (c_avg / count) : 0.0;
    }
}

// ---------------------------------------------------------------------------
// Smooth concentration near inlet/outlet boundaries
//
// FLUID nodes within delta of the inlet/outlet have truncated PD neighborhoods
// (some bonds reach into OUTLET/INLET ghost layers). The PD operator mixes
// PD-computed and BC-extrapolated values, causing a slight discontinuity at
// distance=delta from the boundary. This pass replaces the PD result with
// zero-gradient extrapolation from interior neighbors for affected nodes.
// ---------------------------------------------------------------------------
void smooth_boundary_concentration(Fields& f, const Grid& g, const Config& cfg) {
    int N = g.N_total;
    int ax = (DIM == 2) ? 1 : 2;  // axial direction index

    // Physical domain limits in axial direction (from grid geometry)
    double y_min_fluid = -cfg.L_upstream;
    double y_max_fluid = cfg.L_wire + cfg.L_downstream;

    // Use per-node delta for AMR grids, global delta otherwise
    double delta_global = cfg.use_amr ? cfg.delta_coarse : cfg.delta;

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < N; ++i) {
        if (g.node_type[i] != FLUID) continue;

        double delta = (!g.delta_local.empty()) ? g.delta_local[i] : delta_global;
        double y = g.pos[i][ax];
        bool near_inlet = (y - y_min_fluid < delta);
        bool near_outlet = (y_max_fluid - y < delta);

        if (!near_inlet && !near_outlet) continue;

        // Extrapolate C from interior-side fluid neighbors only
        double c_avg = 0.0;
        int count = 0;
        for (int jj = g.nbr_offset[i]; jj < g.nbr_offset[i + 1]; ++jj) {
            int j = g.nbr_index[jj];
            if (g.node_type[j] != FLUID) continue;

            double yj = g.pos[j][ax];
            // Only use neighbors deeper in the interior
            if (near_outlet && yj < y) {
                c_avg += f.C[j];
                count++;
            } else if (near_inlet && yj > y) {
                c_avg += f.C[j];
                count++;
            }
        }

        if (count > 0) {
            f.C[i] = c_avg / count;
        }
    }
}

// ---------------------------------------------------------------------------
// Solid surface BC: zero velocity on Mg pin
// ---------------------------------------------------------------------------
void apply_solid_surface_bc(Fields& f, const Grid& g) {
    int N = g.N_total;

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < N; ++i) {
        if (g.node_type[i] == SOLID_MG) {
            f.vel[i] = vec_zero();
        }
    }
}

// ---------------------------------------------------------------------------
// Update node types after dissolution (SOLID_MG → FLUID)
// ---------------------------------------------------------------------------
void update_node_types_after_dissolution(Grid& g, const Fields& f) {
    int N = g.N_total;
    for (int i = 0; i < N; ++i) {
        if (f.phase[i] == 1 && g.node_type[i] == SOLID_MG) {
            g.node_type[i] = FLUID;
        }
    }
}
