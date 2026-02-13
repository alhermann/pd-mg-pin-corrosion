#include "boundary.h"
#include <cmath>
#include <omp.h>

void apply_inlet_bc(Fields& f, const Grid& g, const Config& cfg) {
    // Parabolic Poiseuille profile at inlet
    // 2D channel: v(x) = (3/2)*U_in*(1 - (x/R_tube)^2)
    // 3D tube:    v(r) = 2*U_in*(1 - (r/R_tube)^2)
    int N = g.N_total;
    double R2 = cfg.R_tube * cfg.R_tube;

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < N; ++i) {
        if (g.node_type[i] == INLET) {
            double px = g.pos[i][0];
            double r2;
            double v_axial;

            if constexpr (DIM == 2) {
                r2 = px * px;
                double r_ratio2 = r2 / R2;
                if (r_ratio2 > 1.0) r_ratio2 = 1.0;
                v_axial = 1.5 * cfg.U_in * (1.0 - r_ratio2);
            } else {
                double py = g.pos[i][1];
                r2 = px * px + py * py;
                double r_ratio2 = r2 / R2;
                if (r_ratio2 > 1.0) r_ratio2 = 1.0;
                v_axial = 2.0 * cfg.U_in * (1.0 - r_ratio2);
            }

            Vec v_in = vec_zero();
            if constexpr (DIM == 2) {
                v_in[1] = v_axial;  // axial direction = y
            } else {
                v_in[2] = v_axial;  // axial direction = z
            }

            f.vel[i] = v_in;
            f.rho[i] = cfg.rho_f;
            f.pressure[i] = 0.0;
            f.C[i] = cfg.C_liquid_init;
        }
    }
}

void apply_outlet_bc(Fields& f, const Grid& g, const Config& cfg) {
    int N = g.N_total;

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < N; ++i) {
        if (g.node_type[i] == OUTLET) {
            f.rho[i] = cfg.rho_f;
            f.pressure[i] = 0.0;
            // Extrapolate velocity from nearest fluid neighbor (zero-gradient)
            bool found = false;
            for (int jj = g.nbr_offset[i]; jj < g.nbr_offset[i + 1]; ++jj) {
                int j = g.nbr_index[jj];
                if (g.node_type[j] == FLUID) {
                    f.vel[i] = f.vel[j];
                    f.C[i] = f.C[j];
                    found = true;
                    break;
                }
            }
            if (!found) {
                Vec v_in = vec_zero();
                if constexpr (DIM == 2) {
                    v_in[1] = cfg.U_in;
                } else {
                    v_in[2] = cfg.U_in;
                }
                f.vel[i] = v_in;
                f.C[i] = cfg.C_liquid_init;
            }
        }
    }
}

static void apply_wall_mirror(Fields& f, const Grid& g, bool use_new) {
    // FNM: wall fictitious nodes mirror nearest fluid velocity with opposite sign
    int N = g.N_total;

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < N; ++i) {
        if (g.node_type[i] == WALL) {
            double best_dist = 1e30;
            int best_j = -1;
            for (int jj = g.nbr_offset[i]; jj < g.nbr_offset[i + 1]; ++jj) {
                int j = g.nbr_index[jj];
                if (g.node_type[j] == FLUID && g.nbr_dist[jj] < best_dist) {
                    best_dist = g.nbr_dist[jj];
                    best_j = j;
                }
            }
            if (best_j >= 0) {
                if (use_new) {
                    for (int d = 0; d < DIM; ++d) {
                        f.vel_new[i][d] = -f.vel_new[best_j][d];
                    }
                } else {
                    for (int d = 0; d < DIM; ++d) {
                        f.vel[i][d] = -f.vel[best_j][d];
                    }
                }
            } else {
                if (use_new) {
                    f.vel_new[i] = vec_zero();
                } else {
                    f.vel[i] = vec_zero();
                }
            }
            f.C[i] = 0.0; // no dissolved Mg at tube wall
        }
    }
}

void apply_wall_bc(Fields& f, const Grid& g) {
    apply_wall_mirror(f, g, false);
}

void apply_wall_bc_new(Fields& f, const Grid& g) {
    apply_wall_mirror(f, g, true);
}

void apply_solid_surface_bc(Fields& f, const Grid& g) {
    int N = g.N_total;

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < N; ++i) {
        if (g.node_type[i] == SOLID_MG) {
            f.vel[i] = vec_zero();
        }
    }
}

void update_node_types_after_dissolution(Grid& g, const Fields& f) {
    int N = g.N_total;
    for (int i = 0; i < N; ++i) {
        if (f.phase[i] == 1 && g.node_type[i] == SOLID_MG) {
            g.node_type[i] = FLUID;
        }
    }
}
