#include "grid.h"
#include <cmath>
#include <cstdio>
#include <omp.h>

bool Grid::in_wire(double x, double y, double z) const {
    // Wire is a cylinder along z-axis (or the last axis in 2D)
    // 2D: x = radial (r), y = axial (z). Wire: |x| <= R_wire, 0 <= y <= L_wire
    // 3D: Wire: x^2+y^2 <= R_wire^2, 0 <= z <= L_wire
    // We use the stored origin to get absolute coordinates
    // But pos[] already stores absolute positions, so caller passes those directly
    if constexpr (DIM == 2) {
        (void)z;
        // x = radial, y = axial
        return (std::abs(x) <= 40.0e-6) && (y >= 0.0) && (y <= 400.0e-6);
    } else {
        double r2 = x * x + y * y;
        return (r2 <= 40.0e-6 * 40.0e-6) && (z >= 0.0) && (z <= 400.0e-6);
    }
}

bool Grid::in_tube(double x, double y, double z) const {
    if constexpr (DIM == 2) {
        (void)z;
        return std::abs(x) <= 150.0e-6;
    } else {
        double r2 = x * x + y * y;
        return r2 <= 150.0e-6 * 150.0e-6;
    }
}

void Grid::build(const Config& cfg) {
    dx = cfg.dx;
    delta = cfg.delta;
    m = cfg.m_ratio;

    // Domain extents
    double z_min = -cfg.L_upstream - m * dx;   // ghost region upstream
    double z_max = cfg.L_wire + cfg.L_downstream + m * dx; // ghost region downstream

    if constexpr (DIM == 2) {
        // 2D r-z: x = radial [-R_tube - m*dx, R_tube + m*dx], y = axial [z_min, z_max]
        double r_min = -cfg.R_tube - m * dx;
        double r_max =  cfg.R_tube + m * dx;

        Nx = static_cast<int>(std::round((r_max - r_min) / dx)) + 1;
        Ny = static_cast<int>(std::round((z_max - z_min) / dx)) + 1;
        Nz = 1;

        origin_x = r_min;
        origin_y = z_min;
        origin_z = 0.0;
    } else {
        // 3D: x,y = cross-section, z = axial
        double xy_min = -cfg.R_tube - m * dx;
        double xy_max =  cfg.R_tube + m * dx;

        Nx = static_cast<int>(std::round((xy_max - xy_min) / dx)) + 1;
        Ny = Nx; // square cross-section grid
        Nz = static_cast<int>(std::round((z_max - z_min) / dx)) + 1;

        origin_x = xy_min;
        origin_y = xy_min;
        origin_z = z_min;
    }

    N_total = Nx * Ny * Nz;
    std::printf("Grid: Nx=%d Ny=%d Nz=%d  N_total=%d\n", Nx, Ny, Nz, N_total);

    pos.resize(N_total);
    node_type.resize(N_total);

    // Fill positions and classify nodes
    #pragma omp parallel for schedule(static)
    for (int n = 0; n < N_total; ++n) {
        int i, j, k;
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

        double px = origin_x + i * dx;
        double py = origin_y + j * dx;
        double pz = (DIM == 3) ? (origin_z + k * dx) : 0.0;

        pos[n] = make_vec(px, py, pz);

        // Classify: axial coordinate
        double axial = (DIM == 2) ? py : pz;
        double radial;
        if constexpr (DIM == 2) {
            radial = std::abs(px);
        } else {
            radial = std::sqrt(px * px + py * py);
        }

        double z_phys_min = -cfg.L_upstream;
        double z_phys_max = cfg.L_wire + cfg.L_downstream;

        // Ghost layers for inlet/outlet
        if (axial < z_phys_min) {
            node_type[n] = INLET;
        } else if (axial > z_phys_max) {
            node_type[n] = OUTLET;
        }
        // Inside tube?
        else if (radial <= cfg.R_tube) {
            // Check if inside wire
            bool wire;
            if constexpr (DIM == 2) {
                wire = (std::abs(px) <= cfg.R_wire) &&
                       (py >= 0.0) && (py <= cfg.L_wire);
            } else {
                wire = (px * px + py * py <= cfg.R_wire * cfg.R_wire) &&
                       (pz >= 0.0) && (pz <= cfg.L_wire);
            }

            if (wire) {
                node_type[n] = SOLID_MG;
            } else {
                node_type[n] = FLUID;
            }
        }
        // Tube wall region (between R_tube and R_tube + m*dx)
        else if (radial <= cfg.R_tube + m * dx + 0.5 * dx) {
            node_type[n] = WALL;
        } else {
            node_type[n] = OUTSIDE;
        }
    }

    // Count node types
    int counts[6] = {};
    for (int n = 0; n < N_total; ++n) counts[node_type[n]]++;
    std::printf("Node types: FLUID=%d SOLID_MG=%d WALL=%d INLET=%d OUTLET=%d OUTSIDE=%d\n",
                counts[0], counts[1], counts[2], counts[3], counts[4], counts[5]);
}

void Grid::build_neighbors() {
    Timer t("build_neighbors");

    // Precompute offset stencil: all (di,dj,dk) with ||(di,dj,dk)||*dx <= delta + 0.5*dx
    struct Offset { int di, dj, dk; double dist; Vec evec; };
    std::vector<Offset> stencil;

    int mext = m + 1; // search slightly beyond to catch partial volumes
    for (int dk = (DIM == 3 ? -mext : 0); dk <= (DIM == 3 ? mext : 0); ++dk) {
        for (int dj = -mext; dj <= mext; ++dj) {
            for (int di = -mext; di <= mext; ++di) {
                if (di == 0 && dj == 0 && dk == 0) continue;
                double r;
                if constexpr (DIM == 2) {
                    r = std::sqrt(double(di * di + dj * dj)) * dx;
                } else {
                    r = std::sqrt(double(di * di + dj * dj + dk * dk)) * dx;
                }
                if (r <= delta + 0.5 * dx) {
                    Offset off;
                    off.di = di;
                    off.dj = dj;
                    off.dk = dk;
                    off.dist = r;
                    // Unit vector
                    off.evec = make_vec(di * dx / r, dj * dx / r, dk * dx / r);
                    stencil.push_back(off);
                }
            }
        }
    }
    std::printf("Neighbor stencil size: %zu\n", stencil.size());

    // Build CSR
    nbr_offset.resize(N_total + 1, 0);

    // First pass: count neighbors per node
    #pragma omp parallel for schedule(static)
    for (int n = 0; n < N_total; ++n) {
        if (node_type[n] == OUTSIDE) {
            nbr_offset[n + 1] = 0;
            continue;
        }

        int i, j, k;
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

        int count = 0;
        for (auto& off : stencil) {
            int ni = i + off.di;
            int nj = j + off.dj;
            int nk = k + off.dk;
            if (ni < 0 || ni >= Nx || nj < 0 || nj >= Ny) continue;
            if constexpr (DIM == 3) {
                if (nk < 0 || nk >= Nz) continue;
            }
            int nn = idx(ni, nj, nk);
            if (node_type[nn] == OUTSIDE) continue;
            count++;
        }
        nbr_offset[n + 1] = count;
    }

    // Prefix sum
    for (int n = 0; n < N_total; ++n) {
        nbr_offset[n + 1] += nbr_offset[n];
    }
    int total_nbrs = nbr_offset[N_total];
    std::printf("Total neighbor entries: %d (avg %.1f per node)\n",
                total_nbrs, (double)total_nbrs / N_total);

    nbr_index.resize(total_nbrs);
    nbr_dist.resize(total_nbrs);
    nbr_evec.resize(total_nbrs);
    nbr_vol.resize(total_nbrs);

    double dx_dim = 1.0;
    for (int d = 0; d < DIM; ++d) dx_dim *= dx;

    // Second pass: fill neighbor data
    #pragma omp parallel for schedule(static)
    for (int n = 0; n < N_total; ++n) {
        if (node_type[n] == OUTSIDE) continue;

        int i, j, k;
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

        int write_pos = nbr_offset[n];
        for (auto& off : stencil) {
            int ni = i + off.di;
            int nj = j + off.dj;
            int nk = k + off.dk;
            if (ni < 0 || ni >= Nx || nj < 0 || nj >= Ny) continue;
            if constexpr (DIM == 3) {
                if (nk < 0 || nk >= Nz) continue;
            }
            int nn = idx(ni, nj, nk);
            if (node_type[nn] == OUTSIDE) continue;

            // Partial volume correction (beta)
            double r = off.dist;
            double beta;
            if (r <= delta - 0.5 * dx) {
                beta = 1.0;
            } else if (r <= delta + 0.5 * dx) {
                beta = (delta + 0.5 * dx - r) / dx;
            } else {
                beta = 0.0;
            }

            nbr_index[write_pos] = nn;
            nbr_dist[write_pos] = r;
            nbr_evec[write_pos] = off.evec;
            nbr_vol[write_pos] = beta * dx_dim;
            write_pos++;
        }
    }

    t.report();
}
