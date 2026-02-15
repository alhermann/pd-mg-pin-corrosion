#include "grid.h"
#include "fields.h"
#include <cmath>
#include <cstdio>
#include <unordered_map>
#include <algorithm>
#include <omp.h>

bool Grid::in_wire(double x, double y, double z) const {
    if constexpr (DIM == 2) {
        (void)z;
        return (std::abs(x) <= R_wire) && (y >= 0.0) && (y <= L_wire);
    } else {
        double r2 = x * x + y * y;
        return (r2 <= R_wire * R_wire) && (z >= 0.0) && (z <= L_wire);
    }
}

bool Grid::in_tube(double x, double y, double z) const {
    if constexpr (DIM == 2) {
        (void)z;
        return std::abs(x) <= R_tube;
    } else {
        double r2 = x * x + y * y;
        return r2 <= R_tube * R_tube;
    }
}

void Grid::build(const Config& cfg) {
    dx = cfg.dx;
    delta = cfg.delta;
    m = cfg.m_ratio;
    R_wire = cfg.R_wire;
    L_wire = cfg.L_wire;
    R_tube = cfg.R_tube;

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

        // Ghost layers for inlet/outlet (only within tube cross-section)
        if (axial < z_phys_min) {
            if (radial <= cfg.R_tube) {
                node_type[n] = INLET;
            } else if (radial <= cfg.R_tube + m * dx + 0.5 * dx) {
                node_type[n] = WALL;
            } else {
                node_type[n] = OUTSIDE;
            }
        } else if (axial > z_phys_max) {
            if (radial <= cfg.R_tube) {
                node_type[n] = OUTLET;
            } else if (radial <= cfg.R_tube + m * dx + 0.5 * dx) {
                node_type[n] = WALL;
            } else {
                node_type[n] = OUTSIDE;
            }
        }
        // Inside physical domain, inside tube?
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

// ============================================================================
// AMR: Two-level static refinement
// Fine grid (dx) near wire, coarse grid (dx_coarse) in far-field fluid.
// Fictitious nodes bridge the two grid levels with IDW interpolation.
// ============================================================================

// Helper: classify a node at position (px, py, pz) given local dx
static NodeType classify_node(double px, double py, double pz,
                               const Config& cfg, int m_local, double dx_local) {
    double axial = (DIM == 2) ? py : pz;
    double radial;
    if constexpr (DIM == 2) {
        radial = std::abs(px);
    } else {
        radial = std::sqrt(px * px + py * py);
    }

    double z_phys_min = -cfg.L_upstream;
    double z_phys_max = cfg.L_wire + cfg.L_downstream;

    if (axial < z_phys_min) {
        if (radial <= cfg.R_tube) return INLET;
        if (radial <= cfg.R_tube + m_local * dx_local + 0.5 * dx_local) return WALL;
        return OUTSIDE;
    }
    if (axial > z_phys_max) {
        if (radial <= cfg.R_tube) return OUTLET;
        if (radial <= cfg.R_tube + m_local * dx_local + 0.5 * dx_local) return WALL;
        return OUTSIDE;
    }
    if (radial <= cfg.R_tube) {
        bool wire;
        if constexpr (DIM == 2) {
            wire = (std::abs(px) <= cfg.R_wire) && (py >= 0.0) && (py <= cfg.L_wire);
        } else {
            wire = (px * px + py * py <= cfg.R_wire * cfg.R_wire) &&
                   (pz >= 0.0) && (pz <= cfg.L_wire);
        }
        return wire ? SOLID_MG : FLUID;
    }
    if (radial <= cfg.R_tube + m_local * dx_local + 0.5 * dx_local) return WALL;
    return OUTSIDE;
}

// Is position (x, y) inside the fine zone?
static bool in_fine_zone(double x, double y, double fine_r, double fine_z_min, double fine_z_max) {
    if constexpr (DIM == 2) {
        return (std::abs(x) <= fine_r) && (y >= fine_z_min) && (y <= fine_z_max);
    } else {
        return (std::sqrt(x * x + y * y) <= fine_r) && (y >= fine_z_min) && (y <= fine_z_max);
    }
}

void Grid::build_amr(const Config& cfg) {
    Timer t("build_amr");

    dx = cfg.dx;
    delta = cfg.delta;
    m = cfg.m_ratio;
    R_wire = cfg.R_wire;
    L_wire = cfg.L_wire;
    R_tube = cfg.R_tube;

    double dx_f = cfg.dx;
    double dx_c = cfg.dx_coarse;
    double delta_f = cfg.delta;
    double delta_c = cfg.delta_coarse;

    // Fine zone bounding box
    double fine_r = cfg.R_wire + cfg.amr_buffer;
    double fine_z_min = -cfg.amr_buffer;
    double fine_z_max = cfg.L_wire + cfg.amr_buffer;

    // Full domain extents (same as uniform grid)
    double z_domain_min = -cfg.L_upstream - m * dx_c;
    double z_domain_max = cfg.L_wire + cfg.L_downstream + m * dx_c;

    double r_domain_min, r_domain_max;
    if constexpr (DIM == 2) {
        r_domain_min = -cfg.R_tube - m * dx_c;
        r_domain_max =  cfg.R_tube + m * dx_c;
    } else {
        r_domain_min = -cfg.R_tube - m * dx_c;
        r_domain_max =  cfg.R_tube + m * dx_c;
    }

    std::printf("AMR: fine zone |x|<%.0fum, z in [%.0f, %.0f] um\n",
                fine_r * 1e6, fine_z_min * 1e6, fine_z_max * 1e6);
    std::printf("AMR: dx_fine=%.1fum, dx_coarse=%.1fum (ratio=%d)\n",
                dx_f * 1e6, dx_c * 1e6, cfg.amr_ratio);

    // Collect nodes: fine zone at dx_f, coarse zone at dx_c
    pos.clear();
    node_type.clear();
    dx_local.clear();
    delta_local.clear();
    grid_level.clear();

    // Step 1: Place fine nodes covering the fine zone only (no expansion).
    // Fictitious nodes will bridge the gap to the coarse zone.
    int n_fine = 0;
    {
        int nx_f = static_cast<int>(std::round((r_domain_max - r_domain_min) / dx_f)) + 1;
        int ny_f = static_cast<int>(std::round((z_domain_max - z_domain_min) / dx_f)) + 1;

        for (int jj = 0; jj < ny_f; ++jj) {
            double py = z_domain_min + jj * dx_f;
            for (int ii = 0; ii < nx_f; ++ii) {
                double px = r_domain_min + ii * dx_f;
                double pz = 0.0;

                // Only place fine nodes within the fine zone
                if (!in_fine_zone(px, py, fine_r, fine_z_min, fine_z_max))
                    continue;

                NodeType nt = classify_node(px, py, pz, cfg, m, dx_f);
                if (nt == OUTSIDE) continue;

                pos.push_back(make_vec(px, py, pz));
                node_type.push_back(nt);
                dx_local.push_back(dx_f);
                delta_local.push_back(delta_f);
                grid_level.push_back(0);
                n_fine++;
            }
        }
    }
    std::printf("AMR: %d fine nodes placed\n", n_fine);

    // Step 2: Place coarse nodes covering the rest of the domain
    int n_coarse = 0;
    {
        int nx_c = static_cast<int>(std::round((r_domain_max - r_domain_min) / dx_c)) + 1;
        int ny_c = static_cast<int>(std::round((z_domain_max - z_domain_min) / dx_c)) + 1;

        for (int jj = 0; jj < ny_c; ++jj) {
            double py = z_domain_min + jj * dx_c;
            for (int ii = 0; ii < nx_c; ++ii) {
                double px = r_domain_min + ii * dx_c;
                double pz = 0.0;

                // Skip nodes inside the fine zone (already covered at fine resolution)
                if (in_fine_zone(px, py, fine_r, fine_z_min, fine_z_max))
                    continue;

                NodeType nt = classify_node(px, py, pz, cfg, m, dx_c);
                if (nt == OUTSIDE) continue;

                pos.push_back(make_vec(px, py, pz));
                node_type.push_back(nt);
                dx_local.push_back(dx_c);
                delta_local.push_back(delta_c);
                grid_level.push_back(1);
                n_coarse++;
            }
        }
    }
    std::printf("AMR: %d coarse nodes placed\n", n_coarse);

    int N_real = n_fine + n_coarse;

    // Step 3: Place auxiliary (fictitious) nodes at correct grid positions
    // Per Shojaei et al. (IJMS 144, 2018):
    //   Auxiliary fine  (grid_level=0): dx_f spacing, OUTSIDE fine zone, IDW from coarse real
    //   Auxiliary coarse (grid_level=1): dx_c spacing, INSIDE fine zone near boundary, IDW from fine real

    // Build spatial hash for fast lookup of IDW source nodes
    double cell_size = std::max(delta_f, delta_c);
    struct CellKey {
        int ix, iy;
        bool operator==(const CellKey& o) const { return ix == o.ix && iy == o.iy; }
    };
    struct CellHash {
        size_t operator()(const CellKey& k) const {
            return std::hash<long long>()(((long long)k.ix << 32) | (unsigned int)k.iy);
        }
    };
    std::unordered_map<CellKey, std::vector<int>, CellHash> cell_map;

    for (int i = 0; i < N_real; ++i) {
        int ix = static_cast<int>(std::floor((pos[i][0] - r_domain_min) / cell_size));
        int iy = static_cast<int>(std::floor((pos[i][1] - z_domain_min) / cell_size));
        cell_map[{ix, iy}].push_back(i);
    }

    struct FictInfo {
        Vec position;
        int fict_grid_level;
        std::vector<int> sources;
        std::vector<double> weights;
    };
    std::vector<FictInfo> fict_nodes;

    auto find_nodes_in_radius = [&](double cx, double cy, double radius,
                                     int required_level) -> std::vector<int> {
        std::vector<int> result;
        int cr = static_cast<int>(std::ceil(radius / cell_size)) + 1;
        int cix = static_cast<int>(std::floor((cx - r_domain_min) / cell_size));
        int ciy = static_cast<int>(std::floor((cy - z_domain_min) / cell_size));

        for (int dy = -cr; dy <= cr; ++dy) {
            for (int ddx = -cr; ddx <= cr; ++ddx) {
                auto it = cell_map.find({cix + ddx, ciy + dy});
                if (it == cell_map.end()) continue;
                for (int idx : it->second) {
                    if (grid_level[idx] != required_level) continue;
                    if (node_type[idx] == OUTSIDE) continue;
                    double dr = pos[idx][0] - cx;
                    double dz = pos[idx][1] - cy;
                    double dist = std::sqrt(dr * dr + dz * dz);
                    if (dist <= radius) result.push_back(idx);
                }
            }
        }
        return result;
    };

    // Auxiliary fine nodes (grid_level=0, dx_f spacing):
    // Fine grid positions OUTSIDE the fine zone but within delta_f + dx_f of it
    {
        double aux_fine_r     = fine_r     + delta_f + dx_f;
        double aux_fine_z_min = fine_z_min - delta_f - dx_f;
        double aux_fine_z_max = fine_z_max + delta_f + dx_f;

        int nx_f = static_cast<int>(std::round((r_domain_max - r_domain_min) / dx_f)) + 1;
        int ny_f = static_cast<int>(std::round((z_domain_max - z_domain_min) / dx_f)) + 1;

        for (int jj = 0; jj < ny_f; ++jj) {
            double py = z_domain_min + jj * dx_f;
            for (int ii = 0; ii < nx_f; ++ii) {
                double px = r_domain_min + ii * dx_f;

                // Must be OUTSIDE the fine zone (real fine nodes already there)
                if (in_fine_zone(px, py, fine_r, fine_z_min, fine_z_max)) continue;
                // Must be within the auxiliary band
                if (!in_fine_zone(px, py, aux_fine_r, aux_fine_z_min, aux_fine_z_max)) continue;
                // Must not be OUTSIDE the domain
                NodeType nt = classify_node(px, py, 0.0, cfg, m, dx_f);
                if (nt == OUTSIDE) continue;

                // IDW sources = nearby coarse REAL nodes
                auto coarse_sources = find_nodes_in_radius(px, py, delta_c, 1);
                if (coarse_sources.empty()) continue;

                FictInfo fi;
                fi.position = make_vec(px, py, 0.0);
                fi.fict_grid_level = 0;
                double W = 0.0;
                for (int s : coarse_sources) {
                    double dr = pos[s][0] - px;
                    double dz = pos[s][1] - py;
                    double d2 = dr * dr + dz * dz;
                    if (d2 < 1e-30) d2 = 1e-30;
                    double w = 1.0 / d2;
                    fi.sources.push_back(s);
                    fi.weights.push_back(w);
                    W += w;
                }
                for (auto& w : fi.weights) w /= W;
                fict_nodes.push_back(fi);
            }
        }
    }

    // Auxiliary coarse nodes (grid_level=1, dx_c spacing):
    // Coarse grid positions INSIDE the fine zone but near its boundary
    {
        double inner_r     = fine_r     - delta_c - dx_c;
        double inner_z_min = fine_z_min + delta_c + dx_c;
        double inner_z_max = fine_z_max - delta_c - dx_c;

        int nx_c = static_cast<int>(std::round((r_domain_max - r_domain_min) / dx_c)) + 1;
        int ny_c = static_cast<int>(std::round((z_domain_max - z_domain_min) / dx_c)) + 1;

        for (int jj = 0; jj < ny_c; ++jj) {
            double py = z_domain_min + jj * dx_c;
            for (int ii = 0; ii < nx_c; ++ii) {
                double px = r_domain_min + ii * dx_c;

                // Must be INSIDE the fine zone
                if (!in_fine_zone(px, py, fine_r, fine_z_min, fine_z_max)) continue;
                // Must be near the boundary (not deep interior)
                if (in_fine_zone(px, py, inner_r, inner_z_min, inner_z_max)) continue;
                // Must not be OUTSIDE the domain
                NodeType nt = classify_node(px, py, 0.0, cfg, m, dx_c);
                if (nt == OUTSIDE) continue;

                // IDW sources = nearby fine REAL nodes
                auto fine_sources = find_nodes_in_radius(px, py, delta_f, 0);
                if (fine_sources.empty()) continue;

                FictInfo fi;
                fi.position = make_vec(px, py, 0.0);
                fi.fict_grid_level = 1;
                double W = 0.0;
                for (int s : fine_sources) {
                    double dr = pos[s][0] - px;
                    double dz = pos[s][1] - py;
                    double d2 = dr * dr + dz * dz;
                    if (d2 < 1e-30) d2 = 1e-30;
                    double w = 1.0 / d2;
                    fi.sources.push_back(s);
                    fi.weights.push_back(w);
                    W += w;
                }
                for (auto& w : fi.weights) w /= W;
                fict_nodes.push_back(fi);
            }
        }
    }

    // Step 4: Append fictitious nodes to the grid arrays
    int n_fict = (int)fict_nodes.size();
    std::printf("AMR: %d fictitious nodes\n", n_fict);

    for (auto& fi : fict_nodes) {
        pos.push_back(fi.position);
        node_type.push_back(FICTITIOUS);
        dx_local.push_back(fi.fict_grid_level == 0 ? dx_f : dx_c);
        delta_local.push_back(fi.fict_grid_level == 0 ? delta_f : delta_c);
        grid_level.push_back(fi.fict_grid_level);
    }

    N_total = (int)pos.size();
    Nx = 0; Ny = 0; Nz = 1;  // Not meaningful for AMR (non-structured)

    // Step 5: Build fictitious IDW interpolation data (CSR format)
    fict_offset.resize(N_total + 1, 0);
    fict_source.clear();
    fict_weight.clear();

    for (int fi = 0; fi < n_fict; ++fi) {
        int global_idx = N_real + fi;
        fict_offset[global_idx + 1] = (int)fict_nodes[fi].sources.size();
        for (int s = 0; s < (int)fict_nodes[fi].sources.size(); ++s) {
            fict_source.push_back(fict_nodes[fi].sources[s]);
            fict_weight.push_back(fict_nodes[fi].weights[s]);
        }
    }
    // Prefix sum
    for (int i = 0; i < N_total; ++i) {
        fict_offset[i + 1] += fict_offset[i];
    }

    // Count node types
    int counts[7] = {};
    for (int n = 0; n < N_total; ++n) counts[node_type[n]]++;
    std::printf("AMR Node types: FLUID=%d SOLID_MG=%d WALL=%d INLET=%d OUTLET=%d OUTSIDE=%d FICT=%d\n",
                counts[0], counts[1], counts[2], counts[3], counts[4], counts[5], counts[6]);
    std::printf("AMR total: %d nodes (fine=%d, coarse=%d, fict=%d)\n",
                N_total, n_fine, n_coarse, n_fict);

    // Store origin for cell-list use
    origin_x = r_domain_min;
    origin_y = z_domain_min;
    origin_z = 0.0;

    t.report();
}

// ============================================================================
// Cell-list based neighbor search for non-uniform AMR grids
// ============================================================================

void Grid::build_neighbors_celllist(const Config& cfg) {
    Timer t("build_neighbors_celllist");

    double dx_f = cfg.dx;
    double dx_c = cfg.dx_coarse;
    double delta_f = cfg.delta;
    double delta_c = cfg.delta_coarse;

    // Cell-list construction
    double h = std::min(delta_f, delta_c) / 2.0;
    if (h < 1e-30) h = dx_f;

    // Compute bounding box
    double xmin = 1e30, xmax = -1e30, ymin = 1e30, ymax = -1e30;
    for (int i = 0; i < N_total; ++i) {
        if (pos[i][0] < xmin) xmin = pos[i][0];
        if (pos[i][0] > xmax) xmax = pos[i][0];
        if (pos[i][1] < ymin) ymin = pos[i][1];
        if (pos[i][1] > ymax) ymax = pos[i][1];
    }

    int ncx = static_cast<int>(std::ceil((xmax - xmin) / h)) + 1;
    int ncy = static_cast<int>(std::ceil((ymax - ymin) / h)) + 1;

    // Build cell list using flat array
    std::vector<std::vector<int>> cells(ncx * ncy);
    for (int i = 0; i < N_total; ++i) {
        if (node_type[i] == OUTSIDE) continue;
        int ix = static_cast<int>(std::floor((pos[i][0] - xmin) / h));
        int iy = static_cast<int>(std::floor((pos[i][1] - ymin) / h));
        ix = std::max(0, std::min(ix, ncx - 1));
        iy = std::max(0, std::min(iy, ncy - 1));
        cells[iy * ncx + ix].push_back(i);
    }

    // First pass: count neighbors
    nbr_offset.assign(N_total + 1, 0);

    // Temporary per-node neighbor lists (need random access for two-pass)
    struct NbrEntry {
        int idx;
        double dist;
        Vec evec;
        double vol;
    };
    std::vector<std::vector<NbrEntry>> temp_nbrs(N_total);

    #pragma omp parallel for schedule(dynamic, 64)
    for (int i = 0; i < N_total; ++i) {
        if (node_type[i] == OUTSIDE) continue;

        double px = pos[i][0], py = pos[i][1];
        double di = delta_local[i];
        double dxi = dx_local[i];
        int glevel_i = grid_level[i];

        int search_r = static_cast<int>(std::ceil(di / h)) + 1;
        int cix = static_cast<int>(std::floor((px - xmin) / h));
        int ciy = static_cast<int>(std::floor((py - ymin) / h));

        auto& my_nbrs = temp_nbrs[i];

        for (int dy = -search_r; dy <= search_r; ++dy) {
            int cy = ciy + dy;
            if (cy < 0 || cy >= ncy) continue;
            for (int ddx = -search_r; ddx <= search_r; ++ddx) {
                int cx = cix + ddx;
                if (cx < 0 || cx >= ncx) continue;

                for (int j : cells[cy * ncx + cx]) {
                    if (j == i) continue;
                    if (node_type[j] == OUTSIDE) continue;

                    // Only include neighbors at same grid level or FICTITIOUS of same level
                    int glevel_j = grid_level[j];
                    if (node_type[j] == FICTITIOUS) {
                        if (glevel_j != glevel_i) continue;
                    } else {
                        if (glevel_j != glevel_i) continue;
                    }

                    double dr = pos[j][0] - px;
                    double dz_val = pos[j][1] - py;
                    double r = std::sqrt(dr * dr + dz_val * dz_val);

                    // Skip coincident nodes (e.g. real and fictitious at same position)
                    if (r < 1e-14) continue;

                    double dxj = dx_local[j];
                    if (r > di + 0.5 * dxj) continue;

                    // Partial volume correction (beta)
                    double beta;
                    if (r <= di - 0.5 * dxj) {
                        beta = 1.0;
                    } else {
                        beta = (di + 0.5 * dxj - r) / dxj;
                    }

                    double V_j = beta;
                    for (int d = 0; d < DIM; ++d) V_j *= dxj;

                    Vec evec;
                    double inv_r = 1.0 / r;
                    evec[0] = dr * inv_r;
                    evec[1] = dz_val * inv_r;
                    if constexpr (DIM >= 3) evec[2] = 0.0;

                    my_nbrs.push_back({j, r, evec, V_j});
                }
            }
        }
    }

    // Build CSR from temp_nbrs
    for (int i = 0; i < N_total; ++i) {
        nbr_offset[i + 1] = (int)temp_nbrs[i].size();
    }
    for (int i = 0; i < N_total; ++i) {
        nbr_offset[i + 1] += nbr_offset[i];
    }

    int total_nbrs = nbr_offset[N_total];
    nbr_index.resize(total_nbrs);
    nbr_dist.resize(total_nbrs);
    nbr_evec.resize(total_nbrs);
    nbr_vol.resize(total_nbrs);

    for (int i = 0; i < N_total; ++i) {
        int wp = nbr_offset[i];
        for (auto& e : temp_nbrs[i]) {
            nbr_index[wp] = e.idx;
            nbr_dist[wp] = e.dist;
            nbr_evec[wp] = e.evec;
            nbr_vol[wp] = e.vol;
            wp++;
        }
    }

    // Statistics
    int active = 0;
    for (int i = 0; i < N_total; ++i) {
        if (node_type[i] != OUTSIDE) active++;
    }
    std::printf("Cell-list neighbors: %d total entries (avg %.1f per active node)\n",
                total_nbrs, active > 0 ? (double)total_nbrs / active : 0.0);

    t.report();
}

// ============================================================================
// Update fictitious node values via IDW interpolation from source nodes
// ============================================================================

void Grid::update_fictitious(Fields& fields) const {
    if (fict_offset.empty()) return;

    for (int i = 0; i < N_total; ++i) {
        if (node_type[i] != FICTITIOUS) continue;

        double C_val = 0.0;
        double rho_val = 0.0;
        Vec vel_val = vec_zero();
        double p_val = 0.0;

        for (int p = fict_offset[i]; p < fict_offset[i + 1]; ++p) {
            int j = fict_source[p];
            double w = fict_weight[p];

            C_val += w * fields.C[j];
            rho_val += w * fields.rho[j];
            p_val += w * fields.pressure[j];
            for (int d = 0; d < DIM; ++d) {
                vel_val[d] += w * fields.vel[j][d];
            }
        }

        fields.C[i] = C_val;
        fields.rho[i] = rho_val;
        fields.pressure[i] = p_val;
        fields.vel[i] = vel_val;
    }
}
