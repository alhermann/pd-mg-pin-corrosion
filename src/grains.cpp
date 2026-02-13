#include "grains.h"
#include <random>
#include <cstdio>
#include <cmath>
#include <limits>
#include <omp.h>

void GrainStructure::generate(const Grid& grid, const Config& cfg, int seed) {
    Timer t("grain_generation");

    int N = grid.N_total;
    grain_id.resize(N, -1);
    is_grain_boundary.resize(N, false);

    // Collect solid node indices
    std::vector<int> solid_nodes;
    solid_nodes.reserve(N / 4);
    for (int i = 0; i < N; ++i) {
        if (grid.node_type[i] == SOLID_MG) {
            solid_nodes.push_back(i);
        }
    }

    if (solid_nodes.empty()) {
        n_grains = 0;
        std::printf("Grain generation: no solid nodes found.\n");
        return;
    }

    // Estimate number of grains from mean grain size
    double d_grain = cfg.grain_size_mean;
    double solid_area = solid_nodes.size() * std::pow(cfg.dx, DIM);
    double grain_area;
    if constexpr (DIM == 2) {
        grain_area = PI / 4.0 * d_grain * d_grain;
    } else {
        grain_area = PI / 6.0 * d_grain * d_grain * d_grain;
    }
    n_grains = std::max(1, static_cast<int>(std::round(solid_area / grain_area)));

    std::printf("Grain generation: %d solid nodes, estimated %d grains\n",
                (int)solid_nodes.size(), n_grains);

    // Scatter seed points randomly among solid nodes
    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> dist(0, (int)solid_nodes.size() - 1);

    std::vector<Vec> seeds(n_grains);
    for (int g = 0; g < n_grains; ++g) {
        int si = solid_nodes[dist(rng)];
        seeds[g] = grid.pos[si];
    }

    // Assign each solid node to nearest seed (Voronoi)
    #pragma omp parallel for schedule(dynamic, 256)
    for (int idx = 0; idx < (int)solid_nodes.size(); ++idx) {
        int ni = solid_nodes[idx];
        const Vec& p = grid.pos[ni];
        double best_dist = std::numeric_limits<double>::max();
        int best_grain = 0;
        for (int g = 0; g < n_grains; ++g) {
            double d = norm(p - seeds[g]);
            if (d < best_dist) {
                best_dist = d;
                best_grain = g;
            }
        }
        grain_id[ni] = best_grain;
    }

    // Identify grain boundaries: any solid node with an IMMEDIATE neighbor
    // (within sqrt(DIM)*dx, i.e. direct grid neighbors only, NOT full PD horizon)
    // of a different grain_id.
    double gb_cutoff = std::sqrt(static_cast<double>(DIM)) * cfg.dx * 1.01; // slightly > diagonal
    #pragma omp parallel for schedule(static)
    for (int idx = 0; idx < (int)solid_nodes.size(); ++idx) {
        int ni = solid_nodes[idx];
        int gi = grain_id[ni];
        for (int jj = grid.nbr_offset[ni]; jj < grid.nbr_offset[ni + 1]; ++jj) {
            int nj = grid.nbr_index[jj];
            if (grid.nbr_dist[jj] > gb_cutoff) continue; // skip non-immediate neighbors
            if (grid.node_type[nj] == SOLID_MG && grain_id[nj] != gi) {
                is_grain_boundary[ni] = true;
                break;
            }
        }
    }

    // Dilate grain boundary mask (using only immediate neighbors)
    for (int pass = 0; pass < cfg.gb_width_cells; ++pass) {
        std::vector<bool> gb_new = is_grain_boundary;
        #pragma omp parallel for schedule(static)
        for (int idx = 0; idx < (int)solid_nodes.size(); ++idx) {
            int ni = solid_nodes[idx];
            if (is_grain_boundary[ni]) continue;
            for (int jj = grid.nbr_offset[ni]; jj < grid.nbr_offset[ni + 1]; ++jj) {
                int nj = grid.nbr_index[jj];
                if (grid.nbr_dist[jj] > gb_cutoff) continue;
                if (is_grain_boundary[nj]) {
                    gb_new[ni] = true;
                    break;
                }
            }
        }
        is_grain_boundary = gb_new;
    }

    int n_gb = 0;
    for (int ni : solid_nodes) {
        if (is_grain_boundary[ni]) n_gb++;
    }
    std::printf("Grain boundaries: %d nodes (%.1f%% of solid)\n",
                n_gb, 100.0 * n_gb / solid_nodes.size());

    t.report();
}
