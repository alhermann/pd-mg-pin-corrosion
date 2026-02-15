#pragma once
#include "utils.h"
#include "config.h"
#include <vector>
#include <cstdint>

struct Fields; // forward declaration

enum NodeType : uint8_t {
    FLUID      = 0,
    SOLID_MG   = 1,
    WALL       = 2,
    INLET      = 3,
    OUTLET     = 4,
    OUTSIDE    = 5,
    FICTITIOUS = 6  // AMR fictitious coupling node
};

struct Grid {
    int Nx, Ny, Nz;
    double dx;
    double delta;
    int m;
    int N_total;

    // Domain origin (lower corner)
    double origin_x, origin_y, origin_z;

    // Geometry (stored from config for helper functions)
    double R_wire, L_wire, R_tube;

    std::vector<Vec> pos;
    std::vector<NodeType> node_type;

    // CSR neighbor list
    std::vector<int> nbr_offset;
    std::vector<int> nbr_index;
    std::vector<double> nbr_dist;
    std::vector<Vec> nbr_evec;
    std::vector<double> nbr_vol;

    // AMR per-node data (empty when use_amr=0)
    std::vector<double> dx_local;     // per-node grid spacing
    std::vector<double> delta_local;  // per-node horizon
    std::vector<int> grid_level;      // 0=fine, 1=coarse

    // Fictitious node IDW interpolation data (CSR format)
    std::vector<int> fict_offset;     // [N_total+1]
    std::vector<int> fict_source;     // source node global indices
    std::vector<double> fict_weight;  // normalized IDW weights

    void build(const Config& cfg);
    void build_neighbors();
    void build_amr(const Config& cfg);
    void build_neighbors_celllist(const Config& cfg);
    void update_fictitious(Fields& fields) const;

    inline int idx(int i, int j, int k = 0) const {
        if constexpr (DIM == 2) {
            return j * Nx + i;
        } else {
            return k * (Nx * Ny) + j * Nx + i;
        }
    }

    bool in_wire(double x, double y, double z = 0.0) const;
    bool in_tube(double x, double y, double z = 0.0) const;
};
