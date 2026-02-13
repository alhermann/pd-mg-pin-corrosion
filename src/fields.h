#pragma once
#include "utils.h"
#include <vector>
#include <cstdint>
#include <algorithm>

struct Fields {
    // Flow
    std::vector<double> rho;
    std::vector<Vec> vel;
    std::vector<double> pressure;

    // Transport
    std::vector<double> C;
    std::vector<double> D_map;

    // Phase: 0=solid, 1=liquid
    std::vector<uint8_t> phase;
    std::vector<int> grain_id;
    std::vector<uint8_t> is_gb;  // grain boundary flag (1=GB, 0=interior)

    // Buffers for time integration
    std::vector<double> rho_new;
    std::vector<Vec> vel_new;
    std::vector<double> C_new;

    void allocate(int N) {
        rho.resize(N, 0.0);
        vel.resize(N, vec_zero());
        pressure.resize(N, 0.0);

        C.resize(N, 0.0);
        D_map.resize(N, 0.0);

        phase.resize(N, 1); // default liquid
        grain_id.resize(N, -1);
        is_gb.resize(N, 0);

        rho_new.resize(N, 0.0);
        vel_new.resize(N, vec_zero());
        C_new.resize(N, 0.0);
    }

    void swap_buffers() {
        std::swap(rho, rho_new);
        std::swap(vel, vel_new);
        std::swap(C, C_new);
    }
};
