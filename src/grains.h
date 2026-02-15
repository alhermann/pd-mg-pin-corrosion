#pragma once
#include "grid.h"
#include "config.h"
#include <vector>

struct GrainStructure {
    std::vector<int> grain_id;
    std::vector<bool> is_grain_boundary;
    std::vector<bool> is_precipitate;
    int n_grains;

    void generate(const Grid& grid, const Config& cfg, int seed = 42);
};
