#pragma once
#include "grid.h"
#include "fields.h"
#include "config.h"
#include <string>
#include <vector>

class VTKWriter {
public:
    void write(const std::string& filename,
               const Grid& grid, const Fields& fields,
               const Config& cfg);

    // Write PVD collection file for time series
    void write_pvd(const std::string& filename) const;

    // Add a timestep entry
    void add_timestep(double time, const std::string& vti_file);

private:
    struct PVDEntry {
        double time;
        std::string file;
    };
    std::vector<PVDEntry> pvd_entries_;
};
