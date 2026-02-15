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

    // Write VTU (UnstructuredGrid) for AMR grids
    void write_vtu(const std::string& filename,
                   const Grid& grid, const Fields& fields,
                   const Config& cfg);

    // Set PVD path so it gets rewritten after each add_timestep() call.
    // This ensures the PVD is always up-to-date (crash-safe).
    void set_pvd_path(const std::string& path);

    // Write PVD collection file for time series
    void write_pvd(const std::string& filename) const;

    // Add a timestep entry and rewrite PVD if path was set
    void add_timestep(double time, const std::string& vti_file);

private:
    struct PVDEntry {
        double time;
        std::string file;
    };
    std::vector<PVDEntry> pvd_entries_;
    std::string pvd_path_;  // if set, PVD is rewritten after each add_timestep()
};
