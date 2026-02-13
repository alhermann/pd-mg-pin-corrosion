#include "vtk_writer.h"
#include <fstream>
#include <cstdio>
#include <sstream>
#include <iomanip>
#include <cmath>

static inline double safe_val(double v) {
    if (std::isnan(v) || std::isinf(v)) return 0.0;
    // Flush subnormal/denormalized values to zero — they produce
    // 3-digit exponents (e.g. 5.4e-323) that break some VTK parsers.
    if (v != 0.0 && std::abs(v) < 1e-300) return 0.0;
    return v;
}

void VTKWriter::write(const std::string& filename,
                      const Grid& grid, const Fields& fields,
                      const Config& cfg) {
    // Check for NaN and report
    int nan_count = 0;
    for (int i = 0; i < grid.N_total; ++i) {
        for (int d = 0; d < DIM; ++d) {
            if (std::isnan(fields.vel[i][d])) { nan_count++; break; }
        }
        if (std::isnan(fields.rho[i]) || std::isnan(fields.C[i]) ||
            std::isnan(fields.pressure[i])) nan_count++;
    }
    if (nan_count > 0) {
        std::fprintf(stderr, "WARNING: %d NaN values detected when writing %s\n",
                     nan_count, filename.c_str());
    }
    std::ofstream out(filename);
    if (!out.is_open()) {
        std::fprintf(stderr, "Error: Cannot open VTI file '%s'\n", filename.c_str());
        return;
    }

    int nx = grid.Nx;
    int ny = grid.Ny;
    int nz = (DIM == 3) ? grid.Nz : 1;

    out << "<?xml version=\"1.0\"?>\n";
    out << "<VTKFile type=\"ImageData\" version=\"1.0\" byte_order=\"LittleEndian\">\n";
    out << "  <ImageData WholeExtent=\"0 " << nx - 1
        << " 0 " << ny - 1
        << " 0 " << nz - 1 << "\""
        << " Origin=\"" << grid.origin_x << " " << grid.origin_y << " "
        << ((DIM == 3) ? grid.origin_z : 0.0) << "\""
        << " Spacing=\"" << grid.dx << " " << grid.dx << " " << grid.dx << "\">\n";
    out << "    <Piece Extent=\"0 " << nx - 1
        << " 0 " << ny - 1
        << " 0 " << nz - 1 << "\">\n";
    out << "      <PointData Scalars=\"phase\" Vectors=\"velocity\">\n";

    // Velocity (always 3-component for VTK)
    // WALL and OUTSIDE nodes are fictitious: zero them for clean visualization.
    out << "        <DataArray type=\"Float64\" Name=\"velocity\" "
        << "NumberOfComponents=\"3\" format=\"ascii\">\n";
    int N = grid.N_total;
    for (int i = 0; i < N; ++i) {
        bool fictitious = (grid.node_type[i] == WALL || grid.node_type[i] == OUTSIDE);
        if constexpr (DIM == 2) {
            out << "          "
                << (fictitious ? 0.0 : safe_val(fields.vel[i][0])) << " "
                << (fictitious ? 0.0 : safe_val(fields.vel[i][1])) << " 0\n";
        } else {
            out << "          "
                << (fictitious ? 0.0 : safe_val(fields.vel[i][0])) << " "
                << (fictitious ? 0.0 : safe_val(fields.vel[i][1])) << " "
                << (fictitious ? 0.0 : safe_val(fields.vel[i][2])) << "\n";
        }
    }
    out << "        </DataArray>\n";

    // Pressure
    // WALL and OUTSIDE: zero for clean visualization
    out << "        <DataArray type=\"Float64\" Name=\"pressure\" format=\"ascii\">\n";
    for (int i = 0; i < N; ++i) {
        bool fictitious = (grid.node_type[i] == WALL || grid.node_type[i] == OUTSIDE);
        out << "          " << (fictitious ? 0.0 : safe_val(fields.pressure[i])) << "\n";
    }
    out << "        </DataArray>\n";

    // Density
    out << "        <DataArray type=\"Float64\" Name=\"density\" format=\"ascii\">\n";
    for (int i = 0; i < N; ++i) {
        out << "          " << safe_val(fields.rho[i]) << "\n";
    }
    out << "        </DataArray>\n";

    // Concentration
    out << "        <DataArray type=\"Float64\" Name=\"concentration\" format=\"ascii\">\n";
    for (int i = 0; i < N; ++i) {
        out << "          " << safe_val(fields.C[i]) << "\n";
    }
    out << "        </DataArray>\n";

    // Phase
    out << "        <DataArray type=\"UInt8\" Name=\"phase\" format=\"ascii\">\n";
    for (int i = 0; i < N; ++i) {
        out << "          " << (int)fields.phase[i] << "\n";
    }
    out << "        </DataArray>\n";

    // Node type
    out << "        <DataArray type=\"UInt8\" Name=\"node_type\" format=\"ascii\">\n";
    for (int i = 0; i < N; ++i) {
        out << "          " << (int)grid.node_type[i] << "\n";
    }
    out << "        </DataArray>\n";

    // Grain ID
    out << "        <DataArray type=\"Int32\" Name=\"grain_id\" format=\"ascii\">\n";
    for (int i = 0; i < N; ++i) {
        out << "          " << fields.grain_id[i] << "\n";
    }
    out << "        </DataArray>\n";

    // Diffusivity map
    out << "        <DataArray type=\"Float64\" Name=\"D_map\" format=\"ascii\">\n";
    for (int i = 0; i < N; ++i) {
        out << "          " << safe_val(fields.D_map[i]) << "\n";
    }
    out << "        </DataArray>\n";

    // Grain boundary flag
    out << "        <DataArray type=\"UInt8\" Name=\"is_grain_boundary\" format=\"ascii\">\n";
    for (int i = 0; i < N; ++i) {
        out << "          " << (int)fields.is_gb[i] << "\n";
    }
    out << "        </DataArray>\n";

    out << "      </PointData>\n";
    out << "    </Piece>\n";
    out << "  </ImageData>\n";
    out << "</VTKFile>\n";

    out.close();
}

void VTKWriter::add_timestep(double time, const std::string& vti_file) {
    pvd_entries_.push_back({time, vti_file});
}

void VTKWriter::write_pvd(const std::string& filename) const {
    std::ofstream out(filename);
    if (!out.is_open()) {
        std::fprintf(stderr, "Error: Cannot open PVD file '%s'\n", filename.c_str());
        return;
    }

    // Get directory of PVD file for relative path computation
    std::string pvd_dir;
    auto slash_pos = filename.find_last_of('/');
    if (slash_pos != std::string::npos) {
        pvd_dir = filename.substr(0, slash_pos + 1); // includes trailing /
    }

    out << "<?xml version=\"1.0\"?>\n";
    out << "<VTKFile type=\"Collection\" version=\"1.0\" byte_order=\"LittleEndian\">\n";
    out << "  <Collection>\n";

    for (auto& entry : pvd_entries_) {
        // Make VTI path relative to PVD file location
        std::string rel_path = entry.file;
        if (!pvd_dir.empty() && rel_path.find(pvd_dir) == 0) {
            rel_path = rel_path.substr(pvd_dir.size());
        }
        out << "    <DataSet timestep=\"" << std::scientific << std::setprecision(6)
            << entry.time << "\" file=\"" << rel_path << "\"/>\n";
    }

    out << "  </Collection>\n";
    out << "</VTKFile>\n";

    out.close();
    std::printf("  Wrote PVD file: %s (%zu timesteps)\n", filename.c_str(), pvd_entries_.size());
}
