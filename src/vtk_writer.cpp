#include "vtk_writer.h"
#include <fstream>
#include <cstdio>
#include <sstream>
#include <iomanip>

void VTKWriter::write(const std::string& filename,
                      const Grid& grid, const Fields& fields,
                      const Config& cfg) {
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
    out << "        <DataArray type=\"Float64\" Name=\"velocity\" "
        << "NumberOfComponents=\"3\" format=\"ascii\">\n";
    int N = grid.N_total;
    for (int i = 0; i < N; ++i) {
        if constexpr (DIM == 2) {
            out << "          " << fields.vel[i][0] << " " << fields.vel[i][1] << " 0\n";
        } else {
            out << "          " << fields.vel[i][0] << " " << fields.vel[i][1]
                << " " << fields.vel[i][2] << "\n";
        }
    }
    out << "        </DataArray>\n";

    // Pressure
    out << "        <DataArray type=\"Float64\" Name=\"pressure\" format=\"ascii\">\n";
    for (int i = 0; i < N; ++i) {
        out << "          " << fields.pressure[i] << "\n";
    }
    out << "        </DataArray>\n";

    // Density
    out << "        <DataArray type=\"Float64\" Name=\"density\" format=\"ascii\">\n";
    for (int i = 0; i < N; ++i) {
        out << "          " << fields.rho[i] << "\n";
    }
    out << "        </DataArray>\n";

    // Concentration
    out << "        <DataArray type=\"Float64\" Name=\"concentration\" format=\"ascii\">\n";
    for (int i = 0; i < N; ++i) {
        out << "          " << fields.C[i] << "\n";
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
        out << "          " << fields.D_map[i] << "\n";
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

    out << "<?xml version=\"1.0\"?>\n";
    out << "<VTKFile type=\"Collection\" version=\"1.0\" byte_order=\"LittleEndian\">\n";
    out << "  <Collection>\n";

    for (auto& entry : pvd_entries_) {
        out << "    <DataSet timestep=\"" << std::scientific << std::setprecision(6)
            << entry.time << "\" file=\"" << entry.file << "\"/>\n";
    }

    out << "  </Collection>\n";
    out << "</VTKFile>\n";

    out.close();
}
