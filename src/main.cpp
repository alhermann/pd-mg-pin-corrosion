#include "config.h"
#include "grid.h"
#include "grains.h"
#include "fields.h"
#include "coupling.h"
#include <cstdio>
#include <string>

static void initialize_fields(Fields& fields, const Grid& grid,
                               const GrainStructure& grains, const Config& cfg) {
    int N = grid.N_total;
    double R2 = cfg.R_tube * cfg.R_tube;

    for (int i = 0; i < N; ++i) {
        switch (grid.node_type[i]) {
            case FLUID: {
                fields.rho[i] = cfg.rho_f;
                fields.C[i] = cfg.C_liquid_init;
                fields.D_map[i] = cfg.D_liquid;
                fields.phase[i] = 1; // liquid

                // Initialize with Poiseuille profile for faster flow convergence
                double px = grid.pos[i][0];
                double v_axial;
                if constexpr (DIM == 2) {
                    double r_ratio2 = (px * px) / R2;
                    if (r_ratio2 > 1.0) r_ratio2 = 1.0;
                    v_axial = 1.5 * cfg.U_in * (1.0 - r_ratio2);
                } else {
                    double py = grid.pos[i][1];
                    double r_ratio2 = (px * px + py * py) / R2;
                    if (r_ratio2 > 1.0) r_ratio2 = 1.0;
                    v_axial = 2.0 * cfg.U_in * (1.0 - r_ratio2);
                }
                Vec v_init = vec_zero();
                if constexpr (DIM == 2) { v_init[1] = v_axial; }
                else                    { v_init[2] = v_axial; }
                fields.vel[i] = v_init;
                break;
            }

            case SOLID_MG:
                fields.rho[i] = cfg.rho_f;  // use fluid density for PD flow equations
                fields.vel[i] = vec_zero();
                fields.C[i] = cfg.C_solid_init;
                fields.phase[i] = 0; // solid

                // Set diffusivity based on grain structure
                if (grains.is_grain_boundary[i]) {
                    fields.D_map[i] = cfg.D_gb;
                } else {
                    fields.D_map[i] = cfg.D_grain;
                }
                break;

            case WALL:
                fields.rho[i] = cfg.rho_f;
                fields.vel[i] = vec_zero();
                fields.C[i] = 0.0;
                fields.D_map[i] = 0.0;
                fields.phase[i] = 1;
                break;

            case INLET: {
                // Parabolic Poiseuille profile
                double px = grid.pos[i][0];
                double r_ratio2;
                double v_axial;
                if constexpr (DIM == 2) {
                    r_ratio2 = (px * px) / R2;
                    if (r_ratio2 > 1.0) r_ratio2 = 1.0;
                    v_axial = 1.5 * cfg.U_in * (1.0 - r_ratio2);
                } else {
                    double py = grid.pos[i][1];
                    r_ratio2 = (px * px + py * py) / R2;
                    if (r_ratio2 > 1.0) r_ratio2 = 1.0;
                    v_axial = 2.0 * cfg.U_in * (1.0 - r_ratio2);
                }
                Vec v_in = vec_zero();
                if constexpr (DIM == 2) { v_in[1] = v_axial; }
                else { v_in[2] = v_axial; }
                fields.rho[i] = cfg.rho_f;
                fields.vel[i] = v_in;
                fields.C[i] = cfg.C_liquid_init;
                fields.D_map[i] = cfg.D_liquid;
                fields.phase[i] = 1;
                break;
            }

            case OUTLET:
                fields.rho[i] = cfg.rho_f;
                fields.vel[i] = vec_zero();  // will be set by outlet BC
                fields.C[i] = cfg.C_liquid_init;
                fields.D_map[i] = cfg.D_liquid;
                fields.phase[i] = 1;
                break;

            case OUTSIDE:
                fields.rho[i] = 0.0;
                fields.vel[i] = vec_zero();
                fields.C[i] = 0.0;
                fields.D_map[i] = 0.0;
                fields.phase[i] = 1;
                break;
        }

        // Copy grain_id and grain boundary flag
        fields.grain_id[i] = grains.grain_id[i];
        fields.is_gb[i] = grains.is_grain_boundary[i] ? 1 : 0;
    }

    // Copy to new buffers
    fields.rho_new = fields.rho;
    fields.vel_new = fields.vel;
    fields.C_new = fields.C;
}

int main(int argc, char* argv[]) {
    // Disable buffering so output appears immediately
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    std::setvbuf(stderr, nullptr, _IONBF, 0);

    std::printf("=== Peridynamic Mg-Pin Corrosion Simulation ===\n");
    std::printf("  Dimension: %dD\n\n", DIM);

    Timer t_init("initialization");

    // Load configuration
    Config cfg;
    if (argc > 1) {
        cfg.load(argv[1]);
    } else {
        cfg.load("params.cfg");
    }
    cfg.print();

    // Build grid
    std::printf("Building grid...\n");
    Grid grid;
    grid.build(cfg);
    grid.build_neighbors();

    // Generate grain structure
    std::printf("Generating grain structure...\n");
    GrainStructure grains;
    grains.generate(grid, cfg);

    // Allocate and initialize fields
    std::printf("Initializing fields...\n");
    Fields fields;
    fields.allocate(grid.N_total);
    initialize_fields(fields, grid, grains, cfg);

    t_init.report();

    // Run coupled simulation
    CoupledSolver solver;
    solver.run(grid, fields, cfg);

    return 0;
}
