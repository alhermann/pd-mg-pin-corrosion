# Peridynamic Mg-Pin Corrosion Simulation

A peridynamic (PD) simulation of biodegradable Mg-4Ag pin corrosion in simulated body fluid (SBF), coupling three physics:

1. **PD Navier-Stokes** -- weakly compressible viscous flow around the pin
2. **PD Advection-Reaction-Diffusion** -- transport of dissolved Mg species
3. **Phase-change dissolution** -- solid Mg nodes corrode and become fluid

The experimental reference is [Reimers et al. (2023)](https://doi.org/10.1016/j.actbio.2023.04.039): 80 um diameter Mg-4Ag wire in a flow cell at 1 mL/min, 37 C, pH 7.4, with ~50% volume loss after 9 hours.

## Physics

### PD Navier-Stokes (weakly compressible)

The flow solver uses peridynamic nonlocal operators following [Zhao & Bobaru (2022)](https://doi.org/10.1016/j.cma.2022.114815):

- **Mass conservation** via nonlocal divergence of momentum
- **Momentum conservation** with nonlocal convection, pressure gradient, and viscous diffusion
- **Tait equation of state** (gamma = 7) for pressure-density coupling
- **Delta-SPH style density diffusion** for numerical stabilization
- Forward Euler time integration with CFL-limited timestep

The Mach number is kept below 0.04 (c0 >= 25 * U_in) to satisfy the weakly compressible assumption.

### PD Advection-Reaction-Diffusion

Transport follows [Song, Chen & Bobaru (2025)](https://doi.org/10.1016/j.cma.2024.117562):

- **Solid nodes**: Surface dissolution model. Only surface nodes (exposed to fluid) dissolve at rate `k_corr * f_exposure`, where `f_exposure` is the fraction of fluid neighbors. Grain boundary nodes corrode faster by `gb_corr_factor`.
- **Fluid nodes**: Full PD advection-diffusion with hybrid upwind/downwind advection scheme. Bonds to solid and wall nodes are skipped; instead, a dissolution source term accounts for Mg entering the fluid.
- **Phase change**: When solid concentration drops below `C_thresh`, the node converts from solid to fluid.

### Coupling Strategy

Operator-splitting (weak coupling):

```
while t < T_final:
    1. Solve flow to steady state (PD-NS)
    2. Freeze velocity field
    3. Run ARD for corrosion_steps_per_check steps
    4. Check phase changes (dissolution)
    5. If nodes dissolved: update types, rebuild neighbors
    6. Output diagnostics
    7. Repeat
```

This is valid because the flow timescale (ms) is much shorter than the corrosion timescale (hours).

### Boundary Conditions

- **Walls**: Fictitious Node Method (FNM) with geometric reflection across the tube surface. Antisymmetric velocity (no-slip), symmetric density.
- **Inlet**: Prescribed parabolic Poiseuille profile, `rho = rho_f`, `C = 0`.
- **Outlet**: Zero pressure (`rho = rho_f`), velocity and concentration averaged over all fluid neighbors (zero-gradient extrapolation).
- **Solid surface**: No-slip velocity. Dissolved nodes reclassified as fluid.

## Features

- **Compile-time dimension switch**: Build for 2D (r-z axisymmetric) or 3D via `-DPD_DIM=2` or `-DPD_DIM=3`
- **Voronoi grain structure**: Random seed-based tessellation with configurable grain size, grain boundary detection, and enhanced GB corrosion
- **CSR neighbor lists** with partial-volume (beta) correction at the horizon boundary
- **Poiseuille initialization** for faster initial flow convergence
- **OpenMP parallelism**: SoA field layout, read/write buffer separation (no race conditions), dynamic scheduling
- **VTI output** for ParaView with PVD time series collection
- **Diagnostics CSV**: time, volume loss, solid node count, max velocity, max concentration
- **Zero external dependencies** beyond a C++17 compiler and OpenMP

## Project Structure

```
Corrosion/
├── CMakeLists.txt                # Build system (CMake 3.14+)
├── params.cfg                    # Runtime parameters
├── LICENSE
├── README.md
└── src/
    ├── main.cpp                  # Entry point, field initialization
    ├── config.h / config.cpp     # Parameter file parser + Config struct
    ├── utils.h                   # Vec type, operators, Timer, constants
    ├── grid.h / grid.cpp         # Uniform grid, node classification, CSR neighbors
    ├── fields.h                  # Structure-of-Arrays field storage
    ├── grains.h / grains.cpp     # Voronoi grain structure on Mg pin
    ├── pd_ns.h / pd_ns.cpp       # PD Navier-Stokes solver
    ├── pd_ard.h / pd_ard.cpp     # PD advection-reaction-diffusion solver
    ├── boundary.h / boundary.cpp # Boundary conditions (FNM walls, inlet/outlet)
    ├── coupling.h / coupling.cpp # Weak coupling orchestrator
    └── vtk_writer.h / vtk_writer.cpp  # VTI + PVD output
```

## Building

### Prerequisites

- C++17 compiler (GCC 7+, Clang 5+, MSVC 2017+)
- CMake 3.14+
- OpenMP runtime (on macOS: `brew install libomp`)

### 2D build (r-z axisymmetric)

```bash
mkdir build && cd build
cmake .. -DPD_DIM=2 -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

### 3D build

```bash
mkdir build && cd build
cmake .. -DPD_DIM=3 -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

### Debug build (with AddressSanitizer)

```bash
cmake .. -DPD_DIM=2 -DCMAKE_BUILD_TYPE=Debug
make -j$(nproc)
```

## Running

```bash
./build/pd_corrosion params.cfg
```

Output is written to the `output/` directory:
- `corr_NNNNNN_tX.Xs.vti` -- VTK ImageData snapshots with physical time
- `simulation.pvd` -- ParaView time series collection
- `diagnostics.csv` -- time-series data for post-processing

Open `output/simulation.pvd` in [ParaView](https://www.paraview.org/) to visualize.

## Configuration

All parameters are set in `params.cfg`. Key parameters:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `dx` | 5e-6 m | Grid spacing (~16 nodes across wire diameter) |
| `m_ratio` | 3 | Horizon ratio (delta = m * dx) |
| `R_wire` | 40e-6 m | Wire radius |
| `L_wire` | 400e-6 m | Wire length |
| `R_tube` | 150e-6 m | Flow tube radius |
| `rho_f` | 1000 kg/m^3 | Fluid density (SBF at 37 C) |
| `mu_f` | 1e-3 Pa.s | Dynamic viscosity |
| `Q_flow` | 1.667e-8 m^3/s | Volumetric flow rate (1 mL/min) |
| `D_liquid` | 1e-9 m^2/s | Mg ion diffusivity in fluid |
| `D_grain` | 5e-11 m^2/s | Diffusivity through Mg lattice |
| `D_gb` | 5e-9 m^2/s | Diffusivity along grain boundaries |
| `k_corr` | 1e-3 /s | Base surface corrosion rate |
| `gb_corr_factor` | 3.0 | GB corrosion enhancement |
| `C_thresh` | 0.2 | Dissolution threshold (0 = intact, 1 = fully corroded) |
| `grain_size_mean` | 40e-6 m | Mean Voronoi grain size |
| `cfl_factor` | 0.05 | CFL number for flow solver |
| `cfl_factor_corr` | 0.25 | CFL number for corrosion solver |
| `T_final` | 10 s | Total simulation time |
| `flow_conv_tol` | 5e-6 | Flow steady-state convergence tolerance |
| `corrosion_steps_per_check` | 200000 | ARD steps between phase-change checks |

## Physical Parameters

| Quantity | Value | Notes |
|----------|-------|-------|
| Wire diameter | 80 um | Reimers et al. (2023) |
| Reynolds number | ~19 | Based on wire diameter |
| Mach number | < 0.04 | Weakly compressible regime |
| Kinematic viscosity | 1e-6 m^2/s | Water at 37 C |
| Mg density | 1738 kg/m^3 | Pure Mg |
| Peclet number | ~1180 | Before artificial diffusion |

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
