# Peridynamic Mg-Pin Corrosion Simulation — Project Log

**Started:** 2026-02-12
**Language:** C++17 with OpenMP
**Build:** CMake (tested on macOS with Apple Clang + Homebrew libomp)
**Output:** VTI (VTK ImageData) for ParaView

---

## 1. Project Overview

Peridynamic (PD) simulation of biodegradation of Mg-4Ag pins in simulated body
fluid (SBF), coupling three physics:

1. **PD Navier-Stokes** — weakly compressible viscous flow around the pin
2. **PD Advection-Diffusion** — transport of dissolved Mg species in the fluid
3. **Phase-change dissolution** — corrosion: solid Mg nodes become liquid when
   concentration drops below a threshold

The experimental reference is Reimers et al. (2023): 80 um diameter Mg-4Ag wire
in a flow cell at 1 mL/min, 37 deg C, pH 7.4. Volume loss ~50% after 9 h.

The PD equations follow:
- Zhao & Bobaru (2022) for nonlocal Navier-Stokes
- Song, Chen & Bobaru (2025) for coupled advection-reaction-diffusion

---

## 2. Project Structure

```
Corrosion/
+-- CMakeLists.txt              # build system (CMake 3.14+)
+-- params.cfg                  # runtime parameters
+-- PROJECT_LOG.md              # this file
+-- src/
|   +-- main.cpp                # entry point, field initialization
|   +-- config.h / config.cpp   # parameter file parser + Config struct
|   +-- utils.h                 # Vec type, operators, Timer, constants
|   +-- grid.h / grid.cpp       # uniform grid, node classification, CSR neighbors
|   +-- fields.h                # SoA field storage
|   +-- grains.h / grains.cpp   # Voronoi grain structure on Mg pin
|   +-- pd_ns.h / pd_ns.cpp     # PD Navier-Stokes solver
|   +-- pd_ard.h / pd_ard.cpp   # PD advection-reaction-diffusion solver
|   +-- boundary.h / boundary.cpp  # all boundary conditions
|   +-- coupling.h / coupling.cpp  # weak coupling orchestrator
|   +-- vtk_writer.h / vtk_writer.cpp  # VTI + PVD output
+-- build/                      # out-of-source build directory
+-- output/                     # simulation output (VTI, CSV)
```

**Total:** ~1805 lines of C++17. Zero external dependencies (compiler + OpenMP only).

---

## 3. What Has Been Implemented

### 3.1 Grid and Geometry (grid.h/cpp)

- Uniform Cartesian grid with compile-time dimension switch (PD_DIM=2 or 3)
- **2D r-z mode:** Rectangle in (r, z). r in [-R_tube-m*dx, R_tube+m*dx],
  z in [-L_upstream-m*dx, L_wire+L_downstream+m*dx]
- **3D mode:** Box with square cross-section and axial extent
- Six node types: `FLUID`, `SOLID_MG`, `WALL`, `INLET`, `OUTLET`, `OUTSIDE`
- Ghost layers (thickness = m*dx) for inlet/outlet boundaries
- Tube wall region between R_tube and R_tube+m*dx classified as `WALL`

**CSR Neighbor List Construction:**
- Precompute stencil of all integer offsets (di,dj,dk) within delta+0.5*dx
- Two-pass build: count per node -> prefix sum -> fill
- Partial-volume correction (beta weights):
  ```
  beta(r) = 1.0                        if r <= delta - dx/2
           = (delta + dx/2 - r) / dx   if delta - dx/2 < r <= delta + dx/2
           = 0.0                        otherwise
  V_ij = beta * dx^DIM
  ```

### 3.2 Grain Structure (grains.h/cpp)

- Voronoi tessellation from random seed points within solid region
- Number of grains estimated from mean grain size:
  N_grains = solid_area / (pi/4 * d_grain^2)  [2D]
- Each solid node assigned to nearest seed (brute force, fine for moderate N)
- Grain boundaries: nodes with any neighbor of different grain_id
- Dilation of GB mask by `gb_width_cells` passes
- Diffusivity map: D_gb at boundaries, D_grain in interior, D_liquid in fluid

### 3.3 PD Navier-Stokes (pd_ns.h/cpp)

**Governing equations** (weakly compressible):

Mass conservation:
```
  drho/dt = -(d/V_H) * SUM_j [ (rho_j*v_j - rho_i*v_i) . e_ij / |xi| ] V_j
            + density_diffusion
```

Momentum conservation:
```
  dv/dt = (1/rho_i) * [
    -(d/V_H) * SUM_j { (rho_j v_j (x) v_j - rho_i v_i (x) v_i) . e_ij / |xi| * V_j }   [convection]
    -(d/V_H) * SUM_j { (p_j - p_i) / |xi|^2 * xi_ij * V_j }                               [pressure]
    +(mu * alpha_mu / V_H) * SUM_j { (I - e_ij e_ij^T).(v_j - v_i) / |xi|^2 * V_j }       [viscosity]
  ]
```

where:
- `xi_ij = x_j - x_i`, `|xi| = ||xi_ij||`, `e_ij = xi_ij / |xi|`
- `d = DIM` (spatial dimension)
- `V_H = pi*delta^2` (2D) or `(4/3)*pi*delta^3` (3D) — horizon volume
- `alpha_mu = 16/3` (2D) or `10` (3D) — viscous PD coefficient

**IMPORTANT: PD equations are computed ONLY for FLUID nodes.** All other node
types (WALL, SOLID_MG, INLET, OUTLET, OUTSIDE) have their values set by
boundary conditions, not PD evolution. This prevents non-physical density/pressure
buildup in wall and solid nodes.

**Equation of state** (Tait EOS for water):
```
  p = (rho_0 * c0^2 / gamma) * [ (rho/rho_0)^gamma - 1 ]
```
with gamma = 7.0, c0 chosen so Mach = U_in/c0 < 0.04.

**Density diffusion** (numerical stabilization, delta-SPH style):
```
  d_rho/dt += SUM_j [ d_coeff * (rho_j - rho_i) / |xi|^2 * V_j ]
```
where `d_coeff = 4 * D_v / (pi * delta^2)` [2D], `D_v = eta * c0 * delta`.

**Time stepping:**
- Forward Euler (explicit)
- dt = CFL * min(dt_cfl, dt_visc, dt_dens) where:
  - dt_cfl = dx / (c0 + v_max)
  - dt_visc = 0.25 * dx^2 / nu
  - dt_dens = 0.25 * dx^2 / D_v
- Density clamped to [0.5*rho_f, 2.0*rho_f] for stability
- Pressure clamped via density ratio [0.5, 2.0] in EOS

**Steady-state convergence:**
```
  epsilon = sqrt( SUM |v_new - v_old|^2 / SUM |v_old|^2 )
  converged when epsilon < flow_conv_tol (default 5e-6)
```

**Poiseuille initialization:** Fluid nodes are initialized with the analytical
Poiseuille profile (instead of zero) for faster convergence. The flow solver
now converges in ~22k iterations for the first cycle and 100-600 for subsequent
cycles.

### 3.4 PD Advection-Reaction-Diffusion (pd_ard.h/cpp)

**Two distinct update rules depending on node type:**

#### SOLID nodes (surface dissolution, no PD):
Solid concentration represents structural integrity (1=intact, 0=corroded).
Only surface nodes (those exposed to fluid) dissolve:
```
  C_new = C - dt * k_eff * f_exposure
```
where:
- `k_eff = k_corr` (or `k_corr * gb_corr_factor` for grain boundary nodes)
- `f_exposure = n_fluid_neighbors / n_total_neighbors` (fraction exposed to fluid)
- No PD bonds are evaluated for solid nodes

#### FLUID nodes (PD advection-diffusion + dissolution source):
PD transport is computed ONLY over bonds to other fluid-type nodes (FLUID,
INLET, OUTLET). Bonds to SOLID_MG and WALL are skipped for transport.

**PD Laplacian (diffusion):**
```
  diff = SUM_j [ beta_coeff * D_avg * (C_j - C_i) / |xi|^2 ] V_j
```
where:
- `beta_coeff = 4 / (pi * delta^2)` [2D] or `12 / (pi * delta^2)` [3D]
- `D_avg = 2 * D_i * D_j / (D_i + D_j)` — harmonic mean of local diffusivities
- **IMPORTANT:** No extra `1/V_H` factor. The beta_coeff IS the complete
  PD Laplacian normalization coefficient.

**PD Divergence (advection):**
```
  div(Cv) = (d/V_H) * SUM_j [ (C_j*v_j - C_i*v_i) . e_ij / |xi| ] V_j
```
Hybrid downwind/upwind scheme:
```
  adv = (d/V_H) * [ w * adv_downwind + (1-w) * adv_upwind ]
```
with w = 0.8 (params: w_advect).

**Dissolution source term:** Fluid nodes adjacent to the solid pin receive a
source term representing dissolved Mg entering the fluid:
```
  source = avg_k * interface_frac * (rho_m / rho_f)
```
where:
- `avg_k` = average corrosion rate of neighboring solid nodes
- `interface_frac = n_solid_neighbors / n_total_neighbors`
- `rho_m / rho_f` accounts for the density ratio of dissolving material

**Concentration clamp:** [0.0, 1.0] to prevent unphysical values.

**Phase change rule:**
- Only applies to nodes with phase=SOLID_MG
- If C[i] < C_thresh (default 0.2): dissolve
  - Set phase=liquid, node_type=FLUID
  - Set D_map = D_liquid, rho = rho_f, vel = 0
  - C set to C_thresh (not 0) for continuity

### 3.5 Boundary Conditions (boundary.h/cpp)

**Fictitious Node Method (FNM) for walls — proper geometric reflection:**

Wall nodes use mirror-point reflection across the tube wall surface. For each
WALL node, the mirror point is computed by reflecting across the physical tube
boundary (`|x| = R_tube` in 2D, `r = R_tube` in 3D). The mirror point falls
exactly on a grid node for a uniform grid when R_tube is grid-aligned.

- **Velocity:** Antisymmetric `v_wall = -v_mirror` (enforces no-slip at wall midplane)
- **Density:** Symmetric `rho_wall = rho_mirror` (ensures smooth pressure transition)
- Applied to both current (vel/rho) and new (vel_new/rho_new) buffers
- Fallback to nearest-fluid-neighbor if the mirror point is not found
- Wall concentration: `C = 0` (Dirichlet BC, tube walls are inert sinks)

**Inlet (upstream ghost layer):**
- Prescribed parabolic Poiseuille velocity:
  2D: v(x) = (3/2)*U_in*(1 - (x/R_tube)^2)
  3D: v(r) = 2*U_in*(1 - (r/R_tube)^2)
- rho = rho_f, p = 0, C = C_liquid_init = 0

**Outlet (downstream ghost layer) — pressure outlet per Song et al. (2025):**
- Density: `rho = rho_f` (enforces p=0 via Tait EOS, anchors pressure field)
- Velocity: axial component extrapolated from fluid neighbors (zero-gradient);
  transverse component set to zero
- Concentration: extrapolated from fluid neighbors (zero-gradient)
- Fallback to uniform inlet velocity if no fluid neighbors found

**Solid Mg surface:**
- Velocity forced to zero (no-slip)
- When dissolved: reclassified as FLUID

### 3.6 Coupling Strategy (coupling.h/cpp)

Operator-splitting (weak coupling):
```
  while t_corr < T_final:
    1. Solve flow to steady state (PD-NS)
    2. Freeze velocity field
    3. Run ARD for corrosion_steps_per_check steps
    4. Check phase changes (dissolution)
    5. If nodes dissolved:
       - Update node types
       - Rebuild neighbor list (if >10 nodes changed)
    6. Output diagnostics
    7. Loop back to step 1
```

### 3.7 Output (vtk_writer.h/cpp)

- **VTI format** (VTK ImageData XML, ASCII)
- Fields written: velocity (3-component), pressure, density, concentration,
  phase, node_type, grain_id, D_map
- **PVD collection** file for ParaView time series
- **diagnostics.csv**: time_s, time_h, volume_loss, solid_nodes, v_max, C_max

---

## 4. Physical Parameters (Default Values)

| Parameter | Symbol | Value | Unit | Source/Notes |
|-----------|--------|-------|------|-------------|
| Grid spacing | dx | 5e-6 | m | ~16 nodes across wire diameter |
| Horizon ratio | m | 3 | - | delta = 15 um |
| Wire radius | R_wire | 40e-6 | m | Reimers et al. (2023) |
| Wire length | L_wire | 400e-6 | m | Shortened for testing (real: 1 mm) |
| Tube radius | R_tube | 150e-6 | m | Flow cell geometry |
| Fluid density | rho_f | 1000 | kg/m^3 | Water/SBF at 37C |
| Viscosity | mu_f | 1e-3 | Pa.s | Water at 37C |
| Flow rate | Q_flow | 1.667e-8 | m^3/s | 1 mL/min |
| Inlet velocity | U_in | 0.236 | m/s | Derived from Q/(pi*R^2) |
| Sound speed | c0 | >=25*U_in | m/s | Weakly compressible, Ma<0.04 |
| EOS exponent | gamma | 7.0 | - | Tait EOS for water |
| Density diffusion | eta | 0.1 | - | Numerical stabilization |
| Mg density | rho_m | 1738 | kg/m^3 | Pure Mg |
| Liquid diffusivity | D_liquid | 1e-9 | m^2/s | Mg ions in water |
| Grain diffusivity | D_grain | 5e-11 | m^2/s | Through Mg lattice |
| GB diffusivity | D_gb | 5e-9 | m^2/s | Along grain boundaries |
| Dissolution threshold | C_thresh | 0.2 | - | Dimensionless |
| Mean grain size | d_grain | 40e-6 | m | Reimers: 35-45 um |
| CFL factor | CFL | 0.05 | - | Conservative for stability |
| Reynolds number | Re | ~19 | - | Based on wire diameter |

---

## 5. Problems Encountered and Solutions

### 5.1 OpenMP on macOS (Apple Clang)

**Problem:** Apple Clang does not ship with OpenMP support by default.
`find_package(OpenMP REQUIRED)` fails.

**Solution:** Detect macOS and set OpenMP hints for Homebrew's `libomp`:
```cmake
if(APPLE)
    execute_process(COMMAND brew --prefix libomp ...)
    set(OpenMP_CXX_FLAGS "-Xpreprocessor -fopenmp -I${LIBOMP_PREFIX}/include")
    set(OpenMP_omp_LIBRARY "${LIBOMP_PREFIX}/lib/libomp.dylib")
endif()
```

### 5.2 Vec Initializer Lists in if-constexpr Branches

**Problem:** `std::array<double,2>` cannot be assigned from a 3-element brace
initializer `{x, y, z}`, even inside `if constexpr (DIM==3)` dead branches.
Apple Clang checks aggregate initializer validity before discarding dead branches.

**Solution:** Created `make_vec(x, y, z)` helper that assigns components
individually using `if constexpr`, avoiding aggregate initialization entirely.

### 5.3 Flow Solver Divergence — Initial Velocity Field

**Problem:** Initializing all fluid nodes with uniform velocity `v = U_in`
caused explosive growth in the first few iterations. The convective term
`rho*v*v` created huge imbalances at the solid boundary where v_solid=0 but
v_fluid=U_in, driving density negative and velocity to NaN within ~10 steps.

**Solution:** Initialize fluid velocity to **zero everywhere** and let the flow
develop from the inlet BC. The velocity propagates inward over thousands of
iterations, allowing the pressure field to equilibrate naturally.

### 5.4 Flow Solver Divergence — Mach Number Too High

**Problem:** With `c0 = 0.5 m/s` and `U_in = 0.236 m/s`, the Mach number was
0.47, violating the weakly compressible assumption (Ma << 1). Density
fluctuations exceeded 30%, causing nonlinear instability and NaN.

**Solution:** Auto-increase c0 to at least 25x U_in in `compute_derived()`,
giving Ma = 0.04 and density fluctuations < 0.5%. This requires smaller dt
(~7e-9 s) and more iterations (~50k for initial convergence) but is stable.

### 5.5 Flow Solver Divergence — Density Diffusion Too Weak

**Problem:** Even with Ma=0.1 (10x U_in), density still fluctuated 20% and
eventually diverged around iteration 5600.

**Solution:** Increased `eta_density` from 0.1 to 0.5 AND increased c0 to 25x
U_in. The combined effect keeps density within [998.8, 1004.6] (~0.5%).

### 5.6 Flow Solver — FLUID-Only PD Computation

**Problem (v1):** Wall and solid nodes were skipped entirely in the PD step.
**Problem (v2):** Evolving wall density via PD caused non-physical pressure
buildup and boundary discontinuities.

**Solution (final):** PD equations are computed ONLY for FLUID nodes. All
other node types have their values set entirely by boundary conditions:
- WALL: geometric reflection mirroring (symmetric rho, antisymmetric vel)
- SOLID_MG: vel=0, rho/C managed by dissolution model
- INLET/OUTLET: prescribed by BCs

### 5.7 ARD Solver — Instant Dissolution (1/V_H Bug)

**Problem:** All 1296 solid nodes dissolved in the very first corrosion batch.
The diffusion term was orders of magnitude too large, causing concentration to
plummet instantly.

**Root cause:** The PD Laplacian coefficient `beta_coeff = 4/(pi*delta^2)` was
being multiplied by an additional `1/V_H = 1/(pi*delta^2)` factor, effectively
squaring the normalization. This made the Laplacian ~10^9 times too large.

**Key insight:** The PD Laplacian and PD divergence use DIFFERENT normalization:
- **Laplacian:** `nabla^2 C = SUM_j [ d_0 * (C_j - C_i) / |xi|^2 ] V_j`
  where `d_0 = 4/(pi*delta^2)` [2D] is the COMPLETE coefficient. No 1/V_H.
- **Divergence:** `div(F) = (d/V_H) * SUM_j [ (F_j - F_i).e / |xi| ] V_j`
  where the `d/V_H` factor IS needed.

**Solution:** Removed `inv_VH` from the diffusion term. Now the diffusion
correctly produces `dC/dt ~ O(D/dx^2)` which gives reasonable timescales.

### 5.8 Grain Boundary Coverage Too High

**Problem:** 99.5% of solid nodes were classified as grain boundaries with
only 26 grains. This makes nearly the entire pin have D_gb diffusivity.

**Cause:** With 5 um grid spacing and mean grain size 40 um, each grain is only
about 8 cells across. After GB detection + dilation, most interior cells are
within one neighbor hop of a boundary, so almost everything gets marked as GB.

**Status:** Known issue. For better grain resolution, use a finer grid (dx=2 um)
or larger grain size, or reduce gb_width_cells to 0 (no dilation).

### 5.9 Corrosion Timescale — Resolved via Surface Dissolution Model

**Problem:** The pin dissolved almost completely within ~0.4 seconds due to
PD diffusion across the solid-liquid interface being too aggressive.

**Root cause:** PD diffusion was computed for solid-solid bonds using D_grain
and D_gb, which caused the entire solid concentration to rapidly equilibrate
with the liquid. Additionally, the harmonic mean D_avg for solid-liquid bonds
was dominated by D_liquid, making the interface diffusion too fast.

**Solution:** Implemented a proper surface dissolution model:
- **SOLID nodes**: No PD equations at all. Only a surface corrosion term:
  `C_new = C - dt * k_eff * f_exposure`
  where `f_exposure` = fraction of fluid neighbors, and `k_eff = k_corr`
  (or `k_corr * gb_corr_factor` for grain boundary nodes)
- **FLUID nodes**: Full PD advection-diffusion with fluid neighbors
- This gives realistic timescales: ~22 min per GB layer, ~67 min for grain
  interior with k_corr=1e-3.

### 5.10 Inlet/Outlet Ghost Layer Classification Bug

**Problem:** All nodes with axial position in the ghost layer were classified
as INLET or OUTLET regardless of their radial position. Nodes outside the
tube wall in the ghost layers got INLET/OUTLET type instead of WALL/OUTSIDE.
This created spurious flow at the boundary.

**Solution:** Modified `grid.cpp::build()` to check radial position when
classifying inlet/outlet ghost layers. Nodes outside R_tube in the ghost
layers are now correctly classified as WALL (between R_tube and R_tube+m*dx)
or OUTSIDE (beyond that).

### 5.11 Wall Concentration BC

**Problem (v1):** Wall nodes had C=0 which acts as a sink.
**Decision (v2):** Changed to no-flux mirror (C_wall = C_nearest_fluid).
**Decision (v3):** Changed back to C=0 (Dirichlet). The tube wall is far
from the pin, so the concentration at the wall should be negligible. Using
C=0 also prevents concentration buildup artifacts in the wall region.
The PD-ARD solver skips bonds to WALL nodes, so the wall C value mainly
affects diagnostics rather than the transport computation.

### 5.12 Output File Naming

**Problem:** Output files used sequential frame numbers (e.g., `corr_000560.vti`)
with no time information, making it impossible to connect files to physical time.

**Solution:** Changed output naming to include time in seconds:
`{prefix}_{frame:06d}_t{time:.1f}s.vti`, e.g., `corr_000012_t6.1s.vti`.

### 5.13 Wall BC — Boundary Discontinuities (Asymmetric Concentration)

**Problem:** The flow boundary conditions caused jumps and non-smooth
transitions in velocity, pressure, and concentration at the tube wall and
inlet/outlet boundaries. This manifested as asymmetric concentration fields
even for a symmetric geometry. The root cause was the nearest-neighbor
mirroring approach for wall BCs: wall nodes simply copied the velocity/density
of their nearest fluid neighbor, creating discrete jumps.

**Solution:** Implemented proper geometric reflection for wall BCs:
- For each WALL node, compute its mirror point by reflecting across the
  physical tube boundary (`|x| = R_tube` in 2D, `r = R_tube` in 3D)
- The mirror point falls exactly on a grid node for a uniform grid when
  R_tube is grid-aligned
- Velocity: antisymmetric `v_wall = -v_mirror` (no-slip)
- Density: symmetric `rho_wall = rho_mirror` (smooth pressure transition)
- Fallback to nearest-fluid-neighbor if mirror point not found

This ensures continuous velocity and pressure profiles across the wall.

### 5.14 Outlet BC — Single-Neighbor Extrapolation

**Problem:** The outlet BC took values from the FIRST fluid neighbor found,
which could be at an arbitrary angle to the outflow direction, causing
asymmetric conditions.

**Solution:** Average velocity and concentration over ALL fluid neighbors
of each outlet node (weighted equally). This gives a smoother zero-gradient
extrapolation.

### 5.15 PVD Duplicate Timestamps (Static ParaView Animation)

**Problem:** Both flow-debug VTI files and corrosion VTI files were added to
the PVD time series collection. Since flow files have the same timestamp as
the subsequent corrosion file, ParaView sees duplicate timestamps and the
animation appears static.

**Solution:** Flow-debug VTI files are still written (for debugging) but NOT
added to the PVD collection. Only corrosion snapshots and the initial/final
states are in the PVD, with monotonically increasing timestamps.

### 5.16 PD-ARD Solid-Fluid Bond Artifacts

**Problem:** Fluid nodes adjacent to the solid pin had PD bonds reaching into
solid nodes. The resulting diffusion across the solid-fluid interface was
unphysical: solid nodes had C~1.0 while fluid had C~0.0, creating a massive
gradient that drove concentration in adjacent fluid nodes to 1.5 (the old
clamp limit). This invalidated the concentration field.

**Solution:** In the PD-ARD step for FLUID nodes, bonds to SOLID_MG and WALL
neighbors are skipped for transport. Instead, solid neighbors contribute a
dissolution source term. The concentration clamp was tightened from [0, 1.5]
to [0, 1.0].

### 5.17 Diagnostics CSV Issues

**Problem:** (a) Using `static bool header_written` combined with
`std::ios::app` caused missing headers when the output directory had stale
files from previous runs. (b) `write_diagnostics()` was called both inside
the corrosion loop and after phase change, creating duplicate entries.

**Solution:** (a) CSV files are now initialized with headers using truncation
mode at the start of each run. (b) Removed the duplicate call; diagnostics
are written only inside the corrosion output loop.

### 5.18 Outlet BC — Pressure Outlet (p=0) per Song et al. (2025)

**Problem:** The zero-gradient density extrapolation at the outlet let the pressure
field float, causing a pressure bump near the outlet boundary. The pressure
gradient for Poiseuille flow had 12.7% error, and the flow solver needed 26.5k
iterations to converge.

**Root cause:** In weakly compressible methods, the pressure is determined by
density via the EOS. Without anchoring the density at the outlet, pressure
waves reflect off the boundary and create artifacts. Song et al. (2025)
Section 3.2 explicitly state: "the pressure of the virtual nodes at the outlet
is uniformly set to 0."

**Solution:** Changed the outlet BC to:
- **Density:** `rho = rho_f` (enforces p=0 via Tait EOS)
- **Velocity:** Axial component extrapolated from fluid neighbors (zero-gradient);
  transverse component set to zero (suppresses spurious cross-flow)
- **Concentration:** Extrapolated from fluid neighbors (zero-gradient)

**Results (Poiseuille flow without pin, L_upstream=1500 um):**
- Pressure gradient error: 12.7% → **2.1%**
- Flow convergence: 26.5k → **9.3k iterations** (~3x faster)
- Velocity L2 error at midpoint: 1.7% → 1.9% (slight increase near outlet,
  but interior and pressure field are much better)

---

## 6. What Still Needs To Be Done

### 6.1 High Priority

- [ ] **Parameter calibration**: Tune D_liquid, D_grain, D_gb, C_thresh to
      match the Reimers et al. (2023) experimental volume loss curve (~50% at
      9h). May need to reduce D_gb or increase C_thresh significantly.
- [ ] **Grain resolution**: With dx=5 um and grain_size=40 um, grains are only
      8 cells across → 46% of nodes are GB (with gb_width_cells=0, no dilation).
      For thinner boundaries: dx=2 um or grain_size=80 um.
- [x] **Poiseuille validation error**: Resolved. The 16.5% error was caused by
      the pin's upstream hydrodynamic influence, not a solver defect. Without
      pin (R_wire=0), L2 velocity error is 0.8–1.9% in the interior.
      Pressure gradient error reduced from 12.7% to 2.1% by fixing the
      outlet BC (see 5.18 below). Remaining error is from PD horizon
      truncation near walls (δ/R_tube = 10%).
- [ ] **Concentration saturation at pin surface**: C_max_fluid = 1.0 (the
      clamp limit) near the pin. The dissolution source term
      (~5.7e-4 /s) overwhelms PD diffusion (~1.1e-7 /s), saturating the
      first fluid layer. May need source scaling or higher diffusivity.
- [ ] **Performance**: Full 9h simulation requires ~130M ARD steps (18+ hours
      wall time for 2D). Consider implicit time integration or AMR.

### 6.2 Medium Priority

- [x] **Parabolic inlet profile**: Implemented.
- [x] **PVD time series**: Implemented with correct relative paths. Flow
      files excluded from PVD to prevent duplicate timestamps.
- [x] **Output naming**: Files now include physical time in seconds.
- [x] **Separate CFL for corrosion**: cfl_factor_corr=0.25 (vs 0.05 for flow).
- [x] **Concentration source term**: Dissolution source term in fluid nodes
      adjacent to solid pin. Mass conservation: solid loses C via k_corr,
      fluid gains C from dissolution source proportional to interface fraction.
- [x] **Wall BC smoothness**: Proper geometric reflection mirroring replaces
      nearest-neighbor approach. Symmetric density, antisymmetric velocity.
- [x] **Outlet BC smoothness**: Averaging over all fluid neighbors instead
      of single nearest neighbor. Then upgraded to pressure outlet (p=0)
      per Song et al. (2025) — see 5.18.
- [x] **Poiseuille initialization**: Fluid nodes initialized with analytical
      Poiseuille profile for ~10x faster initial flow convergence.
- [ ] **3D validation**: Build with `-DPD_DIM=3` and verify.
- [x] **Poiseuille validation** without pin (R_wire=0). Velocity L2 < 2%,
      pressure gradient error 2.1%. See `params_poiseuille.cfg` and
      `plot_poiseuille.py` for the validation setup.
- [ ] **Flow around cylinder validation**.
- [ ] **Pure diffusion test**.
- [x] **GitHub repository**: Set up at github.com/alhermann/pd-mg-pin-corrosion.

### 6.3 Low Priority / Future Enhancements

- [ ] **Binary VTI output**: ASCII → binary for 5-10x smaller files.
- [ ] **Adaptive time stepping** for corrosion.
- [ ] **Selective neighbor rebuild**.
- [ ] **Pit formation modeling**.
- [ ] **Temperature coupling**.
- [ ] **Mg-4Ag alloy**: Model Ag-rich secondary phases.
- [ ] **Hydrogen evolution**: Model H2 gas bubbles.
- [ ] **Protective film**: Mg(OH)2 / MgCO3 surface film.

---

## 7. Build and Run Instructions

### Prerequisites
- C++17 compiler (GCC 7+, Clang 5+, MSVC 2017+)
- CMake 3.14+
- OpenMP (on macOS: `brew install libomp`)

### Build (2D mode)
```bash
cd Corrosion
mkdir build && cd build
cmake .. -DPD_DIM=2 -DCMAKE_BUILD_TYPE=Release
make -j8
```

### Build (3D mode)
```bash
cmake .. -DPD_DIM=3 -DCMAKE_BUILD_TYPE=Release
make -j8
```

### Run
```bash
cd Corrosion
./build/pd_corrosion params.cfg
```

Output goes to `output/` directory. Open `output/simulation.pvd` in ParaView.

### Key Parameters to Adjust
- `dx`: Grid spacing. Smaller = more accurate but slower. 5 um for 3D, 2 um for 2D.
- `cfl_factor`: Stability. 0.05 is conservative. Can try 0.1 if stable.
- `flow_max_iters`: Max flow iterations per coupling cycle.
- `corrosion_steps_per_check`: More steps = faster corrosion progression per cycle.
- `C_thresh`: Dissolution threshold. Higher = slower dissolution.
- `D_gb`, `D_grain`: Grain boundary vs interior diffusivity ratio controls
  preferential corrosion.

---

## 8. References

1. **Reimers et al. (2023)** — Experimental setup: 80 um Mg-4Ag wire in SBF
   flow cell, 1 mL/min, 37C, pH 7.4.
2. **Zhao & Bobaru (2022)** — Peridynamic formulation for Navier-Stokes:
   nonlocal divergence, gradient, and Laplacian operators.
3. **Song, Chen & Bobaru (2025)** — Coupled PD advection-reaction-diffusion
   for flow and transport. Hybrid upwind/downwind advection scheme. FNM boundary
   conditions.
4. **Madenci & Oterkus (2014)** — Peridynamic Theory and Its Applications.
   Foundation for PD operator calibration coefficients.
5. **Monaghan (1994)** — Weakly compressible SPH: Tait equation of state,
   density diffusion for pressure noise reduction.

---

## 9. Code Architecture Notes

### Dimension Handling
The code uses `PD_DIM` (set via CMake `-DPD_DIM=2` or `3`) as a compile-time
constant. All dimension-dependent code uses `if constexpr (DIM == 2)` for
zero-overhead branching. The `Vec` type is `std::array<double, DIM>`.

### Memory Layout
Structure-of-Arrays (SoA) in `Fields` for cache-friendly access during the
neighbor traversal loops. The CSR neighbor list format ensures contiguous
memory access per node.

### OpenMP Strategy
- Read/write buffer separation: each timestep reads `rho/vel/C` and writes to
  `rho_new/vel_new/C_new`. No race conditions.
- Main loops: `#pragma omp parallel for schedule(dynamic, 256)`
- Reductions for diagnostics (max velocity, convergence check)
- Grain generation: embarrassingly parallel Voronoi assignment

### Operator Splitting
The flow and corrosion are coupled weakly: flow is solved to steady state with
frozen morphology, then corrosion runs with frozen velocity. This is valid
because the flow timescale (ms) is much shorter than the corrosion timescale
(hours), so the flow adjusts quasi-instantaneously to morphological changes.
