# Changelog

## 2026-02-16: AMR ratio=2 calibration + bug fixes

### AMR improvements
- **Reduced AMR ratio from 3 to 2** (`amr_ratio=2`, `dx_coarse=5.0um`):
  smoother coarse-fine transition, reduces numerical smearing at AMR interface.
  New config: `config/params_amr_r2.cfg`
- **Sharper IDW interpolation** for fictitious nodes: changed weighting from
  `1/d^2` to `1/d^4` (Shepard p=4), reducing relay-induced concentration
  smearing at the AMR transition zone (`src/grid.cpp`)

### Bug fixes
- **VTK wall pressure output**: removed artificial `p=0` zeroing for WALL
  nodes in both VTI and VTU writers. Walls now show the actual EOS-computed
  pressure from mirrored density (`src/vtk_writer.cpp`)

### Calibration (Reimers et al. 2023)
- Tuned `D_grain=4.5e-16`, `D_gb=2.2e-15`, `D_precip=2.5e-14` for
  `amr_ratio=2` grid. Best match at 3.5h: **-1.6%** error vs experiment.
  Late-time (4.23h): **-6.8%** error (was -8.8% with ratio=3)
- Increased `C_sat` from 0.9 to 0.95 (less salt-layer blocking, sustains
  late-stage corrosion)
- Enabled volume-loss-dependent interface decay (`corrosion_decay_l=10`)
  in ratio=3 config (Hermann et al. 2022, Eq. 42)
- Fixed comment with corrected decay factor values

### Other
- Capped dissolution rate plot y-axis at 10 %/h (`scripts/plot_volume_loss.py`)

## 2026-02-15: Implicit solver fixes for AMR

### Bug fixes
- **compute_adaptive_dt column-major bug**: `Eigen::InnerIterator(M_, k)`
  iterated column k (Eigen default is ColMajor). Replaced with full SpMV:
  `Eigen::VectorXd MC = M_ * C_vec` (`src/pd_ard_implicit.cpp`)
- **Wall mirror BC for AMR**: `flat_to_ijk()` divided by `Nx=0` for AMR
  grids (undefined behavior). Fixed to use `g.pos[n]` directly with PD
  neighborhood search for mirror point (`src/boundary.cpp`)
- **Missing update_fictitious**: added `grid.update_fictitious(fields)`
  after `smooth_boundary_concentration` in the implicit coupling loop
  (`src/coupling.cpp`)

## 2026-02-14: AMR + implicit solver

- Adaptive mesh refinement (AMR) with fine/coarse zones and fictitious
  node IDW coupling
- Backward Euler implicit ARD solver with GMRES + ILU preconditioner
- M-matrix upwind stabilization for monotone solutions at high Peclet
- Per-bond anisotropic artificial diffusion (upwind only)
- Cell-list neighbor construction for AMR grids
- VTU (unstructured) output for AMR grids

## 2026-02-13: Initial open-source release

- PD Navier-Stokes (weakly compressible, Tait EOS)
- PD advection-reaction-diffusion with bi-material diffusion
- Voronoi grain structure with GB preferential dissolution
- Salt-layer blocking model (C_sat threshold)
- Fictitious node method (FNM) wall BCs
- Pressure outlet BC (Song et al. 2025)
- VTI output with PVD time series
