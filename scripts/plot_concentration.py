#!/usr/bin/env python3
"""
Plot dissolved Mg concentration transport from peridynamic corrosion simulation.

Reads VTI (ASCII VTK ImageData) files from build/output_viz/ and produces a
multi-panel publication-quality figure:
  - Top:          2D concentration contour at t = 2.0 s
  - Bottom-left:  Centerline concentration vs axial position for several times
  - Bottom-right: Transverse concentration profiles at multiple y-locations

Output: build/output_viz/concentration_transport.png (200 DPI)
"""

import xml.etree.ElementTree as ET
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import Normalize
from matplotlib.patches import Rectangle
from pathlib import Path
import re

# ---------------------------------------------------------------------------
# Plot style
# ---------------------------------------------------------------------------
plt.rcParams.update({
    "font.family":       "serif",
    "font.size":         10,
    "axes.labelsize":    10,
    "axes.titlesize":    11,
    "legend.fontsize":   8,
    "xtick.labelsize":   9,
    "ytick.labelsize":   9,
    "xtick.direction":   "in",
    "ytick.direction":   "in",
    "xtick.top":         True,
    "ytick.right":       True,
    "axes.linewidth":    0.7,
    "grid.linewidth":    0.4,
    "lines.linewidth":   1.4,
    "savefig.dpi":       200,
})

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
VIZ_DIR  = BASE_DIR / "build" / "output_viz"
OUTPUT   = VIZ_DIR / "concentration_transport.png"

NX, NY   = 67, 287
ORIGIN_X = -0.000165   # m
ORIGIN_Y = -0.000515   # m
DX       = 5e-6        # m

# Node-type enum
FLUID    = 0
SOLID_MG = 1
WALL     = 2
INLET    = 3
OUTLET   = 4
OUTSIDE  = 5

# Physical coordinates in micrometres
x_um = np.arange(NX) * (DX * 1e6) + (ORIGIN_X * 1e6)
y_um = np.arange(NY) * (DX * 1e6) + (ORIGIN_Y * 1e6)

# ---------------------------------------------------------------------------
# VTI reader
# ---------------------------------------------------------------------------

def read_vti(filepath):
    """Parse an ASCII VTI file and return dict of 2-D arrays (Ny, Nx)."""
    tree = ET.parse(filepath)
    root = tree.getroot()
    fields = {}
    for da in root.findall(".//DataArray"):
        name  = da.get("Name")
        ncomp = int(da.get("NumberOfComponents", "1"))
        dtype = da.get("type")
        if ncomp != 1:
            continue  # skip velocity (3-component)
        raw = da.text.strip().split()
        if "Float" in dtype:
            arr = np.array(raw, dtype=np.float64)
        else:
            arr = np.array(raw, dtype=np.int32)
        fields[name] = arr.reshape((NY, NX))
    return fields


# ---------------------------------------------------------------------------
# Select snapshot files
# ---------------------------------------------------------------------------

def parse_time(fname):
    """Extract simulation time (seconds) from file name."""
    m = re.search(r"_t([\d.]+)s\.vti$", fname)
    return float(m.group(1)) if m else None


# Desired times for centerline plot
desired_times = [0.1, 0.6, 1.0, 1.7, 2.0]

vti_files = sorted(VIZ_DIR.glob("corr_*.vti")) + sorted(VIZ_DIR.glob("final_*.vti"))
time_file_map = {}
for f in vti_files:
    t = parse_time(f.name)
    if t is not None:
        time_file_map[t] = f

# Pick closest available file for each desired time
def pick_closest(desired, available):
    avail = sorted(available)
    return min(avail, key=lambda a: abs(a - desired))

selected = {}
for td in desired_times:
    tc = pick_closest(td, time_file_map.keys())
    selected[td] = (tc, time_file_map[tc])

# Final snapshot (t = 2.0 s) -- prefer the final_* file if it exists
final_file = list(VIZ_DIR.glob("final_*.vti"))
if final_file:
    final_path = final_file[0]
    final_time = parse_time(final_path.name)
else:
    final_time = 2.0
    final_path = time_file_map[pick_closest(2.0, time_file_map.keys())]

# ---------------------------------------------------------------------------
# Read data
# ---------------------------------------------------------------------------
print(f"Reading final snapshot: {final_path.name}")
final_data = read_vti(final_path)
C_final    = final_data["concentration"]
ntype      = final_data["node_type"]

# Read snapshots for centerline plot
centerline_data = {}
for td, (tc, fpath) in selected.items():
    print(f"  snapshot t={tc:.1f}s  ({fpath.name})")
    d = read_vti(fpath)
    centerline_data[tc] = d["concentration"]

# ---------------------------------------------------------------------------
# Masks and derived quantities
# ---------------------------------------------------------------------------
fluid_mask = (ntype == FLUID)
solid_mask = (ntype == SOLID_MG)

# Masked concentration (NaN outside fluid)
C_masked = np.where(fluid_mask, C_final, np.nan)

# Pin bounding box (in um)
pin_ys = y_um[np.any(solid_mask, axis=1)]
pin_xs = x_um[np.any(solid_mask, axis=0)]
pin_x0 = pin_xs.min() - DX * 1e6 / 2
pin_x1 = pin_xs.max() + DX * 1e6 / 2
pin_y0 = pin_ys.min() - DX * 1e6 / 2
pin_y1 = pin_ys.max() + DX * 1e6 / 2

# Max concentration in fluid (for colorbar)
C_max = C_final[fluid_mask].max()

# ---------------------------------------------------------------------------
# Figure layout
# ---------------------------------------------------------------------------
fig = plt.figure(figsize=(11, 13))

# Outer GridSpec: top row for the 2D map, bottom row for the two line plots
outer = gridspec.GridSpec(2, 1, height_ratios=[1.55, 1.0],
                          hspace=0.25,
                          left=0.07, right=0.97, top=0.95, bottom=0.05)

# The 2D map uses a sub-gridspec so we can place the colorbar neatly
gs_top = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=outer[0],
                                          width_ratios=[1, 0.035], wspace=0.03)
ax_2d  = fig.add_subplot(gs_top[0, 0])
ax_cb  = fig.add_subplot(gs_top[0, 1])

# Bottom two panels
gs_bot = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=outer[1],
                                          wspace=0.32)
ax_cl   = fig.add_subplot(gs_bot[0, 0])
ax_prof = fig.add_subplot(gs_bot[0, 1])

# ===== Panel 1: 2-D concentration field ===================================
cmap = plt.cm.hot_r.copy()
cmap.set_bad(color="white")

# Cell edges for pcolormesh
half = DX * 1e6 / 2
xe = np.append(x_um - half, x_um[-1] + half)
ye = np.append(y_um - half, y_um[-1] + half)

pcm = ax_2d.pcolormesh(xe, ye, C_masked,
                        cmap=cmap, shading="flat",
                        norm=Normalize(vmin=0, vmax=C_max),
                        rasterized=True)

# Pin rectangle
pin_rect = Rectangle((pin_x0, pin_y0), pin_x1 - pin_x0, pin_y1 - pin_y0,
                      facecolor="0.45", edgecolor="black", linewidth=0.9,
                      zorder=3, label="Mg pin")
ax_2d.add_patch(pin_rect)

# Tube wall indicators
wall_cols = np.where(np.any(ntype == WALL, axis=0))[0]
ax_2d.axvline(x_um[wall_cols[0]] - half,  color="0.25", lw=0.7, ls="--",
              label="Tube wall")
ax_2d.axvline(x_um[wall_cols[-1]] + half, color="0.25", lw=0.7, ls="--")

# Horizontal reference lines for transverse-profile y-positions
prof_ys = [425, 500, 700]
prof_styles = [("C0", "-"), ("C1", "--"), ("C3", ":")]
for y_ref, (col, ls) in zip(prof_ys, prof_styles):
    ax_2d.axhline(y_ref, color=col, lw=0.8, ls=ls, alpha=0.75)
    ax_2d.text(xe[-1] + 2, y_ref, f"{y_ref}", fontsize=7, color=col,
               va="center", ha="left", clip_on=False)

# Colorbar
cb = fig.colorbar(pcm, cax=ax_cb)
cb.set_label("Dissolved Mg concentration, $C$", fontsize=10)

ax_2d.set_xlabel(r"Transverse position $x$ ($\mu$m)")
ax_2d.set_ylabel(r"Axial position $y$ ($\mu$m)")
ax_2d.set_title(r"Dissolved Mg Concentration Transport  ---  $t = 2.0\;$s",
                fontsize=12, fontweight="bold", pad=10)
ax_2d.set_xlim(xe[0], xe[-1])
ax_2d.set_ylim(ye[0], ye[-1])
ax_2d.set_aspect("equal")
ax_2d.legend(loc="upper left", fontsize=8, framealpha=0.92,
             edgecolor="0.6")

# Label panels
ax_2d.text(-0.04, 1.01, "(a)", transform=ax_2d.transAxes,
           fontsize=12, fontweight="bold", va="bottom")

# ===== Panel 2: Centerline concentration vs y =============================
ix_center = NX // 2  # x = 0

colors_cl = plt.cm.viridis(np.linspace(0.12, 0.92, len(centerline_data)))

for idx, (tc, C_snap) in enumerate(sorted(centerline_data.items())):
    c_line = C_snap[:, ix_center].copy()
    c_line[ntype[:, ix_center] != FLUID] = np.nan
    ax_cl.plot(y_um, c_line, color=colors_cl[idx], lw=1.5,
               label=f"$t = {tc:.1f}$ s")

# Shade pin region on centerline
ax_cl.axvspan(pin_y0, pin_y1, color="0.88", zorder=0, label="Mg pin")

ax_cl.set_xlabel(r"Axial position $y$ ($\mu$m)")
ax_cl.set_ylabel(r"Concentration $C$ at $x = 0$")
ax_cl.set_title("Centerline concentration evolution", fontweight="bold")
ax_cl.legend(fontsize=8, loc="upper right", framealpha=0.92, edgecolor="0.6")
ax_cl.set_xlim(y_um[0], y_um[-1])
ax_cl.set_ylim(bottom=0)
ax_cl.grid(True, ls=":", alpha=0.45)

ax_cl.text(-0.02, 1.02, "(b)", transform=ax_cl.transAxes,
           fontsize=12, fontweight="bold", va="bottom")

# ===== Panel 3: Transverse concentration profiles at selected y ============

for y_ref, (col, ls) in zip(prof_ys, prof_styles):
    iy = np.argmin(np.abs(y_um - y_ref))
    c_trans = C_final[iy, :].copy()
    c_trans[ntype[iy, :] != FLUID] = np.nan
    ax_prof.plot(x_um, c_trans, color=col, ls=ls, lw=1.6,
                 label=rf"$y = {y_um[iy]:.0f}\;\mu$m")

# Shade pin cross-section
ax_prof.axvspan(pin_x0, pin_x1, color="0.85", zorder=0, label="Mg pin")

ax_prof.set_xlabel(r"Transverse position $x$ ($\mu$m)")
ax_prof.set_ylabel("Concentration $C$")
ax_prof.set_title("Transverse profiles downstream of pin", fontweight="bold")
ax_prof.legend(fontsize=8, loc="upper right", framealpha=0.92, edgecolor="0.6")
ax_prof.set_xlim(x_um[0], x_um[-1])
ax_prof.set_ylim(bottom=0)
ax_prof.grid(True, ls=":", alpha=0.45)

ax_prof.text(-0.02, 1.02, "(c)", transform=ax_prof.transAxes,
             fontsize=12, fontweight="bold", va="bottom")

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------
fig.savefig(str(OUTPUT), dpi=200, bbox_inches="tight")
plt.close(fig)
print(f"\nFigure saved to {OUTPUT}")
