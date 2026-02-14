#!/usr/bin/env python3
"""
Poiseuille flow validation: compare PD-NS numerical solution to analytical profile.

Reads VTI (VTK ImageData) output and plots velocity profiles at multiple axial
stations against the analytical 2D Poiseuille solution:

    v_z(x) = (3/2) * U_in * (1 - (x / R_tube)^2)

Usage:
    python3 plot_poiseuille.py [output_dir] [vti_file]

If vti_file not given, uses the last flow*.vti or state*.vti in the output dir.
"""

import sys
import os
import glob
import xml.etree.ElementTree as ET
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


def parse_vti(filepath):
    """Parse an ASCII VTI file and return grid info + field data."""
    tree = ET.parse(filepath)
    root = tree.getroot()

    image_data = root.find(".//ImageData")
    extent_str = image_data.get("WholeExtent").split()
    origin_str = image_data.get("Origin").split()
    spacing_str = image_data.get("Spacing").split()

    ext = [int(x) for x in extent_str]  # x0 x1 y0 y1 z0 z1
    origin = [float(x) for x in origin_str]
    spacing = [float(x) for x in spacing_str]

    nx = ext[1] - ext[0] + 1
    ny = ext[3] - ext[2] + 1
    nz = ext[5] - ext[4] + 1
    N = nx * ny * nz

    # Build coordinate arrays (VTK: i varies fastest, then j, then k)
    x = np.zeros(N)
    y = np.zeros(N)
    for n in range(N):
        j = n // nx
        i = n % nx
        x[n] = origin[0] + i * spacing[0]
        y[n] = origin[1] + j * spacing[1]

    # Parse data arrays
    fields = {}
    for da in root.findall(".//DataArray"):
        name = da.get("Name")
        ncomp = int(da.get("NumberOfComponents", "1"))
        text = da.text.strip()
        vals = np.array([float(v) for v in text.split()])

        if ncomp == 3:
            vals = vals.reshape(-1, 3)
        fields[name] = vals

    return {
        "nx": nx, "ny": ny, "nz": nz, "N": N,
        "origin": origin, "spacing": spacing,
        "x": x, "y": y,
        "fields": fields,
    }


def find_vti_file(output_dir):
    """Find the best VTI file for analysis (prefer flow, then state)."""
    patterns = ["flow_*.vti", "state_*.vti", "corr_*.vti", "final_*.vti"]
    for pat in patterns:
        files = sorted(glob.glob(os.path.join(output_dir, pat)))
        if files:
            return files[-1]  # last one (highest iteration)
    return None


def main():
    output_dir = sys.argv[1] if len(sys.argv) > 1 else "output_poiseuille"
    vti_file = sys.argv[2] if len(sys.argv) > 2 else find_vti_file(output_dir)

    if vti_file is None:
        print(f"No VTI files found in {output_dir}")
        sys.exit(1)

    print(f"Parsing: {vti_file}")
    data = parse_vti(vti_file)
    print(f"Grid: {data['nx']} x {data['ny']}, N={data['N']}")
    print(f"Origin: {data['origin']}")
    print(f"Spacing: {data['spacing']}")

    x = data["x"]  # radial position
    y = data["y"]  # axial position
    dx = data["spacing"][0]
    vel = data["fields"]["velocity"]  # (N, 3): vx, vy, vz
    node_type = data["fields"]["node_type"]
    pressure = data["fields"]["pressure"]

    vx = vel[:, 0]  # radial velocity
    vy = vel[:, 1]  # axial velocity (flow direction in 2D r-z)

    # Determine geometry from params (infer from grid)
    # R_tube: max |x| of FLUID nodes
    fluid_mask = node_type == 0  # FLUID
    if not np.any(fluid_mask):
        print("No FLUID nodes found!")
        sys.exit(1)

    R_tube = np.max(np.abs(x[fluid_mask]))
    y_min_fluid = np.min(y[fluid_mask])
    y_max_fluid = np.max(y[fluid_mask])
    print(f"R_tube (from grid): {R_tube*1e6:.1f} um")
    print(f"Axial extent: [{y_min_fluid*1e6:.0f}, {y_max_fluid*1e6:.0f}] um")

    # Compute U_in from mean axial velocity of inlet nodes (type 3)
    inlet_mask = node_type == 3  # INLET
    if np.any(inlet_mask):
        # Mean velocity = integral of parabolic / width = (2/3) * v_max for 2D
        U_in = np.mean(vy[inlet_mask])
        print(f"U_in (from inlet mean): {U_in:.4f} m/s")
    else:
        # Fallback: estimate from fluid nodes far from walls
        center_mask = fluid_mask & (np.abs(x) < 0.2 * R_tube)
        U_in = np.mean(vy[center_mask]) / 1.5  # center vel ~ 1.5 * U_mean for 2D
        print(f"U_in (estimated): {U_in:.4f} m/s")

    # Axial stations for profile extraction
    y_range = y_max_fluid - y_min_fluid
    # Pick stations at 25%, 50%, 75% of the fluid domain
    stations_frac = [0.10, 0.25, 0.50, 0.75, 0.90]
    stations_y = [y_min_fluid + f * y_range for f in stations_frac]
    station_labels = [f"z = {s*1e6:.0f} um ({f*100:.0f}%)" for s, f in zip(stations_y, stations_frac)]

    # Analytical Poiseuille profile for 2D planar flow
    r_analytical = np.linspace(-R_tube, R_tube, 200)
    v_analytical = 1.5 * U_in * (1.0 - (r_analytical / R_tube) ** 2)

    # =========================================================================
    # Plot 1: Velocity profiles at multiple axial stations
    # =========================================================================
    fig = plt.figure(figsize=(14, 10))
    gs = GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.3)

    ax1 = fig.add_subplot(gs[0, 0])
    colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(stations_y)))
    l2_errors = []

    for idx, (y_station, label, color) in enumerate(zip(stations_y, station_labels, colors)):
        # Select fluid nodes near this axial station
        mask = fluid_mask & (np.abs(y - y_station) < 0.6 * dx)
        if np.sum(mask) < 3:
            print(f"  Station {label}: too few nodes ({np.sum(mask)}), skipping")
            l2_errors.append(np.nan)
            continue

        x_prof = x[mask]
        v_prof = vy[mask]

        # Sort by x for clean plot
        sort_idx = np.argsort(x_prof)
        x_prof = x_prof[sort_idx]
        v_prof = v_prof[sort_idx]

        # Compute L2 error (only for |x| < R_tube)
        in_tube = np.abs(x_prof) < R_tube * 0.99
        v_ana_at_pts = 1.5 * U_in * (1.0 - (x_prof[in_tube] / R_tube) ** 2)
        v_num_at_pts = v_prof[in_tube]

        l2_err = np.sqrt(np.sum((v_num_at_pts - v_ana_at_pts)**2) /
                         np.sum(v_ana_at_pts**2))
        l2_errors.append(l2_err)

        ax1.plot(v_prof * 1e3, x_prof * 1e6, 'o', color=color, markersize=3, alpha=0.7,
                 label=f"{label} (L2={l2_err:.1%})")
        print(f"  Station {label}: {np.sum(mask)} nodes, L2 error = {l2_err:.4f} ({l2_err:.1%})")

    ax1.plot(v_analytical * 1e3, r_analytical * 1e6, 'k-', linewidth=2, label="Analytical")
    ax1.set_xlabel("Axial velocity [mm/s]")
    ax1.set_ylabel("Radial position [um]")
    ax1.set_title("Velocity profiles at various axial stations")
    ax1.legend(fontsize=7, loc="center left")
    ax1.axhline(y=R_tube * 1e6, color='gray', linestyle='--', alpha=0.5, linewidth=0.8)
    ax1.axhline(y=-R_tube * 1e6, color='gray', linestyle='--', alpha=0.5, linewidth=0.8)
    ax1.grid(True, alpha=0.3)

    # =========================================================================
    # Plot 2: L2 error vs axial position
    # =========================================================================
    ax2 = fig.add_subplot(gs[0, 1])
    valid = ~np.isnan(l2_errors)
    ax2.plot(np.array(stations_y)[valid] * 1e6, np.array(l2_errors)[valid] * 100,
             'bo-', linewidth=2, markersize=8)
    ax2.set_xlabel("Axial position [um]")
    ax2.set_ylabel("L2 relative error [%]")
    ax2.set_title("Profile error vs. axial position")
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(bottom=0)

    # =========================================================================
    # Plot 3: Detailed profile at the midpoint (largest axial station with data)
    # =========================================================================
    ax3 = fig.add_subplot(gs[1, 0])

    # Use the 50% station
    y_mid = stations_y[2]
    mask_mid = fluid_mask & (np.abs(y - y_mid) < 0.6 * dx)
    if np.sum(mask_mid) > 3:
        x_mid = x[mask_mid]
        v_mid = vy[mask_mid]
        sort_idx = np.argsort(x_mid)
        x_mid = x_mid[sort_idx]
        v_mid = v_mid[sort_idx]

        v_ana_mid = 1.5 * U_in * (1.0 - (x_mid / R_tube) ** 2)

        ax3.plot(x_mid * 1e6, v_mid * 1e3, 'ro', markersize=5, label="PD numerical")
        ax3.plot(r_analytical * 1e6, v_analytical * 1e3, 'k-', linewidth=2, label="Analytical")
        ax3.set_xlabel("Radial position [um]")
        ax3.set_ylabel("Axial velocity [mm/s]")
        ax3.set_title(f"Midpoint profile (z = {y_mid*1e6:.0f} um)")
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # Inset: error distribution
        inset = ax3.inset_axes([0.55, 0.55, 0.4, 0.4])
        in_tube = np.abs(x_mid) < R_tube * 0.99
        error_pct = (v_mid[in_tube] - v_ana_mid[in_tube]) / (v_ana_mid[in_tube] + 1e-30) * 100
        inset.plot(x_mid[in_tube] * 1e6, error_pct, 'r-', linewidth=1)
        inset.set_xlabel("r [um]", fontsize=7)
        inset.set_ylabel("Error [%]", fontsize=7)
        inset.tick_params(labelsize=6)
        inset.grid(True, alpha=0.3)
        inset.set_title("Local error", fontsize=7)

    # =========================================================================
    # Plot 4: Pressure field along centerline
    # =========================================================================
    ax4 = fig.add_subplot(gs[1, 1])
    center_mask = fluid_mask & (np.abs(x) < 0.6 * dx)
    if np.sum(center_mask) > 3:
        y_center = y[center_mask]
        p_center = pressure[center_mask]
        sort_idx = np.argsort(y_center)
        y_center = y_center[sort_idx]
        p_center = p_center[sort_idx]

        ax4.plot(y_center * 1e6, p_center, 'b-', linewidth=1.5)
        ax4.set_xlabel("Axial position [um]")
        ax4.set_ylabel("Pressure [Pa]")
        ax4.set_title("Centerline pressure")
        ax4.grid(True, alpha=0.3)

        # Expected dp/dz for Poiseuille: dp/dz = -3 mu U_in / R^2 (2D planar)
        mu = 1e-3
        dpdz_analytical = -3 * mu * U_in / (R_tube ** 2)
        # Fit linear pressure gradient
        if len(y_center) > 5:
            # Use middle 50% to avoid boundary effects
            n = len(y_center)
            mid_slice = slice(n // 4, 3 * n // 4)
            coeffs = np.polyfit(y_center[mid_slice], p_center[mid_slice], 1)
            dpdz_numerical = coeffs[0]
            ax4.plot(y_center, np.polyval(coeffs, y_center), 'r--', linewidth=1,
                     label=f"Linear fit: dp/dz = {dpdz_numerical:.1f} Pa/m")
            ax4.legend(fontsize=8)
            print(f"\nPressure gradient:")
            print(f"  Analytical: dp/dz = {dpdz_analytical:.1f} Pa/m")
            print(f"  Numerical:  dp/dz = {dpdz_numerical:.1f} Pa/m")
            print(f"  Error: {abs(dpdz_numerical - dpdz_analytical)/abs(dpdz_analytical)*100:.1f}%")

    fig.suptitle("Poiseuille Flow Validation (PD Navier-Stokes, 2D planar, no pin)",
                 fontsize=13, fontweight="bold")

    outpath = os.path.join(output_dir, "poiseuille_validation.png")
    fig.savefig(outpath, dpi=150, bbox_inches="tight")
    print(f"\nPlot saved to: {outpath}")

    # Also save a summary
    summary_path = os.path.join(output_dir, "validation_summary.txt")
    with open(summary_path, "w") as f:
        f.write("Poiseuille Flow Validation Summary\n")
        f.write("=" * 40 + "\n")
        f.write(f"VTI file: {vti_file}\n")
        f.write(f"Grid: {data['nx']} x {data['ny']}\n")
        f.write(f"R_tube: {R_tube*1e6:.1f} um\n")
        f.write(f"U_in: {U_in:.4f} m/s\n\n")
        f.write("L2 errors by station:\n")
        for label, err in zip(station_labels, l2_errors):
            if not np.isnan(err):
                f.write(f"  {label}: {err:.4f} ({err:.1%})\n")
    print(f"Summary saved to: {summary_path}")


if __name__ == "__main__":
    main()
