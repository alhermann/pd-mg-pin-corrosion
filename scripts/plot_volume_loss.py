#!/usr/bin/env python3
"""Plot simulated vs experimental volume loss curves.

Usage:
    python3 scripts/plot_volume_loss.py [output_dir]

Reads:
    - <output_dir>/mass_loss.csv   (simulated data)
    - config/metadata.csv          (Reimers et al. 2023 experimental data)

Produces:
    - <output_dir>/volume_loss_comparison.png
"""

import csv
import sys
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def load_simulation(path):
    t, ml = [], []
    with open(path) as f:
        reader = csv.reader(f)
        next(reader)  # skip header
        for row in reader:
            t.append(float(row[0]))
            ml.append(float(row[1]))
    return np.array(t), np.array(ml)


def load_experimental(path):
    t, vl = [], []
    with open(path) as f:
        reader = csv.reader(f)
        next(reader)  # skip header
        for row in reader:
            t.append(float(row[4]))   # Degradation Time (h)
            vl.append(float(row[1]))  # Volume Loss (%)
    return np.array(t), np.array(vl)


def main():
    # Paths
    output_dir = sys.argv[1] if len(sys.argv) > 1 else "output"
    sim_path = os.path.join(output_dir, "mass_loss.csv")
    exp_path = os.path.join("config", "metadata.csv")

    if not os.path.exists(sim_path):
        print(f"Error: {sim_path} not found")
        sys.exit(1)
    if not os.path.exists(exp_path):
        print(f"Error: {exp_path} not found")
        sys.exit(1)

    t_sim, ml_sim = load_simulation(sim_path)
    t_exp, vl_exp = load_experimental(exp_path)

    # --- Figure ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Left panel: volume loss curves
    ax1.plot(t_sim, ml_sim, "-", color="#2563eb", linewidth=1.5,
             label="PD simulation")
    ax1.plot(t_exp, vl_exp, "o", color="#dc2626", markersize=7,
             markeredgecolor="black", markeredgewidth=0.5,
             label="Reimers et al. (2023)")
    ax1.set_xlabel("Time (h)", fontsize=12)
    ax1.set_ylabel("Volume loss (%)", fontsize=12)
    ax1.set_xlim(0, max(t_sim.max(), t_exp.max()) * 1.05)
    ax1.set_ylim(0, max(ml_sim.max(), vl_exp.max()) * 1.15)
    ax1.legend(fontsize=11, loc="upper left")
    ax1.grid(True, alpha=0.3)
    ax1.set_title("Volume loss comparison", fontsize=13)

    # Right panel: instantaneous rate
    # Simulation rate (smoothed with 10-point window)
    window = min(10, len(t_sim) // 5) if len(t_sim) > 20 else 1
    dt_sim = np.diff(t_sim)
    dml_sim = np.diff(ml_sim)
    rate_sim = dml_sim / (dt_sim + 1e-30)
    t_rate = 0.5 * (t_sim[:-1] + t_sim[1:])

    if window > 1:
        kernel = np.ones(window) / window
        rate_smooth = np.convolve(rate_sim, kernel, mode="valid")
        t_smooth = np.convolve(t_rate, kernel, mode="valid")
    else:
        rate_smooth = rate_sim
        t_smooth = t_rate

    # Experimental rate (finite differences)
    rate_exp = np.diff(vl_exp) / np.diff(t_exp)
    t_rate_exp = 0.5 * (t_exp[:-1] + t_exp[1:])

    ax2.plot(t_smooth, rate_smooth, "-", color="#2563eb", linewidth=1.5,
             label="PD simulation")
    ax2.plot(t_rate_exp, rate_exp, "s", color="#dc2626", markersize=7,
             markeredgecolor="black", markeredgewidth=0.5,
             label="Reimers et al. (2023)")
    ax2.set_xlabel("Time (h)", fontsize=12)
    ax2.set_ylabel("Instantaneous rate (%/h)", fontsize=12)
    ax2.set_xlim(0, max(t_sim.max(), t_exp.max()) * 1.05)
    ax2.set_ylim(0, None)
    ax2.legend(fontsize=11, loc="upper left")
    ax2.grid(True, alpha=0.3)
    ax2.set_title("Dissolution rate", fontsize=13)

    fig.tight_layout()
    out_path = os.path.join(output_dir, "volume_loss_comparison.png")
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"Saved: {out_path}")

    # Print summary table
    print(f"\n{'Time(h)':>8} {'Exp%':>8} {'Sim%':>8} {'RelErr':>8}")
    print("-" * 36)
    for i in range(len(t_exp)):
        idx = np.argmin(np.abs(t_sim - t_exp[i]))
        err = (ml_sim[idx] - vl_exp[i]) / vl_exp[i] * 100
        print(f"{t_exp[i]:8.3f} {vl_exp[i]:8.2f} {ml_sim[idx]:8.2f} {err:7.1f}%")


if __name__ == "__main__":
    main()
