#include "config.h"
#include "utils.h"
#include <fstream>
#include <sstream>
#include <iostream>
#include <algorithm>
#include <cmath>

static std::string trim(const std::string& s) {
    auto start = s.find_first_not_of(" \t\r\n");
    if (start == std::string::npos) return "";
    auto end = s.find_last_not_of(" \t\r\n");
    return s.substr(start, end - start + 1);
}

void Config::load(const std::string& filename) {
    std::ifstream f(filename);
    if (!f.is_open()) {
        std::cerr << "Warning: Cannot open config file '" << filename
                  << "', using defaults.\n";
        compute_derived();
        return;
    }

    std::string line;
    while (std::getline(f, line)) {
        // Strip comments
        auto pos = line.find('#');
        if (pos != std::string::npos) line = line.substr(0, pos);
        line = trim(line);
        if (line.empty()) continue;

        auto eq = line.find('=');
        if (eq == std::string::npos) continue;

        std::string key = trim(line.substr(0, eq));
        std::string val = trim(line.substr(eq + 1));
        if (key.empty() || val.empty()) continue;

        // Match keys
        if      (key == "dx")               dx = std::stod(val);
        else if (key == "m_ratio")           m_ratio = std::stoi(val);
        else if (key == "R_wire")            R_wire = std::stod(val);
        else if (key == "L_wire")            L_wire = std::stod(val);
        else if (key == "R_tube")            R_tube = std::stod(val);
        else if (key == "L_upstream")        L_upstream = std::stod(val);
        else if (key == "L_downstream")      L_downstream = std::stod(val);
        else if (key == "rho_f")             rho_f = std::stod(val);
        else if (key == "mu_f")              mu_f = std::stod(val);
        else if (key == "gamma_eos")         gamma_eos = std::stod(val);
        else if (key == "c0")               c0 = std::stod(val);
        else if (key == "eta_density")       eta_density = std::stod(val);
        else if (key == "Q_flow")            Q_flow = std::stod(val);
        else if (key == "rho_m")             rho_m = std::stod(val);
        else if (key == "D_liquid")          D_liquid = std::stod(val);
        else if (key == "D_grain")           D_grain = std::stod(val);
        else if (key == "D_gb")              D_gb = std::stod(val);
        else if (key == "C_solid_init")      C_solid_init = std::stod(val);
        else if (key == "C_liquid_init")     C_liquid_init = std::stod(val);
        else if (key == "C_thresh")          C_thresh = std::stod(val);
        else if (key == "C_sat")             C_sat = std::stod(val);
        else if (key == "w_advect")          w_advect = std::stod(val);
        else if (key == "alpha_art_diff")    alpha_art_diff = std::stod(val);
        else if (key == "grain_size_mean")   grain_size_mean = std::stod(val);
        else if (key == "grain_size_std")    grain_size_std = std::stod(val);
        else if (key == "gb_width_cells")    gb_width_cells = std::stoi(val);
        else if (key == "k_corr")            k_corr = std::stod(val);
        else if (key == "gb_corr_factor")    gb_corr_factor = std::stod(val);
        else if (key == "cfl_factor")        cfl_factor = std::stod(val);
        else if (key == "cfl_factor_corr")  cfl_factor_corr = std::stod(val);
        else if (key == "flow_max_iters")    flow_max_iters = std::stoi(val);
        else if (key == "flow_conv_tol")     flow_conv_tol = std::stod(val);
        else if (key == "T_final")           T_final = std::stod(val);
        else if (key == "corrosion_steps_per_check") corrosion_steps_per_check = std::stoi(val);
        else if (key == "output_every_flow") output_every_flow = std::stoi(val);
        else if (key == "output_every_corr") output_every_corr = std::stoi(val);
        else if (key == "output_dir")        output_dir = val;
        else if (key == "use_implicit")          use_implicit = std::stoi(val);
        else if (key == "implicit_dt_fraction")  implicit_dt_fraction = std::stod(val);
        else if (key == "implicit_dt_max")       implicit_dt_max = std::stod(val);
        else if (key == "implicit_output_every") implicit_output_every = std::stoi(val);
        else if (key == "newton_tol")            newton_tol = std::stod(val);
        else if (key == "newton_max_iter")       newton_max_iter = std::stoi(val);
        else {
            std::cerr << "Warning: Unknown config key '" << key << "'\n";
        }
    }

    compute_derived();
}

void Config::compute_derived() {
    delta = m_ratio * dx;

    // Compute inlet velocity from volumetric flow rate
    // U_in = Q / A where A = pi * R_tube^2 (circular tube cross-section)
    U_in = Q_flow / (PI * R_tube * R_tube);

    // Ensure c0 is at least 25x U_in for weakly compressible assumption (Ma² < 0.002)
    if (c0 < 25.0 * U_in) {
        c0 = 25.0 * U_in;
        std::printf("NOTE: Increased c0 to %.4e (25x U_in) for stability.\n", c0);
    }
}

void Config::print() const {
    std::printf("=== Configuration ===\n");
    std::printf("  DIM          = %d\n", DIM);
    std::printf("  dx           = %.2e m\n", dx);
    std::printf("  delta        = %.2e m (m=%d)\n", delta, m_ratio);
    std::printf("  R_wire       = %.2e m\n", R_wire);
    std::printf("  L_wire       = %.2e m\n", L_wire);
    std::printf("  R_tube       = %.2e m\n", R_tube);
    std::printf("  U_in         = %.4e m/s\n", U_in);
    std::printf("  rho_f        = %.1f kg/m3\n", rho_f);
    std::printf("  mu_f         = %.2e Pa.s\n", mu_f);
    std::printf("  Re_wire      = %.2f\n", rho_f * U_in * 2.0 * R_wire / mu_f);
    std::printf("  c0           = %.2f m/s (Mach ~ %.4f)\n", c0, U_in / c0);
    std::printf("  D_liquid     = %.2e m2/s\n", D_liquid);
    std::printf("  D_grain      = %.2e m2/s\n", D_grain);
    std::printf("  D_gb         = %.2e m2/s\n", D_gb);
    std::printf("  C_sat        = %.2f\n", C_sat);
    std::printf("  k_corr       = %.2e 1/s\n", k_corr);
    std::printf("  gb_corr_fac  = %.1f\n", gb_corr_factor);
    std::printf("  T_final      = %.1f s (%.2f h)\n", T_final, T_final / 3600.0);
    std::printf("  output_dir   = %s\n", output_dir.c_str());
    std::printf("=====================\n\n");
}
