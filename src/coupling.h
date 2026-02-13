#pragma once
#include "grid.h"
#include "fields.h"
#include "config.h"
#include "pd_ns.h"
#include "pd_ard.h"
#include "vtk_writer.h"

class CoupledSolver {
public:
    void run(Grid& grid, Fields& fields, const Config& cfg);

private:
    PD_NS_Solver flow_solver_;
    PD_ARD_Solver ard_solver_;
    VTKWriter writer_;

    int initial_solid_count_ = 0;
    int frame_count_ = 0;

    std::string make_filename(const Config& cfg, const std::string& prefix,
                              double time_s, int frame);
    void write_diagnostics(const Grid& grid, const Fields& fields, double t_corr,
                           const Config& cfg);
};
