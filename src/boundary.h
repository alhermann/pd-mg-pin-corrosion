#pragma once
#include "fields.h"
#include "grid.h"
#include "config.h"

void apply_inlet_bc(Fields& f, const Grid& g, const Config& cfg);
void apply_outlet_bc(Fields& f, const Grid& g, const Config& cfg);
void apply_wall_bc(Fields& f, const Grid& g, const Config& cfg);
void apply_wall_bc_new(Fields& f, const Grid& g, const Config& cfg);
void apply_wall_concentration_bc(Fields& f, const Grid& g);
void apply_solid_surface_bc(Fields& f, const Grid& g);
void smooth_boundary_concentration(Fields& f, const Grid& g, const Config& cfg);
void update_node_types_after_dissolution(Grid& g, const Fields& f);
