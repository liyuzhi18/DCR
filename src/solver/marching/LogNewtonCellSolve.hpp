#pragma once

#include "CellSolve.hpp"

namespace dcr::solver {

CellImplicitResult solve_cell_implicit_log_newton(
    const dcr::io::Config& config,
    const dcr::atomic::AtomicData& atomic_data,
    const dcr::state::PlasmaState& plasma,
    const EEDFGridView& grid,
    const BoundaryPhaseResult& boundary,
    const std::vector<dcr::atomic::EnergyLevel>& levels,
    const dcr::base::Vector& nP_old,
    const dcr::base::Vector& flowA_old,
    const dcr::base::Vector& flowM_old,
    double dx_cm,
    double x_left_cm,
    double x_right_cm,
    int cell_index,
    bool detailed_log,
    bool emit_summary_log,
    const dcr::base::Vector* nP_init_override = nullptr,
    const dcr::base::Vector* flowA_init_override = nullptr,
    const dcr::base::Vector* flowM_init_override = nullptr);

} // namespace dcr::solver
