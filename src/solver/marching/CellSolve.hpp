#pragma once

#include "../../atomic/AtomicData.hpp"
#include "../../io/ConfigStructs.hpp"
#include "../../physics/EEDF.hpp"
#include "../../state/PlasmaState.hpp"
#include "../boundary/BoundaryPhase.hpp"
#include "CellAdvance.hpp"
#include "LocalSystemAssembler.hpp"

namespace dcr::solver {

enum class CellImplicitStatus {
    converged,
    max_iter,
    stagnated,
};

// Result container for one implicit cell solve (k -> k+1).
struct CellImplicitResult {
    dcr::base::Vector nP_new;
    dcr::base::Vector flowA_new;
    dcr::base::Vector flowM_new;
    int iterations = 0;
    bool converged = false;
    CellImplicitStatus status = CellImplicitStatus::max_iter;
    double final_rel = 0.0;
    double final_resid_rel = 0.0;
    double elapsed_seconds = 0.0;
    LocalSystem local_final;
    FlowAdvanceResult flow_final;
};

// Expand compact background vector (P block ordering) to full global-state indexing.
dcr::base::Vector make_background_full(const dcr::base::Vector& nP,
                                       const BoundaryPhaseResult& boundary,
                                       int total_states);

// Coupled fixed-point implicit solve for one spatial cell:
// unknowns are (nP_{k+1}, nA_{k+1}, nM_{k+1}), with rates evaluated at x_{k+1}.
// Background solve uses constant nuclei closure each iteration.
CellImplicitResult solve_cell_implicit(
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
