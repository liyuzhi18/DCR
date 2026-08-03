#pragma once

#include "../../atomic/AtomicData.hpp"
#include "../../io/ConfigStructs.hpp"
#include "../../physics/EEDF.hpp"
#include "../../state/PlasmaState.hpp"
#include "../boundary/BoundaryPhase.hpp"
#include "CellAdvance.hpp"
#include "AdaptiveRecycling.hpp"
#include "LocalSystemAssembler.hpp"

#include <limits>

namespace dcr::solver {

// Result container for one implicit cell solve (k -> k+1).
struct CellImplicitResult {
    dcr::base::Vector nP_new;
    dcr::base::Vector flowA_new;
    dcr::base::Vector flowM_new;
    int iterations = 0;
    bool converged = false;
    double final_rel = 0.0;
    double final_resid_rel = 0.0;
    double elapsed_seconds = 0.0;
    LocalSystem local_final;
    FlowAdvanceResult flow_final;
    bool variable_nuclei_balance_closure = false;
    double variable_ion_divergence_nuclei_cm3_s = std::numeric_limits<double>::quiet_NaN();
    double variable_flowA_divergence_nuclei_cm3_s = std::numeric_limits<double>::quiet_NaN();
    double variable_flowM_divergence_nuclei_cm3_s = std::numeric_limits<double>::quiet_NaN();
    double variable_neutral_exhaust_nuclei_cm3_s = std::numeric_limits<double>::quiet_NaN();
    double variable_balance_rhs_cm3_s = std::numeric_limits<double>::quiet_NaN();
    double variable_balance_residual_cm3_s = std::numeric_limits<double>::quiet_NaN();
    double prescribed_nuclei_density_cm3 = std::numeric_limits<double>::quiet_NaN();
    double recycling_source_nuclei_cm3_s = std::numeric_limits<double>::quiet_NaN();
    double local_atom_exhaust_nuclei_cm3_s = std::numeric_limits<double>::quiet_NaN();
    double local_molecule_exhaust_nuclei_cm3_s = std::numeric_limits<double>::quiet_NaN();
    double ion_divergence_closure_nuclei_cm3_s = std::numeric_limits<double>::quiet_NaN();
    double ion_balance_coefficient_s = std::numeric_limits<double>::quiet_NaN();
};

// Expand compact background vector (P block ordering) to full global-state indexing.
dcr::base::Vector make_background_full(const dcr::base::Vector& nP,
                                       const BoundaryPhaseResult& boundary,
                                       int total_states);

// Coupled fixed-point implicit solve for one spatial cell:
// unknowns are (nP_{k+1}, nA_{k+1}, nM_{k+1}), with rates evaluated at x_{k+1}.
// Fixed mode enforces prescribed local nuclei density. Variable mode replaces
// one CR row with a backward/upwind balance using prescribed ion velocities.
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
    const AdaptiveTransportProfile* adaptive_profile = nullptr);

} // namespace dcr::solver
