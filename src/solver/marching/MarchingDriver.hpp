#pragma once

#include "../../atomic/AtomicData.hpp"
#include "../../io/ConfigStructs.hpp"
#include "../../physics/EEDF.hpp"
#include "../../state/PlasmaState.hpp"
#include "../boundary/BoundaryPhase.hpp"
#include "../core/RateAnalysis.hpp"
#include <vector>

namespace dcr::solver {

struct AdaptiveTransportProfile;

struct MarchingHistory {
    // Node locations (cm), size = num_cells.
    std::vector<double> x_cm;
    // Boundary and per-cell marching timing/iteration diagnostics.
    double boundary_elapsed_seconds = 0.0;
    int boundary_iterations = 0;
    std::vector<double> cell_elapsed_seconds;
    std::vector<int> cell_iterations;
    std::vector<int> cell_converged;
    std::vector<double> cell_final_relative_change;
    std::vector<double> cell_final_residual_relative;
    std::vector<double> cell_final_map_residual_norm;
    // Background populations on full state indexing at each node.
    // Shape conceptually: [num_cells, total_states].
    std::vector<dcr::base::Vector> background_full;
    // Recycling-flow block populations at each node.
    // Shapes conceptually:
    // - flowA: [num_cells, |A_indices|]
    // - flowM: [num_cells, |M_indices|]
    std::vector<dcr::base::Vector> flowA;
    std::vector<dcr::base::Vector> flowM;
    // QSS-only reconstructed excited atomic populations at each node.
    // Empty for full_dcr runs.
    std::vector<dcr::base::Vector> qss_local_transient_atomic;
    std::vector<dcr::base::Vector> qss_flow_transient_atomic;
    std::vector<int> qss_local_transient_atomic_indices;
    std::vector<int> qss_flow_transient_atomic_indices;
    // Local effective atomic rates and QSS transport diagnostics at each node.
    std::vector<RateDiagnosticSnapshot> rate_diagnostics;
    // Passive adaptive-recycling diagnostics. These are populated only when
    // numerics.adaptive_recycling_domain.enabled is true; they do not affect
    // the marching solve until the adaptive closure is explicitly wired in.
    bool adaptive_recycling_diagnostics_enabled = false;
    bool adaptive_recycling_L1_found = false;
    int adaptive_recycling_L1_index = -1;
    double adaptive_recycling_L1_cm = 0.0;
    std::vector<double> adaptive_atomic_flow_fraction;
    bool adaptive_recycling_LM_found = false;
    int adaptive_recycling_LM_index = -1;
    double adaptive_recycling_LM_cm = 0.0;
    std::vector<double> adaptive_molecular_flow_fraction;
    bool adaptive_transport_profile_available = false;
    std::vector<double> adaptive_ion_divergence_nuclei_cm3_s;
    std::vector<double> adaptive_pump_exhaust_nuclei_cm3_s;
    std::vector<double> adaptive_neutral_remainder_nuclei_cm3_s;
    std::vector<double> adaptive_atomic_comp_nuclei_cm3_s;
    std::vector<double> adaptive_molecular_comp_nuclei_cm3_s;
    bool adaptive_outer_loop_enabled = false;
    bool adaptive_outer_loop_converged = false;
    int adaptive_outer_iterations = 0;
    std::vector<double> adaptive_outer_L1_history_cm;
    std::vector<double> adaptive_outer_LM_history_cm;
    std::vector<double> adaptive_outer_F_A_end_history;
    std::vector<double> adaptive_outer_F_M_end_history;
    // Per-node aggregate transport/exhaust diagnostics.
    // Entry 0 is NaN because the backward/upwind balance applies to cells k>0.
    bool variable_nuclei_balance_enabled = false;
    bool individual_ion_flux_divergence_enabled = false;
    std::vector<double> variable_ion_divergence_nuclei_cm3_s;
    std::vector<double> variable_flowA_divergence_nuclei_cm3_s;
    std::vector<double> variable_flowM_divergence_nuclei_cm3_s;
    std::vector<double> variable_neutral_exhaust_nuclei_cm3_s;
    std::vector<double> variable_balance_rhs_cm3_s;
    std::vector<double> variable_balance_residual_cm3_s;
    std::vector<double> individual_hminus_omitted_residual_cm3_s;
    std::vector<double> individual_nuclei_weighted_species_residual_cm3_s;
    std::vector<double> individual_nuclei_identity_relative_error;
    std::vector<int> individual_nuclei_identity_consistent;
    // Local prescribed-nuclei closure diagnostics. Entry 0 is NaN because L_I
    // is derived independently in each marched cell.
    std::vector<double> prescribed_nuclei_density_cm3;
    std::vector<double> recycling_source_nuclei_cm3_s;
    std::vector<double> local_atom_exhaust_nuclei_cm3_s;
    std::vector<double> local_molecule_exhaust_nuclei_cm3_s;
    std::vector<double> ion_divergence_closure_nuclei_cm3_s;
    std::vector<double> ion_balance_coefficient_s;
};

// Full spatial integration from boundary to domain end.
// Each cell performs a coupled implicit solve at x+dx.
MarchingHistory run_full_marching(
    const dcr::io::Config& config,
    const dcr::atomic::AtomicData& atomic_data,
    const dcr::state::PlasmaState& plasma,
    const EEDFGridView& grid,
    const BoundaryPhaseResult& boundary,
    const AdaptiveTransportProfile* adaptive_profile = nullptr);

// Outer iteration for variable nuclei balance. Each pass updates the atomic and
// molecular ion velocity lengths from the preceding march.
MarchingHistory run_adaptive_variable_marching(
    const dcr::io::Config& config,
    const dcr::atomic::AtomicData& atomic_data,
    const dcr::state::PlasmaState& plasma,
    const EEDFGridView& grid,
    const BoundaryPhaseResult& boundary);

} // namespace dcr::solver
