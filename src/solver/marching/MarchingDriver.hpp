#pragma once

#include "../../atomic/AtomicData.hpp"
#include "../../io/ConfigStructs.hpp"
#include "../../physics/EEDF.hpp"
#include "../../state/PlasmaState.hpp"
#include "../boundary/BoundaryPhase.hpp"
#include "../core/RateAnalysis.hpp"
#include <vector>

namespace dcr::solver {

struct MarchingHistory {
    // Node locations (cm), size = num_cells.
    std::vector<double> x_cm;
    // Boundary and per-cell marching timing/iteration diagnostics.
    double boundary_elapsed_seconds = 0.0;
    int boundary_iterations = 0;
    std::vector<double> cell_elapsed_seconds;
    std::vector<int> cell_iterations;
    std::vector<int> cell_converged;
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
};

// Full spatial integration from boundary to domain end.
// Each cell performs a coupled implicit solve at x+dx.
MarchingHistory run_full_marching(
    const dcr::io::Config& config,
    const dcr::atomic::AtomicData& atomic_data,
    const dcr::state::PlasmaState& plasma,
    const EEDFGridView& grid,
    const BoundaryPhaseResult& boundary);

} // namespace dcr::solver
