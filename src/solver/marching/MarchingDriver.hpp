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
    // Background populations on full state indexing at each node.
    // Shape conceptually: [num_cells, total_states].
    std::vector<dcr::base::Vector> background_full;
    // Recycling-flow block populations at each node.
    // Shapes conceptually:
    // - flowA: [num_cells, |A_indices|]
    // - flowM: [num_cells, |M_indices|]
    std::vector<dcr::base::Vector> flowA;
    std::vector<dcr::base::Vector> flowM;
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
