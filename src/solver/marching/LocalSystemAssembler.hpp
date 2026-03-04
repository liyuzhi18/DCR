#pragma once

#include "../../atomic/AtomicData.hpp"
#include "../../base/Types.hpp"
#include "../../physics/EEDF.hpp"
#include "../../state/PlasmaState.hpp"
#include "../boundary/BoundaryPhase.hpp"

namespace dcr::solver {

struct LocalSystem {
    // Population vector used for process-rate evaluation at this spatial location.
    dcr::base::Vector population_for_rates;
    // Full collisional-radiative matrix assembled from all processes.
    dcr::base::Matrix R_full;
    // Recycling source projected on background unknowns (P block rows).
    dcr::base::Vector S_background;
};

// Assemble local-rate objects at the current nonlinear iterate:
// - total population used by process evaluation,
// - full CR matrix R,
// - recycling source S projected on background equations.
LocalSystem assemble_local_system(
    const dcr::atomic::AtomicData& atomic_data,
    const dcr::state::PlasmaState& plasma,
    const EEDFGridView& grid,
    const BoundaryPhaseResult& boundary,
    const dcr::base::Vector& background_population,
    const dcr::base::Vector& flowA,
    const dcr::base::Vector& flowM);

} // namespace dcr::solver
