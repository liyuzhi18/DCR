#pragma once

#include "../../atomic/AtomicData.hpp"
#include "../../base/Types.hpp"
#include "../../physics/EEDF.hpp"
#include "../../state/PlasmaState.hpp"
#include "../boundary/BoundaryPhase.hpp"

namespace dcr::solver {

struct BackgroundRateAssembly {
    // Population vector used for background-only process-rate evaluation.
    dcr::base::Vector population_for_rates;
    // Full collisional-radiative matrix assembled from background-only rates.
    dcr::base::Matrix R_full;
};

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
BackgroundRateAssembly assemble_background_rate_matrix(
    const dcr::io::Config& config,
    const dcr::atomic::AtomicData& atomic_data,
    const dcr::state::PlasmaState& plasma,
    const EEDFGridView& grid,
    const dcr::base::Vector& background_population,
    double x_cm);

LocalSystem assemble_local_system_from_background_rate_matrix(
    const dcr::io::Config& config,
    const dcr::atomic::AtomicData& atomic_data,
    const dcr::state::PlasmaState& plasma,
    const EEDFGridView& grid,
    const BoundaryPhaseResult& boundary,
    const BackgroundRateAssembly& background_rates,
    const dcr::base::Vector& flowA,
    const dcr::base::Vector& flowM,
    double x_cm);

LocalSystem assemble_local_system(
    const dcr::io::Config& config,
    const dcr::atomic::AtomicData& atomic_data,
    const dcr::state::PlasmaState& plasma,
    const EEDFGridView& grid,
    const BoundaryPhaseResult& boundary,
    const dcr::base::Vector& background_population,
    const dcr::base::Vector& flowA,
    const dcr::base::Vector& flowM,
    double x_cm);

} // namespace dcr::solver
