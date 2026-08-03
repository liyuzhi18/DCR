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

struct LocalRateCache {
    dcr::base::Matrix constant;
    dcr::base::Matrix linear_ne;
    dcr::base::Matrix quadratic_ne;
};

struct LocalSystem {
    // Population vector used for process-rate evaluation at this spatial location.
    dcr::base::Vector population_for_rates;
    // Population vector used to evaluate recycling-flow source terms.
    dcr::base::Vector population_for_source;
    // Full collisional-radiative matrix assembled from all processes.
    dcr::base::Matrix R_full;
    // CR matrix evaluated from the combined background and recycling state.
    dcr::base::Matrix R_source_full;
    dcr::base::Matrix dR_source_dne;
    std::vector<int> rate_population_derivative_indices;
    std::vector<dcr::base::Matrix> dR_source_dpopulation;
    // Recycling source projected on background unknowns (P block rows).
    dcr::base::Vector S_background;
};

struct LocalChemistrySources {
    dcr::base::Vector background;
    dcr::base::Vector flowA;
    dcr::base::Vector flowM;
};

LocalChemistrySources assemble_local_chemistry_sources(
    const LocalSystem& local,
    const BoundaryPhaseResult& boundary,
    const dcr::atomic::AtomicData& atomic_data,
    const dcr::base::Vector& background,
    const dcr::base::Vector& flowA,
    const dcr::base::Vector& flowM);

LocalChemistrySources assemble_local_chemistry_sources_from_rate_matrix(
    const dcr::base::Matrix& rates,
    const BoundaryPhaseResult& boundary,
    const dcr::atomic::AtomicData& atomic_data,
    const dcr::base::Vector& background,
    const dcr::base::Vector& flowA,
    const dcr::base::Vector& flowM);

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

LocalRateCache build_local_rate_cache(
    const dcr::io::Config& config,
    const dcr::atomic::AtomicData& atomic_data,
    const dcr::state::PlasmaState& plasma,
    const EEDFGridView& grid,
    double x_cm,
    double density_reference_cm3);

LocalSystem assemble_local_system_from_background_rate_matrix(
    const dcr::io::Config& config,
    const dcr::atomic::AtomicData& atomic_data,
    const dcr::state::PlasmaState& plasma,
    const EEDFGridView& grid,
    const BoundaryPhaseResult& boundary,
    const BackgroundRateAssembly& background_rates,
    const dcr::base::Vector& flowA,
    const dcr::base::Vector& flowM,
    double x_cm,
    bool assemble_rate_derivatives = false,
    const LocalRateCache* rate_cache = nullptr);

LocalSystem assemble_local_system(
    const dcr::io::Config& config,
    const dcr::atomic::AtomicData& atomic_data,
    const dcr::state::PlasmaState& plasma,
    const EEDFGridView& grid,
    const BoundaryPhaseResult& boundary,
    const dcr::base::Vector& background_population,
    const dcr::base::Vector& flowA,
    const dcr::base::Vector& flowM,
    double x_cm,
    bool assemble_rate_derivatives = false,
    const LocalRateCache* rate_cache = nullptr);

} // namespace dcr::solver
