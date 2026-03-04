#pragma once

#include "../../atomic/AtomicData.hpp"
#include "../../base/Types.hpp"
#include "../../io/ConfigStructs.hpp"
#include "../boundary/BoundaryPhase.hpp"
#include "CellAdvance.hpp"
#include "LocalSystemAssembler.hpp"

namespace dcr::solver {

void log_single_step_marching(
    const dcr::io::Config& config,
    const dcr::atomic::AtomicData& atomic_data,
    const BoundaryPhaseResult& boundary,
    const dcr::base::Vector& background_population,
    const dcr::base::Vector& flowA_before,
    const dcr::base::Vector& flowM_before,
    const LocalSystem& local_system,
    const FlowAdvanceResult& advanced,
    double x0_cm,
    double x1_cm);

} // namespace dcr::solver

