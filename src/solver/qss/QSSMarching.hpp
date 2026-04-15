#pragma once

#include "QSSBoundary.hpp"

#include "../marching/MarchingDriver.hpp"

#include "../../atomic/AtomicData.hpp"
#include "../../io/ConfigStructs.hpp"
#include "../../physics/EEDF.hpp"
#include "../../state/PlasmaState.hpp"

namespace dcr::solver {

struct QSSMarchingResult {
    MarchingHistory history;
    dcr::base::Vector retained_background;
    dcr::base::Vector transient_atomic;
    dcr::base::Vector flowA;
    dcr::base::Vector flowM;
};

QSSMarchingResult run_qss_marching(
    const dcr::io::Config& config,
    const dcr::atomic::AtomicData& atomic_data,
    dcr::state::PlasmaState& plasma,
    const EEDFGridView& grid,
    const QSSBoundaryResult& qss_boundary);

} // namespace dcr::solver
