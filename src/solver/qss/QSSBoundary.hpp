#pragma once

#include "QSSModel.hpp"

#include "../../atomic/AtomicData.hpp"
#include "../../io/ConfigStructs.hpp"
#include "../../physics/EEDF.hpp"
#include "../../physics/WallBoundary.hpp"
#include "../../state/PlasmaState.hpp"

namespace dcr::solver {

struct QSSBoundaryResult {
    BoundaryPhaseResult boundary;
    QSSLayout layout;
    dcr::base::Vector retained_background;
    dcr::base::Vector transient_atomic;
};

QSSBoundaryResult solve_qss_boundary(
    const dcr::io::Config& config,
    const dcr::atomic::AtomicData& atomic_data,
    dcr::state::PlasmaState& plasma,
    const EEDFGridView& grid,
    double ion_mass_amu,
    const dcr::physics::WallRecycling& wall);

} // namespace dcr::solver
