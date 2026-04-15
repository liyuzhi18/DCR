#pragma once

#include "../boundary/BoundaryPhase.hpp"
#include "../marching/MarchingDriver.hpp"

#include "../../atomic/AtomicData.hpp"
#include "../../io/ConfigStructs.hpp"
#include "../../physics/EEDF.hpp"
#include "../../state/PlasmaState.hpp"
#include "../../physics/WallBoundary.hpp"

namespace dcr::solver {

// Run the reduced QSS-DCR path and return the boundary + marching results in the
// same shapes used by the existing HDF5/output pipeline.
struct QSSSolveResult {
    BoundaryPhaseResult boundary;
    MarchingHistory history;
};

QSSSolveResult run_qss_dcr(
    const dcr::io::Config& config,
    const dcr::atomic::AtomicData& atomic_data,
    dcr::state::PlasmaState& plasma,
    const EEDFGridView& grid,
    double ion_mass_amu,
    const dcr::physics::WallRecycling& wall);

} // namespace dcr::solver
