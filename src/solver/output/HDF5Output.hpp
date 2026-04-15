#pragma once

#include "../../atomic/AtomicData.hpp"
#include "../../io/ConfigStructs.hpp"
#include "../boundary/BoundaryPhase.hpp"
#include "../marching/MarchingDriver.hpp"

namespace dcr::solver {

// Write boundary + marching populations to an HDF5 file under config.io.output_dir.
// Layout:
// - /grid/{x_cm,n_nuclei_cm3}
// - /states/{labels,type_id,charge,atomicity,internal_id,P_indices,A_indices,M_indices,...}
// - /population/{background_full,flowA,flowM,total_full,...}
// - /rates/{Te_eV,Ti_eV,ne_cm3,atomic_scd_cm3_s,atomic_acd_cm3_s,...}
// - /timing/{boundary_elapsed_seconds,boundary_iterations,marching_cell_elapsed_seconds,...}
// All population datasets are stored in cm^-3.
void write_hdf5_output(
    const dcr::io::Config& config,
    const dcr::atomic::AtomicData& atomic_data,
    const BoundaryPhaseResult& boundary,
    const MarchingHistory& history);

} // namespace dcr::solver
