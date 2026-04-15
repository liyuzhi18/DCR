#pragma once

#include "../../atomic/AtomicData.hpp"
#include "../../io/ConfigStructs.hpp"
#include "../../physics/EEDF.hpp"
#include "../../physics/WallBoundary.hpp"
#include "../../state/PlasmaState.hpp"
#include <limits>
#include <vector>

namespace dcr::solver {

struct BoundaryPhaseResult {
    dcr::base::Vector population;

    std::vector<char> is_recycling;
    std::vector<int> P_indices;
    std::vector<int> R_indices;
    std::vector<int> A_indices;
    std::vector<int> M_indices;
    std::vector<int> ion_indices;
    std::vector<int> atom_bg_indices;
    std::vector<int> mol_bg_indices;

    dcr::base::Vector flowA_last;
    dcr::base::Vector flowM_last;

    bool explicit_recycling = false;
    bool have_flow_last = false;

    int atom_ground = -1;
    int molecule_ground = -1;

    double u_A = 0.0;
    double u_M = 0.0;

    double atom_mass_amu = 1.0;
    double molecule_mass_amu = 2.0;

    bool converged = false;
    int iterations = 0;
    double final_rel_change = std::numeric_limits<double>::infinity();
    double final_residual_rel = std::numeric_limits<double>::infinity();
    double elapsed_seconds = 0.0;
};

BoundaryPhaseResult run_boundary_phase(
    const dcr::io::Config& config,
    const dcr::atomic::AtomicData& atomic_data,
    dcr::state::PlasmaState& plasma,
    const EEDFGridView& grid,
    double ion_mass_amu,
    const dcr::physics::WallRecycling& wall);

} // namespace dcr::solver
