#pragma once

#include "../../boundary/BoundaryPhase.hpp"
#include "../../../atomic/AtomicData.hpp"

#include <vector>

namespace dcr::solver {

struct QSSLayout {
    std::vector<int> retained_p_indices;
    std::vector<int> transient_atomic_indices;
    std::vector<int> retained_positive_ion_indices;
    std::vector<int> retained_negative_ion_indices;
    std::vector<int> retained_molecule_indices;
    std::vector<int> retained_molecular_ion_positions;
    std::vector<int> retained_atomic_ground_indices;
    std::vector<int> full_atomic_donor_indices;
    std::vector<int> transient_atomic_donor_indices;
    int retained_atomic_donor_ground = -1;
    std::vector<int> retained_molecular_donor_indices;
    std::vector<int> full_to_retained_p;
    std::vector<int> full_to_transient_atomic;
};

QSSLayout build_qss_layout(
    const BoundaryPhaseResult& full_boundary,
    const std::vector<dcr::atomic::EnergyLevel>& levels);

} // namespace dcr::solver
