#include "QSSLayout.hpp"

#include "QSSLayoutSupport.hpp"

namespace dcr::solver {
using namespace qss_model_detail;

QSSLayout build_qss_layout(const BoundaryPhaseResult& full_boundary,
                           const std::vector<dcr::atomic::EnergyLevel>& levels) {
    QSSLayout layout;
    layout.full_to_retained_p.assign(levels.size(), -1);
    layout.full_to_transient_atomic.assign(levels.size(), -1);
    const int atom_ground = find_ground_state(full_boundary.atom_bg_indices, levels);
    layout.retained_atomic_donor_ground = find_ground_state(full_boundary.A_indices, levels);
    for (int gi : full_boundary.A_indices) {
        if (gi != layout.retained_atomic_donor_ground) {
            layout.transient_atomic_donor_indices.push_back(gi);
        }
    }
    for (int gi : full_boundary.P_indices) {
        if (gi < 0 || gi >= static_cast<int>(levels.size())) continue;
        const auto& level = levels[static_cast<size_t>(gi)];
        if (is_atomic_neutral(level) && gi != atom_ground) {
            layout.full_to_transient_atomic[static_cast<size_t>(gi)] = static_cast<int>(layout.transient_atomic_indices.size());
            layout.transient_atomic_indices.push_back(gi);
            continue;
        }
        const int retained_pos = static_cast<int>(layout.retained_p_indices.size());
        layout.full_to_retained_p[static_cast<size_t>(gi)] = retained_pos;
        layout.retained_p_indices.push_back(gi);
        if (is_positive_ion(level)) {
            layout.retained_positive_ion_indices.push_back(gi);
            if (level.atomicity > 1) layout.retained_molecular_ion_positions.push_back(retained_pos);
        }
        else if (is_negative_ion(level)) layout.retained_negative_ion_indices.push_back(gi);
        else if (is_molecule_neutral(level)) layout.retained_molecule_indices.push_back(gi);
        else if (is_atomic_neutral(level)) layout.retained_atomic_ground_indices.push_back(gi);
    }
    layout.full_atomic_donor_indices = full_boundary.A_indices;
    layout.retained_molecular_donor_indices = full_boundary.M_indices;
    return layout;
}

} // namespace dcr::solver
