#include "QSSLogLayout.hpp"

#include <iostream>
#include <sstream>

namespace dcr::solver {

namespace {
std::string join_level_labels(const std::vector<int>& indices,
                              const std::vector<dcr::atomic::EnergyLevel>& levels) {
    if (indices.empty()) return "(none)";
    std::ostringstream out;
    for (size_t i = 0; i < indices.size(); ++i) {
        if (i > 0) out << ", ";
        out << ((indices[i] >= 0 && indices[i] < static_cast<int>(levels.size())) ? levels[indices[i]].label : "idx=" + std::to_string(indices[i]));
    }
    return out.str();
}
}

void log_qss_layout(const QSSLayout& layout, const std::vector<dcr::atomic::EnergyLevel>& levels) {
    std::cout << "[DCR_Solver][QSS] Layout counts local{retained=" << layout.retained_p_indices.size()
              << ", transient_atomic=" << layout.transient_atomic_indices.size() << "} flow{A="
              << (layout.retained_atomic_donor_ground >= 0 ? 1 : 0) << ", M="
              << layout.retained_molecular_donor_indices.size() << "}\n";
    std::cout << "[DCR_Solver][QSS] Retained positive ions (" << layout.retained_positive_ion_indices.size() << "): " << join_level_labels(layout.retained_positive_ion_indices, levels) << "\n";
    std::cout << "[DCR_Solver][QSS] Retained atomic ground (" << layout.retained_atomic_ground_indices.size() << "): " << join_level_labels(layout.retained_atomic_ground_indices, levels) << "\n";
    std::cout << "[DCR_Solver][QSS] Retained molecular neutrals (" << layout.retained_molecule_indices.size() << "): " << join_level_labels(layout.retained_molecule_indices, levels) << "\n";
    std::cout << "[DCR_Solver][QSS] Retained negative ions (" << layout.retained_negative_ion_indices.size() << "): " << join_level_labels(layout.retained_negative_ion_indices, levels) << "\n";
    std::cout << "[DCR_Solver][QSS] Eliminated transient atomic (" << layout.transient_atomic_indices.size() << "): " << join_level_labels(layout.transient_atomic_indices, levels) << "\n";
    const std::vector<int> atomic_flow = (layout.retained_atomic_donor_ground >= 0) ? std::vector<int>{layout.retained_atomic_donor_ground} : std::vector<int>{};
    std::cout << "[DCR_Solver][QSS] Atomic flow donor (" << atomic_flow.size() << "): " << join_level_labels(atomic_flow, levels) << "\n";
    std::cout << "[DCR_Solver][QSS] Molecular flow donor (" << layout.retained_molecular_donor_indices.size() << "): " << join_level_labels(layout.retained_molecular_donor_indices, levels) << "\n";
}

} // namespace dcr::solver
