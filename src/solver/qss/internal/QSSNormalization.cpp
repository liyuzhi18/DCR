#include "QSSNormalization.hpp"

#include <limits>

namespace dcr::solver {

int find_ground_state(const std::vector<int>& indices,
                      const std::vector<dcr::atomic::EnergyLevel>& levels) {
    int best = -1;
    double best_energy = std::numeric_limits<double>::infinity();
    for (int gi : indices) if (gi >= 0 && gi < static_cast<int>(levels.size()) &&
        levels[static_cast<size_t>(gi)].energy_eV < best_energy) {
        best_energy = levels[static_cast<size_t>(gi)].energy_eV;
        best = gi;
    }
    return best;
}

void normalize_background_to_target(const BoundaryPhaseResult& boundary,
                                    const std::vector<dcr::atomic::EnergyLevel>& levels,
                                    double target_bg_nuclei,
                                    dcr::base::Vector& retained_background) {
    dcr::base::Vector stoich = dcr::base::Vector::Zero(retained_background.size());
    for (int i = 0; i < retained_background.size(); ++i) {
        const int gi = boundary.P_indices[static_cast<size_t>(i)];
        stoich(i) = (gi >= 0 && gi < static_cast<int>(levels.size())) ? static_cast<double>(std::max(1, levels[static_cast<size_t>(gi)].atomicity)) : 1.0;
    }
    const double current = (stoich.array() * retained_background.array()).sum();
    if (current > 0.0) retained_background *= (target_bg_nuclei / current);
}

double nuclei_from_source(const BoundaryPhaseResult& boundary,
                          const std::vector<dcr::atomic::EnergyLevel>& levels,
                          const dcr::base::Vector& source_retained) {
    double total = 0.0;
    for (int i = 0; i < source_retained.size(); ++i) {
        const int gi = boundary.P_indices[static_cast<size_t>(i)];
        if (gi >= 0 && gi < static_cast<int>(levels.size()))
            total += static_cast<double>(std::max(1, levels[static_cast<size_t>(gi)].atomicity)) * source_retained(i);
    }
    return total;
}

} // namespace dcr::solver
