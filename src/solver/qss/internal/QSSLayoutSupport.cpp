#include "QSSLayoutSupport.hpp"

#include <limits>

namespace dcr::solver::qss_model_detail {

bool is_atomic_neutral(const dcr::atomic::EnergyLevel& level) {
    return level.type == dcr::atomic::SpeciesType::Atom && level.charge == 0 && level.atomicity == 1;
}

bool is_positive_ion(const dcr::atomic::EnergyLevel& level) {
    return level.type == dcr::atomic::SpeciesType::Ion && level.charge > 0;
}

bool is_negative_ion(const dcr::atomic::EnergyLevel& level) {
    return level.type == dcr::atomic::SpeciesType::Ion && level.charge < 0;
}

bool is_molecule_neutral(const dcr::atomic::EnergyLevel& level) {
    return level.type == dcr::atomic::SpeciesType::Molecule && level.charge == 0;
}

int find_ground_state(const std::vector<int>& indices,
                      const std::vector<dcr::atomic::EnergyLevel>& levels) {
    int best = -1;
    double best_energy = std::numeric_limits<double>::infinity();
    for (int gi : indices) {
        if (gi >= 0 && gi < static_cast<int>(levels.size()) &&
            levels[static_cast<size_t>(gi)].energy_eV < best_energy) {
            best_energy = levels[static_cast<size_t>(gi)].energy_eV;
            best = gi;
        }
    }
    return best;
}

} // namespace dcr::solver::qss_model_detail
