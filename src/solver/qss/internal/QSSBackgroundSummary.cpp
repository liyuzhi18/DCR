#include "QSSBackgroundSummary.hpp"

namespace dcr::solver {

BgSubgroupSums summarize_background(const BoundaryPhaseResult& boundary,
                                    const dcr::base::Vector& population,
                                    const std::vector<dcr::atomic::EnergyLevel>& levels) {
    BgSubgroupSums s;
    for (int gi : boundary.P_indices) {
        if (gi < 0 || gi >= static_cast<int>(levels.size()) || gi >= population.size()) continue;
        const auto& lvl = levels[static_cast<size_t>(gi)];
        const double n = std::max(population(gi), 0.0);
        if (lvl.type == dcr::atomic::SpeciesType::Ion && lvl.charge > 0) (lvl.atomicity >= 2 ? s.H2_plus : s.H_plus) += n;
        else if (lvl.type == dcr::atomic::SpeciesType::Ion && lvl.charge < 0) s.H_minus += n;
        else if (lvl.type == dcr::atomic::SpeciesType::Atom && lvl.charge == 0) s.H += n;
        else if (lvl.type == dcr::atomic::SpeciesType::Molecule && lvl.charge == 0) s.H2 += n;
    }
    return s;
}

} // namespace dcr::solver
