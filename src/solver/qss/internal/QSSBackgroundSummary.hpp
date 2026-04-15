#pragma once

#include "../../boundary/BoundaryPhase.hpp"
#include "../../../atomic/AtomicData.hpp"
#include "../../../base/Types.hpp"

namespace dcr::solver {

struct BgSubgroupSums {
    double H_plus = 0.0;
    double H = 0.0;
    double H2 = 0.0;
    double H2_plus = 0.0;
    double H_minus = 0.0;
};

BgSubgroupSums summarize_background(const BoundaryPhaseResult& boundary,
                                    const dcr::base::Vector& population,
                                    const std::vector<dcr::atomic::EnergyLevel>& levels);

} // namespace dcr::solver
