#pragma once

#include "../../boundary/BoundaryPhase.hpp"
#include "../../../atomic/AtomicData.hpp"

namespace dcr::solver {

struct RecyclingModel {
    std::vector<int> ion_p_cols;
    std::vector<double> recA_coeff;
    std::vector<double> recM_coeff;
};

RecyclingModel build_recycling_model(const BoundaryPhaseResult& boundary,
                                     const std::vector<dcr::atomic::EnergyLevel>& levels,
                                     const std::vector<int>& p_pos,
                                     double c_A,
                                     double c_A_base,
                                     double c_M_atom);

} // namespace dcr::solver
