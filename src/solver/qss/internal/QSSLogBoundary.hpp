#pragma once

#include "QSSBackgroundSummary.hpp"
#include "../../../base/Types.hpp"

namespace dcr::solver {

void log_qss_boundary_iter(int iter,
                           double rel,
                           const dcr::base::Vector& flowA,
                           const dcr::base::Vector& flowM,
                           const dcr::base::Vector& bg_full,
                           const BoundaryPhaseResult& boundary,
                           const std::vector<dcr::atomic::EnergyLevel>& levels);

} // namespace dcr::solver
