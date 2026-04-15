#pragma once

#include "QSSLayout.hpp"

namespace dcr::solver {

int find_ground_state(const std::vector<int>& indices,
                      const std::vector<dcr::atomic::EnergyLevel>& levels);
void normalize_background_to_target(const BoundaryPhaseResult& boundary,
                                    const std::vector<dcr::atomic::EnergyLevel>& levels,
                                    double target_bg_nuclei,
                                    dcr::base::Vector& retained_background);
double nuclei_from_source(const BoundaryPhaseResult& boundary,
                          const std::vector<dcr::atomic::EnergyLevel>& levels,
                          const dcr::base::Vector& source_retained);

} // namespace dcr::solver
