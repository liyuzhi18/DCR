#pragma once

#include "../../../atomic/AtomicData.hpp"

#include <vector>

namespace dcr::solver::qss_model_detail {

bool is_atomic_neutral(const dcr::atomic::EnergyLevel& level);
bool is_positive_ion(const dcr::atomic::EnergyLevel& level);
bool is_negative_ion(const dcr::atomic::EnergyLevel& level);
bool is_molecule_neutral(const dcr::atomic::EnergyLevel& level);
int find_ground_state(const std::vector<int>& indices,
                      const std::vector<dcr::atomic::EnergyLevel>& levels);

} // namespace dcr::solver::qss_model_detail
