#pragma once

#include "../../../atomic/AtomicData.hpp"
#include "../../../base/Types.hpp"

#include <vector>

namespace dcr::solver {

double positive_sum(const dcr::base::Vector& v);
double nuclei_sum(const dcr::base::Vector& vec,
                  const std::vector<int>& global_indices,
                  const std::vector<dcr::atomic::EnergyLevel>& levels);
double group_sum(const dcr::base::Vector& full,
                 const std::vector<int>& indices);
double relative_change(const dcr::base::Vector& now,
                       const dcr::base::Vector& old);
void apply_non_negative(dcr::base::Vector& values);

} // namespace dcr::solver
