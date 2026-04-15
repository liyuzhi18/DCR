#pragma once

#include "../../../atomic/AtomicData.hpp"
#include "../../../base/Types.hpp"

#include <vector>

namespace dcr::solver::qss_model_detail {

dcr::base::Matrix extract_block(const dcr::base::Matrix& full,
                                const std::vector<int>& rows,
                                const std::vector<int>& cols);

dcr::base::Matrix extract_source_block(const dcr::base::Matrix& full,
                                       const std::vector<int>& retained_rows,
                                       const std::vector<int>& donor_cols,
                                       const std::vector<dcr::atomic::EnergyLevel>& levels,
                                       bool allow_atomic_ion_rows,
                                       bool allow_atomic_ground_rows,
                                       bool allow_molecular_ion_rows);

} // namespace dcr::solver::qss_model_detail
