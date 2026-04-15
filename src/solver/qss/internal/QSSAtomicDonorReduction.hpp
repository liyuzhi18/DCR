#pragma once

#include "QSSLocalBlocks.hpp"

namespace dcr::solver {

void populate_atomic_donor_reduction(QSSLocalBlocks& out,
                                     const dcr::base::Matrix& R_full,
                                     const QSSLayout& layout,
                                     const std::vector<dcr::atomic::EnergyLevel>& levels);

} // namespace dcr::solver
