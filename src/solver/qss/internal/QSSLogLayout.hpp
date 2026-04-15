#pragma once

#include "QSSLayout.hpp"

namespace dcr::solver {

void log_qss_layout(const QSSLayout& layout,
                    const std::vector<dcr::atomic::EnergyLevel>& levels);

} // namespace dcr::solver
