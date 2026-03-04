#pragma once

#include <algorithm>
#include <cmath>

namespace crm_detail {

// Electron speed in cm/s for energy in eV.
inline double electron_speed(double E_eV) {
    return 5.93e7 * std::sqrt(std::max(0.0, E_eV));
}

} // namespace crm_detail
