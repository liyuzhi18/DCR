#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>

namespace crm_detail {

constexpr std::size_t OMP_PARALLEL_GRID_SIZE_THRESHOLD = 256;

// Electron speed in cm/s for energy in eV.
inline double electron_speed(double E_eV) {
    return 5.93e7 * std::sqrt(std::max(0.0, E_eV));
}

inline int openmp_max_threads() {
    return 1;
}

} // namespace crm_detail

#define CRM_DETAIL_OMP_PARALLEL_FOR_ACC_GRID
#define CRM_DETAIL_OMP_SIMD_FOR_ACC
#define CRM_DETAIL_OMP_SIMD_FOR_SUM
