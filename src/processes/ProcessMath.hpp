#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>

#ifdef DCR_USE_OPENMP
#include <omp.h>
#endif

namespace crm_detail {

constexpr std::size_t OMP_PARALLEL_GRID_SIZE_THRESHOLD = 256;

// Electron speed in cm/s for energy in eV.
inline double electron_speed(double E_eV) {
    return 5.93e7 * std::sqrt(std::max(0.0, E_eV));
}

inline int openmp_max_threads() {
#ifdef DCR_USE_OPENMP
    return omp_get_max_threads();
#else
    return 1;
#endif
}

} // namespace crm_detail

#ifdef DCR_USE_OPENMP
#define CRM_DETAIL_OMP_PARALLEL_FOR_ACC_GRID \
    _Pragma("omp parallel for reduction(+:acc) schedule(static) if(grid.size() >= 256)")
#define CRM_DETAIL_OMP_SIMD_FOR_ACC \
    _Pragma("omp simd reduction(+:acc)")
#define CRM_DETAIL_OMP_SIMD_FOR_SUM \
    _Pragma("omp simd reduction(+:sum)")
#else
#define CRM_DETAIL_OMP_PARALLEL_FOR_ACC_GRID
#define CRM_DETAIL_OMP_SIMD_FOR_ACC
#define CRM_DETAIL_OMP_SIMD_FOR_SUM
#endif
