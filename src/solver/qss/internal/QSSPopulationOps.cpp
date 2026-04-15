#include "QSSPopulationOps.hpp"

#include <algorithm>
#include <cmath>

namespace dcr::solver {

double positive_sum(const dcr::base::Vector& v) {
    double s = 0.0;
    for (int i = 0; i < v.size(); ++i) s += std::max(v(i), 0.0);
    return s;
}

double nuclei_sum(const dcr::base::Vector& vec, const std::vector<int>& global_indices,
                  const std::vector<dcr::atomic::EnergyLevel>& levels) {
    double s = 0.0;
    for (size_t i = 0; i < global_indices.size() && static_cast<int>(i) < vec.size(); ++i)
        if (global_indices[i] >= 0 && global_indices[i] < static_cast<int>(levels.size()))
            s += levels[global_indices[i]].atomicity * std::max(vec(static_cast<int>(i)), 0.0);
    return s;
}

double group_sum(const dcr::base::Vector& full, const std::vector<int>& indices) {
    double s = 0.0;
    for (int gi : indices) if (gi >= 0 && gi < full.size()) s += std::max(full(gi), 0.0);
    return s;
}

double relative_change(const dcr::base::Vector& now, const dcr::base::Vector& old) {
    if (now.size() == 0 || old.size() == 0) return 0.0;
    return (now - old).cwiseAbs().maxCoeff() / std::max(1.0, old.cwiseAbs().maxCoeff());
}

void apply_non_negative(dcr::base::Vector& values) {
    for (int i = 0; i < values.size(); ++i) if (!std::isfinite(values(i)) || values(i) < 0.0) values(i) = 0.0;
}

} // namespace dcr::solver
