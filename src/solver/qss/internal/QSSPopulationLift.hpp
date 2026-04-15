#pragma once

#include "QSSLayout.hpp"

#include "../../../base/Types.hpp"

namespace dcr::solver {

dcr::base::Vector make_qss_retained_background(
    const BoundaryPhaseResult& qss_boundary,
    const dcr::base::Vector& full_population);

dcr::base::Vector lift_qss_background_to_full(
    int total_states,
    const BoundaryPhaseResult& qss_boundary,
    const QSSLayout& layout,
    const dcr::base::Vector& retained_background,
    const dcr::base::Vector& transient_atomic);

} // namespace dcr::solver
