#pragma once

#include "QSSLayout.hpp"

namespace dcr::solver {

BoundaryPhaseResult make_qss_boundary_template(
    const BoundaryPhaseResult& full_boundary,
    const QSSLayout& layout);

} // namespace dcr::solver
