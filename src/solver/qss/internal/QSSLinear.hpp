#pragma once

#include "../../../base/Types.hpp"

namespace dcr::solver {

dcr::base::Vector solve_linear(const dcr::base::Matrix& A,
                               const dcr::base::Vector& b);

} // namespace dcr::solver
