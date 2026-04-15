#pragma once

#include "../../../base/Types.hpp"

namespace dcr::solver::qss_model_detail {

dcr::base::Vector solve_linear(const dcr::base::Matrix& A,
                               const dcr::base::Vector& b);

dcr::base::Matrix solve_linear_matrix(const dcr::base::Matrix& A,
                                      const dcr::base::Matrix& B);

} // namespace dcr::solver::qss_model_detail
