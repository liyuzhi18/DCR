#pragma once

#include "../../../io/ConfigStructs.hpp"

#include <vector>

namespace dcr::solver {

std::vector<double> build_step_sizes_cm(const dcr::io::Config& config);

} // namespace dcr::solver
