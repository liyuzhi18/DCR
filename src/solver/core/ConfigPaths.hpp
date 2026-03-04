#pragma once

#include "../../io/ConfigStructs.hpp"
#include <string>

namespace dcr::solver {

// Resolve config path from environment or default locations.
std::string resolve_config_path();

// Normalize configured input roots (atomic data and data tables) to absolute paths.
void normalize_input_roots(dcr::io::Config& config, const std::string& config_path);

} // namespace dcr::solver
