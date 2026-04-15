#pragma once

#include "../../core/RateAnalysis.hpp"

namespace dcr::solver {

void log_rate_snapshot(double x_cm,
                       const RateDiagnosticSnapshot& snapshot);

} // namespace dcr::solver
