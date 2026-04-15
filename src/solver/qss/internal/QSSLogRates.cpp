#include "QSSLogRates.hpp"

#include <iostream>

namespace dcr::solver {

void log_rate_snapshot(double x_cm, const RateDiagnosticSnapshot& snapshot) {
    if (!snapshot.atomic_effective.valid) return;
    std::cout << "[DCR_Solver][Rates][atomic] x=" << x_cm
              << " cm SCD=" << snapshot.atomic_effective.scd_cm3_s
              << " ACD=" << snapshot.atomic_effective.acd_cm3_s << "\n";
}

} // namespace dcr::solver
