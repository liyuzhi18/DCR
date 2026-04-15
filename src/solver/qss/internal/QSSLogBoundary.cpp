#include "QSSLogBoundary.hpp"

#include "QSSPopulationOps.hpp"

#include <iostream>

namespace dcr::solver {

void log_qss_boundary_iter(int iter, double rel, const dcr::base::Vector& flowA,
                           const dcr::base::Vector& flowM, const dcr::base::Vector& bg_full,
                           const BoundaryPhaseResult& boundary,
                           const std::vector<dcr::atomic::EnergyLevel>& levels) {
    const auto bg = summarize_background(boundary, bg_full, levels);
    std::cout << "[DCR_Solver][QSS] Boundary iter " << iter << " rel=" << rel
              << " bg{H+=" << bg.H_plus << ", H=" << bg.H << ", H2=" << bg.H2
              << ", H2+=" << bg.H2_plus << ", H-=" << bg.H_minus
              << "} flow{A=" << positive_sum(flowA) << ", M=" << positive_sum(flowM) << "}\n";
}

} // namespace dcr::solver
