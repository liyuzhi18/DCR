#include "MarchingDiagnostics.hpp"

#include <algorithm>
#include <iostream>

namespace dcr::solver {

namespace {

double nuclei_total(const dcr::base::Vector& n,
                    const std::vector<int>& indices,
                    const std::vector<dcr::atomic::EnergyLevel>& levels) {
    double total = 0.0;
    for (size_t i = 0; i < indices.size(); ++i) {
        if (static_cast<int>(i) >= n.size()) break;
        const int gi = indices[i];
        if (gi < 0 || gi >= static_cast<int>(levels.size())) continue;
        total += levels[gi].atomicity * std::max(n(static_cast<int>(i)), 0.0);
    }
    return total;
}

} // namespace

void log_single_step_marching(
    const dcr::io::Config& config,
    const dcr::atomic::AtomicData& atomic_data,
    const BoundaryPhaseResult& boundary,
    const dcr::base::Vector& background_population,
    const dcr::base::Vector& flowA_before,
    const dcr::base::Vector& flowM_before,
    const LocalSystem& local_system,
    const FlowAdvanceResult& advanced,
    double x0_cm,
    double x1_cm) {

    if (!config.io.verbose_logging) return;

    const auto& levels = atomic_data.get_levels();
    const double bg_total = nuclei_total(background_population, boundary.P_indices, levels);
    const double flowA0 = nuclei_total(flowA_before, boundary.A_indices, levels);
    const double flowM0 = nuclei_total(flowM_before, boundary.M_indices, levels);
    const double flowA1 = nuclei_total(advanced.flowA_next, boundary.A_indices, levels);
    const double flowM1 = nuclei_total(advanced.flowM_next, boundary.M_indices, levels);

    std::cout << "[DCR_Solver] Marching step x=" << x0_cm << " -> " << x1_cm << " cm\n";
    std::cout << "[DCR_Solver] Marching totals (nuclei cm^-3): "
              << "background=" << bg_total
              << " flowA_before=" << flowA0
              << " flowM_before=" << flowM0
              << " flowA_after=" << flowA1
              << " flowM_after=" << flowM1 << "\n";

    const double S_norm = local_system.S_background.norm();
    const double R_norm = local_system.R_full.cwiseAbs().sum();
    std::cout << "[DCR_Solver] Marching diagnostics: |R|_sum=" << R_norm
              << " |S_bg|_2=" << S_norm
              << " c_s_A=" << advanced.c_s_A
              << " c_s_M=" << advanced.c_s_M << "\n";
    std::cout << "[DCR_Solver] Marching note: one-cell implicit solve uses "
                 "x+dx rates for both flow advance and local background solve.\n";

    // Print background populations at the first marched cell (x = x1).
    std::cout << "[DCR_Solver] Background states at first cell (x=" << x1_cm
              << " cm, cm^-3):\n";
    for (size_t pi = 0; pi < boundary.P_indices.size(); ++pi) {
        const int gi = boundary.P_indices[pi];
        if (gi < 0 || gi >= background_population.size()) continue;
        if (gi >= static_cast<int>(levels.size())) continue;
        std::cout << "  [P] i=" << gi
                  << " n=" << background_population(gi)
                  << " label=\"" << levels[gi].label << "\"\n";
    }

    // Print only non-negligible flow states to keep logs readable.
    std::cout << "[DCR_Solver] Flow A states after one step:\n";
    for (size_t i = 0; i < boundary.A_indices.size(); ++i) {
        if (static_cast<int>(i) >= advanced.flowA_next.size()) break;
        const double n0 = (static_cast<int>(i) < flowA_before.size()) ? flowA_before(static_cast<int>(i)) : 0.0;
        const double n1 = advanced.flowA_next(static_cast<int>(i));
        if (std::max(std::abs(n0), std::abs(n1)) < 1e-20) continue;
        const int gi = boundary.A_indices[i];
        if (gi < 0 || gi >= static_cast<int>(levels.size())) continue;
        std::cout << "  [A] i=" << gi
                  << " n0=" << n0
                  << " n1=" << n1
                  << " label=\"" << levels[gi].label << "\"\n";
    }
    std::cout << "[DCR_Solver] Flow M states after one step:\n";
    for (size_t i = 0; i < boundary.M_indices.size(); ++i) {
        if (static_cast<int>(i) >= advanced.flowM_next.size()) break;
        const double n0 = (static_cast<int>(i) < flowM_before.size()) ? flowM_before(static_cast<int>(i)) : 0.0;
        const double n1 = advanced.flowM_next(static_cast<int>(i));
        if (std::max(std::abs(n0), std::abs(n1)) < 1e-20) continue;
        const int gi = boundary.M_indices[i];
        if (gi < 0 || gi >= static_cast<int>(levels.size())) continue;
        std::cout << "  [M] i=" << gi
                  << " n0=" << n0
                  << " n1=" << n1
                  << " label=\"" << levels[gi].label << "\"\n";
    }
}

} // namespace dcr::solver
