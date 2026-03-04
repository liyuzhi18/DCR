#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdlib>
#include <iostream>

#include "../src/atomic/AtomicData.hpp"
#include "../src/physics/WallBoundary.hpp"
#include "../src/solver/boundary/BoundaryPhase.hpp"
#include "../src/solver/marching/MarchingDriver.hpp"
#include "TestDCRSetup.hpp"

namespace {

struct BgGroups {
    double H_plus = 0.0;
    double H_minus = 0.0;
    double H = 0.0;
    double H2 = 0.0;
    double H2_plus = 0.0;
};

BgGroups summarize_bg_groups(const dcr::base::Vector& bg_full,
                             const std::vector<int>& p_indices,
                             const std::vector<dcr::atomic::EnergyLevel>& levels) {
    BgGroups g;
    for (int gi : p_indices) {
        if (gi < 0 || gi >= bg_full.size() || gi >= static_cast<int>(levels.size())) continue;
        const auto& lvl = levels[gi];
        const double n = std::max(bg_full(gi), 0.0);
        if (lvl.type == dcr::atomic::SpeciesType::Ion && lvl.charge > 0) {
            if (lvl.atomicity >= 2) g.H2_plus += n;
            else g.H_plus += n;
        } else if (lvl.type == dcr::atomic::SpeciesType::Ion && lvl.charge < 0) {
            g.H_minus += n;
        } else if (lvl.type == dcr::atomic::SpeciesType::Atom && lvl.charge == 0) {
            g.H += n;
        } else if (lvl.type == dcr::atomic::SpeciesType::Molecule && lvl.charge == 0) {
            g.H2 += n;
        }
    }
    return g;
}

} // namespace

int main() {
    std::cout << "--- Testing Reference Regression Case ---\n";

    auto cfg = test_dcr::load_reference_config(/*num_cells=*/3, /*verbose_logging=*/false);
    cfg.grid.length_cm = 0.5;
    cfg.grid.type = "linear";
    cfg.grid.first_cell_cm = 0.0;
    cfg.numerics.max_iterations = 40;
    cfg.numerics.tolerance = std::max(1e-8, cfg.numerics.tolerance);

    dcr::atomic::AtomicData atomic_data(cfg);
    auto plasma = test_dcr::make_plasma_state(cfg, atomic_data.get_total_states());
    test_dcr::EEDFContext eedf(cfg.plasma.Te_eV);

    const double ion_mass_amu = test_dcr::estimate_ion_mass_amu(cfg);
    const auto wall = dcr::physics::compute_wall_recycling(
        cfg.wall.material,
        cfg.plasma.Te_eV,
        cfg.plasma.Ti_eV,
        ion_mass_amu,
        cfg.wall.sheath_potential_drop
    );

    const auto boundary = dcr::solver::run_boundary_phase(
        cfg, atomic_data, plasma, eedf.grid, ion_mass_amu, wall
    );
    const auto history = dcr::solver::run_full_marching(cfg, atomic_data, plasma, eedf.grid, boundary);
    const auto& levels = atomic_data.get_levels();

    const double bg_boundary = test_dcr::nuclei_total_selected(boundary.population, boundary.P_indices, levels);
    double flow_boundary = 0.0;
    if (boundary.explicit_recycling) {
        flow_boundary = test_dcr::nuclei_total_selected(boundary.population, boundary.R_indices, levels);
    } else {
        flow_boundary = test_dcr::nuclei_total_compact(boundary.flowA_last, boundary.A_indices, levels)
                      + test_dcr::nuclei_total_compact(boundary.flowM_last, boundary.M_indices, levels);
    }
    const double total_boundary = bg_boundary + flow_boundary;
    const double boundary_flow_frac = flow_boundary / std::max(1.0, total_boundary);

    const size_t k1 = std::min<size_t>(1, history.x_cm.size() - 1);
    const size_t kf = history.x_cm.size() - 1;
    const double flowA_k1 = test_dcr::nuclei_total_compact(history.flowA[k1], boundary.A_indices, levels);
    const double flowM_k1 = test_dcr::nuclei_total_compact(history.flowM[k1], boundary.M_indices, levels);
    const double total_k1 = test_dcr::nuclei_total_full(history.background_full[k1], levels) + flowA_k1 + flowM_k1;
    const double flowA_k1_frac = flowA_k1 / std::max(1.0, total_k1);
    const double flowM_k1_frac = flowM_k1 / std::max(1.0, total_k1);

    const double flowA_kf = test_dcr::nuclei_total_compact(history.flowA[kf], boundary.A_indices, levels);
    const double flowM_kf = test_dcr::nuclei_total_compact(history.flowM[kf], boundary.M_indices, levels);
    const double total_kf = test_dcr::nuclei_total_full(history.background_full[kf], levels) + flowA_kf + flowM_kf;
    const auto bg_last = summarize_bg_groups(history.background_full[kf], boundary.P_indices, levels);
    const double hplus_bg_frac = bg_last.H_plus / std::max(1.0, total_kf);
    const double h2_bg_frac = bg_last.H2 / std::max(1.0, total_kf);

    // Coarse regression envelope: catches broken chemistry/mass-balance while
    // tolerating expected physics-model tuning.
    assert(boundary_flow_frac > 0.05 && boundary_flow_frac < 0.95);
    assert(flowA_k1_frac >= 0.0 && flowA_k1_frac < 0.95);
    assert(flowM_k1_frac >= 0.0 && flowM_k1_frac < 0.95);
    assert(hplus_bg_frac >= 0.0 && hplus_bg_frac < 0.95);
    assert(h2_bg_frac >= 0.0 && h2_bg_frac < 0.95);

    for (size_t k = 0; k < history.x_cm.size(); ++k) {
        const double total_k = test_dcr::nuclei_total_full(history.background_full[k], levels)
                             + test_dcr::nuclei_total_compact(history.flowA[k], boundary.A_indices, levels)
                             + test_dcr::nuclei_total_compact(history.flowM[k], boundary.M_indices, levels);
        assert(std::isfinite(total_k));
        assert(total_k > 0.0);
        assert(total_k < 1e3 * std::max(1.0, total_boundary));
    }

    if (std::getenv("DCR_TEST_DUMP")) {
        std::cout << "boundary_flow_frac=" << boundary_flow_frac << "\n";
        std::cout << "flowA_k1_frac=" << flowA_k1_frac << "\n";
        std::cout << "flowM_k1_frac=" << flowM_k1_frac << "\n";
        std::cout << "hplus_bg_frac=" << hplus_bg_frac << "\n";
        std::cout << "h2_bg_frac=" << h2_bg_frac << "\n";
    }

    std::cout << "[PASS] Reference regression checks.\n";
    return 0;
}
