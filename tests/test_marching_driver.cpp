#include <cassert>
#include <cmath>
#include <iostream>

#include "../src/atomic/AtomicData.hpp"
#include "../src/physics/WallBoundary.hpp"
#include "../src/solver/boundary/BoundaryPhase.hpp"
#include "../src/solver/marching/MarchingDriver.hpp"
#include "TestDCRSetup.hpp"

int main() {
    std::cout << "--- Testing MarchingDriver ---\n";

    auto cfg = test_dcr::load_reference_config(/*num_cells=*/8, /*verbose_logging=*/false);
    cfg.grid.length_cm = 1.0;
    cfg.grid.type = "linear";
    cfg.grid.first_cell_cm = 0.0;

    dcr::atomic::AtomicData atomic_data(cfg);
    auto plasma = test_dcr::make_plasma_state(cfg, atomic_data.get_total_states());
    const auto boundary_temperatures = test_dcr::plasma_temperatures_at(cfg, 0.0);
    test_dcr::EEDFContext eedf(boundary_temperatures.electron_eV);

    const double ion_mass_amu = test_dcr::estimate_ion_mass_amu(cfg);
    const auto wall = dcr::physics::compute_wall_recycling(
        cfg.wall.material,
        boundary_temperatures.electron_eV,
        boundary_temperatures.ion_eV,
        ion_mass_amu,
        cfg.wall.sheath_potential_drop
    );

    const auto boundary = dcr::solver::run_boundary_phase(
        cfg, atomic_data, plasma, eedf.grid, ion_mass_amu, wall
    );
    const auto history = dcr::solver::run_full_marching(cfg, atomic_data, plasma, eedf.grid, boundary);

    const auto& levels = atomic_data.get_levels();
    const size_t n_nodes = static_cast<size_t>(cfg.grid.num_cells);

    assert(history.x_cm.size() == n_nodes);
    assert(history.background_full.size() == n_nodes);
    assert(history.flowA.size() == n_nodes);
    assert(history.flowM.size() == n_nodes);
    assert(history.rate_diagnostics.size() == n_nodes);

    for (size_t k = 1; k < history.x_cm.size(); ++k) {
        assert(history.x_cm[k] > history.x_cm[k - 1]);
    }
    assert(std::abs(history.x_cm.back() - cfg.grid.length_cm) < 1e-10);

    const int total_states = atomic_data.get_total_states();
    for (size_t k = 0; k < n_nodes; ++k) {
        const auto& bg = history.background_full[k];
        const auto& a = history.flowA[k];
        const auto& m = history.flowM[k];

        assert(bg.size() == total_states);
        assert(a.size() == static_cast<int>(boundary.A_indices.size()));
        assert(m.size() == static_cast<int>(boundary.M_indices.size()));
        test_dcr::assert_all_finite_nonnegative(bg);
        test_dcr::assert_all_finite_nonnegative(a);
        test_dcr::assert_all_finite_nonnegative(m);

        const auto& rates = history.rate_diagnostics[k];
        assert(std::isfinite(rates.electron_temperature_eV));
        assert(std::isfinite(rates.ion_temperature_eV));
        assert(std::isfinite(rates.electron_density_cm3));
        assert(rates.electron_density_cm3 >= 0.0);
        assert(std::isfinite(rates.atomic_effective.scd_cm3_s));
        assert(std::isfinite(rates.atomic_effective.acd_cm3_s));
        assert(rates.atomic_effective.scd_cm3_s >= 0.0);
        assert(rates.atomic_effective.acd_cm3_s >= 0.0);
        assert(std::isfinite(rates.atomic_qss.max_transport_to_local_ratio));
        assert(std::isfinite(rates.atomic_qss.max_transport_to_loss_frequency_ratio));

        const double total_nuclei =
            test_dcr::nuclei_total_full(bg, levels) +
            test_dcr::nuclei_total_compact(a, boundary.A_indices, levels) +
            test_dcr::nuclei_total_compact(m, boundary.M_indices, levels);

        const double rel_err = std::abs(total_nuclei - cfg.plasma.total_density)
                             / std::max(1.0, cfg.plasma.total_density);
        assert(std::isfinite(rel_err));
        assert(rel_err < 1e-2);
    }

    // Ensure marching produced a state update from x=0 to first interior node.
    if (n_nodes > 1) {
        const double diff_bg = (history.background_full[1] - history.background_full[0]).norm();
        const double diff_a = (history.flowA[1] - history.flowA[0]).norm();
        const double diff_m = (history.flowM[1] - history.flowM[0]).norm();
        assert((diff_bg + diff_a + diff_m) > 0.0);
    }

    std::cout << "[PASS] MarchingDriver checks.\n";
    return 0;
}
