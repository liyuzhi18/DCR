#include <cassert>
#include <cmath>
#include <iostream>

#include "../src/atomic/AtomicData.hpp"
#include "../src/physics/WallBoundary.hpp"
#include "../src/solver/boundary/BoundaryPhase.hpp"
#include "TestDCRSetup.hpp"

int main() {
    std::cout << "--- Testing BoundaryPhase ---\n";

    auto cfg = test_dcr::load_reference_config(/*num_cells=*/4, /*verbose_logging=*/false);
    dcr::atomic::AtomicData atomic_data(cfg);
    assert(atomic_data.get_total_states() > 0);

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

    const auto& levels = atomic_data.get_levels();
    const int total_states = atomic_data.get_total_states();

    assert(boundary.population.size() == total_states);
    assert(boundary.P_indices.size() + boundary.R_indices.size() == static_cast<size_t>(total_states));
    assert(!boundary.P_indices.empty());
    assert(!boundary.A_indices.empty());
    assert(!boundary.M_indices.empty());
    assert(boundary.u_A > 0.0);
    assert(boundary.u_M > 0.0);

    test_dcr::assert_all_finite_nonnegative(boundary.population);

    if (!boundary.explicit_recycling) {
        assert(boundary.have_flow_last);
        assert(boundary.flowA_last.size() == static_cast<int>(boundary.A_indices.size()));
        assert(boundary.flowM_last.size() == static_cast<int>(boundary.M_indices.size()));
        test_dcr::assert_all_finite_nonnegative(boundary.flowA_last);
        test_dcr::assert_all_finite_nonnegative(boundary.flowM_last);
    }

    const double bg_total = test_dcr::nuclei_total_selected(boundary.population, boundary.P_indices, levels);
    double flow_total = 0.0;
    if (boundary.explicit_recycling) {
        flow_total = test_dcr::nuclei_total_selected(boundary.population, boundary.R_indices, levels);
    } else {
        flow_total = test_dcr::nuclei_total_compact(boundary.flowA_last, boundary.A_indices, levels)
                   + test_dcr::nuclei_total_compact(boundary.flowM_last, boundary.M_indices, levels);
    }

    const double total = bg_total + flow_total;
    assert(std::isfinite(total));
    assert(total > 0.0);

    const double target = cfg.plasma.total_density;
    const double rel_err = std::abs(total - target) / std::max(1.0, target);
    assert(rel_err < 5e-4);

    std::cout << "[PASS] BoundaryPhase checks.\n";
    return 0;
}
