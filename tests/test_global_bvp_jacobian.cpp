#include <iostream>
#include <stdexcept>

#include "../src/atomic/AtomicData.hpp"
#include "../src/physics/WallBoundary.hpp"
#include "../src/solver/boundary/BoundaryPhase.hpp"
#include "../src/solver/marching/GlobalBVP.hpp"
#include "TestDCRSetup.hpp"

int main() {
    auto config = test_dcr::load_reference_config(2, false);
    config.grid.length_cm = 0.01;
    config.global_bvp.ion_velocity_transition_length_cm = 1.0;
    dcr::atomic::AtomicData atomic_data(config);
    auto plasma = test_dcr::make_plasma_state(config, atomic_data.get_total_states());
    const auto temperatures = test_dcr::plasma_temperatures_at(config, 0.0);
    test_dcr::EEDFContext eedf(temperatures.electron_eV);
    const double ion_mass_amu = test_dcr::estimate_ion_mass_amu(config);
    const auto wall = dcr::physics::compute_wall_recycling(
        config.wall.material, temperatures.electron_eV, temperatures.ion_eV,
        ion_mass_amu, config.wall.sheath_potential_drop);
    const auto boundary = dcr::solver::run_boundary_phase(
        config, atomic_data, plasma, eedf.grid, ion_mass_amu, wall);
    if (!boundary.converged) throw std::runtime_error("Boundary phase did not converge.");

    const auto diagnostics = dcr::solver::check_global_bvp_initial_jacobian(
        config, atomic_data, plasma, eedf.grid, boundary, wall);
    bool failed = false;
    for (size_t i = 0; i < diagnostics.labels.size(); ++i) {
        std::cout << diagnostics.labels[i] << " relative_error="
                  << diagnostics.relative_errors[i]
                  << " observed_order=" << diagnostics.observed_orders[i] << "\n";
        if (!(diagnostics.relative_errors[i] < 1.0e-6) ||
            !(diagnostics.observed_orders[i] > 1.5)) {
            failed = true;
        }
    }
    if (failed) throw std::runtime_error("Matrix-free Jacobian directional check failed.");
    std::cout << "[PASS] Global BVP matrix-free Jacobian directional checks.\n";
    return 0;
}
