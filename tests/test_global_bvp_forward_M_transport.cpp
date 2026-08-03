#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>

#include "../src/atomic/AtomicData.hpp"
#include "../src/physics/WallBoundary.hpp"
#include "../src/solver/boundary/BoundaryPhase.hpp"
#include "../src/solver/marching/GlobalBVP.hpp"
#include "TestDCRSetup.hpp"

int main() {
    auto config = test_dcr::load_reference_config(4, false);
    config.grid.length_cm = 0.01;
    dcr::atomic::AtomicData atomic_data(config);
    auto plasma = test_dcr::make_plasma_state(config, atomic_data.get_total_states());
    const auto temperatures = test_dcr::plasma_temperatures_at(config, 0.0);
    test_dcr::EEDFContext eedf(temperatures.electron_eV);
    const double ion_mass = test_dcr::estimate_ion_mass_amu(config);
    const auto wall = dcr::physics::compute_wall_recycling(
        config.wall.material, temperatures.electron_eV, temperatures.ion_eV,
        ion_mass, config.wall.sheath_potential_drop);
    const auto boundary = dcr::solver::run_boundary_phase(
        config, atomic_data, plasma, eedf.grid, ion_mass, wall);
    if (!boundary.converged) throw std::runtime_error("Boundary phase did not converge.");
    const auto diagnostic = dcr::solver::check_global_bvp_forward_m_transport(
        config, atomic_data, plasma, eedf.grid, boundary, wall);
    const double launch_error = std::abs(diagnostic.target_ground_flux -
        diagnostic.expected_target_ground_flux) /
        std::max(1.0, diagnostic.expected_target_ground_flux);
    if (!diagnostic.finite_nonnegative || !diagnostic.no_floor_activation ||
        diagnostic.density.size() != 4 || launch_error > 1.0e-13 ||
        diagnostic.maximum_excited_target_flux != 0.0 ||
        diagnostic.maximum_scaled_midpoint_balance > 1.0e-10 ||
        diagnostic.upstream_total_flux > diagnostic.target_total_flux * (1.0 + 1.0e-12)) {
        throw std::runtime_error("Forward molecular transport diagnostic failed.");
    }
    std::cout << "[PASS] target_Gamma=" << diagnostic.target_ground_flux
              << " upstream_Gamma=" << diagnostic.upstream_total_flux
              << " midpoint_balance=" << diagnostic.maximum_scaled_midpoint_balance << "\n";
    return 0;
}
