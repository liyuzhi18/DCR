#include <iostream>
#include <stdexcept>

#include "../src/atomic/AtomicData.hpp"
#include "../src/physics/WallBoundary.hpp"
#include "../src/solver/boundary/BoundaryPhase.hpp"
#include "../src/solver/marching/GlobalBVP.hpp"
#include "TestDCRSetup.hpp"

namespace {

void require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

} // namespace

int main() {
    auto config = test_dcr::load_reference_config(10, false);
    config.grid.length_cm = 0.01;
    config.numerics.marching_solver = "global_sparse_newton";
    config.numerics.marching_tolerance = 1.0e-8;
    config.numerics.marching_max_iterations = 120;
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
    require(boundary.converged, "Boundary phase did not converge.");

    auto two_node_config = config;
    two_node_config.grid.num_cells = 2;
    auto two_node_plasma = test_dcr::make_plasma_state(
        two_node_config, atomic_data.get_total_states());
    const auto two_node = dcr::solver::solve_global_target_conditioned_bvp(
        two_node_config, atomic_data, two_node_plasma, eedf.grid, boundary, wall);
    require(two_node.converged, "Two-node coarse seed did not converge.");

    auto five_node_config = config;
    five_node_config.grid.num_cells = 5;
    auto five_node_plasma = test_dcr::make_plasma_state(
        five_node_config, atomic_data.get_total_states());
    const auto five_node = dcr::solver::solve_global_target_conditioned_bvp(
        five_node_config, atomic_data, five_node_plasma, eedf.grid,
        boundary, wall, &two_node);
    require(five_node.converged, "Five-node coarse seed did not converge.");

    const auto result = dcr::solver::solve_global_target_conditioned_bvp(
        config, atomic_data, plasma, eedf.grid, boundary, wall, &five_node);
    require(result.converged, "Ten-node global BVP did not converge.");
    require(result.residual_norm < 1.0e-8, "Ten-node scaled residual exceeds 1e-8.");
    require(result.history.x_cm.size() == 10, "Ten-node history has the wrong size.");
    require(result.max_chemistry_nuclei_relative_error < 1.0e-10,
        "Midpoint chemistry violates nuclei conservation.");
    require(result.integrated_nuclei_balance_relative_error < 1.0e-7,
        "Integrated nuclei balance does not close.");
    require(result.ion_flux_chemistry_relative_error < 1.0e-7,
        "Ion flux change does not match integrated ion chemistry.");

    std::cout << "[PASS] Ten-node global BVP lambda=1 residual="
              << result.residual_norm
              << " chemistry_nuclei=" << result.max_chemistry_nuclei_relative_error
              << " integrated_nuclei=" << result.integrated_nuclei_balance_relative_error
              << " ion_flux_chemistry=" << result.ion_flux_chemistry_relative_error
              << "\n";
    return 0;
}
