#include <iostream>
#include <stdexcept>

#include "../src/atomic/AtomicData.hpp"
#include "../src/physics/WallBoundary.hpp"
#include "../src/solver/boundary/BoundaryPhase.hpp"
#include "../src/solver/marching/GlobalBVP.hpp"
#include "TestDCRSetup.hpp"

int main() {
    auto config = test_dcr::load_reference_config(3, false);
    config.grid.length_cm = 0.005;
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
    const auto diagnostic = dcr::solver::check_global_bvp_fd_m_coupling(
        config, atomic_data, plasma, eedf.grid, boundary, wall);
    if (!(diagnostic.relative_error < 1.0e-6) ||
        !(diagnostic.molecular_profile_relative_change > 0.0)) {
        throw std::runtime_error("Reduced FD action omitted forward-M coupling.");
    }
    std::cout << "[PASS] FD-through-M relative_error=" << diagnostic.relative_error
              << " M_profile_change=" << diagnostic.molecular_profile_relative_change << "\n";
    return 0;
}
