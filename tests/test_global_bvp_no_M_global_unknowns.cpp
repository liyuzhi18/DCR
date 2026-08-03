#include <iostream>
#include <stdexcept>

#include "../src/atomic/AtomicData.hpp"
#include "../src/physics/WallBoundary.hpp"
#include "../src/solver/boundary/BoundaryPhase.hpp"
#include "../src/solver/marching/GlobalBVP.hpp"
#include "TestDCRSetup.hpp"

int main() {
    auto config = test_dcr::load_reference_config(4, false);
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
    const auto count = dcr::solver::global_bvp_count_diagnostics(
        config, atomic_data, boundary);
    const int expected = count.nodes *
        (count.background_states + count.recycling_atom_states) + 1;
    if (count.recycling_molecule_states <= 0 || count.unknown_count != expected ||
        count.residual_count != expected || count.molecular_unknown_count != 0 ||
        count.molecular_residual_rows != 0) {
        throw std::runtime_error("Global reduced count still contains molecular unknowns or rows.");
    }
    std::cout << "[PASS] reduced unknowns=" << count.unknown_count
              << " M_metadata=" << count.recycling_molecule_states
              << " M_unknowns=" << count.molecular_unknown_count
              << " M_rows=" << count.molecular_residual_rows << "\n";
    return 0;
}
