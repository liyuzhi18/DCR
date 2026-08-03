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
    for (int nodes : {2, 5}) {
        auto config = test_dcr::load_reference_config(nodes, false);
        config.numerics.adaptive_recycling_domain.closure_mode = "variable_nuclei_balance";
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

        const auto count = dcr::solver::global_bvp_count_diagnostics(
            config, atomic_data, boundary);
        require(count.nodes == nodes, "Configured global node count is incorrect.");
        require(count.intervals == nodes - 1, "Global interval count is incorrect.");
        require(count.unknown_count == count.residual_count, "Global BVP is not square.");
        require(count.ion_interval_rows == count.intervals * count.positive_ions,
            "Ion interval-row count is incorrect.");
        require(count.ion_upstream_boundary_rows == count.positive_ions,
            "Ion upstream boundary count is incorrect.");
        require(count.recycling_target_boundary_rows == count.recycling_atom_states,
            "Target recycling boundary count is incorrect.");
        require(count.molecular_unknown_count == 0 && count.molecular_residual_rows == 0,
            "Molecular transport leaked into the reduced global system.");
        require(count.target_density_rows == 1, "Expected one target density row.");
        require(count.spatial_density_rows == 0, "Found a pointwise density constraint.");
        require(count.legacy_ion_closure_rows == 0, "Found a legacy L_I closure row.");
        require(count.q_up_coupled_rows == 1, "q_up must couple to exactly one row.");
        require(count.proton_global_index >= 0, "No unique proton carrier selected.");
        std::cout << "nodes=" << count.nodes
                  << " intervals=" << count.intervals
                  << " unknowns=" << count.unknown_count
                  << " equations=" << count.residual_count
                  << " upstream_ion_bc=" << count.ion_upstream_boundary_rows
                  << " target_recycling_bc=" << count.recycling_target_boundary_rows
                  << " target_density_bc=" << count.target_density_rows << "\n";
    }
    std::cout << "[PASS] Global BVP boundary and equation counts are square.\n";
    return 0;
}
