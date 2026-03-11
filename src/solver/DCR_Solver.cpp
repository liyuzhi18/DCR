#include "DCR_Solver.hpp"

#include "boundary/BoundaryPhase.hpp"
#include "core/ConfigPaths.hpp"
#include "core/TemperatureProfile.hpp"
#include "marching/MarchingDriver.hpp"
#include "output/HDF5Output.hpp"

#include "../atomic/AtomicData.hpp"
#include "../io/ConfigLoader.hpp"
#include "../physics/EEDF.hpp"
#include "../physics/EnergyGrid.hpp"
#include "../physics/WallBoundary.hpp"
#include "../state/PlasmaState.hpp"

#include <iostream>
#include <stdexcept>
#include <vector>

namespace dcr::solver {

void DCR_Solver::solve() {
    // Phase 0: configuration and path resolution.
    const std::string config_path = resolve_config_path();
    dcr::io::Config config = dcr::io::ConfigLoader::load(config_path);
    normalize_input_roots(config, config_path);

    if (config.grid.num_cells <= 0) {
        throw std::runtime_error("Invalid grid.num_cells in config.");
    }

    dcr::atomic::AtomicData atomic_data(config);
    if (atomic_data.get_total_states() <= 0) {
        throw std::runtime_error("No atomic/molecular states loaded.");
    }

    const int total_states = atomic_data.get_total_states();

    dcr::state::PlasmaState plasma(config.grid.num_cells, total_states);
    plasma.init_Te().setConstant(config.plasma.Te_eV);
    plasma.init_Ti().setConstant(config.plasma.Ti_eV);
    plasma.init_ne().setConstant(config.plasma.total_density);

    if (config.io.verbose_logging) {
        std::cout << "[DCR_Solver] Config: " << config_path << "\n";
        std::cout << "[DCR_Solver] States: " << total_states << "\n";
        std::cout << "[DCR_Solver] Processes: " << atomic_data.get_processes().size() << "\n";
        std::cout << "[DCR_Solver] Cells: " << config.grid.num_cells << "\n";
    }

    // --- Wall recycling coefficients (atoms reflect, molecules desorb) ---
    double ion_mass_amu = 1.0;
    for (const auto& sp : config.species) {
        if (!sp.is_molecule && sp.charge > 0) {
            ion_mass_amu = sp.mass_amu;
            break;
        }
    }
    const auto boundary_temperatures = evaluate_plasma_temperatures(config, 0.0);
    plasma.init_Te().setConstant(boundary_temperatures.electron_eV);
    plasma.init_Ti().setConstant(boundary_temperatures.ion_eV);
    const auto wall = dcr::physics::compute_wall_recycling(
        config.wall.material,
        boundary_temperatures.electron_eV,
        boundary_temperatures.ion_eV,
        ion_mass_amu,
        config.wall.sheath_potential_drop
    );
    if (config.io.verbose_logging) {
        std::cout << "[DCR_Solver] Wall recycling: T{e=" << boundary_temperatures.electron_eV
                  << ", i=" << boundary_temperatures.ion_eV
                  << "} E0=" << wall.ion_impact_energy_ev
                  << " alpha_atom=" << wall.alpha_atom
                  << " alpha_molecule=" << wall.alpha_molecule << "\n";
    }

    // --- Build legacy flychk electron-energy grid for EEDF integrations ---
    std::vector<double> energies;
    std::vector<double> weights;
    {
        const auto energy_grid = dcr::physics::make_default_energy_grid();
        energies.reserve(energy_grid.size());
        weights.reserve(energy_grid.size());
        for (const auto& p : energy_grid) {
            energies.push_back(p.energy_eV);
            weights.push_back(p.width_eV);
        }
    }

    EEDFConfig eedf_cfg;
    EEDF eedf(boundary_temperatures.electron_eV, eedf_cfg);
    eedf.normalize_on_grid(energies, weights);
    EEDFGridView grid(energies, weights, &eedf);

    // Phase 1: boundary solve at x=0.
    const auto boundary = run_boundary_phase(
        config, atomic_data, plasma, grid, ion_mass_amu, wall
    );

    // Phase 2: full spatial marching using the converged boundary as initial condition.
    const auto marching_history = run_full_marching(config, atomic_data, plasma, grid, boundary);

    // Phase 3: persistent output for post-processing/visualization.
    write_hdf5_output(config, atomic_data, boundary, marching_history);

    // --- Assemble rate matrix using all processes at converged boundary population ---
    dcr::base::Matrix R = dcr::base::Matrix::Zero(total_states, total_states);
    for (const auto& proc : atomic_data.get_processes()) {
        if (!proc) continue;
        proc->apply(plasma, grid, boundary.population, R, nullptr);
    }

    if (config.io.verbose_logging) {
        std::cout << "[DCR_Solver] Rate matrix |R|_sum: " << R.cwiseAbs().sum() << "\n";
    }
}

} // namespace dcr::solver
