#include "DCR_Solver.hpp"

#include "boundary/BoundaryPhase.hpp"
#include "core/ConfigPaths.hpp"
#include "core/TemperatureProfile.hpp"
#include "marching/MarchingDriver.hpp"
#ifdef DCR_ENABLE_EXPERIMENTAL_GLOBAL_BVP
#include "marching/GlobalBVP.hpp"
#endif
#include "output/HDF5Output.hpp"
#include "qss/QSSSolver.hpp"

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

namespace {

EEDFConfig make_eedf_config(const dcr::io::Config& config) {
    EEDFConfig eedf_cfg;
    eedf_cfg.type = config.plasma.eedf.type;
    eedf_cfg.power_p = config.plasma.eedf.power_p;
    eedf_cfg.hot_fraction = config.plasma.eedf.hot_fraction;
    eedf_cfg.hot_temperature_factor = config.plasma.eedf.hot_temperature_factor;
    return eedf_cfg;
}

} // namespace

void DCR_Solver::solve() {
    // Phase 0: configuration and path resolution.
    const std::string config_path = resolve_config_path();
    dcr::io::Config config = dcr::io::ConfigLoader::load(config_path);
    normalize_input_roots(config, config_path);

#ifndef DCR_ENABLE_EXPERIMENTAL_GLOBAL_BVP
    if (config.solver.spatial_method == "experimental_global_bvp") {
        throw std::runtime_error(
            "experimental_global_bvp was requested, but this build was configured "
            "without DCR_ENABLE_EXPERIMENTAL_GLOBAL_BVP");
    }
#endif

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

    const EEDFConfig eedf_cfg = make_eedf_config(config);
    EEDF eedf(boundary_temperatures.electron_eV, eedf_cfg);
    eedf.normalize_on_grid(energies, weights);
    EEDFGridView grid(energies, weights, &eedf);

    BoundaryPhaseResult final_boundary;
    MarchingHistory marching_history;
    if (config.solver.mode == "qss_dcr") {
        if (config.io.verbose_logging) {
            std::cout << "[DCR_Solver] Solver mode: qss_dcr\n";
        }
        const auto qss = run_qss_dcr(
            config, atomic_data, plasma, grid, ion_mass_amu, wall
        );
        final_boundary = qss.boundary;
        marching_history = qss.history;
    } else {
        if (config.io.verbose_logging) {
            std::cout << "[DCR_Solver] Solver mode: full_dcr\n";
        }
        // Phase 1: boundary solve at x=0.
        const auto boundary = run_boundary_phase(
            config, atomic_data, plasma, grid, ion_mass_amu, wall
        );
        final_boundary = boundary;
        if (!boundary.converged) {
            throw std::runtime_error(
                "Boundary phase did not converge; aborting before marching."
            );
        }
        // Phase 2: production uses the target-to-upstream local marcher.
        if (config.solver.spatial_method == "experimental_global_bvp") {
#ifdef DCR_ENABLE_EXPERIMENTAL_GLOBAL_BVP
            const auto global = solve_global_target_conditioned_bvp(
                config, atomic_data, plasma, grid, boundary, wall
            );
            if (!global.converged) {
                throw std::runtime_error("Global target-conditioned BVP did not converge.");
            }
            final_boundary = global.boundary;
            marching_history = global.history;
#else
            throw std::runtime_error(
                "experimental_global_bvp was requested, but this build was configured "
                "without DCR_ENABLE_EXPERIMENTAL_GLOBAL_BVP");
#endif
        } else {
            if (config.numerics.adaptive_recycling_domain.enabled &&
                config.numerics.adaptive_recycling_domain.closure_mode ==
                    "variable_nuclei_balance") {
                marching_history = run_adaptive_variable_marching(
                    config, atomic_data, plasma, grid, boundary
                );
            } else {
                marching_history = run_full_marching(
                    config, atomic_data, plasma, grid, boundary
                );
            }
        }
    }

    // Phase 3: persistent output for post-processing/visualization.
    write_hdf5_output(config, atomic_data, final_boundary, marching_history);

    // --- Assemble rate matrix using all processes at converged boundary population ---
    dcr::base::Matrix R = dcr::base::Matrix::Zero(total_states, total_states);
    for (const auto& proc : atomic_data.get_processes()) {
        if (!proc) continue;
        proc->apply(plasma, grid, final_boundary.population, R, nullptr);
    }

    if (config.io.verbose_logging) {
        std::cout << "[DCR_Solver] Rate matrix |R|_sum: " << R.cwiseAbs().sum() << "\n";
    }
}

} // namespace dcr::solver
