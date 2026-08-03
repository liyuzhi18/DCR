#include <algorithm>
#include <cmath>
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
    auto config = test_dcr::load_reference_config(2, false);
    config.grid.length_cm = 0.001;
    config.numerics.marching_solver = "global_sparse_newton";
    config.numerics.marching_tolerance = 1.0e-8;
    config.numerics.marching_max_iterations = 100;
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

    const auto result = dcr::solver::solve_global_target_conditioned_bvp(
        config, atomic_data, plasma, eedf.grid, boundary, wall);
    require(result.converged, "Two-node global BVP did not converge.");
    require(result.residual_norm < 1.0e-8, "Two-node scaled residual exceeds 1e-8.");
    require(result.history.x_cm.size() == 2, "Two-node history has the wrong size.");
    require(result.upstream_proton_flux_cm2_s > 0.0,
        "Solved upstream proton supply is not positive.");
    const auto shifted_result = dcr::solver::solve_global_target_conditioned_bvp(
        config, atomic_data, plasma, eedf.grid, boundary, wall, nullptr, std::log(0.5));
    require(shifted_result.converged && shifted_result.residual_norm < 1.0e-8,
        "Two-node BVP did not converge from the second initial guess.");
    require(std::abs(shifted_result.upstream_proton_flux_cm2_s -
        result.upstream_proton_flux_cm2_s) / result.upstream_proton_flux_cm2_s < 1.0e-6,
        "Two initial guesses converged to different upstream proton supplies.");
    for (size_t k = 0; k < result.history.background_full.size(); ++k) {
        const double relative_difference =
            (shifted_result.history.background_full[k] -
                result.history.background_full[k]).norm() /
            std::max(1.0, result.history.background_full[k].norm());
        require(relative_difference < 1.0e-6,
            "Two initial guesses converged to different population profiles.");
    }

    const auto& levels = atomic_data.get_levels();
    double target_nuclei = test_dcr::nuclei_total_full(
        result.history.background_full.front(), levels);
    target_nuclei += test_dcr::nuclei_total_compact(
        result.history.flowA.front(), boundary.A_indices, levels);
    target_nuclei += test_dcr::nuclei_total_compact(
        result.history.flowM.front(), boundary.M_indices, levels);
    require(std::abs(target_nuclei - config.plasma.total_density) /
        config.plasma.total_density < 1.0e-8, "Target nuclei condition is not closed.");

    int proton_gi = -1;
    for (int gi : boundary.ion_indices) {
        const auto& level = levels[static_cast<size_t>(gi)];
        if (level.type == dcr::atomic::SpeciesType::Ion &&
            level.charge > 0 && level.atomicity == 1) {
            require(proton_gi < 0, "Ambiguous proton carrier in test data.");
            proton_gi = gi;
        }
    }
    require(proton_gi >= 0, "No proton carrier found.");
    const double proton_flux = dcr::solver::global_ion_velocity_cm_s(
        config, levels[static_cast<size_t>(proton_gi)], config.grid.length_cm) *
        result.history.background_full.back()(proton_gi);
    const double proton_bc_error = std::abs(
        proton_flux + result.upstream_proton_flux_cm2_s) /
        result.upstream_proton_flux_cm2_s;
    require(proton_bc_error < 1.0e-8, "Upstream proton boundary condition is not closed.");

    for (int gi : boundary.ion_indices) {
        if (gi == proton_gi) continue;
        const double flux = dcr::solver::global_ion_velocity_cm_s(
            config, levels[static_cast<size_t>(gi)], config.grid.length_cm) *
            result.history.background_full.back()(gi);
        require(std::abs(flux) / result.upstream_proton_flux_cm2_s < 1.0e-8,
            "A minor positive ion has nonzero upstream influx.");
    }
    for (size_t k = 0; k < result.history.background_full.size(); ++k) {
        auto check_positive = [&](const dcr::base::Vector& values) {
            for (int i = 0; i < values.size(); ++i) {
                require(std::isfinite(values(i)) && values(i) > 0.0,
                    "Global BVP produced a non-positive or non-finite population.");
            }
        };
        check_positive(result.history.background_full[k]);
        check_positive(result.history.flowA[k]);
        for (int i = 0; i < result.history.flowM[k].size(); ++i) {
            require(std::isfinite(result.history.flowM[k](i)) &&
                result.history.flowM[k](i) >= 0.0,
                "Derived molecular population is negative or non-finite.");
        }
    }

    std::cout << "[PASS] Two-node global BVP reached lambda=1 with residual="
              << result.residual_norm << " q_up="
              << result.upstream_proton_flux_cm2_s << " cm^-2 s^-1.\n";
    return 0;
}
