#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>

#include "../src/atomic/AtomicData.hpp"
#include "../src/physics/Sheath.hpp"
#include "../src/solver/core/TemperatureProfile.hpp"
#include "../src/solver/marching/GlobalBVP.hpp"
#include "TestDCRSetup.hpp"

namespace {

bool close(double actual, double expected, double relative_tolerance = 1.0e-12) {
    return std::abs(actual - expected) <= relative_tolerance *
        std::max({1.0, std::abs(actual), std::abs(expected)});
}

void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

} // namespace

int main() {
    auto config = test_dcr::load_reference_config(4, false);
    config.grid.length_cm = 2.0;
    config.global_bvp.ion_velocity_transition_length_cm = 10.0;
    config.global_bvp.ion_upstream_speed_fraction = 0.2;

    dcr::atomic::AtomicData atomic_data(config);
    const auto& levels = atomic_data.get_levels();
    const dcr::atomic::EnergyLevel* ion = nullptr;
    for (const auto& level : levels) {
        if (level.type == dcr::atomic::SpeciesType::Ion && level.charge > 0) {
            ion = &level;
            break;
        }
    }
    require(ion != nullptr, "No positive-ion state found.");

    for (double x_cm : {0.0, 5.0, 9.0, 20.0}) {
        const auto temperatures = dcr::solver::evaluate_plasma_temperatures(config, x_cm);
        const double xi = std::clamp(x_cm / 10.0, 0.0, 1.0);
        const double shape = 1.0 - 0.8 * xi;
        const double expected = -dcr::physics::calculate_Bohm_speed(
            temperatures.electron_eV, temperatures.ion_eV, ion->mass_amu) * shape;
        const double actual = dcr::solver::global_ion_velocity_cm_s(config, *ion, x_cm);
        require(actual < 0.0, "Global ion velocity is not target-directed.");
        require(close(actual, expected), "Global ion velocity differs from prescribed profile.");
    }

    // Manufactured negative-velocity ion balance: the global path must retain
    // the signed face flux rather than silently replacing it with density.
    const double dx_cm = 0.25;
    const double u_left = -4.0e5;
    const double u_right = -3.0e5;
    const double n_left = 2.0e12;
    const double prescribed_source = 8.0e17;
    const double flux_left = u_left * n_left;
    const double n_right = (flux_left + prescribed_source * dx_cm) / u_right;
    const double ion_divergence = dcr::solver::conservative_flux_divergence_cm3_s(
        u_left, n_left, u_right, n_right, dx_cm);
    require(close(ion_divergence, prescribed_source), "Signed ion flux divergence is incorrect.");

    // The same conservative operator must retain the opposite sign for a
    // target-launched recycling flow.
    const double neutral_divergence = dcr::solver::conservative_flux_divergence_cm3_s(
        2.0e5, 3.0e11, 2.0e5, 2.5e11, 0.5);
    require(close(neutral_divergence, -2.0e16), "Recycling-flow flux divergence is incorrect.");

    bool rejected_zero_dx = false;
    try {
        (void)dcr::solver::conservative_flux_divergence_cm3_s(-1.0, 1.0, -1.0, 1.0, 0.0);
    } catch (const std::invalid_argument&) {
        rejected_zero_dx = true;
    }
    require(rejected_zero_dx, "Flux divergence accepted zero cell width.");

    std::cout << "[PASS] Global BVP prescribed-velocity transport checks.\n";
    return 0;
}
