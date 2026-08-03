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
    config.global_bvp.ion_velocity_transition_length_cm = 50.0;
    config.global_bvp.ion_upstream_speed_fraction = 0.1;
    dcr::atomic::AtomicData atomic_data(config);

    const dcr::atomic::EnergyLevel* proton = nullptr;
    const dcr::atomic::EnergyLevel* molecular_ion = nullptr;
    for (const auto& level : atomic_data.get_levels()) {
        if (level.charge <= 0) continue;
        if (level.atomicity == 1 && proton == nullptr) proton = &level;
        if (level.atomicity == 2 && molecular_ion == nullptr) molecular_ion = &level;
    }
    require(proton != nullptr, "No proton state found.");
    require(molecular_ion != nullptr, "No molecular-ion state found.");

    for (const auto* ion : {proton, molecular_ion}) {
        for (const auto& point : std::initializer_list<std::pair<double, double>>{
                 {0.0, 1.0}, {25.0, 0.55}, {50.0, 0.1}, {75.0, 0.1}}) {
            const auto temperatures = dcr::solver::evaluate_plasma_temperatures(
                config, point.first);
            const double bohm_speed = dcr::physics::calculate_Bohm_speed(
                temperatures.electron_eV, temperatures.ion_eV, ion->mass_amu);
            const double velocity = dcr::solver::global_ion_velocity_cm_s(
                config, *ion, point.first);
            require(close(velocity, -point.second * bohm_speed),
                "Fixed-scale ion velocity has the wrong value.");
        }
    }

    const double x_cm = 3.5;
    double reference_velocity = 0.0;
    for (double current_domain_cm : {5.0, 20.0, 50.0}) {
        auto domain_config = config;
        domain_config.grid.length_cm = current_domain_cm;
        const double velocity = dcr::solver::global_ion_velocity_cm_s(
            domain_config, *proton, x_cm);
        if (reference_velocity == 0.0) reference_velocity = velocity;
        require(close(velocity, reference_velocity),
            "Continuation-domain length changed the ion velocity.");
    }
    const auto temperatures = dcr::solver::evaluate_plasma_temperatures(config, x_cm);
    const double bohm_speed = dcr::physics::calculate_Bohm_speed(
        temperatures.electron_eV, temperatures.ion_eV, proton->mass_amu);
    require(close(std::abs(reference_velocity) / bohm_speed, 0.937),
        "The 3.5 cm ion-speed ratio is not 0.937.");

    std::cout << "[PASS] Global BVP fixed-scale ion velocity is domain invariant.\n";
    return 0;
}
