#include <cassert>
#include <cmath>
#include <iostream>
#include <vector>

#include "../src/io/ConfigStructs.hpp"
#include "../src/physics/EEDF.hpp"
#include "../src/solver/core/TemperatureProfile.hpp"
#include "../src/state/PlasmaState.hpp"

int main() {
    std::cout << "--- Testing Temperature Profile ---\n";

    dcr::io::Config cfg;
    cfg.plasma.Te_eV = 5.0;
    cfg.plasma.Ti_eV = 2.0;

    {
        const auto temps = dcr::solver::evaluate_plasma_temperatures(cfg, 1.0);
        assert(temps.electron_eV == 5.0);
        assert(temps.ion_eV == 2.0);
    }

    cfg.plasma.electron_temperature_profile.enabled = true;
    cfg.plasma.electron_temperature_profile.type = "linear_x";
    cfg.plasma.electron_temperature_profile.x_start_cm = 0.0;
    cfg.plasma.electron_temperature_profile.x_end_cm = 2.0;
    cfg.plasma.electron_temperature_profile.value_start_eV = 1.0;
    cfg.plasma.electron_temperature_profile.value_end_eV = 5.0;

    cfg.plasma.ion_temperature_profile.enabled = true;
    cfg.plasma.ion_temperature_profile.type = "constant";
    cfg.plasma.ion_temperature_profile.value_eV = 1.5;

    {
        const auto t0 = dcr::solver::evaluate_plasma_temperatures(cfg, -0.1);
        const auto t1 = dcr::solver::evaluate_plasma_temperatures(cfg, 1.0);
        const auto t2 = dcr::solver::evaluate_plasma_temperatures(cfg, 5.0);
        assert(std::abs(t0.electron_eV - 1.0) < 1e-12);
        assert(std::abs(t1.electron_eV - 3.0) < 1e-12);
        assert(std::abs(t2.electron_eV - 5.0) < 1e-12);
        assert(std::abs(t0.ion_eV - 1.5) < 1e-12);
        assert(std::abs(t1.ion_eV - 1.5) < 1e-12);
        assert(std::abs(t2.ion_eV - 1.5) < 1e-12);
    }

    cfg.plasma.electron_temperature_profile.type = "constant_heat_flux";
    cfg.plasma.electron_temperature_profile.value_start_eV = 2.0;
    cfg.plasma.electron_temperature_profile.value_end_eV = 5.0;

    {
        const auto t0 = dcr::solver::evaluate_plasma_temperatures(cfg, -0.1);
        const auto t1 = dcr::solver::evaluate_plasma_temperatures(cfg, 1.0);
        const auto t2 = dcr::solver::evaluate_plasma_temperatures(cfg, 5.0);
        const double expected_mid = std::pow(
            0.5 * (std::pow(2.0, 3.5) + std::pow(5.0, 3.5)),
            2.0 / 7.0
        );
        assert(std::abs(t0.electron_eV - 2.0) < 1e-12);
        assert(std::abs(t1.electron_eV - expected_mid) < 1e-12);
        assert(std::abs(t2.electron_eV - 5.0) < 1e-12);
    }

    dcr::state::PlasmaState plasma(1, 1);
    std::vector<double> energies{1.0, 2.0, 3.0};
    std::vector<double> weights{1.0, 1.0, 1.0};
    EEDF template_eedf(5.0, EEDFConfig{});
    template_eedf.normalize_on_grid(energies, weights);
    EEDFGridView template_grid(energies, weights, &template_eedf);

    const double template_value = template_grid.value_at(1);
    const dcr::solver::LocalKineticContext local(plasma, template_grid, 2.0, 4.0, 7.0);
    assert(std::abs(local.plasma().electron_temperature_ev() - 2.0) < 1e-12);
    assert(std::abs(local.plasma().ion_temperature_ev() - 4.0) < 1e-12);
    assert(std::abs(local.plasma().electron_density_cm3() - 7.0) < 1e-12);
    assert(local.grid().valid());
    assert(std::abs(local.grid().value_at(1) - template_value) > 1e-8);

    std::cout << "[PASS] Temperature profile checks.\n";
    return 0;
}
