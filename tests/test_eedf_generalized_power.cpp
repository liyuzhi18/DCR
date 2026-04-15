#include <algorithm>
#include <cassert>
#include <cmath>
#include <iostream>
#include <vector>

#include "../src/physics/EEDF.hpp"
#include "../src/physics/EnergyGrid.hpp"

namespace {

double integrate_weighted(const std::vector<double>& energies,
                          const std::vector<double>& weights,
                          const EEDFGridView& grid,
                          bool include_energy) {
    double total = 0.0;
    const size_t n = std::min(energies.size(), weights.size());
    for (size_t i = 0; i < n; ++i) {
        const double factor = include_energy ? energies[i] : 1.0;
        total += factor * grid.value_at(i) * weights[i];
    }
    return total;
}

} // namespace

int main() {
    std::cout << "--- Testing Generalized-Power EEDF ---\n";

    std::vector<double> energies;
    std::vector<double> weights;
    for (const auto& p : dcr::physics::make_default_energy_grid()) {
        energies.push_back(p.energy_eV);
        weights.push_back(p.width_eV);
    }

    for (double power_p : {0.9, 1.0, 2.0}) {
        EEDFConfig cfg;
        cfg.type = "generalized_power";
        cfg.power_p = power_p;

        EEDF eedf(5.0, cfg);
        eedf.normalize_on_grid(energies, weights);
        EEDFGridView grid(energies, weights, &eedf);

        const double integral = integrate_weighted(energies, weights, grid, /*include_energy=*/false);
        const double mean_energy = integrate_weighted(energies, weights, grid, /*include_energy=*/true);

        assert(std::isfinite(eedf.generalized_power_theta_ev()));
        assert(eedf.generalized_power_theta_ev() > 0.0);
        assert(std::abs(integral - 1.0) < 1e-10);
        assert(std::abs(mean_energy - 7.5) < 1e-8);

        for (size_t i = 0; i < grid.size(); ++i) {
            const double value = grid.value_at(i);
            assert(std::isfinite(value));
            assert(value >= 0.0);
        }
    }

    for (double hot_fraction : {0.05, 0.1}) {
        EEDFConfig cfg;
        cfg.type = "bi_maxwellian";
        cfg.hot_fraction = hot_fraction;
        cfg.hot_temperature_factor = 2.5;

        EEDF eedf(5.0, cfg);
        eedf.normalize_on_grid(energies, weights);
        EEDFGridView grid(energies, weights, &eedf);

        const double integral = integrate_weighted(energies, weights, grid, /*include_energy=*/false);
        const double mean_energy = integrate_weighted(energies, weights, grid, /*include_energy=*/true);

        assert(std::abs(integral - 1.0) < 1e-10);
        assert(std::abs(mean_energy - 7.5) < 1e-3);

        for (size_t i = 0; i < grid.size(); ++i) {
            const double value = grid.value_at(i);
            assert(std::isfinite(value));
            assert(value >= 0.0);
        }
    }

    EEDF default_maxwell(5.0, EEDFConfig{});
    default_maxwell.normalize_on_grid(energies, weights);
    EEDFGridView default_grid(energies, weights, &default_maxwell);

    EEDFConfig explicit_cfg;
    explicit_cfg.type = "maxwellian";
    EEDF explicit_maxwell(5.0, explicit_cfg);
    explicit_maxwell.normalize_on_grid(energies, weights);
    EEDFGridView explicit_grid(energies, weights, &explicit_maxwell);

    for (size_t i = 0; i < default_grid.size(); ++i) {
        assert(std::abs(default_grid.value_at(i) - explicit_grid.value_at(i)) < 1e-14);
    }

    std::cout << "[PASS] Generalized-Power EEDF checks.\n";
    return 0;
}
