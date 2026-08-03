#pragma once

#include "../../io/ConfigStructs.hpp"
#include "../../physics/EEDF.hpp"
#include "../../state/PlasmaState.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>

namespace dcr::solver {

struct PlasmaTemperatures {
    double electron_eV = 0.0;
    double ion_eV = 0.0;
};

inline double evaluate_temperature_profile(const dcr::io::TemperatureProfileConfig& profile,
                                           double fallback_eV,
                                           double x_cm) {
    if (!profile.enabled) return fallback_eV;

    if (profile.type == "constant") {
        return profile.value_eV;
    }

    if (profile.type == "linear" || profile.type == "linear_x") {
        const double x0 = profile.x_start_cm;
        const double x1 = profile.x_end_cm;
        if (x1 <= x0) return profile.value_end_eV;
        if (x_cm <= x0) return profile.value_start_eV;
        if (x_cm >= x1) return profile.value_end_eV;
        const double t = (x_cm - x0) / (x1 - x0);
        return profile.value_start_eV + t * (profile.value_end_eV - profile.value_start_eV);
    }

    if (profile.type == "constant_heat_flux") {
        const double x0 = profile.x_start_cm;
        const double x1 = profile.x_end_cm;
        if (x1 <= x0) return profile.value_end_eV;
        if (x_cm <= x0) return profile.value_start_eV;
        if (x_cm >= x1) return profile.value_end_eV;
        if (profile.value_start_eV < 0.0 || profile.value_end_eV < 0.0) {
            throw std::runtime_error("constant_heat_flux temperature profile requires non-negative endpoint temperatures");
        }

        const double t = (x_cm - x0) / (x1 - x0);
        const double t0_pow = std::pow(profile.value_start_eV, 3.5);
        const double t1_pow = std::pow(profile.value_end_eV, 3.5);
        return std::pow(t0_pow + t * (t1_pow - t0_pow), 2.0 / 7.0);
    }

    throw std::runtime_error("Unsupported temperature profile type: " + profile.type);
}

inline PlasmaTemperatures evaluate_plasma_temperatures(const dcr::io::Config& config,
                                                       double x_cm) {
    PlasmaTemperatures out;
    out.electron_eV = evaluate_temperature_profile(
        config.plasma.electron_temperature_profile,
        config.plasma.Te_eV,
        x_cm
    );
    out.ion_eV = evaluate_temperature_profile(
        config.plasma.ion_temperature_profile,
        config.plasma.Ti_eV,
        x_cm
    );
    return out;
}

class LocalKineticContext {
public:
    LocalKineticContext(const dcr::state::PlasmaState& plasma_template,
                        const EEDFGridView& template_grid,
                        double electron_temperature_eV,
                        double ion_temperature_eV,
                        double electron_density_cm3)
        : plasma_(plasma_template),
          eedf_(template_grid.eedf() ? *template_grid.eedf() : EEDF(electron_temperature_eV, EEDFConfig{})),
          grid_(template_grid.energies(), template_grid.weights(), &eedf_) {
        plasma_.init_Te().setConstant(electron_temperature_eV);
        plasma_.init_Ti().setConstant(ion_temperature_eV);
        plasma_.init_ne().setConstant(std::max(0.0, electron_density_cm3));
        eedf_.set_main_temperature(electron_temperature_eV);
        eedf_.normalize_on_grid(grid_.energies(), grid_.weights());
    }

    const dcr::state::PlasmaState& plasma() const { return plasma_; }
    const EEDFGridView& grid() const { return grid_; }

private:
    dcr::state::PlasmaState plasma_;
    EEDF eedf_;
    EEDFGridView grid_;
};

} // namespace dcr::solver
