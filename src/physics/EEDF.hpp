#pragma once

#include <algorithm>
#include <cmath>
#include <vector>

// Minimal EEDF implementation (ported from flychk-flow-new).
// All energies in eV. The distribution is normalized on a provided grid.

struct MaxwellianComponent {
    double temperature_ev = 0.0;
    double fraction = 0.0;
};

struct GaussianComponent {
    double mean_energy_ev = 0.0;
    double fwhm_ev = 0.0;
    double fraction = 0.0;
};

struct EEDFConfig {
    double main_component_fraction = 1.0;
    std::vector<MaxwellianComponent> fixed_temp_components;
    std::vector<GaussianComponent> gaussian_components;
};

class EEDF {
public:
    EEDF(double main_temperature_ev, const EEDFConfig& cfg)
        : main_component_{main_temperature_ev, cfg.main_component_fraction},
          fixed_temp_components_(cfg.fixed_temp_components),
          gaussian_components_(cfg.gaussian_components) {
        for (auto& comp : fixed_temp_components_) {
            if (comp.temperature_ev <= 0.0) comp.temperature_ev = main_temperature_ev;
        }
    }

    // Evaluate f(E) at energy_ev (eV). Returns normalized value.
    double operator()(double energy_ev) const { return norm_factor_ * raw_f(energy_ev); }

    void set_main_temperature(double T_ev) { main_component_.temperature_ev = T_ev; }

    // Normalize so that sum_i f(E_i) dE_i = 1 on the provided grid.
    void normalize_on_grid(const std::vector<double>& E_eV,
                           const std::vector<double>& dE_eV) {
        const size_t N = std::min(E_eV.size(), dE_eV.size());
        if (N == 0) { norm_factor_ = 1.0; return; }

        double integral = 0.0;
        for (size_t i = 0; i < N; ++i) {
            integral += raw_f(E_eV[i]) * dE_eV[i];
        }
        norm_factor_ = (integral > 0.0) ? (1.0 / integral) : 1.0;
    }

    double normalization_factor() const { return norm_factor_; }

private:
    double raw_f(double energy_ev) const {
        if (energy_ev <= 0.0) return 0.0;
        double f = 0.0;
        const double sqrtE = std::sqrt(energy_ev);

        if (main_component_.temperature_ev > 0.0 && main_component_.fraction > 0.0) {
            const double T = main_component_.temperature_ev;
            const double T32 = std::pow(T, 1.5);
            f += main_component_.fraction * (TWO_OVER_SQRT_PI / T32) * sqrtE * std::exp(-energy_ev / T);
        }

        for (const auto& comp : fixed_temp_components_) {
            if (comp.temperature_ev <= 0.0 || comp.fraction <= 0.0) continue;
            const double T = comp.temperature_ev;
            const double T32 = std::pow(T, 1.5);
            f += comp.fraction * (TWO_OVER_SQRT_PI / T32) * sqrtE * std::exp(-energy_ev / T);
        }

        for (const auto& tail : gaussian_components_) {
            if (tail.fwhm_ev <= 0.0 || tail.fraction <= 0.0) continue;
            const double sigma = tail.fwhm_ev * FWHM_TO_SIGMA;
            const double x = (energy_ev - tail.mean_energy_ev) / sigma;
            const double norm = 1.0 / (sigma * std::sqrt(2.0 * M_PI));
            f += tail.fraction * norm * std::exp(-0.5 * x * x);
        }
        return f;
    }

    MaxwellianComponent main_component_{};
    std::vector<MaxwellianComponent> fixed_temp_components_;
    std::vector<GaussianComponent> gaussian_components_;

    static constexpr double TWO_OVER_SQRT_PI = 1.1283791670955126; // 2/sqrt(pi)
    static constexpr double FWHM_TO_SIGMA = 0.42466090014400953;   // 1 / 2.35482004503

    double norm_factor_ = 1.0;
};

class EEDFGridView {
public:
    EEDFGridView(const std::vector<double>& energies,
                 const std::vector<double>& weights,
                 const EEDF* eedf_ptr)
        : energies_(energies), weights_(weights), eedf_(eedf_ptr) {}

    size_t size() const { return std::min(energies_.size(), weights_.size()); }
    double energy(size_t i) const { return energies_[i]; }
    double weight(size_t i) const { return weights_[i]; }
    double value_at(size_t i) const { return eedf_ ? (*eedf_)(energies_[i]) : 0.0; }
    double eval(double E) const { return eedf_ ? (*eedf_)(E) : 0.0; }
    bool valid() const { return eedf_ != nullptr; }
    const std::vector<double>& energies() const { return energies_; }
    const std::vector<double>& weights() const { return weights_; }
    const EEDF* eedf() const { return eedf_; }

private:
    const std::vector<double>& energies_;
    const std::vector<double>& weights_;
    const EEDF* eedf_ = nullptr;
};
