#pragma once

#include <algorithm>
#include <cmath>
#include <string>
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
    std::string type = "maxwellian";
    double power_p = 1.0;
    double hot_fraction = 0.0;
    double hot_temperature_factor = 2.5;
    double main_component_fraction = 1.0;
    std::vector<MaxwellianComponent> fixed_temp_components;
    std::vector<GaussianComponent> gaussian_components;
};

class EEDF {
public:
    EEDF(double main_temperature_ev, const EEDFConfig& cfg)
        : type_(cfg.type),
          power_p_(cfg.power_p),
          hot_fraction_(cfg.hot_fraction),
          hot_temperature_factor_(cfg.hot_temperature_factor),
          target_temperature_ev_(main_temperature_ev),
          generalized_power_theta_ev_(main_temperature_ev),
          main_component_{main_temperature_ev, cfg.main_component_fraction},
          fixed_temp_components_(cfg.fixed_temp_components),
          gaussian_components_(cfg.gaussian_components) {
        for (auto& comp : fixed_temp_components_) {
            if (comp.temperature_ev <= 0.0) comp.temperature_ev = main_temperature_ev;
        }
    }

    // Evaluate f(E) at energy_ev (eV). Returns normalized value.
    double operator()(double energy_ev) const { return norm_factor_ * raw_f(energy_ev); }

    void set_main_temperature(double T_ev) {
        target_temperature_ev_ = T_ev;
        main_component_.temperature_ev = T_ev;
    }

    // Normalize so that sum_i f(E_i) dE_i = 1 on the provided grid.
    void normalize_on_grid(const std::vector<double>& E_eV,
                           const std::vector<double>& dE_eV) {
        const size_t N = std::min(E_eV.size(), dE_eV.size());
        if (N == 0) { norm_factor_ = 1.0; return; }

        if (is_generalized_power()) {
            generalized_power_theta_ev_ = solve_generalized_power_theta(E_eV, dE_eV);
        }

        double integral = 0.0;
        for (size_t i = 0; i < N; ++i) {
            integral += raw_f(E_eV[i]) * dE_eV[i];
        }
        norm_factor_ = (integral > 0.0) ? (1.0 / integral) : 1.0;
    }

    double normalization_factor() const { return norm_factor_; }
    double generalized_power_theta_ev() const { return generalized_power_theta_ev_; }

private:
    double raw_f(double energy_ev) const {
        if (energy_ev <= 0.0) return 0.0;
        if (is_generalized_power()) {
            return raw_generalized_power(energy_ev, generalized_power_theta_ev_);
        }
        if (is_bi_maxwellian()) {
            return raw_bi_maxwellian(energy_ev);
        }
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

    bool is_generalized_power() const { return type_ == "generalized_power"; }
    bool is_bi_maxwellian() const { return type_ == "bi_maxwellian"; }

    double raw_generalized_power(double energy_ev, double theta_ev) const {
        if (energy_ev <= 0.0 || theta_ev <= 0.0 || power_p_ <= 0.0) return 0.0;
        const double x = energy_ev / theta_ev;
        return std::sqrt(energy_ev) * std::exp(-std::pow(x, power_p_));
    }

    double maxwellian_energy_component(double energy_ev, double temperature_ev, double fraction) const {
        if (energy_ev <= 0.0 || temperature_ev <= 0.0 || fraction <= 0.0) return 0.0;
        const double sqrtE = std::sqrt(energy_ev);
        const double T32 = std::pow(temperature_ev, 1.5);
        return fraction * (TWO_OVER_SQRT_PI / T32) * sqrtE * std::exp(-energy_ev / temperature_ev);
    }

    double raw_bi_maxwellian(double energy_ev) const {
        const double hot_fraction = std::clamp(hot_fraction_, 0.0, 1.0);
        const double bulk_fraction = 1.0 - hot_fraction;
        const double temperature_factor = std::max(hot_temperature_factor_, 1.0);
        const double bulk_temperature = std::max(
            target_temperature_ev_ / std::max(1.0 + hot_fraction * (temperature_factor - 1.0), 1e-12),
            1e-12
        );
        const double hot_temperature = temperature_factor * bulk_temperature;
        return maxwellian_energy_component(energy_ev, bulk_temperature, bulk_fraction) +
               maxwellian_energy_component(energy_ev, hot_temperature, hot_fraction);
    }

    double generalized_power_mean_energy(const std::vector<double>& E_eV,
                                         const std::vector<double>& dE_eV,
                                         double theta_ev) const {
        double integral = 0.0;
        double weighted_energy = 0.0;
        const size_t N = std::min(E_eV.size(), dE_eV.size());
        for (size_t i = 0; i < N; ++i) {
            const double weight = dE_eV[i];
            if (weight <= 0.0) continue;
            const double raw = raw_generalized_power(E_eV[i], theta_ev);
            integral += raw * weight;
            weighted_energy += E_eV[i] * raw * weight;
        }
        if (integral <= 0.0) return 0.0;
        return weighted_energy / integral;
    }

    double solve_generalized_power_theta(const std::vector<double>& E_eV,
                                         const std::vector<double>& dE_eV) const {
        const double target_mean_energy_ev = 1.5 * std::max(target_temperature_ev_, 1e-12);
        if (target_mean_energy_ev <= 0.0 || power_p_ <= 0.0) {
            return std::max(target_temperature_ev_, 1.0);
        }

        double low = std::max(1e-8, target_temperature_ev_ * 1e-3);
        double high = std::max(target_temperature_ev_, 1e-6);
        double mean_low = generalized_power_mean_energy(E_eV, dE_eV, low);
        double mean_high = generalized_power_mean_energy(E_eV, dE_eV, high);

        while (mean_low > target_mean_energy_ev && low > 1e-12) {
            low *= 0.5;
            mean_low = generalized_power_mean_energy(E_eV, dE_eV, low);
        }
        while (mean_high < target_mean_energy_ev && high < 1e6) {
            high *= 2.0;
            mean_high = generalized_power_mean_energy(E_eV, dE_eV, high);
        }

        if (mean_high < target_mean_energy_ev) {
            return high;
        }

        for (int iter = 0; iter < 80; ++iter) {
            const double mid = 0.5 * (low + high);
            const double mean_mid = generalized_power_mean_energy(E_eV, dE_eV, mid);
            if (mean_mid < target_mean_energy_ev) {
                low = mid;
            } else {
                high = mid;
            }
        }
        return 0.5 * (low + high);
    }

    std::string type_ = "maxwellian";
    double power_p_ = 1.0;
    double hot_fraction_ = 0.0;
    double hot_temperature_factor_ = 2.5;
    double target_temperature_ev_ = 0.0;
    double generalized_power_theta_ev_ = 0.0;
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
