#pragma once

#include "../base/ProcessBase.hpp"
#include "../base/Types.hpp"
#include "../state/PlasmaState.hpp"
#include "../physics/EEDF.hpp"
#include "ProcessMath.hpp"
#include <array>
#include <algorithm>
#include <cmath>
#include <vector>

namespace crm_detail {

// Atomic excitation process using cross-section fits from the data file.
class AtomicExcitationProcess final : public dcr::ProcessBase {
public:
    AtomicExcitationProcess(dcr::base::Index from,
                            dcr::base::Index to,
                            double threshold_ev,
                            double g_from,
                            double g_to,
                            double e_from_ev,
                            double e_to_ev,
                            double oscillator_strength,
                            int flag,
                            const std::array<double, 8>& params)
        : from_(from),
          to_(to),
          threshold_ev_(threshold_ev),
          g_from_(g_from),
          g_to_(g_to),
          e_from_ev_(e_from_ev),
          e_to_ev_(e_to_ev),
          oscillator_strength_(oscillator_strength),
          flag_(flag),
          params_(params) {}

    void apply(const dcr::state::PlasmaState& plasma,
               const EEDFGridView& grid,
               const Eigen::VectorXd& population,
               Eigen::MatrixXd& R,
               ReactionAccumulator* accumulator) const override {
        (void)plasma;
        (void)grid;
        (void)population;
        (void)accumulator;
        if (from_ < 0 || to_ < 0) return;
        if (from_ >= R.rows() || from_ >= R.cols()) return;
        if (to_ >= R.rows() || to_ >= R.cols()) return;
        if (!grid.valid()) return;

        const double k_up = integrate_excitation(grid);
        const double r_up = k_up * plasma.electron_density_cm3();
        if (r_up > 0.0) {
            R(to_, from_) += r_up;
            R(from_, from_) -= r_up;
        }

        const double k_dn = integrate_deexcitation(grid);
        const double r_dn = k_dn * plasma.electron_density_cm3();
        if (r_dn > 0.0) {
            R(from_, to_) += r_dn;
            R(to_, to_) -= r_dn;
        }

        // Spontaneous emission (radiative) term.
        const double delta_e = std::max({1e-6, e_to_ev_ - e_from_ev_, threshold_ev_});
        if (delta_e > 0.0 && oscillator_strength_ > 0.0 && g_to_ > 0.0 && g_from_ > 0.0) {
            const double rate_se = oscillator_strength_ * SE_RATE_CONST * delta_e * delta_e * (g_from_ / g_to_);
            if (rate_se > 0.0) {
                R(from_, to_) += rate_se;
                R(to_, to_) -= rate_se;
            }
        }
    }

private:
    // Integrate excitation rate coefficient: ∫ σ(E) v(E) f(E) dE.
    double integrate_excitation(const EEDFGridView& grid) const {
        const double fallback_gap = std::max(0.0, e_to_ev_ - e_from_ev_);
        const double Eth = std::max({1e-6, threshold_ev_, fallback_gap});
        if (oscillator_strength_ <= 0.0) return 0.0;
        double acc = 0.0;
        for (size_t i = 0; i < grid.size(); ++i) {
            const double E = grid.energy(i);
            if (E <= Eth) continue;
            const double sigma = cross_section(E);
            if (sigma <= 0.0) continue;
            const double val = grid.value_at(i);
            if (val <= 0.0) continue;
            acc += sigma * electron_speed(E) * val * grid.weight(i);
        }
        return acc;
    }

    // Detailed-balance de-excitation coefficient using excitation cross-section.
    double integrate_deexcitation(const EEDFGridView& grid) const {
        if (g_to_ <= 0.0 || oscillator_strength_ <= 0.0) return 0.0;
        const double fallback_gap = std::max(0.0, e_to_ev_ - e_from_ev_);
        const double Eth = std::max({1e-6, threshold_ev_, fallback_gap});
        double acc = 0.0;
        for (size_t i = 0; i < grid.size(); ++i) {
            const double E = grid.energy(i);
            if (E < 0.0) continue;
            const double sigma = deexcitation_cross_section(E, Eth);
            if (sigma <= 0.0) continue;
            const double val = grid.value_at(i);
            if (val <= 0.0) continue;
            acc += sigma * electron_speed(E) * val * grid.weight(i);
        }
        return acc * (g_from_ / std::max(g_to_, 1.0));
    }

    double deexcitation_cross_section(double E, double Eth) const {
        if (E <= 0.0) return 0.0;
        const double sigma_ex = cross_section(E + Eth);
        if (sigma_ex <= 0.0) return 0.0;
        const double factor = (E + Eth) / std::max(E, 1e-12);
        return sigma_ex * factor;
    }

    // Bound-bound excitation cross-section (fit form from the data file).
    double cross_section(double E) const {
        if (oscillator_strength_ <= 0.0) return 0.0;
        const double fallback_gap = std::max(0.0, e_to_ev_ - e_from_ev_);
        const double Eth = std::max({1e-6, threshold_ev_, fallback_gap});
        const double X = std::max(E / Eth, 1e-6);

        double gaunt = 0.0;
        if (flag_ == 99) {
            const double Y = X + params_[4];
            if (Y <= 0.0) return 0.0;
            gaunt = params_[0] * std::log(X) + params_[1] + params_[2] / Y + params_[3] / (Y * Y);
        } else {
            gaunt = gaunt_factor(X);
        }

        double sigma = (EX_CONST * oscillator_strength_ / (Eth * Eth)) * gaunt / X;

        // Relativistic correction (legacy fit).
        const double gamma = 1.0 + (X * Eth) * REL_GAMMA_COEFF;
        const double beta2 = 1.0 - 1.0 / (gamma * gamma);
        if (beta2 > 0.0 && beta2 < 1.0) {
            const double rel_bracket = std::log((beta2 / (1.0 - beta2)) * (REL_LOG_FACTOR / Eth)) - beta2;
            const double sigma_rel = REL_SIGMA_FACTOR * (oscillator_strength_ / (Eth * std::max(beta2, 1e-12))) * rel_bracket;
            const double sfact = 1.0 / (1.0 + std::exp(REL_SWITCH_COEFF * (REL_SWITCH_THRESHOLD - X * Eth)));
            sigma = sigma * (1.0 - sfact) + sfact * sigma_rel;
        }

        return (sigma > 0.0) ? sigma : 0.0;
    }

    double gaunt_factor(double X) const {
        if (X > 3.0746418) return 0.276 * std::log(X);
        return 0.31;
    }

    dcr::base::Index from_ = -1;
    dcr::base::Index to_ = -1;
    double threshold_ev_ = 0.0;
    double g_from_ = 0.0;
    double g_to_ = 0.0;
    double e_from_ev_ = 0.0;
    double e_to_ev_ = 0.0;
    double oscillator_strength_ = 0.0;
    int flag_ = 0;
    std::array<double, 8> params_{};

    static constexpr double EX_CONST = 2.36304887970842e-13;
    static constexpr double REL_GAMMA_COEFF = 1.95695119687198e-6;
    static constexpr double REL_LOG_FACTOR = 255499.47326188;
    static constexpr double REL_SIGMA_FACTOR = 2.54954953927045e-19;
    static constexpr double REL_SWITCH_COEFF = 1.0e-5;
    static constexpr double REL_SWITCH_THRESHOLD = 1.0e5;
    static constexpr double SE_RATE_CONST = 4.34327e7;
};

// Atomic photo-recombination process (Einstein-Milne from photo cross-sections).
class AtomicPhotoProcess final : public dcr::ProcessBase {
public:
    AtomicPhotoProcess(dcr::base::Index from,
                       dcr::base::Index to,
                       double threshold_ev,
                       int flag,
                       const std::array<double, 7>& params,
                       double g_lower,
                       double g_upper)
        : from_(from),
          to_(to),
          threshold_ev_(threshold_ev),
          flag_(flag),
          params_(params),
          g_lower_(g_lower),
          g_upper_(g_upper) {
        if (flag_ == 98) {
            const double logx = std::log(std::max(params_[6], 1e-12) / std::max(threshold_ev_, 1e-12));
            const double y = params_[0]
                           + logx * (params_[1]
                           + logx * (params_[2]
                           + logx * params_[3]));
            cached_cthres_ = 1.0e-18 * params_[4] * std::exp(y) * (13.606 / std::max(params_[5], 1e-12));
        }
    }

    void apply(const dcr::state::PlasmaState& plasma,
               const EEDFGridView& grid,
               const Eigen::VectorXd& population,
               Eigen::MatrixXd& R,
               ReactionAccumulator* accumulator) const override {
        (void)population;
        (void)accumulator;
        if (from_ < 0 || to_ < 0) return;
        if (!grid.valid()) return;

        const double rr_coeff = integrate_rr(grid);
        const double rr_rate = rr_coeff * plasma.electron_density_cm3();
        if (rr_rate > 0.0) {
            R(from_, to_) += rr_rate;
            R(to_, to_) -= rr_rate;
        }
    }

private:
    // Radiative recombination coefficient.
    double integrate_rr(const EEDFGridView& grid) const {
        const double Eth = std::max(1e-6, threshold_ev_);
        double acc = 0.0;
        for (size_t i = 0; i < grid.size(); ++i) {
            const double E = grid.energy(i);
            if (E < 0.0) continue;
            const double X = E / Eth + 1.0;
            const double sigma_rr = recombination_cross_section(X, Eth);
            if (sigma_rr <= 0.0) continue;
            const double val = grid.value_at(i);
            if (val <= 0.0) continue;
            acc += sigma_rr * SQRT_2E_M * E * val * grid.weight(i);
        }
        return acc;
    }

    double recombination_cross_section(double X, double Eth) const {
        if (X <= 1.0) return 0.0;
        const double sigma_photo = photo_cross_section(X, Eth);
        const double U = Eth * (X - 1.0);
        const double sigma_rr = (g_lower_ / std::max(g_upper_, 1.0))
                              * (U + Eth) * (U + Eth)
                              / std::max(U, 1e-12)
                              / EINSTEIN_MILNE_FACTOR
                              * sigma_photo;
        return std::max(sigma_rr, 0.0);
    }

    double photo_cross_section(double X, double Eth) const {
        if (flag_ == 99) {
            return KRAMER * std::pow(Eth, 2.5) * params_[0] / std::max(params_[1], 1e-12)
                   / std::pow(X * Eth, 3.0);
        }
        if (flag_ == 98) {
            const double logX = std::log(X);
            double Y = params_[0] + logX * (params_[1] + logX * (params_[2] + logX * params_[3]));
            double sigma = 1.0e-18 * params_[4] * std::exp(Y) * (13.606 / std::max(params_[5], 1e-12));
            if (X > std::max(params_[6], 1.0) / Eth) {
                const double ratio = params_[6] / Eth;
                sigma = cached_cthres_ * std::pow(ratio, 3.0) / std::pow(X, 3.0);
            }
            return sigma;
        }
        return (params_[0] + params_[1] / X + params_[2] / (X * X))
             * std::pow(X, (-3.5 - params_[3]));
    }

    dcr::base::Index from_ = -1;
    dcr::base::Index to_ = -1;
    double threshold_ev_ = 0.0;
    int flag_ = 0;
    std::array<double, 7> params_{};
    double g_lower_ = 1.0;
    double g_upper_ = 1.0;
    double cached_cthres_ = 0.0;

    static constexpr double SQRT_2E_M = 5.93097e7;
    static constexpr double EINSTEIN_MILNE_FACTOR = 1.02e6;
    static constexpr double KRAMER = 2.9165e-17;
};

// Atomic electron-impact ionization + three-body recombination.
class AtomicIonizationProcess final : public dcr::ProcessBase {
public:
    AtomicIonizationProcess(dcr::base::Index from,
                            dcr::base::Index to,
                            double threshold_ev,
                            int flag,
                            const std::array<double, 4>& params,
                            double g_lower,
                            double g_upper)
        : from_(from),
          to_(to),
          threshold_ev_(threshold_ev),
          flag_(flag),
          params_(params),
          g_lower_(g_lower),
          g_upper_(g_upper) {}

    void apply(const dcr::state::PlasmaState& plasma,
               const EEDFGridView& grid,
               const Eigen::VectorXd& population,
               Eigen::MatrixXd& R,
               ReactionAccumulator* accumulator) const override {
        (void)population;
        (void)accumulator;
        if (from_ < 0 || to_ < 0) return;
        if (!grid.valid()) return;

        const double k_eii = integrate_eii(grid);
        const double r_eii = k_eii * plasma.electron_density_cm3();
        if (r_eii > 0.0) {
            R(to_, from_) += r_eii;
            R(from_, from_) -= r_eii;
        }

        const double k_tbr = integrate_tbr(grid);
        const double n_e = plasma.electron_density_cm3();
        const double r_tbr = k_tbr * n_e * n_e;
        if (r_tbr > 0.0) {
            R(from_, to_) += r_tbr;
            R(to_, to_) -= r_tbr;
        }
    }

private:
    double integrate_eii(const EEDFGridView& grid) const {
        const double Eth = std::max(1e-6, threshold_ev_);
        double acc = 0.0;
        for (size_t i = 0; i < grid.size(); ++i) {
            const double E = grid.energy(i);
            if (E <= Eth) continue;
            const double sigma = ionization_cross_section(E / Eth);
            if (sigma <= 0.0) continue;
            const double val = grid.value_at(i);
            if (val <= 0.0) continue;
            acc += sigma * electron_speed(E) * val * grid.weight(i);
        }
        return acc;
    }

    double integrate_tbr(const EEDFGridView& grid) const {
        if (g_lower_ <= 0.0 || g_upper_ <= 0.0) return 0.0;
        const double Eth = std::max(1e-6, threshold_ev_);
        double acc = 0.0;
        for (size_t i = 0; i < grid.size(); ++i) {
            const double E = grid.energy(i);
            if (E <= Eth) continue;
            const double sigma = ionization_cross_section(E / Eth);
            if (sigma <= 0.0) continue;
            const double conv = pair_convolution(E - Eth, grid);
            if (conv <= 0.0) continue;
            acc += sigma * electron_speed(E) * std::sqrt(E) * conv * grid.weight(i);
        }
        return acc * IONIZATION_CONST_R * (g_lower_ / std::max(g_upper_, 1.0));
    }

    double ionization_cross_section(double X) const {
        if (X <= 1.0) return 0.0;
        if (flag_ == 99) {
            const double beta = get_beta();
            const double W = std::pow(std::log(X), beta / X);
            return IONIZATION_CONST_X * params_[0] * W * std::log(X) / X / (threshold_ev_ * threshold_ev_);
        }
        if (params_[1] < 0.0) {
            const double Y = 1.0 - 1.0 / X;
            const double OM = - params_[1] * std::log(X)
                            + params_[2] * Y * Y
                            + params_[3] * Y / X
                            + params_[0] * Y / (X * X);
            return 3.81e-16 * OM / std::max(g_lower_, 1.0) / threshold_ev_ / X;
        }
        if (std::abs(params_[1] * params_[2] - 1.0) < 1.0e-11) {
            return 4.5e-14 * params_[0] * std::log(X) / X / threshold_ev_ / threshold_ev_;
        }
        const double CI3BI = 6.51e-14 * params_[1] * params_[0] / (threshold_ev_ * threshold_ev_);
        return CI3BI * std::log(X) / X / (1.0 + params_[2] / (X * X) + params_[3] / X);
    }

    double pair_convolution(double excess_energy, const EEDFGridView& grid) const {
        if (excess_energy <= 0.0) return 0.0;
        const auto& secondary_grid = get_secondary_fraction_grid();
        double sum = 0.0;
        for (const auto& node : secondary_grid) {
            const double frac = node.first;
            const double dfrac = node.second;
            const double e2 = frac * excess_energy;
            const double e3 = excess_energy - e2;
            if (e3 < 0.0) continue;
            const double f2 = grid.eval(e2);
            const double f3 = grid.eval(e3);
            if (f2 <= 0.0 || f3 <= 0.0) continue;
            sum += f2 * f3 * dfrac / std::sqrt(e2) / std::sqrt(e3);
        }
        return 2.0 * sum;
    }

    static const std::vector<std::pair<double, double>>& get_secondary_fraction_grid() {
        static std::vector<std::pair<double, double>> grid;
        if (grid.empty()) {
            grid.reserve(159);
            double frac = 0.0;

            auto push_point = [&](double df) {
                frac += df;
                grid.emplace_back(frac, df);
            };

            const double base1 = 1.0e-4 / 2.0;
            for (int i = 0; i < 10; ++i) push_point(base1);

            for (int i = 10; i < 100; ++i) {
                const double df = (1.0e-4 * (i - 9)) / 2.0;
                push_point(df);
            }

            const double base3 = 1.0e-2 / 2.0;
            for (int i = 100; i < 158; ++i) push_point(base3);

            const double last_frac = 0.5;
            double last_df = last_frac - frac;
            if (last_df <= 0.0) last_df = base3;
            frac += last_df;
            grid.emplace_back(last_frac, last_df);
        }
        return grid;
    }

    double get_beta() const {
        const double z = params_[1] - 1.0;
        return 0.25 * std::sqrt((100.0 * z + 91.0) / (4.0 * z + 3.0)) - 1.25;
    }

    dcr::base::Index from_ = -1;
    dcr::base::Index to_ = -1;
    double threshold_ev_ = 0.0;
    int flag_ = 0;
    std::array<double, 4> params_{};
    double g_lower_ = 1.0;
    double g_upper_ = 1.0;

    static constexpr double IONIZATION_CONST_X = 3.745e-14;
    static constexpr double IONIZATION_CONST_R = 1.4679e-22;
};

// Autoionization: constant rate coefficient from file.
class AtomicAutoionizationProcess final : public dcr::ProcessBase {
public:
    AtomicAutoionizationProcess(dcr::base::Index from,
                                dcr::base::Index to,
                                double rate_coeff)
        : from_(from), to_(to), rate_coeff_(rate_coeff) {}

    void apply(const dcr::state::PlasmaState& plasma,
               const EEDFGridView& grid,
               const Eigen::VectorXd& population,
               Eigen::MatrixXd& R,
               ReactionAccumulator* accumulator) const override {
        (void)plasma;
        (void)grid;
        (void)population;
        (void)accumulator;
        if (from_ < 0 || to_ < 0) return;
        if (rate_coeff_ <= 0.0) return;
        R(to_, from_) += rate_coeff_;
        R(from_, from_) -= rate_coeff_;
    }

private:
    dcr::base::Index from_ = -1;
    dcr::base::Index to_ = -1;
    double rate_coeff_ = 0.0;
};

} // namespace crm_detail
