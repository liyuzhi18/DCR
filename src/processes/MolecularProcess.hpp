#pragma once

#include "../base/ProcessBase.hpp"
#include "../base/Types.hpp"
#include "../state/PlasmaState.hpp"
#include "../physics/EEDF.hpp"
#include "../physics/Quadrature.hpp"
#include "ProcessMath.hpp"
#include <array>
#include <memory>
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace crm_detail {

constexpr double DEFAULT_REDUCED_MASS = 0.5;
constexpr double MCX_REDUCED_MASS = 2.0 / 3.0;
constexpr double BOHR_RADIUS_CM = 5.29e-9;
constexpr double BOHR_RADIUS_CM2 = BOHR_RADIUS_CM * BOHR_RADIUS_CM;
constexpr double EV_MAX_ENERGY = 500.0;
constexpr double MCX_FACTOR = 1.5674707423419676e6;
constexpr double ED_FACTOR = MCX_FACTOR;
constexpr double ED_EB = 0.754;

struct HydrogenDRBranchingSpec {
    bool enabled = false;
    int vibrational_quantum = -1;
    std::array<int, 4> discrete_state_indices{{-1, -1, -1, -1}}; // n = 2..5
    std::vector<int> high_n_state_indices; // n >= 6
};

namespace hydrogen_dr_tables {

constexpr std::array<double, 8> ENERGIES_EV = {0.5, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 10.0};

using ProbRow = std::array<double, 5>;
using ProbTable = std::array<ProbRow, ENERGIES_EV.size()>;

constexpr ProbTable V0 = {{
    ProbRow{1.00, 0.00, 0.00, 0.00, 0.00},
    ProbRow{1.00, 0.00, 0.00, 0.00, 0.00},
    ProbRow{0.84, 0.11, 0.04, 0.01, 0.00},
    ProbRow{0.66, 0.18, 0.10, 0.02, 0.04},
    ProbRow{0.50, 0.24, 0.14, 0.04, 0.08},
    ProbRow{0.31, 0.30, 0.21, 0.06, 0.12},
    ProbRow{0.13, 0.33, 0.27, 0.09, 0.18},
    ProbRow{0.09, 0.32, 0.26, 0.11, 0.22}
}};

constexpr ProbTable V1 = {{
    ProbRow{1.00, 0.00, 0.00, 0.00, 0.00},
    ProbRow{0.85, 0.15, 0.00, 0.00, 0.00},
    ProbRow{0.57, 0.20, 0.11, 0.06, 0.06},
    ProbRow{0.23, 0.29, 0.18, 0.10, 0.20},
    ProbRow{0.15, 0.30, 0.24, 0.10, 0.21},
    ProbRow{0.11, 0.31, 0.26, 0.11, 0.22},
    ProbRow{0.07, 0.32, 0.28, 0.11, 0.22},
    ProbRow{0.07, 0.32, 0.28, 0.11, 0.22}
}};

constexpr ProbTable V2 = {{
    ProbRow{1.00, 0.00, 0.00, 0.00, 0.00},
    ProbRow{0.65, 0.35, 0.00, 0.00, 0.00},
    ProbRow{0.25, 0.32, 0.19, 0.08, 0.16},
    ProbRow{0.14, 0.32, 0.23, 0.10, 0.21},
    ProbRow{0.08, 0.32, 0.27, 0.11, 0.22},
    ProbRow{0.07, 0.31, 0.27, 0.12, 0.23},
    ProbRow{0.06, 0.31, 0.27, 0.12, 0.24},
    ProbRow{0.06, 0.31, 0.27, 0.12, 0.24}
}};

constexpr ProbTable V3 = {{
    ProbRow{0.57, 0.43, 0.00, 0.00, 0.00},
    ProbRow{0.34, 0.54, 0.10, 0.00, 0.00},
    ProbRow{0.22, 0.42, 0.21, 0.07, 0.10},
    ProbRow{0.15, 0.38, 0.26, 0.09, 0.14},
    ProbRow{0.10, 0.34, 0.29, 0.10, 0.18},
    ProbRow{0.07, 0.33, 0.29, 0.11, 0.21},
    ProbRow{0.06, 0.32, 0.29, 0.11, 0.22},
    ProbRow{0.06, 0.32, 0.29, 0.11, 0.22}
}};

constexpr ProbTable V4 = {{
    ProbRow{0.32, 0.68, 0.00, 0.00, 0.00},
    ProbRow{0.22, 0.54, 0.24, 0.00, 0.00},
    ProbRow{0.14, 0.40, 0.25, 0.07, 0.14},
    ProbRow{0.08, 0.34, 0.28, 0.09, 0.18},
    ProbRow{0.07, 0.33, 0.28, 0.11, 0.21},
    ProbRow{0.07, 0.32, 0.27, 0.11, 0.22},
    ProbRow{0.06, 0.31, 0.27, 0.12, 0.24},
    ProbRow{0.06, 0.31, 0.27, 0.12, 0.24}
}};

constexpr ProbTable V5 = {{
    ProbRow{0.24, 0.74, 0.02, 0.00, 0.00},
    ProbRow{0.16, 0.55, 0.24, 0.02, 0.03},
    ProbRow{0.10, 0.38, 0.25, 0.09, 0.18},
    ProbRow{0.08, 0.34, 0.25, 0.11, 0.22},
    ProbRow{0.07, 0.33, 0.24, 0.12, 0.24},
    ProbRow{0.07, 0.32, 0.24, 0.12, 0.25},
    ProbRow{0.08, 0.32, 0.23, 0.12, 0.25},
    ProbRow{0.08, 0.32, 0.23, 0.12, 0.25}
}};

constexpr ProbTable V10 = {{
    ProbRow{0.08, 0.43, 0.25, 0.08, 0.16},
    ProbRow{0.07, 0.36, 0.24, 0.11, 0.22},
    ProbRow{0.07, 0.33, 0.24, 0.12, 0.24},
    ProbRow{0.07, 0.33, 0.24, 0.12, 0.24},
    ProbRow{0.07, 0.33, 0.24, 0.12, 0.24},
    ProbRow{0.07, 0.32, 0.24, 0.12, 0.25},
    ProbRow{0.08, 0.32, 0.23, 0.12, 0.25},
    ProbRow{0.08, 0.32, 0.23, 0.12, 0.25}
}};

inline const ProbTable& table_for_v(int v) {
    if (v <= 0) return V0;
    switch (v) {
        case 1: return V1;
        case 2: return V2;
        case 3: return V3;
        case 4: return V4;
        case 5: return V5;
        default:
            return V10; // use v=10 table for v>=6
    }
}

inline ProbRow interpolate(const ProbTable& table, double energy_ev) {
    const double min_E = ENERGIES_EV.front();
    const double max_E = ENERGIES_EV.back();
    double clamped_E = energy_ev;
    if (clamped_E < min_E) clamped_E = min_E;
    if (clamped_E > max_E) clamped_E = max_E;

    if (clamped_E <= ENERGIES_EV.front()) return table.front();
    if (clamped_E >= ENERGIES_EV.back()) return table.back();

    size_t upper = 1;
    for (; upper < ENERGIES_EV.size(); ++upper) {
        if (clamped_E <= ENERGIES_EV[upper]) break;
    }
    const size_t lower = upper - 1;
    const double span = ENERGIES_EV[upper] - ENERGIES_EV[lower];
    double t = 0.0;
    if (span > 0.0) {
        t = (clamped_E - ENERGIES_EV[lower]) / span;
    }

    ProbRow result{};
    for (size_t i = 0; i < result.size(); ++i) {
        result[i] = table[lower][i] + t * (table[upper][i] - table[lower][i]);
    }
    return result;
}

inline ProbRow lookup_probabilities(int v, double energy_ev) {
    return interpolate(table_for_v(v), energy_ev);
}

} // namespace hydrogen_dr_tables

struct TabulatedCrossSection {
    std::vector<double> energies;
    std::vector<double> sigmas;

    bool valid() const {
        return energies.size() > 1 && energies.size() == sigmas.size();
    }

    double sample(double E) const {
        if (!valid()) return 0.0;
        if (E <= energies.front()) return sigmas.front();
        if (E >= energies.back()) return sigmas.back();
        auto it = std::lower_bound(energies.begin(), energies.end(), E);
        if (it == energies.end()) return sigmas.back();
        const size_t upper = static_cast<size_t>(it - energies.begin());
        if (upper == 0) return sigmas.front();
        const size_t lower = upper - 1;
        const double e0 = energies[lower];
        const double e1 = energies[upper];
        const double s0 = sigmas[lower];
        const double s1 = sigmas[upper];
        const double span = e1 - e0;
        if (span <= 0.0) return s0;
        const double t = (E - e0) / span;
        return s0 + t * (s1 - s0);
    }
};

inline TabulatedCrossSection load_cross_section_table(const std::string& path) {
    TabulatedCrossSection table;
    std::ifstream in(path);
    if (!in) {
        std::cout << "[MolecularVE] Table not found: " << path << "\n";
        return table;
    }

    std::string line;
    std::vector<std::pair<double, double>> pairs;
    while (std::getline(in, line)) {
        if (line.empty()) continue;
        std::istringstream iss(line);
        std::string token;
        iss >> token;
        if (token.empty()) continue;
        if (token[0] == '#') continue;
        iss.clear();
        iss.str(line);
        double energy = 0.0;
        double sigma = 0.0;
        if (!(iss >> energy >> sigma)) continue;
        pairs.emplace_back(energy, sigma);
    }

    if (pairs.empty()) return table;
    std::sort(pairs.begin(), pairs.end(),
              [](const auto& a, const auto& b) { return a.first < b.first; });
    table.energies.reserve(pairs.size());
    table.sigmas.reserve(pairs.size());
    for (const auto& p : pairs) {
        table.energies.push_back(p.first);
        table.sigmas.push_back(p.second);
    }
    return table;
}

class MolecularMIDEProcess;

class MolecularMIDEPool {
public:
    void add_member(const MolecularMIDEProcess* proc);
    void remove_member(const MolecularMIDEProcess* proc);
    double normalization_factor(double E) const;

private:
    std::vector<const MolecularMIDEProcess*> members_;
};

// Base class for molecular processes (no extra API for now).
class MolecularProcess : public dcr::ProcessBase {
};

// Molecular vibrational excitation (ve) using tabulated cross sections.
class MolecularVEProcess final : public MolecularProcess {
public:
    MolecularVEProcess(dcr::base::Index from,
                       dcr::base::Index to,
                       const std::string& table_path)
        : from_(from), to_(to), table_path_(table_path),
          table_(load_cross_section_table(table_path)) {}

    void apply(const dcr::state::PlasmaState& plasma,
               const EEDFGridView& grid,
               const Eigen::VectorXd& population,
               Eigen::MatrixXd& R,
               ReactionAccumulator* accumulator) const override {
        (void)population;
        (void)accumulator;
        if (from_ < 0 || to_ < 0) return;
        if (!grid.valid()) return;
        if (!table_.valid()) return;

        const double k = integrate(grid);
        const double r_up = k * plasma.electron_density_cm3();
        if (r_up > 0.0) {
            R(to_, from_) += r_up;
            R(from_, from_) -= r_up;
        }
    }

private:
    double integrate(const EEDFGridView& grid) const {
        double acc = 0.0;
        for (size_t i = 0; i < grid.size(); ++i) {
            const double E = grid.energy(i);
            const double sigma = table_.sample(E);
            if (!(sigma > 0.0)) continue;
            const double val = grid.value_at(i);
            if (val <= 0.0) continue;
            acc += sigma * electron_speed(E) * val * grid.weight(i);
        }
        return acc;
    }

    dcr::base::Index from_ = -1;
    dcr::base::Index to_ = -1;
    std::string table_path_;
    TabulatedCrossSection table_;
};

// Molecular excitation process (ev) using analytic fit coefficients.
class MolecularExcitationProcess final : public MolecularProcess {
public:
    MolecularExcitationProcess(dcr::base::Index from,
                               dcr::base::Index to,
                               double threshold_ev,
                               const std::array<double, 6>& params)
        : from_(from), to_(to), threshold_ev_(threshold_ev), params_(params) {}

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
        if (!grid.valid()) return;

        const double k = integrate(grid);
        const double r_up = k * plasma.electron_density_cm3();
        if (r_up > 0.0) {
            R(to_, from_) += r_up;
            R(from_, from_) -= r_up;
        }
    }

private:
    // Integrate excitation rate coefficient: ∫ σ(E) v(E) f(E) dE.
    double integrate(const EEDFGridView& grid) const {
        const double Eth = std::max(1e-6, threshold_ev_);
        double acc = 0.0;
        for (size_t i = 0; i < grid.size(); ++i) {
            const double E = grid.energy(i);
            if (E <= Eth) continue;
            if (E > EV_MAX_ENERGY) continue;
            const double sigma = cross_section(E, Eth);
            if (!(sigma > 0.0)) continue;
            const double val = grid.value_at(i);
            if (val <= 0.0) continue;
            acc += sigma * electron_speed(E) * val * grid.weight(i);
        }
        return acc;
    }

    // Analytic excitation cross-section fit (Janev-style form).
    double cross_section(double E, double Eth) const {
        if (E <= Eth || Eth <= 0.0) return 0.0;
        const double X = E / Eth;
        if (X <= 1.0) return 0.0;
        const double invX = 1.0 / X;
        const double prefactor = (X - 1.0) / X;
        if (!(prefactor > 0.0)) return 0.0;

        double nested = params_[5];
        for (int idx = 4; idx >= 1; --idx) {
            nested = params_[idx] + invX * nested;
        }

        const double term_log = params_[0] * params_[0] * std::log(X) * invX;
        const double term_poly = invX * nested;
        double sigma = prefactor * (term_log + term_poly);
        sigma = std::abs(sigma) * BOHR_RADIUS_CM2;
        return (sigma > 0.0) ? sigma : 0.0;
    }

    dcr::base::Index from_ = -1;
    dcr::base::Index to_ = -1;
    double threshold_ev_ = 0.0;
    std::array<double, 6> params_{};
};

// Molecular ionization process (mi) using a simple semi-empirical fit.
class MolecularMIProcess final : public MolecularProcess {
public:
    MolecularMIProcess(dcr::base::Index from,
                       dcr::base::Index to,
                       double threshold_ev,
                       const std::array<double, 7>& params)
        : from_(from), to_(to), threshold_ev_(threshold_ev), params_(params) {}

    void apply(const dcr::state::PlasmaState& plasma,
               const EEDFGridView& grid,
               const Eigen::VectorXd& population,
               Eigen::MatrixXd& R,
               ReactionAccumulator* accumulator) const override {
        (void)population;
        (void)accumulator;
        if (from_ < 0 || to_ < 0) return;
        if (!grid.valid()) return;

        const double k = integrate(grid);
        const double r = k * plasma.electron_density_cm3();
        if (r <= 0.0) return;

        // Franck-Condon-like branching over H2+ vibrational ladder (v'=0..18).
        const int n_rows = static_cast<int>(R.rows());
        for (size_t i = 0; i < MI_FRACTIONS.size(); ++i) {
            const int target = to_ + static_cast<int>(i);
            if (target < 0 || target >= n_rows) break;
            const double frac = MI_FRACTIONS[i];
            if (frac <= 0.0) continue;
            const double weight = frac / MI_FRAC_NORMALISATION;
            R(target, from_) += r * weight;
        }
        R(from_, from_) -= r;
    }

private:
    // Integrate ionization rate coefficient: ∫ σ(E) v(E) f(E) dE.
    double integrate(const EEDFGridView& grid) const {
        const double Eth = std::max(1e-6, threshold_ev_);
        double acc = 0.0;
        for (size_t i = 0; i < grid.size(); ++i) {
            const double E = grid.energy(i);
            if (E <= Eth) continue;
            const double X = E / Eth;
            const double sigma = cross_section(X);
            if (sigma <= 0.0) continue;
            const double val = grid.value_at(i);
            if (val <= 0.0) continue;
            acc += sigma * electron_speed(E) * val * grid.weight(i);
        }
        return acc;
    }

    // Legacy 7-parameter MI fit used in flychk-flow-new.
    double cross_section(double X) const {
        if (X <= 1.0) return 0.0;
        const double invX = 1.0 / X;
        const double one_minus = 1.0 - invX;
        if (one_minus <= 0.0) return 0.0;

        double sum = 0.0;
        for (int k = 1; k <= 5; ++k) {
            sum += params_[k] * std::pow(one_minus, static_cast<double>(k));
        }

        double sigma = invX * (params_[0] * std::log(X) + sum);
        sigma *= std::exp(-std::pow(invX, params_[6]));
        sigma *= BOHR_RADIUS_CM2;
        return (sigma > 0.0) ? sigma : 0.0;
    }

    dcr::base::Index from_ = -1;
    dcr::base::Index to_ = -1;
    double threshold_ev_ = 0.0;
    std::array<double, 7> params_{};

    // Hydrogen MI branching fractions (v'=0..18), reused from legacy model.
    // Normalization is handled dynamically over available target states.
    static constexpr std::array<double, 19> MI_FRACTIONS = {
        0.119, 0.190, 0.188, 0.152, 0.125,
        0.075, 0.052, 0.037, 0.024, 0.016,
        0.0117, 0.0082, 0.0057, 0.00374, 0.00258,
        0.00175, 0.00109, 0.00056, 0.00012
    };
    static constexpr double MI_FRAC_NORMALISATION = 1.01344;
};

// Molecular dissociative recombination (dr) using analytic fit coefficients.
class MolecularDRProcess final : public MolecularProcess {
public:
    MolecularDRProcess(dcr::base::Index from,
                       dcr::base::Index product_a,
                       dcr::base::Index product_b,
                       const std::array<double, 6>& params,
                       const HydrogenDRBranchingSpec& branching = {})
        : from_(from),
          product_a_(product_a),
          product_b_(product_b),
          params_(params),
          hydrogen_branching_(branching) {}

    void apply(const dcr::state::PlasmaState& plasma,
               const EEDFGridView& grid,
               const Eigen::VectorXd& population,
               Eigen::MatrixXd& R,
               ReactionAccumulator* accumulator) const override {
        (void)population;
        (void)accumulator;
        if (from_ < 0) return;
        if (product_a_ < 0 || product_b_ < 0) return;
        if (!grid.valid()) return;

        const double k = integrate(grid);
        const double rate = k * plasma.electron_density_cm3();
        if (rate <= 0.0) return;

        R(product_a_, from_) += rate;
        R(from_, from_) -= rate;

        if (hydrogen_branching_.enabled) {
            apply_hydrogen_branching(plasma, R, rate);
            return;
        }

        R(product_b_, from_) += rate;
    }

private:
    void apply_hydrogen_branching(const dcr::state::PlasmaState& plasma,
                                  Eigen::MatrixXd& R,
                                  double total_rate) const {
        if (product_a_ < 0) return;
        if (hydrogen_branching_.high_n_state_indices.empty()) return;
        if (hydrogen_branching_.vibrational_quantum < 0) return;

        const auto probs = hydrogen_dr_tables::lookup_probabilities(
            hydrogen_branching_.vibrational_quantum,
            plasma.electron_temperature_ev()
        );

        auto add_branch = [&](int target_idx, double probability) {
            if (target_idx < 0) return;
            if (target_idx >= R.rows()) return;
            if (!(probability > 0.0)) return;
            const double coeff = total_rate * probability;
            if (coeff <= 0.0) return;
            R(target_idx, from_) += coeff;
        };

        for (size_t i = 0; i < hydrogen_branching_.discrete_state_indices.size(); ++i) {
            add_branch(hydrogen_branching_.discrete_state_indices[i], probs[i]);
        }

        const double tail_prob = probs.back();
        if (!(tail_prob > 0.0)) return;
        const size_t tail_states = hydrogen_branching_.high_n_state_indices.size();
        if (tail_states == 0) return;
        const double share = tail_prob / static_cast<double>(tail_states);
        for (int idx : hydrogen_branching_.high_n_state_indices) {
            add_branch(idx, share);
        }
    }

    double integrate(const EEDFGridView& grid) const {
        double acc = 0.0;
        for (size_t i = 0; i < grid.size(); ++i) {
            const double E = grid.energy(i);
            if (!(E > 0.0)) continue;
            const double sigma = cross_section(E);
            if (sigma <= 0.0) continue;
            const double val = grid.value_at(i);
            if (val <= 0.0) continue;
            acc += sigma * electron_speed(E) * val * grid.weight(i);
        }
        return acc;
    }

    double cross_section(double E) const {
        if (E <= 0.0) return 0.0;
        double term_pow = std::pow(E, 0.665);
        if (term_pow <= 0.0) term_pow = 1e-12;
        double denom = term_pow * (1.0 + 1.1 * std::pow(E, 0.512) + params_[1] * std::pow(E, params_[2]));
        if (denom <= 0.0) return 0.0;
        const double gaussian = params_[3] * std::exp(-params_[4] * (E - params_[5]) * (E - params_[5]));
        const double sigma = params_[0] * (1.0 / denom + gaussian) * 1.0e-16;
        return (sigma > 0.0) ? sigma : 0.0;
    }

    dcr::base::Index from_ = -1;
    dcr::base::Index product_a_ = -1;
    dcr::base::Index product_b_ = -1;
    std::array<double, 6> params_{};
    HydrogenDRBranchingSpec hydrogen_branching_;
};

// Molecular radiative attachment (ra) with EEDF integration.
class MolecularRAProcess final : public MolecularProcess {
public:
    MolecularRAProcess(dcr::base::Index from,
                       dcr::base::Index to,
                       double coeff)
        : from_(from), to_(to), coeff_(coeff) {}

    void apply(const dcr::state::PlasmaState& plasma,
               const EEDFGridView& grid,
               const Eigen::VectorXd& population,
               Eigen::MatrixXd& R,
               ReactionAccumulator* accumulator) const override {
        (void)population;
        (void)accumulator;
        if (from_ < 0 || to_ < 0) return;
        if (!grid.valid()) return;

        const double k = integrate(grid);
        if (k <= 0.0) return;

        const double rate = (k / 10000.0) * plasma.electron_density_cm3();
        if (rate <= 0.0) return;

        R(to_, from_) += rate;
        R(from_, from_) -= rate;
    }

private:
    double integrate(const EEDFGridView& grid) const {
        double acc = 0.0;
        for (size_t i = 0; i < grid.size(); ++i) {
            const double E = grid.energy(i);
            if (E > RA_EMAX) break;
            if (E <= 0.0) continue;
            const double sigma = coeff_ * std::sqrt(E) / (ED_EB + E) * 1.0e-18;
            if (!(sigma > 0.0)) continue;
            const double val = grid.value_at(i);
            if (val <= 0.0) continue;
            acc += sigma * electron_speed(E) * val * grid.weight(i);
        }
        return acc;
    }

    dcr::base::Index from_ = -1;
    dcr::base::Index to_ = -1;
    double coeff_ = 0.0;

    static constexpr double RA_EMAX = 10.0;
};

// Molecular dissociative attachment (da) fit.
class MolecularDAProcess final : public MolecularProcess {
public:
    MolecularDAProcess(dcr::base::Index from,
                       dcr::base::Index product_a,
                       dcr::base::Index product_b,
                       double peak_energy,
                       double scale)
        : from_(from),
          product_a_(product_a),
          product_b_(product_b),
          peak_energy_(peak_energy),
          scale_(scale) {}

    void apply(const dcr::state::PlasmaState& plasma,
               const EEDFGridView& grid,
               const Eigen::VectorXd& population,
               Eigen::MatrixXd& R,
               ReactionAccumulator* accumulator) const override {
        (void)population;
        (void)accumulator;
        if (from_ < 0) return;
        if (product_a_ < 0 || product_b_ < 0) return;
        if (!grid.valid()) return;

        const double k = integrate(grid);
        const double rate = k * plasma.electron_density_cm3();
        if (rate <= 0.0) return;

        R(product_a_, from_) += rate;
        R(product_b_, from_) += rate;
        R(from_, from_) -= rate;
    }

private:
    double integrate(const EEDFGridView& grid) const {
        double acc = 0.0;
        for (size_t i = 0; i < grid.size(); ++i) {
            const double E = grid.energy(i);
            if (E <= 0.0) continue;
            const double sigma = scale_ * std::exp(-(E - peak_energy_) / 0.45) * 1.0e-16;
            if (!(sigma > 0.0)) continue;
            const double val = grid.value_at(i);
            if (val <= 0.0) continue;
            acc += sigma * electron_speed(E) * val * grid.weight(i);
        }
        return acc;
    }

    dcr::base::Index from_ = -1;
    dcr::base::Index product_a_ = -1;
    dcr::base::Index product_b_ = -1;
    double peak_energy_ = 0.0;
    double scale_ = 0.0;
};

// Molecular dissociation (de) using a temperature fit (no EEDF needed).
class MolecularDEProcess final : public MolecularProcess {
public:
    MolecularDEProcess(dcr::base::Index from,
                       dcr::base::Index product_a,
                       dcr::base::Index product_b,
                       double threshold_ev,
                       const std::array<double, 6>& params)
        : from_(from),
          product_a_(product_a),
          product_b_(product_b),
          threshold_ev_(threshold_ev),
          params_(params) {}

    void apply(const dcr::state::PlasmaState& plasma,
               const EEDFGridView& grid,
               const Eigen::VectorXd& population,
               Eigen::MatrixXd& R,
               ReactionAccumulator* accumulator) const override {
        (void)grid;
        (void)population;
        (void)accumulator;
        if (from_ < 0) return;
        if (product_a_ < 0 || product_b_ < 0) return;

        const double k = rate_coefficient(plasma.electron_temperature_ev());
        const double rate = k * plasma.electron_density_cm3();
        if (rate <= 0.0) return;

        R(product_a_, from_) += rate;
        R(product_b_, from_) += rate;
        R(from_, from_) -= rate;
    }

private:
    double rate_coefficient(double Te_ev) const {
        (void)threshold_ev_;
        const double Te = std::max(Te_ev / 1000.0 * 11606.0, 1.0);
        const double logTe = std::log(Te);
        const double sum = params_[0] * std::pow(Te, -params_[1])
                         + params_[2] * std::pow(Te, -params_[3])
                         + params_[4] * std::exp(-params_[5] * logTe * logTe);
        return std::exp(sum);
    }

    dcr::base::Index from_ = -1;
    dcr::base::Index product_a_ = -1;
    dcr::base::Index product_b_ = -1;
    double threshold_ev_ = 0.0;
    std::array<double, 6> params_{};
};

// Molecular dissociation (ed) with electron- or ion-driven fits.
class MolecularEDProcess final : public MolecularProcess {
public:
    MolecularEDProcess(dcr::base::Index primary,
                       dcr::base::Index partner,
                       std::vector<dcr::base::Index> products,
                       int flag,
                       const std::array<double, 11>& params,
                       const std::vector<QuadraturePoint>* quadrature)
        : primary_(primary),
          partner_(partner),
          products_(std::move(products)),
          flag_(flag),
          params_(params),
          quadrature_(quadrature) {
        if (!products_.empty()) primary_product_ = products_.front();
        if (products_.size() > 1) partner_product_ = products_[1];
    }

    void apply(const dcr::state::PlasmaState& plasma,
               const EEDFGridView& grid,
               const Eigen::VectorXd& population,
               Eigen::MatrixXd& R,
               ReactionAccumulator* accumulator) const override {
        (void)accumulator;
        if (primary_ < 0) return;

        const double rate_coeff = (flag_ == 99) ? integrate_eedf(grid) : integrate_ion(plasma);
        if (rate_coeff <= 0.0) return;

        if (flag_ == 99) {
            if (primary_product_ < 0) return;
            const double coeff = rate_coeff * plasma.electron_density_cm3();
            if (coeff <= 0.0) return;
            R(primary_product_, primary_) += coeff;
            R(primary_, primary_) -= coeff;
            return;
        }

        const double primary_density = density(population, primary_);
        const double partner_density = (partner_ >= 0) ? density(population, partner_) : 0.0;
        if (primary_density <= 0.0 && partner_density <= 0.0) return;

        const double coeff_primary = rate_coeff * std::max(partner_density, 0.0);
        const double coeff_partner = rate_coeff * std::max(primary_density, 0.0);

        switch (flag_) {
            case 98: {
                if (coeff_primary > 0.0) {
                    if (primary_product_ >= 0) R(primary_product_, primary_) += coeff_primary;
                    R(primary_, primary_) -= coeff_primary;
                }
                if (coeff_partner > 0.0) {
                    if (partner_product_ >= 0) R(partner_product_, partner_) += coeff_partner;
                    R(partner_, partner_) -= coeff_partner;
                }
                break;
            }
            case 97:
            case 96: {
                if (primary_product_ >= 0) {
                    const double share_primary = 0.5 * coeff_primary;
                    const double share_partner = 0.5 * coeff_partner;
                    if (share_primary > 0.0) R(primary_product_, primary_) += share_primary;
                    if (share_partner > 0.0) R(primary_product_, partner_) += share_partner;
                }
                if (coeff_primary > 0.0) R(primary_, primary_) -= coeff_primary;
                if (coeff_partner > 0.0) R(partner_, partner_) -= coeff_partner;
                break;
            }
            case 95: {
                if (coeff_partner > 0.0) {
                    if (partner_product_ >= 0) R(partner_product_, partner_) += coeff_partner;
                    R(partner_, partner_) -= coeff_partner;
                }
                break;
            }
            default: {
                if (coeff_primary > 0.0) {
                    if (primary_product_ >= 0) R(primary_product_, primary_) += coeff_primary;
                    R(primary_, primary_) -= coeff_primary;
                }
                break;
            }
        }
    }

private:
    double density(const Eigen::VectorXd& population, int idx) const {
        if (idx < 0 || idx >= population.size()) return 0.0;
        return std::max(population(idx), 0.0);
    }

    double cross_section(double E) const {
        if (E <= 0.0) return 0.0;
        switch (flag_) {
            case 99: {
                if (E <= ED_EB) return 0.0;
                const double log_term = std::log(std::exp(1.0) + params_[1] * E);
                const double exp_term = std::exp(-params_[2] / std::pow(E, params_[3]));
                const double corr = 1.0 - std::pow(ED_EB / E, 2.0);
                const double sigma = params_[0] / E * log_term * exp_term * corr * 1.0e-13;
                return (sigma > 0.0) ? sigma : 0.0;
            }
            case 98: {
                const double logE = std::log(E);
                const double poly = params_[0]
                                  + logE * (params_[1]
                                  + logE * (params_[2]
                                  + logE * (params_[3]
                                  + logE * (params_[4]
                                  + logE * (params_[5]
                                  + logE * (params_[6]
                                  + logE * (params_[7]
                                  + logE * params_[8])))))));
                const double sigma = std::exp(poly);
                return (sigma > 0.0) ? sigma : 0.0;
            }
            case 97: {
                double sigma = params_[0] / std::pow(E, params_[1]);
                sigma /= (1.0 + params_[2] * std::pow(E, params_[3]));
                sigma *= 1.0e-16;
                return (sigma > 0.0) ? sigma : 0.0;
            }
            case 96:
            case 95: {
                const double logE = std::log(E);
                const double poly = params_[0]
                                  + logE * (params_[1]
                                  + logE * (params_[2]
                                  + logE * (params_[3]
                                  + logE * (params_[4]
                                  + logE * (params_[5]
                                  + logE * (params_[6]
                                  + logE * (params_[7]
                                  + logE * params_[8])))))));
                const double sigma = std::exp(poly);
                return (sigma > 0.0) ? sigma : 0.0;
            }
            default:
                return 0.0;
        }
    }

    double integrate_eedf(const EEDFGridView& grid) const {
        if (flag_ != 99 || !grid.valid()) return 0.0;
        double acc = 0.0;
        for (size_t i = 0; i < grid.size(); ++i) {
            const double E = grid.energy(i);
            const double sigma = cross_section(E);
            if (sigma <= 0.0) continue;
            const double val = grid.value_at(i);
            if (val <= 0.0) continue;
            acc += sigma * electron_speed(E) * val * grid.weight(i);
        }
        return acc;
    }

    double integrate_ion(const dcr::state::PlasmaState& plasma) const {
        if (flag_ == 99) return 0.0;
        if (!quadrature_ || quadrature_->empty()) return 0.0;
        const double temp = std::max(plasma.ion_temperature_ev(), 1e-3);
        double acc = 0.0;
        for (const auto& qp : *quadrature_) {
            const double E = qp.point * temp;
            const double sigma = cross_section(E);
            if (sigma <= 0.0) continue;
            acc += sigma * std::sqrt(temp / DEFAULT_REDUCED_MASS) * ED_FACTOR * qp.weight;
        }
        return acc;
    }

    dcr::base::Index primary_ = -1;
    dcr::base::Index partner_ = -1;
    std::vector<dcr::base::Index> products_;
    dcr::base::Index primary_product_ = -1;
    dcr::base::Index partner_product_ = -1;
    int flag_ = 0;
    std::array<double, 11> params_{};
    const std::vector<QuadraturePoint>* quadrature_ = nullptr;
};

// Molecular charge-exchange process (ion temperature integration).
class MolecularMCXProcess final : public MolecularProcess {
public:
    MolecularMCXProcess(dcr::base::Index reactant_a,
                        dcr::base::Index reactant_b,
                        dcr::base::Index product_a,
                        dcr::base::Index product_b,
                        int flag,
                        const std::array<double, 11>& params,
                        double threshold_ev,
                        const std::vector<QuadraturePoint>* quadrature,
                        double reduced_mass,
                        int Z_hint,
                        double high_v_factor = 0.0)
        : reactant_a_(reactant_a),
          reactant_b_(reactant_b),
          product_a_(product_a),
          product_b_(product_b),
          flag_(flag),
          params_(params),
          threshold_ev_(threshold_ev),
          quadrature_(quadrature),
          reduced_mass_(reduced_mass),
          Z_hint_(Z_hint),
          high_vibrational_factor_(high_v_factor) {}

    void apply(const dcr::state::PlasmaState& plasma,
               const EEDFGridView& grid,
               const Eigen::VectorXd& population,
               Eigen::MatrixXd& R,
               ReactionAccumulator* accumulator) const override {
        (void)grid;
        (void)accumulator;

        const double rate_coeff = integrate_ion(plasma);
        if (rate_coeff <= 0.0) return;
        if (reactant_a_ < 0) return;

        const double primary_density = population_density(population, reactant_a_);
        if (primary_density <= 0.0) return;

        double partner_density = 0.0;
        if (reactant_b_ >= 0) {
            partner_density = population_density(population, reactant_b_);
        }
        if (partner_density <= 0.0) return;

        const double coeff_primary = rate_coeff * partner_density;
        if (coeff_primary > 0.0) {
            if (product_a_ >= 0) R(product_a_, reactant_a_) += coeff_primary;
            R(reactant_a_, reactant_a_) -= coeff_primary;
        }

        if (reactant_b_ >= 0) {
            const double coeff_partner = rate_coeff * primary_density;
            if (coeff_partner > 0.0) {
                if (product_b_ >= 0) R(product_b_, reactant_b_) += coeff_partner;
                R(reactant_b_, reactant_b_) -= coeff_partner;
            }
        }
    }

private:
    double population_density(const Eigen::VectorXd& population, int idx) const {
        if (idx < 0 || idx >= population.size()) return 0.0;
        return std::max(population(idx), 0.0);
    }

    double cross_section(double E) const {
        if (E < threshold_ev_) return 0.0;
        switch (flag_) {
            case 99: {
                if (E <= threshold_ev_) return 0.0;
                const double term = params_[0] * std::pow(E, params_[1]);
                const double bracket = 1.0 - std::pow(threshold_ev_ / E, params_[2]);
                if (bracket <= 0.0) return 0.0;
                const double expo = std::exp(-params_[4] * std::pow(E, params_[5]));
                const double sigma = term * std::pow(bracket, params_[3]) * expo * 1.0e-16;
                return (sigma > 0.0) ? sigma : 0.0;
            }
            case 98: {
                if (E <= threshold_ev_) return 0.0;
                const double denom = params_[3] * std::pow(E, params_[4])
                                   + params_[5] * std::pow(E, params_[6])
                                   + params_[7] * std::pow(E, params_[8])
                                   + params_[9] * std::pow(E, params_[10]);
                if (denom <= 0.0) return 0.0;
                const double expo = std::exp(-params_[1] / std::pow(E, params_[2]));
                const double sigma = params_[0] * expo / denom * 1.0e-16;
                return (sigma > 0.0) ? sigma : 0.0;
            }
            case 90: {
                if (E <= 0.0 || high_vibrational_factor_ <= 0.0) return 0.0;
                const double term1 = std::pow(E, 0.033);
                const double term2 = 9.85e-10 * std::pow(E, 2.16);
                const double term3 = (1.66 * high_vibrational_factor_) * 1.0e-25 * std::pow(E, 5.25);
                const double denom = term1 + term2 + term3;
                if (denom <= 0.0) return 0.0;
                const double sigma = (27.0 * high_vibrational_factor_ / denom) * 1.0e-16;
                return (sigma > 0.0) ? sigma : 0.0;
            }
            default:
                return 0.0;
        }
    }

    double integrate_ion(const dcr::state::PlasmaState& plasma) const {
        if (!quadrature_ || quadrature_->empty()) return 0.0;
        const double temp = std::max(plasma.ion_temperature_ev(), 1e-3);
        double acc = 0.0;
        for (const auto& qp : *quadrature_) {
            const double E = qp.point * temp;
            const double sigma = cross_section(E);
            if (sigma <= 0.0) continue;
            const double prefac = sigma * std::sqrt(temp / std::max(reduced_mass_, 1e-6)) * MCX_FACTOR * qp.weight;
            acc += prefac;
        }
        return acc;
    }

    dcr::base::Index reactant_a_ = -1;
    dcr::base::Index reactant_b_ = -1;
    dcr::base::Index product_a_ = -1;
    dcr::base::Index product_b_ = -1;
    int flag_ = 0;
    std::array<double, 11> params_{};
    double threshold_ev_ = 0.0;
    const std::vector<QuadraturePoint>* quadrature_ = nullptr;
    double reduced_mass_ = DEFAULT_REDUCED_MASS;
    int Z_hint_ = 0;
    double high_vibrational_factor_ = 0.0;
};

// Molecular ion dissociation (MIDE) with shared normalization pool.
class MolecularMIDEProcess final : public MolecularProcess {
public:
    MolecularMIDEProcess(dcr::base::Index from,
                         dcr::base::Index product_a,
                         dcr::base::Index product_b,
                         double threshold_ev,
                         const std::array<double, 3>& params,
                         std::shared_ptr<MolecularMIDEPool> pool)
        : from_(from),
          product_a_(product_a),
          product_b_(product_b),
          threshold_ev_(threshold_ev),
          params_(params),
          pool_(std::move(pool)) {
        if (pool_) pool_->add_member(this);
    }

    ~MolecularMIDEProcess() override {
        if (pool_) pool_->remove_member(this);
    }

    void apply(const dcr::state::PlasmaState& plasma,
               const EEDFGridView& grid,
               const Eigen::VectorXd& population,
               Eigen::MatrixXd& R,
               ReactionAccumulator* accumulator) const override {
        (void)population;
        (void)accumulator;
        if (from_ < 0 || product_a_ < 0 || product_b_ < 0) return;
        if (!grid.valid()) return;

        const double k = integrate(grid);
        const double rate = k * plasma.electron_density_cm3();
        if (rate <= 0.0) return;

        R(product_a_, from_) += rate;
        R(product_b_, from_) += rate;
        R(from_, from_) -= rate;
    }

private:
    double integrate(const EEDFGridView& grid) const {
        double acc = 0.0;
        for (size_t i = 0; i < grid.size(); ++i) {
            const double E = grid.energy(i);
            if (E < threshold_ev_ || E > MIDE_EMAX) continue;
            const double sigma = cross_section(E);
            if (sigma <= 0.0) continue;
            const double val = grid.value_at(i);
            if (val <= 0.0) continue;
            acc += sigma * electron_speed(E) * val * grid.weight(i);
        }
        return acc;
    }

    double cross_section(double E) const {
        const double raw = raw_cross_section(E);
        if (raw <= 0.0) return 0.0;
        double scale = 1.0;
        if (pool_) scale = pool_->normalization_factor(E);
        const double sigma = raw * scale;
        return (sigma > 0.0) ? sigma : 0.0;
    }

    double raw_cross_section(double E) const {
        if (E < threshold_ev_ || E > MIDE_EMAX) return 0.0;
        const double ratio = threshold_ev_ / std::max(E, 1e-12);
        const double base = 1.0 - std::pow(ratio, params_[1]);
        if (base <= 0.0) return 0.0;
        return params_[0] * std::pow(E, -0.805) * std::pow(base, params_[2]) * 1.0e-16;
    }

    dcr::base::Index from_ = -1;
    dcr::base::Index product_a_ = -1;
    dcr::base::Index product_b_ = -1;
    double threshold_ev_ = 0.0;
    std::array<double, 3> params_{};
    std::shared_ptr<MolecularMIDEPool> pool_;

    static constexpr double MIDE_EMAX = 200.0;

public:
    double raw_cross_section_public(double E) const { return raw_cross_section(E); }
    static double total_cross_section(double E) {
        if (E <= 0.0) return 0.0;
        const double numerator = 13.2 * std::log(std::exp(1.0) + 2.55e-4 * E);
        const double denom = std::pow(E, 0.31) * (1.0 + 0.017 * std::pow(E, 0.76));
        if (denom <= 0.0) return 0.0;
        return numerator / denom * 1.0e-16;
    }
    friend class MolecularMIDEPool;
};

inline void MolecularMIDEPool::add_member(const MolecularMIDEProcess* proc) {
    members_.push_back(proc);
}

inline void MolecularMIDEPool::remove_member(const MolecularMIDEProcess* proc) {
    members_.erase(std::remove(members_.begin(), members_.end(), proc), members_.end());
}

inline double MolecularMIDEPool::normalization_factor(double E) const {
    const double sigma_tot = MolecularMIDEProcess::total_cross_section(E);
    if (!(sigma_tot > 0.0)) return 1.0;
    double denom = 0.0;
    for (const auto* proc : members_) {
        const double raw = proc->raw_cross_section_public(E);
        if (raw <= 0.0) continue;
        denom += raw;
    }
    if (denom <= 0.0) return 1.0;
    return sigma_tot / denom;
}

} // namespace crm_detail
