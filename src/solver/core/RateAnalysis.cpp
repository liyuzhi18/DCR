#include "RateAnalysis.hpp"

#include "../../physics/Sheath.hpp"
#include "../../processes/MolecularProcess.hpp"
#include "TemperatureProfile.hpp"

#include <Eigen/LU>
#include <Eigen/QR>

#include <algorithm>
#include <cmath>
#include <limits>

namespace dcr::solver {

namespace {

double quasineutral_electron_density(const dcr::base::Vector& population,
                                     const std::vector<dcr::atomic::EnergyLevel>& levels) {
    const int n = std::min<int>(population.size(), static_cast<int>(levels.size()));
    double ne = 0.0;
    for (int i = 0; i < n; ++i) {
        const int z = levels[static_cast<size_t>(i)].charge;
        if (z == 0) continue;
        ne += static_cast<double>(z) * std::max(population(i), 0.0);
    }
    return std::max(0.0, ne);
}

dcr::base::Vector solve_linear(const dcr::base::Matrix& A,
                               const dcr::base::Vector& b) {
    if (A.rows() == A.cols()) {
        Eigen::FullPivLU<dcr::base::Matrix> lu(A);
        if (lu.rank() == A.rows()) {
            return lu.solve(b);
        }
    }
    return A.colPivHouseholderQr().solve(b);
}

dcr::base::Matrix extract_submatrix(const dcr::base::Matrix& full,
                                    const std::vector<int>& indices) {
    const int n = static_cast<int>(indices.size());
    dcr::base::Matrix sub = dcr::base::Matrix::Zero(n, n);
    for (int i = 0; i < n; ++i) {
        const int gi = indices[static_cast<size_t>(i)];
        if (gi < 0 || gi >= full.rows()) continue;
        for (int j = 0; j < n; ++j) {
            const int gj = indices[static_cast<size_t>(j)];
            if (gj < 0 || gj >= full.cols()) continue;
            sub(i, j) = full(gi, gj);
        }
    }
    return sub;
}

dcr::base::Vector extract_subvector(const dcr::base::Vector& full,
                                    const std::vector<int>& indices) {
    const int n = static_cast<int>(indices.size());
    dcr::base::Vector sub = dcr::base::Vector::Zero(n);
    for (int i = 0; i < n; ++i) {
        const int gi = indices[static_cast<size_t>(i)];
        if (gi < 0 || gi >= full.size()) continue;
        sub(i) = std::max(full(gi), 0.0);
    }
    return sub;
}

bool is_atomic_neutral_state(const dcr::atomic::EnergyLevel& level) {
    return level.atomicity == 1 &&
           level.type == dcr::atomic::SpeciesType::Atom &&
           level.charge == 0;
}

bool is_positive_atomic_ion_state(const dcr::atomic::EnergyLevel& level) {
    return level.atomicity == 1 &&
           level.type == dcr::atomic::SpeciesType::Ion &&
           level.charge > 0;
}

double sum_neutral_atomic_rows(const dcr::base::Matrix& rates,
                               const dcr::base::Vector& population,
                               const std::vector<dcr::atomic::EnergyLevel>& levels) {
    const int rows = std::min<int>(rates.rows(), static_cast<int>(levels.size()));
    const int cols = std::min<int>(rates.cols(), population.size());
    double total = 0.0;
    for (int row = 0; row < rows; ++row) {
        if (!is_atomic_neutral_state(levels[static_cast<size_t>(row)])) continue;
        for (int col = 0; col < cols; ++col) {
            const double term = rates(row, col) * std::max(population(col), 0.0);
            if (term > 0.0) total += term;
        }
    }
    return total;
}

double sum_neutral_atomic_rows_with_column_weights(
    const dcr::base::Matrix& rates,
    const dcr::base::Vector& population,
    const std::vector<dcr::atomic::EnergyLevel>& levels,
    const std::vector<double>& column_weights) {

    const int rows = std::min<int>(rates.rows(), static_cast<int>(levels.size()));
    const int cols = std::min({static_cast<int>(rates.cols()),
                               static_cast<int>(population.size()),
                               static_cast<int>(column_weights.size())});
    double total = 0.0;
    for (int row = 0; row < rows; ++row) {
        if (!is_atomic_neutral_state(levels[static_cast<size_t>(row)])) continue;
        for (int col = 0; col < cols; ++col) {
            const double weight = std::clamp(column_weights[static_cast<size_t>(col)], 0.0, 1.0);
            if (weight <= 0.0) continue;
            const double term = rates(row, col) * std::max(population(col), 0.0) * weight;
            if (term > 0.0) total += term;
        }
    }
    return total;
}

bool is_positive_molecular_ion_state(const dcr::atomic::EnergyLevel& level) {
    return level.atomicity == 2 &&
           level.type == dcr::atomic::SpeciesType::Ion &&
           level.charge > 0;
}

double sum_positive_ion_rows_from_columns(
    const dcr::base::Matrix& rates,
    const dcr::base::Vector& population,
    const std::vector<dcr::atomic::EnergyLevel>& levels,
    const std::vector<int>& columns) {

    const int rows = std::min<int>(rates.rows(), static_cast<int>(levels.size()));
    double total = 0.0;
    for (int row = 0; row < rows; ++row) {
        const auto& row_level = levels[static_cast<size_t>(row)];
        if (row_level.type != dcr::atomic::SpeciesType::Ion || row_level.charge <= 0) continue;
        for (int col : columns) {
            if (col < 0 || col >= rates.cols() || col >= population.size()) continue;
            const double term = rates(row, col) * std::max(population(col), 0.0);
            if (term > 0.0) total += term;
        }
    }
    return total;
}

bool is_mar_like_molecular_process(const dcr::ProcessBase& process) {
    return dynamic_cast<const crm_detail::MolecularDRProcess*>(&process) != nullptr ||
           dynamic_cast<const crm_detail::MolecularMCXProcess*>(&process) != nullptr ||
           dynamic_cast<const crm_detail::MolecularMIDEProcess*>(&process) != nullptr ||
           dynamic_cast<const crm_detail::MolecularDAProcess*>(&process) != nullptr ||
           dynamic_cast<const crm_detail::MolecularRAProcess*>(&process) != nullptr;
}

bool is_molecular_ionization_process(const dcr::ProcessBase& process) {
    return dynamic_cast<const crm_detail::MolecularMIProcess*>(&process) != nullptr;
}

bool is_molecular_dissociative_recombination_process(const dcr::ProcessBase& process) {
    return dynamic_cast<const crm_detail::MolecularDRProcess*>(&process) != nullptr;
}

bool is_molecular_charge_exchange_process(const dcr::ProcessBase& process) {
    return dynamic_cast<const crm_detail::MolecularMCXProcess*>(&process) != nullptr;
}

} // namespace

AtomicRateCalculator::AtomicRateCalculator(const dcr::atomic::AtomicData& atomic_data)
    : atomic_data_(atomic_data),
      levels_(atomic_data.get_levels()),
      atomic_global_to_subspace_(levels_.size(), -1) {

    double atom_ground_energy = std::numeric_limits<double>::infinity();
    double ion_ground_energy = std::numeric_limits<double>::infinity();
    double first_excited_energy = std::numeric_limits<double>::infinity();

    // Build the atomic-only subspace used for OpenADAS-style effective rates:
    // retain H/H+ states only, then identify the lowest neutral and ion levels
    // as the reduced "ground" states.
    for (size_t gi = 0; gi < levels_.size(); ++gi) {
        const auto& level = levels_[gi];
        if (is_atomic_neutral_state(level) || is_positive_atomic_ion_state(level)) {
            atomic_global_to_subspace_[gi] = static_cast<int>(atomic_subspace_indices_.size());
            atomic_subspace_indices_.push_back(static_cast<int>(gi));
        }
        if (is_atomic_neutral_state(level) && level.energy_eV < atom_ground_energy) {
            atom_ground_energy = level.energy_eV;
            atom_ground_index_ = static_cast<int>(gi);
        }
        if (is_positive_atomic_ion_state(level) && level.energy_eV < ion_ground_energy) {
            ion_ground_energy = level.energy_eV;
            ion_ground_index_ = static_cast<int>(gi);
        }
    }

    // QSS diagnostics are reported for excited atomic neutrals only:
    // ground H and ground H+ are the retained states of the reduced system.
    for (int gi : atomic_subspace_indices_) {
        if (gi == atom_ground_index_) continue;
        if (gi == ion_ground_index_) continue;
        if (gi >= 0 &&
            gi < static_cast<int>(levels_.size()) &&
            is_atomic_neutral_state(levels_[static_cast<size_t>(gi)])) {
            qss_excited_indices_.push_back(gi);
            if (levels_[static_cast<size_t>(gi)].energy_eV < first_excited_energy) {
                first_excited_energy = levels_[static_cast<size_t>(gi)].energy_eV;
                first_excited_index_ = gi;
            }
        }
    }
}

RateDiagnosticSnapshot AtomicRateCalculator::evaluate(
    const dcr::io::Config& config,
    const BoundaryPhaseResult& boundary,
    const LocalSystem& local_system,
    const dcr::base::Vector& background_population,
    double x_cm) const {

    RateDiagnosticSnapshot snapshot;
    const auto temperatures = evaluate_plasma_temperatures(config, x_cm);
    snapshot.electron_temperature_eV = temperatures.electron_eV;
    snapshot.ion_temperature_eV = temperatures.ion_eV;
    snapshot.electron_density_cm3 = quasineutral_electron_density(background_population, levels_);

    snapshot.atomic_effective.atom_ground_index = atom_ground_index_;
    snapshot.atomic_effective.ion_ground_index = ion_ground_index_;

    if (atom_ground_index_ < 0 || ion_ground_index_ < 0 || atomic_subspace_indices_.size() < 2) {
        snapshot.atomic_qss.excited_indices = qss_excited_indices_;
        return snapshot;
    }

    // M_atomic is the local atomic CR matrix with the same sign convention as
    // the assembled global matrix: off-diagonals are source terms into a row,
    // and the diagonal is the total loss frequency from that state.
    const dcr::base::Matrix M_atomic = extract_submatrix(local_system.R_full, atomic_subspace_indices_);
    const dcr::base::Vector n_atomic = extract_subvector(background_population, atomic_subspace_indices_);

    const int g_pos = atomic_global_to_subspace_[static_cast<size_t>(atom_ground_index_)];
    const int i_pos = atomic_global_to_subspace_[static_cast<size_t>(ion_ground_index_)];
    if (g_pos < 0 || i_pos < 0) {
        snapshot.atomic_qss.excited_indices = qss_excited_indices_;
        return snapshot;
    }

    std::vector<int> eliminated_positions;
    eliminated_positions.reserve(atomic_subspace_indices_.size());
    for (int pos = 0; pos < static_cast<int>(atomic_subspace_indices_.size()); ++pos) {
        if (pos == g_pos || pos == i_pos) continue;
        eliminated_positions.push_back(pos);
    }

    if (snapshot.electron_density_cm3 > 0.0) {
        // Start from the direct ground <-> ion coupling already present in the
        // full atomic matrix, then fold in all excited-state pathways below.
        double eff_ig = M_atomic(i_pos, g_pos);
        double eff_ii = M_atomic(i_pos, i_pos);

        if (!eliminated_positions.empty()) {
            const int nx = static_cast<int>(eliminated_positions.size());
            dcr::base::Matrix M_xx = dcr::base::Matrix::Zero(nx, nx);
            dcr::base::Vector M_xg = dcr::base::Vector::Zero(nx);
            dcr::base::Vector M_xi = dcr::base::Vector::Zero(nx);
            dcr::base::Vector M_ix = dcr::base::Vector::Zero(nx);
            for (int r = 0; r < nx; ++r) {
                const int xr = eliminated_positions[static_cast<size_t>(r)];
                M_xg(r) = M_atomic(xr, g_pos);
                M_xi(r) = M_atomic(xr, i_pos);
                M_ix(r) = M_atomic(i_pos, xr);
                for (int c = 0; c < nx; ++c) {
                    const int xc = eliminated_positions[static_cast<size_t>(c)];
                    M_xx(r, c) = M_atomic(xr, xc);
                }
            }

            // Schur complement reduction with excited states in QSS:
            //   0 = M_xg n_g + M_xx n_x + M_xi n_i
            // so n_x = -M_xx^{-1}(M_xg n_g + M_xi n_i),
            // which corrects the retained ion equation coefficients.
            const dcr::base::Vector y_g = solve_linear(M_xx, M_xg);
            const dcr::base::Vector y_i = solve_linear(M_xx, M_xi);
            if (y_g.allFinite() && y_i.allFinite()) {
                eff_ig -= M_ix.dot(y_g);
                eff_ii -= M_ix.dot(y_i);
            }
        }

        snapshot.atomic_effective.valid = std::isfinite(eff_ig) && std::isfinite(eff_ii);
        if (snapshot.atomic_effective.valid) {
            // Reduced ion equation:
            //   dn_i/dt = ne * SCD * n_g - ne * ACD * n_i
            // so SCD is the effective g->i coupling divided by ne, and ACD is
            // minus the retained ion self-loss coefficient divided by ne.
            snapshot.atomic_effective.scd_cm3_s =
                std::max(0.0, eff_ig / snapshot.electron_density_cm3);
            snapshot.atomic_effective.acd_cm3_s =
                std::max(0.0, -eff_ii / snapshot.electron_density_cm3);

            const double n_ion_ground =
                (ion_ground_index_ >= 0 && ion_ground_index_ < background_population.size())
                    ? std::max(background_population(ion_ground_index_), 0.0)
                    : 0.0;
            snapshot.atomic_sources.effective_eir_rate_cm3_s =
                snapshot.electron_density_cm3 * snapshot.atomic_effective.acd_cm3_s * n_ion_ground;
        }
    }

    const int pn = std::min<int>(local_system.S_background.size(),
                                 static_cast<int>(boundary.P_indices.size()));
    for (int pi = 0; pi < pn; ++pi) {
        const int gi = boundary.P_indices[static_cast<size_t>(pi)];
        if (gi < 0 || gi >= static_cast<int>(levels_.size())) continue;
        if (!is_atomic_neutral_state(levels_[static_cast<size_t>(gi)])) continue;
        snapshot.atomic_sources.flow_h_source_rate_cm3_s +=
            std::max(0.0, local_system.S_background(pi));
    }

    AtomicQSSDiagnostics qss;
    qss.excited_indices = qss_excited_indices_;
    qss.first_excited_index = first_excited_index_;
    const int n_excited = static_cast<int>(qss.excited_indices.size());
    qss.transport_rate_cm3_s = dcr::base::Vector::Zero(n_excited);
    qss.local_source_rate_cm3_s = dcr::base::Vector::Zero(n_excited);
    qss.local_loss_rate_cm3_s = dcr::base::Vector::Zero(n_excited);
    qss.local_loss_frequency_s = dcr::base::Vector::Zero(n_excited);
    qss.transport_to_local_ratio = dcr::base::Vector::Zero(n_excited);
    qss.transport_to_loss_frequency_ratio = dcr::base::Vector::Zero(n_excited);

    if (n_excited > 0) {
        // Use the same background atomic exhaust model used in the local solve:
        // transport frequency = c_s,A / w.
        const double c_s_A = dcr::physics::calculate_thermal_speed(
            config.plasma.neutral_atom_temperature_eV,
            boundary.atom_mass_amu
        );
        const double w_local = std::max(config.grid.poloidal_width_cm, 1e-12);
        qss.transport_frequency_s = c_s_A / w_local;
        qss.valid = true;

        constexpr double tiny_density = 1e-30;
        constexpr double tiny_rate = 1e-300;
        for (int j = 0; j < n_excited; ++j) {
            const int gi = qss.excited_indices[static_cast<size_t>(j)];
            if (gi < 0 || gi >= static_cast<int>(levels_.size())) continue;
            const int pos = (gi >= 0 && gi < static_cast<int>(atomic_global_to_subspace_.size()))
                ? atomic_global_to_subspace_[static_cast<size_t>(gi)]
                : -1;
            if (pos < 0 || pos >= n_atomic.size()) continue;

            // For one excited state j:
            // - loss_frequency = -M_jj
            // - source_rate = sum_{k!=j} M_jk n_k
            // - transport_rate = (c_s,A / w) n_j
            const double n_j = std::max(n_atomic(pos), 0.0);
            const double loss_frequency = std::max(0.0, -M_atomic(pos, pos));
            double source_rate = 0.0;
            for (int k = 0; k < M_atomic.cols(); ++k) {
                if (k == pos) continue;
                const double coeff = M_atomic(pos, k);
                if (coeff <= 0.0) continue;
                source_rate += coeff * std::max(n_atomic(k), 0.0);
            }

            const double loss_rate = loss_frequency * n_j;
            const double transport_rate = qss.transport_frequency_s * n_j;

            qss.transport_rate_cm3_s(j) = transport_rate;
            qss.local_source_rate_cm3_s(j) = source_rate;
            qss.local_loss_rate_cm3_s(j) = loss_rate;
            qss.local_loss_frequency_s(j) = loss_frequency;
            if (gi == first_excited_index_) {
                qss.first_excited_local_loss_frequency_s = loss_frequency;
                if (loss_frequency > tiny_rate && boundary.u_A > 0.0) {
                    qss.relaxation_length_cm = boundary.u_A / loss_frequency;
                }
            }

            if (n_j > tiny_density) {
                // Two complementary QSS checks:
                // 1) transport / local volumetric rate scale
                // 2) transport frequency / local relaxation frequency
                const double local_scale = std::max({source_rate, loss_rate, tiny_rate});
                qss.transport_to_local_ratio(j) = transport_rate / local_scale;
                qss.transport_to_loss_frequency_ratio(j) =
                    qss.transport_frequency_s / std::max(loss_frequency, tiny_rate);
                qss.max_transport_to_local_ratio =
                    std::max(qss.max_transport_to_local_ratio, qss.transport_to_local_ratio(j));
                qss.max_transport_to_loss_frequency_ratio =
                    std::max(qss.max_transport_to_loss_frequency_ratio,
                             qss.transport_to_loss_frequency_ratio(j));
            }
        }
    }

    snapshot.atomic_qss = std::move(qss);
    return snapshot;
}

RateDiagnosticSnapshot AtomicRateCalculator::evaluate(
    const dcr::io::Config& config,
    const BoundaryPhaseResult& boundary,
    const LocalSystem& local_system,
    const dcr::base::Vector& background_population,
    double x_cm,
    const dcr::state::PlasmaState& plasma,
    const EEDFGridView& grid,
    double h2plus_transport_rate_cm3_s) const {

    RateDiagnosticSnapshot snapshot = evaluate(
        config, boundary, local_system, background_population, x_cm
    );

    if (snapshot.electron_density_cm3 <= 0.0 || local_system.population_for_rates.size() == 0) {
        return snapshot;
    }

    const LocalKineticContext plasma_local(
        plasma,
        grid,
        snapshot.electron_temperature_eV,
        snapshot.ion_temperature_eV,
        snapshot.electron_density_cm3
    );

    std::vector<double> h2plus_mcx_source_factor(levels_.size(), 0.0);
    if (local_system.population_for_source.size() > 0 && !boundary.M_indices.empty()) {
        double h2plus_mi_source_total = 0.0;
        double h2plus_mcx_source_total = 0.0;

        for (const auto& proc : atomic_data_.get_processes()) {
            if (!proc) continue;
            const bool is_mi = is_molecular_ionization_process(*proc);
            const bool is_mcx = is_molecular_charge_exchange_process(*proc);
            if (!is_mi && !is_mcx) continue;

            dcr::base::Matrix rates = dcr::base::Matrix::Zero(
                static_cast<int>(levels_.size()),
                static_cast<int>(levels_.size())
            );
            proc->apply(
                plasma_local.plasma(),
                plasma_local.grid(),
                local_system.population_for_source,
                rates,
                nullptr
            );
            const double rate = sum_positive_ion_rows_from_columns(
                rates,
                local_system.population_for_source,
                levels_,
                boundary.M_indices
            );
            if (is_mi) {
                snapshot.atomic_sources.molecular_flow_ionization_rate_cm3_s += rate;
                h2plus_mi_source_total += rate;
            } else {
                snapshot.atomic_sources.molecular_flow_charge_exchange_rate_cm3_s += rate;
                h2plus_mcx_source_total += rate;
            }
        }

        const double h2plus_transport = std::max(0.0, h2plus_transport_rate_cm3_s);
        const double h2plus_total_supply =
            h2plus_transport + h2plus_mi_source_total + h2plus_mcx_source_total;
        const double global_mcx_source_factor = (h2plus_total_supply > 0.0)
            ? h2plus_mcx_source_total / h2plus_total_supply
            : 0.0;

        for (size_t i = 0; i < levels_.size(); ++i) {
            if (!is_positive_molecular_ion_state(levels_[i])) continue;
            h2plus_mcx_source_factor[i] = global_mcx_source_factor;
        }
    }

    for (const auto& proc : atomic_data_.get_processes()) {
        if (!proc || !is_mar_like_molecular_process(*proc)) continue;
        dcr::base::Matrix rates = dcr::base::Matrix::Zero(
            static_cast<int>(levels_.size()),
            static_cast<int>(levels_.size())
        );
        proc->apply(
            plasma_local.plasma(),
            plasma_local.grid(),
            local_system.population_for_rates,
            rates,
            nullptr
        );

        if (is_molecular_dissociative_recombination_process(*proc)) {
            snapshot.atomic_sources.mar_h_source_rate_cm3_s +=
                sum_neutral_atomic_rows_with_column_weights(
                    rates,
                    local_system.population_for_rates,
                    levels_,
                    h2plus_mcx_source_factor
                );
        } else {
            snapshot.atomic_sources.mar_h_source_rate_cm3_s +=
                sum_neutral_atomic_rows(rates, local_system.population_for_rates, levels_);
        }
    }

    return snapshot;
}

} // namespace dcr::solver
