#include "LocalSystemAssembler.hpp"

#include "../core/TemperatureProfile.hpp"

#include <algorithm>
#include <set>

namespace dcr::solver {

namespace {

// Electron density from quasi-neutrality: ne = sum_i (Z_i * n_i), allowing negative-ion contribution.
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

void redirect_h2plus_dr_products_to_ground(const dcr::io::Config& config,
                                           const std::vector<dcr::atomic::EnergyLevel>& levels,
                                           dcr::base::Matrix& rates) {
    if (!config.numerics.disable_h2plus_dr) return;
    if (rates.rows() != rates.cols()) return;

    const int total_states = static_cast<int>(levels.size());
    if (rates.rows() != total_states) return;

    int ground_row = -1;
    for (int i = 0; i < total_states; ++i) {
        const auto& level = levels[static_cast<size_t>(i)];
        if (level.type == dcr::atomic::SpeciesType::Atom &&
            level.charge == 0 &&
            level.internal_id == 1) {
            ground_row = i;
            break;
        }
    }
    if (ground_row < 0) return;

    for (int col = 0; col < total_states; ++col) {
        const auto& source = levels[static_cast<size_t>(col)];
        if (source.atomicity != 2 || source.charge != 1 || !source.is_background) continue;
        for (int row = 0; row < total_states; ++row) {
            const auto& target = levels[static_cast<size_t>(row)];
            if (target.type != dcr::atomic::SpeciesType::Atom) continue;
            if (target.charge != 0 || target.internal_id <= 1) continue;
            rates(ground_row, col) += rates(row, col);
            rates(row, col) = 0.0;
        }
    }
}

dcr::base::Vector sanitize_background_population(const dcr::base::Vector& background_population,
                                                 int total_states) {
    if (background_population.size() == total_states) return background_population;
    return dcr::base::Vector::Zero(total_states);
}

dcr::base::Matrix assemble_cached_rate_matrix(
    const dcr::io::Config& config,
    const dcr::atomic::AtomicData& atomic_data,
    const dcr::state::PlasmaState& plasma,
    const EEDFGridView& grid,
    const LocalRateCache& cache,
    const dcr::base::Vector& population,
    double electron_temperature_eV,
    double ion_temperature_eV,
    double electron_density_cm3) {
    dcr::base::Matrix rates = cache.constant +
        electron_density_cm3 * cache.linear_ne +
        electron_density_cm3 * electron_density_cm3 * cache.quadratic_ne;
    const LocalKineticContext context(
        plasma, grid, electron_temperature_eV, ion_temperature_eV, electron_density_cm3);
    for (const auto& proc : atomic_data.get_processes()) {
        if (!proc || proc->population_dependencies().empty()) continue;
        proc->apply(context.plasma(), context.grid(), population, rates, nullptr);
    }
    redirect_h2plus_dr_products_to_ground(config, atomic_data.get_levels(), rates);
    return rates;
}

} // namespace

BackgroundRateAssembly assemble_background_rate_matrix(
    const dcr::io::Config& config,
    const dcr::atomic::AtomicData& atomic_data,
    const dcr::state::PlasmaState& plasma,
    const EEDFGridView& grid,
    const dcr::base::Vector& background_population,
    double x_cm) {
    BackgroundRateAssembly out;
    const int total_states = atomic_data.get_total_states();
    const auto& levels = atomic_data.get_levels();
    const auto temperatures = evaluate_plasma_temperatures(config, x_cm);
    out.population_for_rates = sanitize_background_population(background_population, total_states);
    out.R_full = dcr::base::Matrix::Zero(total_states, total_states);

    const double ne_local = quasineutral_electron_density(out.population_for_rates, levels);
    const LocalKineticContext plasma_local(
        plasma,
        grid,
        temperatures.electron_eV,
        temperatures.ion_eV,
        ne_local
    );
    for (const auto& proc : atomic_data.get_processes()) {
        if (!proc) continue;
        proc->apply(
            plasma_local.plasma(),
            plasma_local.grid(),
            out.population_for_rates,
            out.R_full,
            nullptr
        );
    }
    redirect_h2plus_dr_products_to_ground(config, levels, out.R_full);
    return out;
}

LocalRateCache build_local_rate_cache(
    const dcr::io::Config& config,
    const dcr::atomic::AtomicData& atomic_data,
    const dcr::state::PlasmaState& plasma,
    const EEDFGridView& grid,
    double x_cm,
    double density_reference_cm3) {
    const int total_states = atomic_data.get_total_states();
    const auto temperatures = evaluate_plasma_temperatures(config, x_cm);
    const dcr::base::Vector zero_population = dcr::base::Vector::Zero(total_states);
    const double reference = std::max(density_reference_cm3, 1.0);
    auto assemble = [&](double ne) {
        const LocalKineticContext context(
            plasma, grid, temperatures.electron_eV, temperatures.ion_eV, ne);
        dcr::base::Matrix rates = dcr::base::Matrix::Zero(total_states, total_states);
        for (const auto& proc : atomic_data.get_processes()) {
            if (!proc || !proc->population_dependencies().empty()) continue;
            proc->apply(context.plasma(), context.grid(), zero_population, rates, nullptr);
        }
        redirect_h2plus_dr_products_to_ground(config, atomic_data.get_levels(), rates);
        return rates;
    };
    const dcr::base::Matrix at_zero = assemble(0.0);
    const dcr::base::Matrix at_reference = assemble(reference);
    const dcr::base::Matrix at_twice_reference = assemble(2.0 * reference);
    LocalRateCache cache;
    cache.constant = at_zero;
    cache.quadratic_ne =
        (at_twice_reference - 2.0 * at_reference + at_zero) /
        (2.0 * reference * reference);
    cache.linear_ne =
        (at_reference - at_zero - reference * reference * cache.quadratic_ne) /
        reference;
    return cache;
}

LocalSystem assemble_local_system_from_background_rate_matrix(
    const dcr::io::Config& config,
    const dcr::atomic::AtomicData& atomic_data,
    const dcr::state::PlasmaState& plasma,
    const EEDFGridView& grid,
    const BoundaryPhaseResult& boundary,
    const BackgroundRateAssembly& background_rates,
    const dcr::base::Vector& flowA,
    const dcr::base::Vector& flowM,
    double x_cm,
    bool assemble_rate_derivatives,
    const LocalRateCache* rate_cache) {
    LocalSystem out;
    const int total_states = atomic_data.get_total_states();
    const auto& levels = atomic_data.get_levels();
    const auto temperatures = evaluate_plasma_temperatures(config, x_cm);
    out.population_for_rates = background_rates.population_for_rates;
    out.R_full = background_rates.R_full;

    // For S evaluation, use local total population = background + recycling-flow states.
    dcr::base::Vector population_for_source = out.population_for_rates;
    for (size_t i = 0; i < boundary.A_indices.size(); ++i) {
        const int gj = boundary.A_indices[i];
        if (gj < 0 || gj >= total_states) continue;
        const double nA = (static_cast<int>(i) < flowA.size())
            ? std::max(flowA(static_cast<int>(i)), 0.0)
            : 0.0;
        if (boundary.explicit_recycling) {
            population_for_source(gj) = nA;
        } else {
            population_for_source(gj) += nA;
        }
    }
    for (size_t i = 0; i < boundary.M_indices.size(); ++i) {
        const int gj = boundary.M_indices[i];
        if (gj < 0 || gj >= total_states) continue;
        const double nM = (static_cast<int>(i) < flowM.size())
            ? std::max(flowM(static_cast<int>(i)), 0.0)
            : 0.0;
        if (boundary.explicit_recycling) {
            population_for_source(gj) = nM;
        } else {
            population_for_source(gj) += nM;
        }
    }

    const double ne_source = quasineutral_electron_density(population_for_source, levels);
    out.population_for_source = population_for_source;
    const LocalKineticContext plasma_source(
        plasma,
        grid,
        temperatures.electron_eV,
        temperatures.ion_eV,
        ne_source
    );
    dcr::base::Matrix R_for_S;
    if (rate_cache) {
        R_for_S = assemble_cached_rate_matrix(
            config, atomic_data, plasma, grid, *rate_cache, population_for_source,
            temperatures.electron_eV, temperatures.ion_eV, ne_source);
    } else {
        R_for_S = dcr::base::Matrix::Zero(total_states, total_states);
        for (const auto& proc : atomic_data.get_processes()) {
            if (!proc) continue;
            proc->apply(
                plasma_source.plasma(),
                plasma_source.grid(),
                population_for_source,
                R_for_S,
                nullptr
            );
        }
        redirect_h2plus_dr_products_to_ground(config, levels, R_for_S);
    }
    out.R_source_full = R_for_S;

    auto assemble_source_rates = [&](const dcr::base::Vector& population, double ne) {
        const LocalKineticContext context(
            plasma, grid, temperatures.electron_eV, temperatures.ion_eV, ne);
        dcr::base::Matrix rates = dcr::base::Matrix::Zero(total_states, total_states);
        for (const auto& proc : atomic_data.get_processes()) {
            if (proc) proc->apply(context.plasma(), context.grid(), population, rates, nullptr);
        }
        redirect_h2plus_dr_products_to_ground(config, levels, rates);
        return rates;
    };

    if (assemble_rate_derivatives) {
        if (rate_cache) {
            out.dR_source_dne = rate_cache->linear_ne +
                2.0 * ne_source * rate_cache->quadratic_ne;
        } else {
            const double ne_step = std::max(1.0, 1.0e-6 * std::max(ne_source, 1.0));
            out.dR_source_dne =
                (assemble_source_rates(population_for_source, ne_source + ne_step) - R_for_S) /
                ne_step;
        }

        std::set<int> dependency_indices;
        for (const auto& proc : atomic_data.get_processes()) {
            if (!proc) continue;
            for (int gi : proc->population_dependencies()) {
                if (gi >= 0 && gi < total_states) dependency_indices.insert(gi);
            }
        }
        auto assemble_population_dependent_rates = [&](const dcr::base::Vector& population) {
            dcr::base::Matrix rates = dcr::base::Matrix::Zero(total_states, total_states);
            for (const auto& proc : atomic_data.get_processes()) {
                if (!proc || proc->population_dependencies().empty()) continue;
                proc->apply(
                    plasma_source.plasma(), plasma_source.grid(), population, rates, nullptr);
            }
            return rates;
        };
        const dcr::base::Matrix dependency_rates =
            assemble_population_dependent_rates(population_for_source);
        for (int gi : dependency_indices) {
            const double step = std::max(1.0, 1.0e-6 * std::max(population_for_source(gi), 1.0));
            dcr::base::Vector perturbed = population_for_source;
            perturbed(gi) += step;
            out.rate_population_derivative_indices.push_back(gi);
            out.dR_source_dpopulation.push_back(
                (assemble_population_dependent_rates(perturbed) - dependency_rates) / step);
        }
    }

    // Build S on background rows with the same routing policy as boundary solve:
    // I rows: IA + IM, a rows: aM only, m rows: no recycling-flow source.
    const int Pn = static_cast<int>(boundary.P_indices.size());
    out.S_background = dcr::base::Vector::Zero(Pn);
    for (int pi = 0; pi < Pn; ++pi) {
        const int gi = boundary.P_indices[pi];
        if (gi < 0 || gi >= total_states) continue;

        const auto& row_lvl = levels[gi];
        const bool positive_ion_row = row_lvl.type == dcr::atomic::SpeciesType::Ion &&
            row_lvl.charge > 0;
        const bool neutral_atom_row = row_lvl.type == dcr::atomic::SpeciesType::Atom &&
            row_lvl.charge == 0;
        const bool use_A_source = positive_ion_row;
        const bool use_M_source = positive_ion_row || neutral_atom_row;

        double Si = 0.0;
        if (use_A_source) {
            for (size_t j = 0; j < boundary.A_indices.size(); ++j) {
                const int gj = boundary.A_indices[j];
                if (gj < 0 || gj >= total_states) continue;
                const double nA = (static_cast<int>(j) < flowA.size())
                    ? std::max(flowA(static_cast<int>(j)), 0.0)
                    : 0.0;
                Si += R_for_S(gi, gj) * nA;
            }
        }
        if (use_M_source) {
            for (size_t j = 0; j < boundary.M_indices.size(); ++j) {
                const int gj = boundary.M_indices[j];
                if (gj < 0 || gj >= total_states) continue;
                const double nM = (static_cast<int>(j) < flowM.size())
                    ? std::max(flowM(static_cast<int>(j)), 0.0)
                    : 0.0;
                Si += R_for_S(gi, gj) * nM;
            }
        }
        out.S_background(pi) = Si;
    }

    return out;
}

LocalChemistrySources assemble_local_chemistry_sources_from_rate_matrix(
    const dcr::base::Matrix& rates,
    const BoundaryPhaseResult& boundary,
    const dcr::atomic::AtomicData& atomic_data,
    const dcr::base::Vector& background,
    const dcr::base::Vector& flowA,
    const dcr::base::Vector& flowM) {
    LocalChemistrySources out;
    const auto& levels = atomic_data.get_levels();
    const int total_states = atomic_data.get_total_states();
    out.background = dcr::base::Vector::Zero(background.size());
    out.flowA = dcr::base::Vector::Zero(flowA.size());
    out.flowM = dcr::base::Vector::Zero(flowM.size());

    for (int pi = 0; pi < background.size(); ++pi) {
        const int gi = boundary.P_indices[static_cast<size_t>(pi)];
        if (gi < 0 || gi >= total_states) continue;
        for (int pj = 0; pj < background.size(); ++pj) {
            const int gj = boundary.P_indices[static_cast<size_t>(pj)];
            if (gj >= 0 && gj < total_states) out.background(pi) += rates(gi, gj) * background(pj);
        }

        const auto& row = levels[static_cast<size_t>(gi)];
        const bool charged_row = row.type == dcr::atomic::SpeciesType::Ion && row.charge != 0;
        const bool atom_row = row.type == dcr::atomic::SpeciesType::Atom && row.charge == 0;
        const bool molecule_row = row.type == dcr::atomic::SpeciesType::Molecule && row.charge == 0;
        if (charged_row || molecule_row) {
            for (int ai = 0; ai < flowA.size(); ++ai) {
                const int gj = boundary.A_indices[static_cast<size_t>(ai)];
                out.background(pi) += rates(gi, gj) * flowA(ai);
            }
        }
        if (charged_row || atom_row) {
            for (int mi = 0; mi < flowM.size(); ++mi) {
                const int gj = boundary.M_indices[static_cast<size_t>(mi)];
                out.background(pi) += rates(gi, gj) * flowM(mi);
            }
        }
    }

    for (int ai = 0; ai < flowA.size(); ++ai) {
        const int gi = boundary.A_indices[static_cast<size_t>(ai)];
        for (int aj = 0; aj < flowA.size(); ++aj) {
            const int gj = boundary.A_indices[static_cast<size_t>(aj)];
            out.flowA(ai) += rates(gi, gj) * flowA(aj);
        }
    }
    for (int mi = 0; mi < flowM.size(); ++mi) {
        const int gi = boundary.M_indices[static_cast<size_t>(mi)];
        for (int mj = 0; mj < flowM.size(); ++mj) {
            const int gj = boundary.M_indices[static_cast<size_t>(mj)];
            out.flowM(mi) += rates(gi, gj) * flowM(mj);
        }
    }
    return out;
}

LocalChemistrySources assemble_local_chemistry_sources(
    const LocalSystem& local,
    const BoundaryPhaseResult& boundary,
    const dcr::atomic::AtomicData& atomic_data,
    const dcr::base::Vector& background,
    const dcr::base::Vector& flowA,
    const dcr::base::Vector& flowM) {
    const dcr::base::Matrix& rates = local.R_source_full.size() > 0
        ? local.R_source_full : local.R_full;
    return assemble_local_chemistry_sources_from_rate_matrix(
        rates, boundary, atomic_data, background, flowA, flowM);
}

LocalSystem assemble_local_system(
    const dcr::io::Config& config,
    const dcr::atomic::AtomicData& atomic_data,
    const dcr::state::PlasmaState& plasma,
    const EEDFGridView& grid,
    const BoundaryPhaseResult& boundary,
    const dcr::base::Vector& background_population,
    const dcr::base::Vector& flowA,
    const dcr::base::Vector& flowM,
    double x_cm,
    bool assemble_rate_derivatives,
    const LocalRateCache* rate_cache) {
    BackgroundRateAssembly background_rates;
    if (rate_cache) {
        const auto temperatures = evaluate_plasma_temperatures(config, x_cm);
        background_rates.population_for_rates = sanitize_background_population(
            background_population, atomic_data.get_total_states());
        const double ne = quasineutral_electron_density(
            background_rates.population_for_rates, atomic_data.get_levels());
        background_rates.R_full = assemble_cached_rate_matrix(
            config, atomic_data, plasma, grid, *rate_cache,
            background_rates.population_for_rates,
            temperatures.electron_eV, temperatures.ion_eV, ne);
    } else {
        background_rates = assemble_background_rate_matrix(
            config, atomic_data, plasma, grid, background_population, x_cm);
    }
    return assemble_local_system_from_background_rate_matrix(
        config,
        atomic_data,
        plasma,
        grid,
        boundary,
        background_rates,
        flowA,
        flowM,
        x_cm,
        assemble_rate_derivatives,
        rate_cache
    );
}

} // namespace dcr::solver
