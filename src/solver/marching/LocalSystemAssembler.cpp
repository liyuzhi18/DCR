#include "LocalSystemAssembler.hpp"

#include "../core/TemperatureProfile.hpp"

#include <algorithm>

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

} // namespace

// Build local matrix/source objects from the current iterate.
LocalSystem assemble_local_system(
    const dcr::io::Config& config,
    const dcr::atomic::AtomicData& atomic_data,
    const dcr::state::PlasmaState& plasma,
    const EEDFGridView& grid,
    const BoundaryPhaseResult& boundary,
    const dcr::base::Vector& background_population,
    const dcr::base::Vector& flowA,
    const dcr::base::Vector& flowM,
    double x_cm) {

    LocalSystem out;

    const int total_states = atomic_data.get_total_states();
    out.population_for_rates = background_population;
    if (out.population_for_rates.size() != total_states) {
        out.population_for_rates = dcr::base::Vector::Zero(total_states);
    }

    // Use background-only population for the main R assembly (match boundary behavior).
    // Recycling-flow populations are not injected into out.R_full.

    // Update ne from quasi-neutrality at this spatial/iterative state.
    const auto& levels = atomic_data.get_levels();
    const auto temperatures = evaluate_plasma_temperatures(config, x_cm);
    const double ne_local = quasineutral_electron_density(out.population_for_rates, levels);
    const LocalKineticContext plasma_local(
        plasma,
        grid,
        temperatures.electron_eV,
        temperatures.ion_eV,
        ne_local
    );

    out.R_full = dcr::base::Matrix::Zero(total_states, total_states);
    for (const auto& proc : atomic_data.get_processes()) {
        if (!proc) continue;
        proc->apply(plasma_local.plasma(), plasma_local.grid(), out.population_for_rates, out.R_full, nullptr);
    }

    // For S evaluation, use local total population = background + recycling-flow states.
    // This keeps source coupling responsive to local cell populations without changing the
    // main background-only R block used in the linear background solve.
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
    dcr::base::Matrix R_for_S = dcr::base::Matrix::Zero(total_states, total_states);
    {
        const double ne_source = quasineutral_electron_density(population_for_source, levels);
        const LocalKineticContext plasma_source(
            plasma,
            grid,
            temperatures.electron_eV,
            temperatures.ion_eV,
            ne_source
        );
        for (const auto& proc : atomic_data.get_processes()) {
            if (!proc) continue;
            proc->apply(plasma_source.plasma(), plasma_source.grid(), population_for_source, R_for_S, nullptr);
        }
    }

    // Build S on background rows with the same routing policy as boundary solve:
    // I rows: IA + IM, a rows: aM only, m rows: no recycling source.
    const int Pn = static_cast<int>(boundary.P_indices.size());
    out.S_background = dcr::base::Vector::Zero(Pn);
    for (int pi = 0; pi < Pn; ++pi) {
        const int gi = boundary.P_indices[pi];
        if (gi < 0 || gi >= total_states) continue;

        bool use_A_source = false;
        bool use_M_source = false;
        const auto& row_lvl = levels[gi];
        if (row_lvl.type == dcr::atomic::SpeciesType::Ion && row_lvl.charge > 0) {
            use_A_source = true;
            use_M_source = true;
        } else if (row_lvl.type == dcr::atomic::SpeciesType::Atom && row_lvl.charge == 0) {
            use_A_source = false;
            use_M_source = true;
        } else if (row_lvl.type == dcr::atomic::SpeciesType::Molecule && row_lvl.charge == 0) {
            use_A_source = false;
            use_M_source = false;
        }

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

} // namespace dcr::solver
