#include "LocalSystemAssembler.hpp"

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
    const dcr::atomic::AtomicData& atomic_data,
    const dcr::state::PlasmaState& plasma,
    const EEDFGridView& grid,
    const BoundaryPhaseResult& boundary,
    const dcr::base::Vector& background_population,
    const dcr::base::Vector& flowA,
    const dcr::base::Vector& flowM) {

    LocalSystem out;

    const int total_states = atomic_data.get_total_states();
    out.population_for_rates = background_population;
    if (out.population_for_rates.size() != total_states) {
        out.population_for_rates = dcr::base::Vector::Zero(total_states);
    }

    // Build total population used in process-rate evaluation.
    // Explicit recycling: dedicated flow states -> overwrite those entries.
    // Implicit recycling: flow shares neutral indices -> add onto background entries.
    for (size_t i = 0; i < boundary.A_indices.size(); ++i) {
        const int gi = boundary.A_indices[i];
        if (gi < 0 || gi >= total_states) continue;
        const double nA = (static_cast<int>(i) < flowA.size()) ? std::max(flowA(static_cast<int>(i)), 0.0) : 0.0;
        if (boundary.explicit_recycling) {
            out.population_for_rates(gi) = nA;
        } else {
            out.population_for_rates(gi) += nA;
        }
    }
    for (size_t i = 0; i < boundary.M_indices.size(); ++i) {
        const int gi = boundary.M_indices[i];
        if (gi < 0 || gi >= total_states) continue;
        const double nM = (static_cast<int>(i) < flowM.size()) ? std::max(flowM(static_cast<int>(i)), 0.0) : 0.0;
        if (boundary.explicit_recycling) {
            out.population_for_rates(gi) = nM;
        } else {
            out.population_for_rates(gi) += nM;
        }
    }

    // Update ne from quasi-neutrality at this spatial/iterative state.
    const auto& levels = atomic_data.get_levels();
    const double ne_local = quasineutral_electron_density(out.population_for_rates, levels);
    dcr::state::PlasmaState plasma_local = plasma;
    plasma_local.init_ne().setConstant(ne_local);

    out.R_full = dcr::base::Matrix::Zero(total_states, total_states);
    for (const auto& proc : atomic_data.get_processes()) {
        if (!proc) continue;
        proc->apply(plasma_local, grid, out.population_for_rates, out.R_full, nullptr);
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
                Si += out.R_full(gi, gj) * nA;
            }
        }
        if (use_M_source) {
            for (size_t j = 0; j < boundary.M_indices.size(); ++j) {
                const int gj = boundary.M_indices[j];
                if (gj < 0 || gj >= total_states) continue;
                const double nM = (static_cast<int>(j) < flowM.size())
                    ? std::max(flowM(static_cast<int>(j)), 0.0)
                    : 0.0;
                Si += out.R_full(gi, gj) * nM;
            }
        }
        out.S_background(pi) = Si;
    }

    return out;
}

} // namespace dcr::solver
