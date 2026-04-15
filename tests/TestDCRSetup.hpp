#pragma once

#include <algorithm>
#include <cassert>
#include <cmath>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <vector>

#include "../src/atomic/EnergyLevel.hpp"
#include "../src/io/ConfigLoader.hpp"
#include "../src/physics/EEDF.hpp"
#include "../src/physics/EnergyGrid.hpp"
#include "../src/physics/WallBoundary.hpp"
#include "../src/solver/boundary/BoundaryPhase.hpp"
#include "../src/solver/core/ConfigPaths.hpp"
#include "../src/solver/core/TemperatureProfile.hpp"
#include "../src/state/PlasmaState.hpp"

namespace test_dcr {

namespace fs = std::filesystem;

inline fs::path source_root() {
#ifdef DCR_SOURCE_DIR
    return fs::path(DCR_SOURCE_DIR);
#else
    return fs::current_path();
#endif
}

inline dcr::io::Config load_reference_config(int num_cells, bool verbose_logging = false) {
    const fs::path cfg_path = source_root() / "config" / "hydro_am.yaml";
    if (!fs::exists(cfg_path)) {
        throw std::runtime_error("Missing config file for tests: " + cfg_path.string());
    }

    dcr::io::Config cfg = dcr::io::ConfigLoader::load(cfg_path.string());
    dcr::solver::normalize_input_roots(cfg, cfg_path.string());

    cfg.io.verbose_logging = verbose_logging;
    cfg.grid.num_cells = std::max(2, num_cells);
    cfg.grid.length_cm = (cfg.grid.length_cm > 0.0) ? cfg.grid.length_cm : 1.0;
    cfg.grid.type = "linear";
    cfg.grid.first_cell_cm = 0.0;

    // Keep tests robust and fast.
    cfg.numerics.max_iterations = std::min(400, std::max(100, cfg.numerics.max_iterations));
    cfg.numerics.tolerance = std::max(1e-10, cfg.numerics.tolerance);
    if (cfg.numerics.relaxation <= 0.0 || cfg.numerics.relaxation > 1.0) {
        cfg.numerics.relaxation = 0.9;
    }

    return cfg;
}

inline double estimate_ion_mass_amu(const dcr::io::Config& cfg) {
    for (const auto& sp : cfg.species) {
        if (!sp.is_molecule && sp.charge > 0 && sp.mass_amu > 0.0) {
            return sp.mass_amu;
        }
    }
    return 1.0;
}

inline dcr::solver::PlasmaTemperatures plasma_temperatures_at(const dcr::io::Config& cfg,
                                                              double x_cm) {
    return dcr::solver::evaluate_plasma_temperatures(cfg, x_cm);
}

inline EEDFConfig make_eedf_config(const dcr::io::Config& cfg) {
    EEDFConfig eedf_cfg;
    eedf_cfg.type = cfg.plasma.eedf.type;
    eedf_cfg.power_p = cfg.plasma.eedf.power_p;
    eedf_cfg.hot_fraction = cfg.plasma.eedf.hot_fraction;
    eedf_cfg.hot_temperature_factor = cfg.plasma.eedf.hot_temperature_factor;
    return eedf_cfg;
}

struct EEDFContext {
    std::vector<double> energies;
    std::vector<double> weights;
    EEDF eedf;
    EEDFGridView grid;

    explicit EEDFContext(double Te_eV, const EEDFConfig& cfg = EEDFConfig{})
        : eedf(Te_eV, cfg), grid(energies, weights, &eedf) {
        const auto energy_grid = dcr::physics::make_default_energy_grid();
        energies.reserve(energy_grid.size());
        weights.reserve(energy_grid.size());
        for (const auto& p : energy_grid) {
            energies.push_back(p.energy_eV);
            weights.push_back(p.width_eV);
        }
        eedf.normalize_on_grid(energies, weights);
    }
};

inline dcr::state::PlasmaState make_plasma_state(const dcr::io::Config& cfg, int total_states) {
    dcr::state::PlasmaState plasma(cfg.grid.num_cells, total_states);
    plasma.init_Te().setConstant(cfg.plasma.Te_eV);
    plasma.init_Ti().setConstant(cfg.plasma.Ti_eV);
    plasma.init_ne().setConstant(cfg.plasma.total_density);
    return plasma;
}

inline dcr::base::Vector compact_background_from_boundary(
    const dcr::solver::BoundaryPhaseResult& boundary) {
    const int Pn = static_cast<int>(boundary.P_indices.size());
    dcr::base::Vector nP = dcr::base::Vector::Zero(Pn);
    for (int i = 0; i < Pn; ++i) {
        const int gi = boundary.P_indices[static_cast<size_t>(i)];
        if (gi >= 0 && gi < boundary.population.size()) {
            nP(i) = std::max(boundary.population(gi), 0.0);
        }
    }
    return nP;
}

inline dcr::base::Vector full_background_from_compact(
    const dcr::base::Vector& nP,
    const dcr::solver::BoundaryPhaseResult& boundary,
    int total_states) {
    dcr::base::Vector out = dcr::base::Vector::Zero(total_states);
    for (int i = 0; i < nP.size() && i < static_cast<int>(boundary.P_indices.size()); ++i) {
        const int gi = boundary.P_indices[static_cast<size_t>(i)];
        if (gi >= 0 && gi < total_states) out(gi) = std::max(nP(i), 0.0);
    }
    return out;
}

inline double nuclei_total_compact(const dcr::base::Vector& n,
                                   const std::vector<int>& indices,
                                   const std::vector<dcr::atomic::EnergyLevel>& levels) {
    double total = 0.0;
    for (size_t i = 0; i < indices.size() && static_cast<int>(i) < n.size(); ++i) {
        const int gi = indices[i];
        if (gi < 0 || gi >= static_cast<int>(levels.size())) continue;
        total += levels[gi].atomicity * std::max(n(static_cast<int>(i)), 0.0);
    }
    return total;
}

inline double nuclei_total_selected(const dcr::base::Vector& n_full,
                                    const std::vector<int>& indices,
                                    const std::vector<dcr::atomic::EnergyLevel>& levels) {
    double total = 0.0;
    for (int gi : indices) {
        if (gi < 0 || gi >= n_full.size() || gi >= static_cast<int>(levels.size())) continue;
        total += levels[gi].atomicity * std::max(n_full(gi), 0.0);
    }
    return total;
}

inline double nuclei_total_full(const dcr::base::Vector& n_full,
                                const std::vector<dcr::atomic::EnergyLevel>& levels) {
    double total = 0.0;
    const int n = std::min<int>(n_full.size(), static_cast<int>(levels.size()));
    for (int i = 0; i < n; ++i) {
        total += levels[i].atomicity * std::max(n_full(i), 0.0);
    }
    return total;
}

inline void assert_all_finite_nonnegative(const dcr::base::Vector& v) {
    for (int i = 0; i < v.size(); ++i) {
        assert(std::isfinite(v(i)));
        assert(v(i) >= -1e-12);
    }
}

inline dcr::base::Vector flowA_from_boundary(const dcr::solver::BoundaryPhaseResult& boundary) {
    dcr::base::Vector flowA = dcr::base::Vector::Zero(static_cast<int>(boundary.A_indices.size()));
    if (boundary.explicit_recycling) {
        for (size_t i = 0; i < boundary.A_indices.size(); ++i) {
            const int gi = boundary.A_indices[i];
            if (gi >= 0 && gi < boundary.population.size()) {
                flowA(static_cast<int>(i)) = std::max(boundary.population(gi), 0.0);
            }
        }
    } else if (boundary.have_flow_last) {
        flowA = boundary.flowA_last;
    }
    return flowA;
}

inline dcr::base::Vector flowM_from_boundary(const dcr::solver::BoundaryPhaseResult& boundary) {
    dcr::base::Vector flowM = dcr::base::Vector::Zero(static_cast<int>(boundary.M_indices.size()));
    if (boundary.explicit_recycling) {
        for (size_t i = 0; i < boundary.M_indices.size(); ++i) {
            const int gi = boundary.M_indices[i];
            if (gi >= 0 && gi < boundary.population.size()) {
                flowM(static_cast<int>(i)) = std::max(boundary.population(gi), 0.0);
            }
        }
    } else if (boundary.have_flow_last) {
        flowM = boundary.flowM_last;
    }
    return flowM;
}

} // namespace test_dcr
