#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <map>
#include <string>
#include <vector>

#include "../src/atomic/AtomicData.hpp"
#include "../src/io/ConfigLoader.hpp"
#include "../src/physics/WallBoundary.hpp"
#include "../src/solver/core/ConfigPaths.hpp"
#include "../src/solver/core/RateAnalysis.hpp"
#include "../src/solver/core/TemperatureProfile.hpp"
#include "../src/solver/marching/LocalSystemAssembler.hpp"
#include "../src/state/PlasmaState.hpp"
#include "TestDCRSetup.hpp"

namespace {

struct GroupContribution {
    double net = 0.0;
    double positive = 0.0;
    double negative = 0.0;
};

double quasineutral_electron_density(const dcr::base::Vector& population,
                                     const std::vector<dcr::atomic::EnergyLevel>& levels) {
    const int n = std::min<int>(population.size(), static_cast<int>(levels.size()));
    double ne = 0.0;
    for (int i = 0; i < n; ++i) {
        ne += static_cast<double>(levels[static_cast<size_t>(i)].charge) * std::max(population(i), 0.0);
    }
    return std::max(0.0, ne);
}

std::string classify_source(const dcr::atomic::EnergyLevel& level,
                            int atom_ground,
                            int molecule_ground) {
    if (level.type == dcr::atomic::SpeciesType::Ion && level.charge > 0) {
        return (level.atomicity >= 2) ? "H2+" : "H+";
    }
    if (level.type == dcr::atomic::SpeciesType::Ion && level.charge < 0) {
        return "H-";
    }
    if (level.type == dcr::atomic::SpeciesType::Atom && level.charge == 0) {
        return (level.global_index == atom_ground) ? "H(1s)" : "H(n>1)";
    }
    if (level.type == dcr::atomic::SpeciesType::Molecule && level.charge == 0) {
        return (level.global_index == molecule_ground) ? "H2(v=0)" : "H2(v>0)";
    }
    return "other";
}

void add_group_contribution(std::map<std::string, GroupContribution>& grouped,
                            const std::string& key,
                            double value) {
    auto& bucket = grouped[key];
    bucket.net += value;
    if (value >= 0.0) {
        bucket.positive += value;
    } else {
        bucket.negative += value;
    }
}

} // namespace

int main(int argc, char** argv) {
    try {
        const std::string config_path =
            (argc >= 2) ? argv[1] : (test_dcr::source_root() / "config" / "hydro_am.yaml").string();

        dcr::io::Config cfg = dcr::io::ConfigLoader::load(config_path);
        dcr::solver::normalize_input_roots(cfg, config_path);
        cfg.solver.mode = "full_dcr";
        cfg.io.verbose_logging = false;

        dcr::atomic::AtomicData atomic_data(cfg);
        const auto& levels = atomic_data.get_levels();
        auto plasma = test_dcr::make_plasma_state(cfg, atomic_data.get_total_states());
        const auto temperatures = test_dcr::plasma_temperatures_at(cfg, 0.0);
        test_dcr::EEDFContext eedf(temperatures.electron_eV);

        const double ion_mass_amu = test_dcr::estimate_ion_mass_amu(cfg);
        const auto wall = dcr::physics::compute_wall_recycling(
            cfg.wall.material,
            temperatures.electron_eV,
            temperatures.ion_eV,
            ion_mass_amu,
            cfg.wall.sheath_potential_drop
        );

        const auto boundary = dcr::solver::run_boundary_phase(
            cfg, atomic_data, plasma, eedf.grid, ion_mass_amu, wall
        );

        const dcr::base::Vector nP = test_dcr::compact_background_from_boundary(boundary);
        const dcr::base::Vector bg_full = test_dcr::full_background_from_compact(
            nP, boundary, atomic_data.get_total_states()
        );
        const dcr::base::Vector flowA = test_dcr::flowA_from_boundary(boundary);
        const dcr::base::Vector flowM = test_dcr::flowM_from_boundary(boundary);

        const auto local = dcr::solver::assemble_local_system(
            cfg, atomic_data, plasma, eedf.grid, boundary, bg_full, flowA, flowM, 0.0
        );

        std::vector<int> p_pos(levels.size(), -1);
        for (int pi = 0; pi < static_cast<int>(boundary.P_indices.size()); ++pi) {
            const int gi = boundary.P_indices[static_cast<size_t>(pi)];
            if (gi >= 0 && gi < static_cast<int>(levels.size())) p_pos[static_cast<size_t>(gi)] = pi;
        }

        double n_h_total = 0.0;
        double n_h_plus_total = 0.0;
        double S_h_total = 0.0;
        double direct_ionization_from_h = 0.0;
        double chemistry_h_total = 0.0;
        std::map<std::string, GroupContribution> h_grouped;
        std::vector<std::pair<double, std::string>> positive_sources;
        std::vector<std::pair<double, std::string>> negative_sources;
        for (int gi : boundary.atom_bg_indices) {
            const int pi = (gi >= 0 && gi < static_cast<int>(p_pos.size())) ? p_pos[static_cast<size_t>(gi)] : -1;
            if (pi >= 0 && pi < nP.size()) {
                n_h_total += std::max(nP(pi), 0.0);
                S_h_total += local.S_background(pi);
            }
            for (int pi_col = 0; pi_col < nP.size(); ++pi_col) {
                const int gi_col = boundary.P_indices[static_cast<size_t>(pi_col)];
                if (gi_col < 0 || gi_col >= static_cast<int>(levels.size())) continue;
                const double contrib = local.R_full(gi, gi_col) * std::max(nP(pi_col), 0.0);
                chemistry_h_total += contrib;
                add_group_contribution(
                    h_grouped,
                    classify_source(levels[static_cast<size_t>(gi_col)], boundary.atom_ground, boundary.molecule_ground),
                    contrib
                );
                if (contrib > 0.0) {
                    positive_sources.emplace_back(contrib, levels[static_cast<size_t>(gi_col)].label + " -> H");
                } else if (contrib < 0.0) {
                    negative_sources.emplace_back(contrib, levels[static_cast<size_t>(gi_col)].label + " -> H");
                }
            }
        }
        for (int gi : boundary.ion_indices) {
            const int pi = (gi >= 0 && gi < static_cast<int>(p_pos.size())) ? p_pos[static_cast<size_t>(gi)] : -1;
            if (pi >= 0 && pi < nP.size() && gi < static_cast<int>(levels.size()) && levels[static_cast<size_t>(gi)].atomicity == 1) {
                n_h_plus_total += std::max(nP(pi), 0.0);
            }
        }
        for (int gi_row : boundary.ion_indices) {
            if (gi_row < 0 || gi_row >= static_cast<int>(levels.size())) continue;
            if (levels[static_cast<size_t>(gi_row)].atomicity != 1) continue;
            for (int gi_col : boundary.atom_bg_indices) {
                if (gi_col < 0 || gi_col >= static_cast<int>(levels.size())) continue;
                const int pi_col = p_pos[static_cast<size_t>(gi_col)];
                if (pi_col < 0 || pi_col >= nP.size()) continue;
                direct_ionization_from_h += local.R_full(gi_row, gi_col) * std::max(nP(pi_col), 0.0);
            }
        }

        const double c_s_A = dcr::physics::calculate_thermal_speed(
            cfg.plasma.neutral_atom_temperature_eV, boundary.atom_mass_amu
        );
        const double exA_over_w = c_s_A / std::max(cfg.grid.poloidal_width_cm, 1e-12);
        const double exhaust_h = exA_over_w * n_h_total;

        dcr::solver::AtomicRateCalculator rate_calc(atomic_data);
        const auto snapshot = rate_calc.evaluate(cfg, boundary, local, bg_full, 0.0);
        const double ne = quasineutral_electron_density(bg_full, levels);
        const int g = snapshot.atomic_effective.atom_ground_index;
        const double n_h_ground = (g >= 0 && g < bg_full.size()) ? std::max(bg_full(g), 0.0) : 0.0;
        const double eff_ionization_ground = ne * snapshot.atomic_effective.scd_cm3_s * n_h_ground;

        std::cout << std::setprecision(8) << std::scientific;
        std::cout << "Config: " << config_path << "\n";
        std::cout << "Boundary local H analysis at x=0 cm\n";
        std::cout << "  Te=" << temperatures.electron_eV
                  << " eV Ti=" << temperatures.ion_eV
                  << " eV ne=" << ne << " cm^-3\n";
        std::cout << "  n_H_total=" << n_h_total << " cm^-3\n";
        std::cout << "  n_Hplus_total=" << n_h_plus_total << " cm^-3\n";
        std::cout << "  n_H_total / n_Hplus_total="
                  << (n_h_plus_total > 0.0 ? (n_h_total / n_h_plus_total) : 0.0) << "\n";
        std::cout << "  S_H_total(from recycling-flow source)=" << S_h_total << " cm^-3 s^-1\n";
        std::cout << "  direct ion production from local H manifold="
                  << direct_ionization_from_h << " cm^-3 s^-1\n";
        std::cout << "  effective ground-H ionization ne*SCD*n_H(1s)="
                  << eff_ionization_ground << " cm^-3 s^-1\n";
        std::cout << "  atomic exhaust sink=" << exhaust_h << " cm^-3 s^-1\n";
        std::cout << "  chemistry_total_into_H_manifold=" << chemistry_h_total << " cm^-3 s^-1\n";
        std::cout << "  S_H_total / direct_ionization_from_h="
                  << (direct_ionization_from_h > 0.0 ? (S_h_total / direct_ionization_from_h) : 0.0) << "\n";
        std::cout << "  S_H_total / (ne*SCD*n_H(1s))="
                  << (eff_ionization_ground > 0.0 ? (S_h_total / eff_ionization_ground) : 0.0) << "\n";
        std::cout << "  Chemistry grouped by source state family:\n";
        for (const auto& [name, value] : h_grouped) {
            std::cout << "    " << std::setw(8) << std::left << name
                      << " net=" << std::setw(14) << std::right << value.net
                      << " positive=" << std::setw(14) << value.positive
                      << " negative=" << std::setw(14) << value.negative << "\n";
        }
        std::sort(positive_sources.begin(), positive_sources.end(),
                  [](const auto& a, const auto& b) { return a.first > b.first; });
        std::sort(negative_sources.begin(), negative_sources.end(),
                  [](const auto& a, const auto& b) { return a.first < b.first; });
        std::cout << "  Top positive row-chemistry contributors into H manifold:\n";
        for (size_t i = 0; i < std::min<size_t>(8, positive_sources.size()); ++i) {
            std::cout << "    " << std::setw(14) << positive_sources[i].first
                      << "  " << positive_sources[i].second << "\n";
        }
        std::cout << "  Top negative row-chemistry contributors into H manifold:\n";
        for (size_t i = 0; i < std::min<size_t>(8, negative_sources.size()); ++i) {
            std::cout << "    " << std::setw(14) << negative_sources[i].first
                      << "  " << negative_sources[i].second << "\n";
        }
        return 0;
    } catch (const std::exception& ex) {
        std::cerr << "inspect_boundary_h_balance failed: " << ex.what() << "\n";
        return 1;
    }
}
