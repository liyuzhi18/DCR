#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <map>
#include <numeric>
#include <string>
#include <tuple>
#include <vector>

#include "../src/atomic/AtomicData.hpp"
#include "../src/io/ConfigLoader.hpp"
#include "../src/physics/WallBoundary.hpp"
#include "../src/solver/core/ConfigPaths.hpp"
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

void print_group_table(const std::string& title,
                       const std::map<std::string, GroupContribution>& grouped) {
    std::cout << title << "\n";
    for (const auto& [name, value] : grouped) {
        std::cout << "  " << std::setw(8) << std::left << name
                  << " net=" << std::setw(14) << std::right << value.net
                  << " positive=" << std::setw(14) << value.positive
                  << " negative=" << std::setw(14) << value.negative
                  << "\n";
    }
}

} // namespace

int main(int argc, char** argv) {
    try {
        const std::string config_path =
            (argc >= 2) ? argv[1] : (test_dcr::source_root() / "config" / "hydro_am_qss_quick.yaml").string();

        dcr::io::Config cfg = dcr::io::ConfigLoader::load(config_path);
        dcr::solver::normalize_input_roots(cfg, config_path);
        cfg.solver.mode = "full_dcr";
        cfg.io.verbose_logging = false;
        if (argc >= 3) {
            cfg.numerics.boundary_max_iterations = std::max(1, std::stoi(argv[2]));
        }

        dcr::atomic::AtomicData atomic_data(cfg);
        const auto& levels = atomic_data.get_levels();
        auto plasma = test_dcr::make_plasma_state(cfg, atomic_data.get_total_states());
        const auto boundary_temperatures = test_dcr::plasma_temperatures_at(cfg, 0.0);
        test_dcr::EEDFContext eedf(boundary_temperatures.electron_eV);

        const double ion_mass_amu = test_dcr::estimate_ion_mass_amu(cfg);
        const auto wall = dcr::physics::compute_wall_recycling(
            cfg.wall.material,
            boundary_temperatures.electron_eV,
            boundary_temperatures.ion_eV,
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
            if (gi >= 0 && gi < static_cast<int>(levels.size())) {
                p_pos[static_cast<size_t>(gi)] = pi;
            }
        }

        const double c_s_M = dcr::physics::calculate_thermal_speed(
            cfg.plasma.neutral_molecule_temperature_eV, boundary.molecule_mass_amu
        );
        const double w_local = std::max(cfg.grid.boundary_poloidal_width_cm, 1e-12);
        const double exM_over_w = c_s_M / w_local;

        std::cout << std::setprecision(8) << std::scientific;
        std::cout << "Config: " << config_path << "\n";
        std::cout << "Boundary local H2 analysis at x=0 cm\n";
        std::cout << "  Te=" << boundary_temperatures.electron_eV
                  << " eV Ti=" << boundary_temperatures.ion_eV
                  << " eV exM_over_w=" << exM_over_w << " s^-1\n";

        double n_h2_total = 0.0;
        double h2_group_s_from_recycling = 0.0;
        double h2_group_exhaust = 0.0;
        double h2_group_chem_total = 0.0;
        std::map<std::string, GroupContribution> h2_grouped;

        for (int gi_row : boundary.mol_bg_indices) {
            const int pi_row = (gi_row >= 0 && gi_row < static_cast<int>(p_pos.size()))
                ? p_pos[static_cast<size_t>(gi_row)] : -1;
            if (pi_row < 0 || pi_row >= nP.size()) continue;

            const double n_row = std::max(nP(pi_row), 0.0);
            n_h2_total += n_row;
            h2_group_s_from_recycling += local.S_background(pi_row);
            h2_group_exhaust += exM_over_w * n_row;

            for (int pi_col = 0; pi_col < nP.size(); ++pi_col) {
                const int gi_col = boundary.P_indices[static_cast<size_t>(pi_col)];
                if (gi_col < 0 || gi_col >= static_cast<int>(levels.size())) continue;
                const double contrib = local.R_full(gi_row, gi_col) * std::max(nP(pi_col), 0.0);
                h2_group_chem_total += contrib;
                add_group_contribution(
                    h2_grouped,
                    classify_source(levels[static_cast<size_t>(gi_col)], boundary.atom_ground, boundary.molecule_ground),
                    contrib
                );
            }
        }

        std::cout << "\nTotal local H2 manifold (all retained H2(v))\n";
        std::cout << "  n_H2_total=" << n_h2_total << " cm^-3\n";
        std::cout << "  chemistry_total=" << h2_group_chem_total << " cm^-3 s^-1\n";
        std::cout << "  recycling_source_S=" << h2_group_s_from_recycling << " cm^-3 s^-1\n";
        std::cout << "  exhaust_sink=" << h2_group_exhaust << " cm^-3 s^-1\n";
        std::cout << "  net_balance=(chem + S - exhaust)="
                  << (h2_group_chem_total + h2_group_s_from_recycling - h2_group_exhaust)
                  << " cm^-3 s^-1\n";
        print_group_table("  Chemistry grouped by source state family:", h2_grouped);

        const int gi_h2_ground = boundary.molecule_ground;
        const int pi_h2_ground = (gi_h2_ground >= 0 && gi_h2_ground < static_cast<int>(p_pos.size()))
            ? p_pos[static_cast<size_t>(gi_h2_ground)] : -1;
        if (pi_h2_ground >= 0) {
            const double n_h2_ground = std::max(nP(pi_h2_ground), 0.0);
            std::map<std::string, GroupContribution> h2_ground_grouped;
            std::vector<std::pair<double, std::string>> positive_sources;
            std::vector<std::pair<double, std::string>> negative_sources;
            double chemistry_ground = 0.0;

            for (int pi_col = 0; pi_col < nP.size(); ++pi_col) {
                const int gi_col = boundary.P_indices[static_cast<size_t>(pi_col)];
                if (gi_col < 0 || gi_col >= static_cast<int>(levels.size())) continue;
                const double contrib = local.R_full(gi_h2_ground, gi_col) * std::max(nP(pi_col), 0.0);
                chemistry_ground += contrib;
                add_group_contribution(
                    h2_ground_grouped,
                    classify_source(levels[static_cast<size_t>(gi_col)], boundary.atom_ground, boundary.molecule_ground),
                    contrib
                );
                if (contrib > 0.0) {
                    positive_sources.emplace_back(contrib, levels[static_cast<size_t>(gi_col)].label);
                } else if (contrib < 0.0) {
                    negative_sources.emplace_back(contrib, levels[static_cast<size_t>(gi_col)].label);
                }
            }

            std::sort(positive_sources.begin(), positive_sources.end(),
                      [](const auto& a, const auto& b) { return a.first > b.first; });
            std::sort(negative_sources.begin(), negative_sources.end(),
                      [](const auto& a, const auto& b) { return a.first < b.first; });

            std::cout << "\nGround molecular row: " << levels[static_cast<size_t>(gi_h2_ground)].label << "\n";
            std::cout << "  n_H2(v=0)=" << n_h2_ground << " cm^-3\n";
            std::cout << "  chemistry_row_total=" << chemistry_ground << " cm^-3 s^-1\n";
            std::cout << "  recycling_source_S=" << local.S_background(pi_h2_ground) << " cm^-3 s^-1\n";
            std::cout << "  exhaust_sink=" << (exM_over_w * n_h2_ground) << " cm^-3 s^-1\n";
            std::cout << "  net_balance=(chem + S - exhaust)="
                      << (chemistry_ground + local.S_background(pi_h2_ground) - exM_over_w * n_h2_ground)
                      << " cm^-3 s^-1\n";
            print_group_table("  Row chemistry grouped by source state family:", h2_ground_grouped);

            const size_t top_n = std::min<size_t>(8, positive_sources.size());
            std::cout << "  Top positive source states into H2(v=0):\n";
            for (size_t i = 0; i < top_n; ++i) {
                std::cout << "    " << std::setw(14) << positive_sources[i].first
                          << "  " << positive_sources[i].second << "\n";
            }

            const size_t top_sinks = std::min<size_t>(8, negative_sources.size());
            std::cout << "  Top background sink states from H2(v=0) row chemistry:\n";
            for (size_t i = 0; i < top_sinks; ++i) {
                std::cout << "    " << std::setw(14) << negative_sources[i].first
                          << "  " << negative_sources[i].second << "\n";
            }
        }

        return 0;
    } catch (const std::exception& ex) {
        std::cerr << "inspect_boundary_h2_balance failed: " << ex.what() << "\n";
        return 1;
    }
}
