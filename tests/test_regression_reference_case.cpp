#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <vector>

#include "../src/atomic/AtomicData.hpp"
#include "../src/physics/WallBoundary.hpp"
#include "../src/solver/boundary/BoundaryPhase.hpp"
#include "../src/solver/marching/MarchingDriver.hpp"
#include "TestDCRSetup.hpp"

namespace {

struct BgGroups {
    double H_plus = 0.0;
    double H_minus = 0.0;
    double H = 0.0;
    double H2 = 0.0;
    double H2_plus = 0.0;
};

BgGroups summarize_bg_groups(const dcr::base::Vector& bg_full,
                             const std::vector<int>& p_indices,
                             const std::vector<dcr::atomic::EnergyLevel>& levels) {
    BgGroups g;
    for (int gi : p_indices) {
        if (gi < 0 || gi >= bg_full.size() || gi >= static_cast<int>(levels.size())) continue;
        const auto& lvl = levels[gi];
        const double n = std::max(bg_full(gi), 0.0);
        if (lvl.type == dcr::atomic::SpeciesType::Ion && lvl.charge > 0) {
            if (lvl.atomicity >= 2) g.H2_plus += n;
            else g.H_plus += n;
        } else if (lvl.type == dcr::atomic::SpeciesType::Ion && lvl.charge < 0) {
            g.H_minus += n;
        } else if (lvl.type == dcr::atomic::SpeciesType::Atom && lvl.charge == 0) {
            g.H += n;
        } else if (lvl.type == dcr::atomic::SpeciesType::Molecule && lvl.charge == 0) {
            g.H2 += n;
        }
    }
    return g;
}

double integrate_trapezoid(const std::vector<double>& x,
                           const std::vector<double>& values) {
    double integral = 0.0;
    for (size_t k = 1; k < x.size() && k < values.size(); ++k) {
        integral += 0.5 * (x[k] - x[k - 1]) * (values[k - 1] + values[k]);
    }
    return integral;
}

void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

void require_close(double actual, double expected, double relative_tolerance,
                   const char* message) {
    const double scale = std::max({1.0, std::abs(actual), std::abs(expected)});
    require(std::isfinite(actual) && std::abs(actual - expected) <= relative_tolerance * scale,
            message);
}

} // namespace

int main() {
    std::cout << "--- Testing Reference Regression Case ---\n";

    auto cfg = test_dcr::load_reference_config(/*num_cells=*/3, /*verbose_logging=*/false);
    cfg.grid.length_cm = 0.5;
    cfg.grid.type = "linear";
    cfg.grid.first_cell_cm = 0.0;
    cfg.numerics.max_iterations = 400;
    cfg.numerics.tolerance = std::max(1e-8, cfg.numerics.tolerance);

    dcr::atomic::AtomicData atomic_data(cfg);
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
    const auto history = dcr::solver::run_full_marching(cfg, atomic_data, plasma, eedf.grid, boundary);
    const auto& levels = atomic_data.get_levels();

    const double bg_boundary = test_dcr::nuclei_total_selected(boundary.population, boundary.P_indices, levels);
    double flow_boundary = 0.0;
    if (boundary.explicit_recycling) {
        flow_boundary = test_dcr::nuclei_total_selected(boundary.population, boundary.R_indices, levels);
    } else {
        flow_boundary = test_dcr::nuclei_total_compact(boundary.flowA_last, boundary.A_indices, levels)
                      + test_dcr::nuclei_total_compact(boundary.flowM_last, boundary.M_indices, levels);
    }
    const double total_boundary = bg_boundary + flow_boundary;
    const double boundary_flow_frac = flow_boundary / std::max(1.0, total_boundary);

    const size_t k1 = std::min<size_t>(1, history.x_cm.size() - 1);
    const size_t kf = history.x_cm.size() - 1;
    const double flowA_k1 = test_dcr::nuclei_total_compact(history.flowA[k1], boundary.A_indices, levels);
    const double flowM_k1 = test_dcr::nuclei_total_compact(history.flowM[k1], boundary.M_indices, levels);
    const double total_k1 = test_dcr::nuclei_total_full(history.background_full[k1], levels) + flowA_k1 + flowM_k1;
    const double flowA_k1_frac = flowA_k1 / std::max(1.0, total_k1);
    const double flowM_k1_frac = flowM_k1 / std::max(1.0, total_k1);

    const double flowA_kf = test_dcr::nuclei_total_compact(history.flowA[kf], boundary.A_indices, levels);
    const double flowM_kf = test_dcr::nuclei_total_compact(history.flowM[kf], boundary.M_indices, levels);
    const double total_kf = test_dcr::nuclei_total_full(history.background_full[kf], levels) + flowA_kf + flowM_kf;
    const auto bg_last = summarize_bg_groups(history.background_full[kf], boundary.P_indices, levels);
    const double hplus_bg_frac = bg_last.H_plus / std::max(1.0, total_kf);
    const double h2_bg_frac = bg_last.H2 / std::max(1.0, total_kf);

    std::vector<BgGroups> bg_groups;
    std::vector<double> flowA_nuclei;
    std::vector<double> flowM_nuclei;
    std::vector<double> ionization_rates;
    std::vector<double> recombination_rates;
    bg_groups.reserve(history.x_cm.size());
    flowA_nuclei.reserve(history.x_cm.size());
    flowM_nuclei.reserve(history.x_cm.size());
    ionization_rates.reserve(history.x_cm.size());
    recombination_rates.reserve(history.x_cm.size());
    for (size_t k = 0; k < history.x_cm.size(); ++k) {
        bg_groups.push_back(summarize_bg_groups(
            history.background_full[k], boundary.P_indices, levels));
        flowA_nuclei.push_back(test_dcr::nuclei_total_compact(
            history.flowA[k], boundary.A_indices, levels));
        flowM_nuclei.push_back(test_dcr::nuclei_total_compact(
            history.flowM[k], boundary.M_indices, levels));
        const auto& rates = history.rate_diagnostics[k];
        const int atom_index = rates.atomic_effective.atom_ground_index;
        const int ion_index = rates.atomic_effective.ion_ground_index;
        const double atom_density = (atom_index >= 0 && atom_index < history.background_full[k].size())
            ? history.background_full[k](atom_index) : 0.0;
        const double ion_density = (ion_index >= 0 && ion_index < history.background_full[k].size())
            ? history.background_full[k](ion_index) : 0.0;
        ionization_rates.push_back(
            rates.electron_density_cm3 * atom_density * rates.atomic_effective.scd_cm3_s);
        recombination_rates.push_back(
            rates.electron_density_cm3 * ion_density * rates.atomic_effective.acd_cm3_s);
    }
    const double boundary_atom_recycling_flux = boundary.u_A * flowA_nuclei.front();
    const double boundary_molecule_recycling_flux = boundary.u_M * flowM_nuclei.front();
    const double integrated_ionization = integrate_trapezoid(history.x_cm, ionization_rates);
    const double integrated_recombination = integrate_trapezoid(history.x_cm, recombination_rates);

    // Golden values for the e1eefb1 wall-to-upstream marching topology with
    // the historical source routing, corrected chemistry, and separated
    // spatial exhaust width.
    constexpr std::array<std::array<double, 7>, 3> expected_nodes{{
        {{2.1028994567409853e14, 3.0192681550577207e13, 4.7642968924230704e5,
          3.9058472124359678e12, 1.7278929278705434e6, 1.1120024979198969e14,
          6.4050542587771025e14}},
        {{3.9436716179215100e14, 3.8532279216789859e13, 6.3118255282270571e5,
          2.8220179705949321e12, 2.2005664873798518e6, 1.0851357956345220e14,
          4.5294294002348569e14}},
        {{5.4990098177240744e14, 3.4706932224067312e13, 4.4257703874651680e5,
          1.9300268845116836e12, 1.9808645713919101e6, 1.0546717648713417e14,
          3.0606485288134912e14}}
    }};
    constexpr double expected_atom_recycling_flux = 4.5724760851555071e20;
    constexpr double expected_molecule_recycling_flux = 1.4017289371190036e20;
    constexpr double expected_integrated_ionization = 1.8343521327717514e18;
    constexpr double expected_integrated_recombination = 6.8007120039750864e16;
    constexpr double golden_relative_tolerance = 1.0e-4;

    require(boundary.converged, "Reference boundary solve did not converge");
    require(history.x_cm.size() == expected_nodes.size(),
            "Reference marcher node count changed");
    for (size_t k = 0; k < expected_nodes.size(); ++k) {
        const std::array<double, 7> actual{{
            bg_groups[k].H_plus, bg_groups[k].H, bg_groups[k].H2,
            bg_groups[k].H2_plus, bg_groups[k].H_minus,
            flowA_nuclei[k], flowM_nuclei[k]
        }};
        for (size_t j = 0; j < actual.size(); ++j) {
            require_close(actual[j], expected_nodes[k][j], golden_relative_tolerance,
                          "Reference marcher population/flow regression");
        }

        const double total_nuclei =
            test_dcr::nuclei_total_full(history.background_full[k], levels) +
            flowA_nuclei[k] + flowM_nuclei[k];
        require_close(total_nuclei, cfg.plasma.total_density, 1.0e-12,
                      "Reference marcher violated prescribed nuclei density");
    }
    require_close(boundary_atom_recycling_flux, expected_atom_recycling_flux,
                  golden_relative_tolerance, "Atomic recycling flux regression");
    require_close(boundary_molecule_recycling_flux, expected_molecule_recycling_flux,
                  golden_relative_tolerance, "Molecular recycling flux regression");
    require_close(integrated_ionization, expected_integrated_ionization,
                  golden_relative_tolerance, "Integrated ionization regression");
    require_close(integrated_recombination, expected_integrated_recombination,
                  golden_relative_tolerance, "Integrated recombination regression");

    // Coarse regression envelope: catches broken chemistry/mass-balance while
    // tolerating expected physics-model tuning.
    assert(boundary_flow_frac > 0.05 && boundary_flow_frac < 0.95);
    assert(flowA_k1_frac >= 0.0 && flowA_k1_frac < 0.95);
    assert(flowM_k1_frac >= 0.0 && flowM_k1_frac < 0.95);
    assert(hplus_bg_frac >= 0.0 && hplus_bg_frac < 0.95);
    assert(h2_bg_frac >= 0.0 && h2_bg_frac < 0.95);

    for (size_t k = 0; k < history.x_cm.size(); ++k) {
        const double total_k = test_dcr::nuclei_total_full(history.background_full[k], levels)
                             + test_dcr::nuclei_total_compact(history.flowA[k], boundary.A_indices, levels)
                             + test_dcr::nuclei_total_compact(history.flowM[k], boundary.M_indices, levels);
        assert(std::isfinite(total_k));
        assert(total_k > 0.0);
        assert(total_k < 1e3 * std::max(1.0, total_boundary));
    }

    if (std::getenv("DCR_TEST_DUMP")) {
        std::cout << std::setprecision(17);
        std::cout << "boundary_converged=" << boundary.converged
                  << " boundary_iterations=" << boundary.iterations << "\n";
        std::cout << "boundary_flow_frac=" << boundary_flow_frac << "\n";
        std::cout << "flowA_k1_frac=" << flowA_k1_frac << "\n";
        std::cout << "flowM_k1_frac=" << flowM_k1_frac << "\n";
        std::cout << "hplus_bg_frac=" << hplus_bg_frac << "\n";
        std::cout << "h2_bg_frac=" << h2_bg_frac << "\n";
        for (size_t k = 0; k < history.x_cm.size(); ++k) {
            std::cout << "node[" << k << "]="
                      << bg_groups[k].H_plus << ","
                      << bg_groups[k].H << ","
                      << bg_groups[k].H2 << ","
                      << bg_groups[k].H2_plus << ","
                      << bg_groups[k].H_minus << ","
                      << flowA_nuclei[k] << ","
                      << flowM_nuclei[k] << "\n";
        }
        for (size_t k = 1;
             k < history.x_cm.size() &&
             k < history.ion_divergence_closure_nuclei_cm3_s.size();
             ++k) {
            std::cout << "L_I[" << k << "]="
                      << history.ion_divergence_closure_nuclei_cm3_s[k] << "\n";
        }
        std::cout << "boundary_atom_recycling_flux=" << boundary_atom_recycling_flux << "\n";
        std::cout << "boundary_molecule_recycling_flux=" << boundary_molecule_recycling_flux << "\n";
        std::cout << "integrated_ionization=" << integrated_ionization << "\n";
        std::cout << "integrated_recombination=" << integrated_recombination << "\n";
    }

    std::cout << "[PASS] Reference regression checks.\n";
    return 0;
}
