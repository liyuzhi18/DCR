#include <algorithm>
#include <cassert>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>

#include "../src/atomic/AtomicData.hpp"
#include "../src/physics/Sheath.hpp"
#include "../src/physics/WallBoundary.hpp"
#include "../src/solver/boundary/BoundaryPhase.hpp"
#include "../src/solver/marching/AdaptiveRecycling.hpp"
#include "../src/solver/marching/MarchingDriver.hpp"
#include "TestDCRSetup.hpp"

namespace {

void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

} // namespace

int main() {
    std::cout << "--- Testing MarchingDriver ---\n";

    auto cfg = test_dcr::load_reference_config(/*num_cells=*/8, /*verbose_logging=*/false);
    cfg.grid.length_cm = 1.0;
    cfg.grid.type = "linear";
    cfg.grid.first_cell_cm = 0.0;
    cfg.numerics.adaptive_recycling_domain.enabled = true;
    // Use a relaxed cutoff in this compact regression case so the passive
    // diagnostic exercises the in-domain L1-found branch.
    cfg.numerics.adaptive_recycling_domain.epsilon_A = 0.99;
    cfg.numerics.adaptive_recycling_domain.epsilon_M = 0.5;

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
    const auto direct_L1 = dcr::solver::estimate_recycling_layer_from_atomic_flow(
        history.x_cm,
        history.flowA,
        cfg.numerics.adaptive_recycling_domain.epsilon_A
    );
    const auto direct_LM = dcr::solver::estimate_flow_depletion_layer(
        history.x_cm,
        history.flowM,
        cfg.numerics.adaptive_recycling_domain.epsilon_M
    );

    const auto& levels = atomic_data.get_levels();
    const size_t n_nodes = static_cast<size_t>(cfg.grid.num_cells);

    assert(history.x_cm.size() == n_nodes);
    assert(history.background_full.size() == n_nodes);
    assert(history.flowA.size() == n_nodes);
    assert(history.flowM.size() == n_nodes);
    assert(history.rate_diagnostics.size() == n_nodes);
    assert(history.adaptive_recycling_diagnostics_enabled == true);
    assert(history.adaptive_atomic_flow_fraction.size() == n_nodes);
    assert(history.adaptive_recycling_L1_found == direct_L1.found);
    assert(history.adaptive_recycling_L1_found == true);
    assert(history.adaptive_recycling_L1_index == direct_L1.index);
    assert(std::abs(history.adaptive_recycling_L1_cm - direct_L1.L1_cm) < 1e-14);
    assert(history.adaptive_molecular_flow_fraction.size() == n_nodes);
    assert(history.adaptive_recycling_LM_found == direct_LM.found);
    assert(history.adaptive_recycling_LM_found == true);
    assert(history.adaptive_recycling_LM_index == direct_LM.index);
    assert(std::abs(history.adaptive_recycling_LM_cm - direct_LM.length_cm) < 1e-14);

    std::cout << "[DCR_Solver][test] adaptive passive F_A/L1 diagnostics\n";
    std::cout << "  stored: found=" << history.adaptive_recycling_L1_found
              << " index=" << history.adaptive_recycling_L1_index
              << " L1_cm=" << history.adaptive_recycling_L1_cm << "\n";
    std::cout << "  direct: found=" << direct_L1.found
              << " index=" << direct_L1.index
              << " L1_cm=" << direct_L1.L1_cm << "\n";
    if (!history.adaptive_atomic_flow_fraction.empty()) {
        std::cout << "  F_A[0] stored=" << history.adaptive_atomic_flow_fraction.front()
                  << " direct=" << direct_L1.atomic_flow_fraction.front() << "\n";
        std::cout << "  F_A[end] stored=" << history.adaptive_atomic_flow_fraction.back()
                  << " direct=" << direct_L1.atomic_flow_fraction.back() << "\n";
        std::cout << "  F_M[0] stored=" << history.adaptive_molecular_flow_fraction.front()
                  << " direct=" << direct_LM.flow_fraction.front() << "\n";
        std::cout << "  F_M[end] stored=" << history.adaptive_molecular_flow_fraction.back()
                  << " direct=" << direct_LM.flow_fraction.back() << "\n";
    }
    for (size_t k = 0; k < n_nodes; ++k) {
        assert(std::abs(history.adaptive_atomic_flow_fraction[k]
                        - direct_L1.atomic_flow_fraction[k]) < 1e-14);
        assert(std::abs(history.adaptive_molecular_flow_fraction[k]
                        - direct_LM.flow_fraction[k]) < 1e-14);
    }

    std::vector<double> Te_profile;
    Te_profile.reserve(history.rate_diagnostics.size());
    for (const auto& snap : history.rate_diagnostics) {
        Te_profile.push_back(snap.electron_temperature_eV);
    }
    const double c_s_A = dcr::physics::calculate_thermal_speed(
        cfg.plasma.neutral_atom_temperature_eV,
        boundary.atom_mass_amu
    );
    const double c_s_M = dcr::physics::calculate_thermal_speed(
        cfg.plasma.neutral_molecule_temperature_eV,
        boundary.molecule_mass_amu
    );
    const auto adaptive_profile = dcr::solver::build_adaptive_transport_profile(
        history.x_cm,
        history.background_full,
        levels,
        boundary.ion_indices,
        boundary.atom_bg_indices,
        boundary.mol_bg_indices,
        Te_profile,
        history.adaptive_recycling_L1_cm,
        history.adaptive_recycling_LM_found
            ? history.adaptive_recycling_LM_cm
            : cfg.grid.length_cm,
        c_s_A,
        c_s_M,
        cfg.grid.spatial_exhaust_width_cm
    );
    assert(adaptive_profile.ion_divergence_nuclei_cm3_s.size() == n_nodes);
    assert(history.adaptive_transport_profile_available == true);
    assert(history.adaptive_ion_divergence_nuclei_cm3_s.size() == n_nodes);
    assert(history.adaptive_pump_exhaust_nuclei_cm3_s.size() == n_nodes);
    assert(history.adaptive_neutral_remainder_nuclei_cm3_s.size() == n_nodes);
    assert(history.adaptive_atomic_comp_nuclei_cm3_s.size() == n_nodes);
    assert(history.adaptive_molecular_comp_nuclei_cm3_s.size() == n_nodes);
    assert(adaptive_profile.pump_exhaust_nuclei_cm3_s.size() == n_nodes);
    assert(adaptive_profile.neutral_remainder_nuclei_cm3_s.size() == n_nodes);
    assert(adaptive_profile.atomic_group_nuclei_divergence_cm3_s.size() == n_nodes);
    assert(adaptive_profile.molecular_group_nuclei_divergence_cm3_s.size() == n_nodes);

    std::cout << "[DCR_Solver][test] adaptive transport profile table\n";
    std::cout << "  k x_cm F_A F_M ion_div_nuc pump_nuc P_rem atom_comp_nuc mol_comp_nuc balance_err\n";
    for (size_t k = 0; k < n_nodes; ++k) {
        const double ion = adaptive_profile.ion_divergence_nuclei_cm3_s[k];
        const double pump = adaptive_profile.pump_exhaust_nuclei_cm3_s[k];
        const double rem = adaptive_profile.neutral_remainder_nuclei_cm3_s[k];
        const double atom = adaptive_profile.atomic_group_nuclei_divergence_cm3_s[k];
        const double mol = adaptive_profile.molecular_group_nuclei_divergence_cm3_s[k];
        assert(std::abs(history.adaptive_ion_divergence_nuclei_cm3_s[k] - ion)
               <= 1e-12 * std::max(1.0, std::abs(ion)));
        assert(std::abs(history.adaptive_pump_exhaust_nuclei_cm3_s[k] - pump)
               <= 1e-12 * std::max(1.0, std::abs(pump)));
        assert(std::abs(history.adaptive_neutral_remainder_nuclei_cm3_s[k] - rem)
               <= 1e-12 * std::max(1.0, std::abs(rem)));
        assert(std::abs(history.adaptive_atomic_comp_nuclei_cm3_s[k] - atom)
               <= 1e-12 * std::max(1.0, std::abs(atom)));
        assert(std::abs(history.adaptive_molecular_comp_nuclei_cm3_s[k] - mol)
               <= 1e-12 * std::max(1.0, std::abs(mol)));
        const double balance_err = rem - (pump - ion);
        const double partition_err = rem - (atom + mol);
        std::cout << "  " << k
                  << " " << history.x_cm[k]
                  << " " << history.adaptive_atomic_flow_fraction[k]
                  << " " << history.adaptive_molecular_flow_fraction[k]
                  << " " << ion
                  << " " << pump
                  << " " << rem
                  << " " << atom
                  << " " << mol
                  << " " << balance_err
                  << "\n";
        assert(std::abs(balance_err) <= 1e-8 * std::max(1.0, std::abs(rem)));
        assert(std::abs(partition_err) <= 1e-8 * std::max(1.0, std::abs(rem)));
    }

    auto cfg_probe = cfg;
    cfg_probe.io.verbose_logging = false;
    std::cout << "[DCR_Solver][test] rerunning with adaptive profile pointer "
              << "for diagnostic-only cell-solve plumbing\n";
    const auto history_probe = dcr::solver::run_full_marching(
        cfg_probe,
        atomic_data,
        plasma,
        eedf.grid,
        boundary,
        &adaptive_profile
    );
    assert(history_probe.x_cm.size() == history.x_cm.size());
    assert(history_probe.flowA.size() == history.flowA.size());
    for (size_t k = 0; k < history.x_cm.size(); ++k) {
        assert(std::abs(history_probe.x_cm[k] - history.x_cm[k]) < 1e-14);
    }
    assert(history_probe.adaptive_recycling_diagnostics_enabled == true);
    std::cout << "[DCR_Solver][test] diagnostic-only rerun F_A[end]="
              << history_probe.adaptive_atomic_flow_fraction.back()
              << " baseline F_A[end]=" << history.adaptive_atomic_flow_fraction.back()
              << "\n";

    auto cfg_variable = cfg;
    cfg_variable.io.verbose_logging = false;
    cfg_variable.numerics.adaptive_recycling_domain.closure_mode = "variable_nuclei_balance";
    const auto history_variable = dcr::solver::run_full_marching(
        cfg_variable,
        atomic_data,
        plasma,
        eedf.grid,
        boundary,
        &adaptive_profile
    );
    assert(history_variable.background_full.size() == n_nodes);
    assert(history_variable.flowA.size() == n_nodes);
    assert(history_variable.flowM.size() == n_nodes);
    assert(history_variable.variable_nuclei_balance_enabled == true);
    assert(history_variable.variable_ion_divergence_nuclei_cm3_s.size() == n_nodes);
    assert(history_variable.variable_flowA_divergence_nuclei_cm3_s.size() == n_nodes);
    assert(history_variable.variable_flowM_divergence_nuclei_cm3_s.size() == n_nodes);
    assert(history_variable.variable_neutral_exhaust_nuclei_cm3_s.size() == n_nodes);
    assert(history_variable.variable_balance_rhs_cm3_s.size() == n_nodes);
    assert(history_variable.variable_balance_residual_cm3_s.size() == n_nodes);
    std::cout << "[DCR_Solver][test] variable nuclei balance totals\n";
    std::cout << "  k x_cm n_nuc_total rel_to_prescribed\n";
    double max_rel_to_prescribed = 0.0;
    double max_rel_balance_residual = 0.0;
    for (size_t k = 0; k < n_nodes; ++k) {
        const double total_nuclei =
            test_dcr::nuclei_total_full(history_variable.background_full[k], levels) +
            test_dcr::nuclei_total_compact(history_variable.flowA[k], boundary.A_indices, levels) +
            test_dcr::nuclei_total_compact(history_variable.flowM[k], boundary.M_indices, levels);
        const double rel_to_prescribed = std::abs(total_nuclei - cfg.plasma.total_density)
                                      / std::max(1.0, cfg.plasma.total_density);
        max_rel_to_prescribed = std::max(max_rel_to_prescribed, rel_to_prescribed);
        std::cout << "  " << k
                  << " " << history_variable.x_cm[k]
                  << " " << total_nuclei
                  << " " << rel_to_prescribed
                  << "\n";
        require(std::isfinite(total_nuclei), "Marching nuclei total is not finite");
        require(total_nuclei >= 0.0, "Marching nuclei total is negative");
        if (k == 0) {
            assert(std::isnan(history_variable.variable_balance_residual_cm3_s[k]));
        } else {
            assert(std::isfinite(history_variable.variable_ion_divergence_nuclei_cm3_s[k]));
            assert(std::isfinite(history_variable.variable_flowA_divergence_nuclei_cm3_s[k]));
            assert(std::isfinite(history_variable.variable_flowM_divergence_nuclei_cm3_s[k]));
            assert(std::isfinite(history_variable.variable_neutral_exhaust_nuclei_cm3_s[k]));
            assert(std::isfinite(history_variable.variable_balance_rhs_cm3_s[k]));
            assert(std::isfinite(history_variable.variable_balance_residual_cm3_s[k]));
            const double balance_scale = std::max({
                1.0,
                std::abs(history_variable.variable_ion_divergence_nuclei_cm3_s[k]),
                std::abs(history_variable.variable_flowA_divergence_nuclei_cm3_s[k]),
                std::abs(history_variable.variable_flowM_divergence_nuclei_cm3_s[k]),
                std::abs(history_variable.variable_neutral_exhaust_nuclei_cm3_s[k])
            });
            max_rel_balance_residual = std::max(
                max_rel_balance_residual,
                std::abs(history_variable.variable_balance_residual_cm3_s[k]) /
                    balance_scale
            );
        }
    }
    std::cout << "  max_rel_to_prescribed=" << max_rel_to_prescribed << "\n";
    std::cout << "  max_rel_balance_residual=" << max_rel_balance_residual << "\n";
    require(max_rel_to_prescribed > 1e-6,
            "Variable nuclei marcher was projected to the prescribed density");
    require(max_rel_balance_residual < 1e-10,
            "Variable nuclei marcher did not satisfy the local balance row");

    auto cfg_outer = cfg_variable;
    cfg_outer.numerics.adaptive_recycling_domain.max_outer_iterations = 2;
    cfg_outer.numerics.adaptive_recycling_domain.L1_relative_tolerance = 0.0;
    const auto history_outer = dcr::solver::run_adaptive_variable_marching(
        cfg_outer,
        atomic_data,
        plasma,
        eedf.grid,
        boundary
    );
    std::cout << "[DCR_Solver][test] adaptive variable outer loop\n";
    std::cout << "  enabled=" << history_outer.adaptive_outer_loop_enabled
              << " converged=" << history_outer.adaptive_outer_loop_converged
              << " iterations=" << history_outer.adaptive_outer_iterations
              << " L1_first=" << history_outer.adaptive_outer_L1_history_cm.front()
              << " L1_last=" << history_outer.adaptive_outer_L1_history_cm.back()
              << " LM_first=" << history_outer.adaptive_outer_LM_history_cm.front()
              << " LM_last=" << history_outer.adaptive_outer_LM_history_cm.back()
              << " F_A_end_last=" << history_outer.adaptive_outer_F_A_end_history.back()
              << " F_M_end_last=" << history_outer.adaptive_outer_F_M_end_history.back()
              << "\n";
    assert(history_outer.variable_nuclei_balance_enabled == true);
    assert(history_outer.adaptive_outer_loop_enabled == true);
    assert(history_outer.adaptive_outer_iterations == 2);
    assert(history_outer.adaptive_outer_L1_history_cm.size() == 2);
    assert(history_outer.adaptive_outer_LM_history_cm.size() == 2);
    assert(history_outer.adaptive_outer_F_A_end_history.size() == 2);
    assert(history_outer.adaptive_outer_F_M_end_history.size() == 2);
    for (int i = 0; i < history_outer.adaptive_outer_iterations; ++i) {
        assert(std::isfinite(history_outer.adaptive_outer_L1_history_cm[static_cast<size_t>(i)]));
        assert(std::isfinite(history_outer.adaptive_outer_LM_history_cm[static_cast<size_t>(i)]));
        assert(std::isfinite(history_outer.adaptive_outer_F_A_end_history[static_cast<size_t>(i)]));
        assert(std::isfinite(history_outer.adaptive_outer_F_M_end_history[static_cast<size_t>(i)]));
    }

    for (size_t k = 1; k < history.x_cm.size(); ++k) {
        assert(history.x_cm[k] > history.x_cm[k - 1]);
    }
    assert(std::abs(history.x_cm.back() - cfg.grid.length_cm) < 1e-10);

    const int total_states = atomic_data.get_total_states();
    for (size_t k = 0; k < n_nodes; ++k) {
        const auto& bg = history.background_full[k];
        const auto& a = history.flowA[k];
        const auto& m = history.flowM[k];

        assert(bg.size() == total_states);
        assert(a.size() == static_cast<int>(boundary.A_indices.size()));
        assert(m.size() == static_cast<int>(boundary.M_indices.size()));
        test_dcr::assert_all_finite_nonnegative(bg);
        test_dcr::assert_all_finite_nonnegative(a);
        test_dcr::assert_all_finite_nonnegative(m);

        const auto& rates = history.rate_diagnostics[k];
        assert(std::isfinite(rates.electron_temperature_eV));
        assert(std::isfinite(rates.ion_temperature_eV));
        assert(std::isfinite(rates.electron_density_cm3));
        assert(rates.electron_density_cm3 >= 0.0);
        assert(std::isfinite(rates.atomic_effective.scd_cm3_s));
        assert(std::isfinite(rates.atomic_effective.acd_cm3_s));
        assert(rates.atomic_effective.scd_cm3_s >= 0.0);
        assert(rates.atomic_effective.acd_cm3_s >= 0.0);
        assert(std::isfinite(rates.atomic_qss.max_transport_to_local_ratio));
        assert(std::isfinite(rates.atomic_qss.max_transport_to_loss_frequency_ratio));

        const double total_nuclei =
            test_dcr::nuclei_total_full(bg, levels) +
            test_dcr::nuclei_total_compact(a, boundary.A_indices, levels) +
            test_dcr::nuclei_total_compact(m, boundary.M_indices, levels);
        assert(std::isfinite(total_nuclei));
        assert(total_nuclei >= 0.0);
    }

    // Ensure marching produced a state update from x=0 to first interior node.
    if (n_nodes > 1) {
        const double diff_bg = (history.background_full[1] - history.background_full[0]).norm();
        const double diff_a = (history.flowA[1] - history.flowA[0]).norm();
        const double diff_m = (history.flowM[1] - history.flowM[0]).norm();
        assert((diff_bg + diff_a + diff_m) > 0.0);
    }

    std::cout << "[PASS] MarchingDriver checks.\n";
    return 0;
}
