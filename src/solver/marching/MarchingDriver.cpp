#include "MarchingDriver.hpp"

#include "AdaptiveRecycling.hpp"
#include "CellSolve.hpp"
#include "MarchingDiagnostics.hpp"

#include "../core/TemperatureProfile.hpp"
#include "../../physics/Sheath.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <string>
#include <utility>

namespace dcr::solver {

namespace {

// Sum_{i=0}^{n-1} first * ratio^i
double geometric_sum(double first, double ratio, int n_terms) {
    if (n_terms <= 0) return 0.0;
    if (std::abs(ratio - 1.0) < 1e-12) return first * static_cast<double>(n_terms);
    return first * (std::pow(ratio, n_terms) - 1.0) / (ratio - 1.0);
}

// Build per-cell dx from the configured mesh settings.
std::vector<double> build_step_sizes_cm(const dcr::io::Config& config) {
    const int n_steps = std::max(0, config.grid.num_cells - 1);
    std::vector<double> dx(static_cast<size_t>(n_steps), 0.0);
    if (n_steps == 0) return dx;

    const double L = std::max(0.0, config.grid.length_cm);
    if (L <= 0.0) return dx;

    if (config.grid.type == "log" && config.grid.first_cell_cm > 0.0) {
        const double d0 = config.grid.first_cell_cm;
        if (n_steps == 1) {
            dx[0] = L;
            return dx;
        }
        // If requested first cell is too large for a geometric progression over full length, fallback linear.
        if (d0 * static_cast<double>(n_steps) >= L) {
            const double dlin = L / static_cast<double>(n_steps);
            std::fill(dx.begin(), dx.end(), dlin);
            return dx;
        }

        // Solve for geometric ratio r so sum_i d0 r^i = L.
        double lo = 1.0;
        double hi = 2.0;
        while (geometric_sum(d0, hi, n_steps) < L && hi < 1.0e6) {
            hi *= 2.0;
        }
        if (geometric_sum(d0, hi, n_steps) < L) {
            const double dlin = L / static_cast<double>(n_steps);
            std::fill(dx.begin(), dx.end(), dlin);
            return dx;
        }
        for (int it = 0; it < 100; ++it) {
            const double mid = 0.5 * (lo + hi);
            if (geometric_sum(d0, mid, n_steps) < L) lo = mid;
            else hi = mid;
        }
        const double r = 0.5 * (lo + hi);
        double s = 0.0;
        for (int i = 0; i < n_steps; ++i) {
            dx[static_cast<size_t>(i)] = d0 * std::pow(r, i);
            s += dx[static_cast<size_t>(i)];
        }
        // Force exact integral length by adjusting the last cell.
        dx.back() += (L - s);
        if (dx.back() < 0.0) dx.back() = 0.0;
        return dx;
    }

    const double dlin = L / static_cast<double>(n_steps);
    std::fill(dx.begin(), dx.end(), dlin);
    return dx;
}

void log_rate_snapshot(double x_cm,
                       const RateDiagnosticSnapshot& snapshot,
                       const std::vector<dcr::atomic::EnergyLevel>& levels) {
    (void)levels;
    if (!snapshot.atomic_effective.valid) return;
    std::cout << "[DCR_Solver][Rates][atomic] x=" << x_cm
              << " cm SCD=" << snapshot.atomic_effective.scd_cm3_s
              << " ACD=" << snapshot.atomic_effective.acd_cm3_s
              << "\n";
}

double positive_sum(const dcr::base::Vector& v) {
    double total = 0.0;
    for (int i = 0; i < v.size(); ++i) total += std::max(v(i), 0.0);
    return total;
}

struct SubpopulationSummary {
    double H_plus = 0.0;
    double H = 0.0;
    double H2 = 0.0;
    double H2_plus = 0.0;
    double H_minus = 0.0;
    double nuclei_total = 0.0;
};

SubpopulationSummary summarize_subpopulations(
    const dcr::base::Vector& background_full,
    const dcr::base::Vector& flowA,
    const dcr::base::Vector& flowM,
    const BoundaryPhaseResult& boundary,
    const std::vector<dcr::atomic::EnergyLevel>& levels) {

    SubpopulationSummary s;
    const int nmax = std::min<int>(background_full.size(), static_cast<int>(levels.size()));
    for (int gi = 0; gi < nmax; ++gi) {
        const auto& level = levels[static_cast<size_t>(gi)];
        const double n = std::max(background_full(gi), 0.0);
        const double mu = static_cast<double>(std::max(1, level.atomicity));
        s.nuclei_total += mu * n;
        if (level.type == dcr::atomic::SpeciesType::Ion && level.charge > 0) {
            if (level.atomicity >= 2) s.H2_plus += n;
            else s.H_plus += n;
        } else if (level.type == dcr::atomic::SpeciesType::Ion && level.charge < 0) {
            s.H_minus += n;
        } else if (level.type == dcr::atomic::SpeciesType::Atom && level.charge == 0) {
            s.H += n;
        } else if (level.type == dcr::atomic::SpeciesType::Molecule && level.charge == 0) {
            s.H2 += n;
        }
    }

    for (size_t j = 0; j < boundary.A_indices.size() && static_cast<int>(j) < flowA.size(); ++j) {
        const int gi = boundary.A_indices[j];
        if (gi < 0 || gi >= static_cast<int>(levels.size())) continue;
        const double mu = static_cast<double>(std::max(1, levels[static_cast<size_t>(gi)].atomicity));
        s.nuclei_total += mu * std::max(flowA(static_cast<int>(j)), 0.0);
    }
    for (size_t j = 0; j < boundary.M_indices.size() && static_cast<int>(j) < flowM.size(); ++j) {
        const int gi = boundary.M_indices[j];
        if (gi < 0 || gi >= static_cast<int>(levels.size())) continue;
        const double mu = static_cast<double>(std::max(1, levels[static_cast<size_t>(gi)].atomicity));
        s.nuclei_total += mu * std::max(flowM(static_cast<int>(j)), 0.0);
    }
    return s;
}

double molecular_transport_loss_rate(const BoundaryPhaseResult& boundary,
                                      const dcr::base::Vector& flowM_left,
                                      const dcr::base::Vector& flowM_right,
                                      double dx_cm) {
    if (!(boundary.u_M > 0.0) || !(dx_cm > 0.0)) return 0.0;
    const double lhs = boundary.u_M * (positive_sum(flowM_right) - positive_sum(flowM_left)) / dx_cm;
    return std::max(0.0, -lhs);
}

double marching_domain_length_cm(const dcr::io::Config& config,
                                 const MarchingHistory& history) {
    const double history_length = history.x_cm.empty() ? 0.0 : history.x_cm.back();
    return std::max(config.grid.length_cm, history_length);
}

double effective_L1_cm(const dcr::io::Config& config,
                       const MarchingHistory& history) {
    return history.adaptive_recycling_L1_found
        ? history.adaptive_recycling_L1_cm
        : marching_domain_length_cm(config, history);
}

double effective_LM_cm(const dcr::io::Config& config,
                       const MarchingHistory& history) {
    return history.adaptive_recycling_LM_found
        ? history.adaptive_recycling_LM_cm
        : marching_domain_length_cm(config, history);
}

double terminal_fraction(const std::vector<double>& fraction) {
    return fraction.empty() ? std::numeric_limits<double>::quiet_NaN() : fraction.back();
}

double relative_change(double old_value, double new_value) {
    const double denom = std::max({1.0e-30, std::abs(old_value), std::abs(new_value)});
    return std::abs(new_value - old_value) / denom;
}

bool build_profile_from_history(const dcr::io::Config& config,
                                const dcr::atomic::AtomicData& atomic_data,
                                const BoundaryPhaseResult& boundary,
                                const MarchingHistory& history,
                                AdaptiveTransportProfile& profile) {
    if (history.x_cm.empty() || history.background_full.empty()) return false;

    const auto& levels = atomic_data.get_levels();
    std::vector<double> Te_eV;
    std::vector<double> Ti_eV;
    Te_eV.reserve(history.rate_diagnostics.size());
    for (const auto& snap : history.rate_diagnostics) {
        Te_eV.push_back(snap.electron_temperature_eV);
        Ti_eV.push_back(snap.ion_temperature_eV);
    }
    if (Te_eV.empty()) return false;

    const double c_s_M = dcr::physics::calculate_thermal_speed(
        config.plasma.neutral_molecule_temperature_eV,
        boundary.molecule_mass_amu
    );
    profile = build_adaptive_transport_profile(
        history.x_cm,
        history.background_full,
        levels,
        boundary.ion_indices,
        boundary.atom_bg_indices,
        boundary.mol_bg_indices,
        Te_eV,
        Ti_eV,
        effective_L1_cm(config, history),
        effective_LM_cm(config, history),
        c_s_M,
        config.grid.spatial_exhaust_width_cm,
        config.numerics.adaptive_recycling_domain.ion_velocity_floor_fraction
    );
    return true;
}

void append_outer_history(const dcr::io::Config& config,
                          const MarchingHistory& history,
                          std::vector<double>& L1_history_cm,
                          std::vector<double>& LM_history_cm,
                          std::vector<double>& F_A_end_history,
                          std::vector<double>& F_M_end_history) {
    L1_history_cm.push_back(effective_L1_cm(config, history));
    LM_history_cm.push_back(effective_LM_cm(config, history));
    F_A_end_history.push_back(terminal_fraction(history.adaptive_atomic_flow_fraction));
    F_M_end_history.push_back(terminal_fraction(history.adaptive_molecular_flow_fraction));
}

} // namespace

MarchingHistory run_full_marching(
    const dcr::io::Config& config,
    const dcr::atomic::AtomicData& atomic_data,
    const dcr::state::PlasmaState& plasma,
    const EEDFGridView& grid,
    const BoundaryPhaseResult& boundary,
    const AdaptiveTransportProfile* adaptive_profile) {

    MarchingHistory history;

    // Build mesh spacing and x coordinates.
    const auto dx = build_step_sizes_cm(config);
    const int n_nodes = static_cast<int>(dx.size()) + 1;
    if (n_nodes <= 0) return history;
    history.boundary_elapsed_seconds = boundary.elapsed_seconds;
    history.boundary_iterations = boundary.iterations;

    history.x_cm.assign(static_cast<size_t>(n_nodes), 0.0);
    for (int i = 1; i < n_nodes; ++i) {
        history.x_cm[static_cast<size_t>(i)] = history.x_cm[static_cast<size_t>(i - 1)] + dx[static_cast<size_t>(i - 1)];
    }

    const auto& levels = atomic_data.get_levels();
    const int total_states = atomic_data.get_total_states();
    const int Pn = static_cast<int>(boundary.P_indices.size());
    if (Pn == 0) {
        if (config.io.verbose_logging) {
            std::cout << "[DCR_Solver] Marching skipped: no background states in P block.\n";
        }
        return history;
    }

    const bool variable_nuclei_balance_closure =
        config.numerics.adaptive_recycling_domain.closure_mode == "variable_nuclei_balance";
    history.variable_nuclei_balance_enabled = variable_nuclei_balance_closure;
    if (config.io.verbose_logging) {
        std::cout << "[DCR_Solver] Marching mode: "
                  << (variable_nuclei_balance_closure
                          ? "hybrid variable nuclei-balance closure"
                          : "fixed prescribed-density closure")
                  << "\n";
    }

    const AtomicRateCalculator rate_calculator(atomic_data);
    // Initialize compact background state from boundary solution.
    dcr::base::Vector nP = dcr::base::Vector::Zero(Pn);
    for (int i = 0; i < Pn; ++i) {
        const int gi = boundary.P_indices[static_cast<size_t>(i)];
        if (gi >= 0 && gi < boundary.population.size()) {
            nP(i) = std::max(boundary.population(gi), 0.0);
        }
    }

    // Initialize recycling flows from boundary output.
    dcr::base::Vector flowA = dcr::base::Vector::Zero(static_cast<int>(boundary.A_indices.size()));
    dcr::base::Vector flowM = dcr::base::Vector::Zero(static_cast<int>(boundary.M_indices.size()));
    if (boundary.explicit_recycling) {
        for (size_t i = 0; i < boundary.A_indices.size(); ++i) {
            const int gi = boundary.A_indices[i];
            if (gi >= 0 && gi < boundary.population.size()) {
                flowA(static_cast<int>(i)) = std::max(boundary.population(gi), 0.0);
            }
        }
        for (size_t i = 0; i < boundary.M_indices.size(); ++i) {
            const int gi = boundary.M_indices[i];
            if (gi >= 0 && gi < boundary.population.size()) {
                flowM(static_cast<int>(i)) = std::max(boundary.population(gi), 0.0);
            }
        }
    } else if (boundary.have_flow_last) {
        flowA = boundary.flowA_last;
        flowM = boundary.flowM_last;
    }

    const dcr::base::Vector nP_initial = nP;
    const dcr::base::Vector flowA_initial = flowA;
    const dcr::base::Vector flowM_initial = flowM;

    history.background_full.reserve(static_cast<size_t>(n_nodes));
    history.flowA.reserve(static_cast<size_t>(n_nodes));
    history.flowM.reserve(static_cast<size_t>(n_nodes));
    history.rate_diagnostics.reserve(static_cast<size_t>(n_nodes));
    history.cell_elapsed_seconds.reserve(dx.size());
    history.cell_iterations.reserve(dx.size());
    history.cell_converged.reserve(dx.size());
    if (variable_nuclei_balance_closure) {
        history.variable_ion_divergence_nuclei_cm3_s.reserve(static_cast<size_t>(n_nodes));
        history.variable_flowA_divergence_nuclei_cm3_s.reserve(static_cast<size_t>(n_nodes));
        history.variable_flowM_divergence_nuclei_cm3_s.reserve(static_cast<size_t>(n_nodes));
        history.variable_neutral_exhaust_nuclei_cm3_s.reserve(static_cast<size_t>(n_nodes));
        history.variable_balance_rhs_cm3_s.reserve(static_cast<size_t>(n_nodes));
        history.variable_balance_residual_cm3_s.reserve(static_cast<size_t>(n_nodes));
        history.prescribed_nuclei_density_cm3.reserve(static_cast<size_t>(n_nodes));
        history.recycling_source_nuclei_cm3_s.reserve(static_cast<size_t>(n_nodes));
        history.local_atom_exhaust_nuclei_cm3_s.reserve(static_cast<size_t>(n_nodes));
        history.local_molecule_exhaust_nuclei_cm3_s.reserve(static_cast<size_t>(n_nodes));
        history.ion_divergence_closure_nuclei_cm3_s.reserve(static_cast<size_t>(n_nodes));
        history.ion_balance_coefficient_s.reserve(static_cast<size_t>(n_nodes));
    }

    const dcr::base::Vector bg_full_boundary = make_background_full(nP, boundary, total_states);
    history.background_full.push_back(bg_full_boundary);
    history.flowA.push_back(flowA);
    history.flowM.push_back(flowM);
    if (variable_nuclei_balance_closure) {
        const double nan = std::numeric_limits<double>::quiet_NaN();
        history.variable_ion_divergence_nuclei_cm3_s.push_back(nan);
        history.variable_flowA_divergence_nuclei_cm3_s.push_back(nan);
        history.variable_flowM_divergence_nuclei_cm3_s.push_back(nan);
        history.variable_neutral_exhaust_nuclei_cm3_s.push_back(nan);
        history.variable_balance_rhs_cm3_s.push_back(nan);
        history.variable_balance_residual_cm3_s.push_back(nan);
        history.prescribed_nuclei_density_cm3.push_back(config.plasma.total_density);
        history.recycling_source_nuclei_cm3_s.push_back(nan);
        history.local_atom_exhaust_nuclei_cm3_s.push_back(nan);
        history.local_molecule_exhaust_nuclei_cm3_s.push_back(nan);
        history.ion_divergence_closure_nuclei_cm3_s.push_back(nan);
        history.ion_balance_coefficient_s.push_back(nan);
    }
    const auto boundary_local = assemble_local_system(
        config, atomic_data, plasma, grid, boundary, bg_full_boundary, flowA, flowM, 0.0
    );
    history.rate_diagnostics.push_back(
        rate_calculator.evaluate(config, boundary, boundary_local, bg_full_boundary, 0.0, plasma, grid)
    );
    if (config.io.verbose_logging) {
        log_rate_snapshot(0.0, history.rate_diagnostics.back(), levels);
    }

    // March cell-by-cell.
    for (size_t k = 0; k < dx.size(); ++k) {
        const dcr::base::Vector flowA_before = flowA;
        const dcr::base::Vector flowM_before = flowM;
        const dcr::base::Vector nP_before = nP;

        const int cell_index = static_cast<int>(k + 1);
        const std::string marching_solver = config.numerics.marching_solver.empty()
            ? "picard" : config.numerics.marching_solver;
        const bool detailed_log_cell =
            (k == 0) || (marching_solver.find("newton") != std::string::npos);
        const double x_left = history.x_cm[k];
        const double x_right = history.x_cm[k + 1];
        CellImplicitResult step = solve_cell_implicit(
            config,
            atomic_data,
            plasma,
            grid,
            boundary,
            levels,
            nP_before,
            flowA_before,
            flowM_before,
            dx[k],
            x_left,
            x_right,
            cell_index,
            detailed_log_cell,
            true,
            adaptive_profile
        );

        nP = step.nP_new;
        flowA = step.flowA_new;
        flowM = step.flowM_new;
        history.cell_elapsed_seconds.push_back(step.elapsed_seconds);
        history.cell_iterations.push_back(step.iterations);
        history.cell_converged.push_back(step.converged ? 1 : 0);

        if (config.io.verbose_logging) {
            if (!step.converged) {
                std::cout << "[DCR_Solver] Marching cell " << cell_index
                          << ": failed to converge in a single fixed-dx solve.\n";
            }
        }
        if (!step.converged && config.numerics.abort_on_marching_nonconvergence) {
            throw std::runtime_error(
                "Marching cell " + std::to_string(cell_index) +
                " reached marching_max_iterations without convergence."
            );
        }

        const dcr::base::Vector bg_full = make_background_full(nP, boundary, total_states);
        history.background_full.push_back(bg_full);
        history.flowA.push_back(flowA);
        history.flowM.push_back(flowM);
        if (variable_nuclei_balance_closure) {
            history.variable_ion_divergence_nuclei_cm3_s.push_back(
                step.variable_ion_divergence_nuclei_cm3_s);
            history.variable_flowA_divergence_nuclei_cm3_s.push_back(
                step.variable_flowA_divergence_nuclei_cm3_s);
            history.variable_flowM_divergence_nuclei_cm3_s.push_back(
                step.variable_flowM_divergence_nuclei_cm3_s);
            history.variable_neutral_exhaust_nuclei_cm3_s.push_back(
                step.variable_neutral_exhaust_nuclei_cm3_s);
            history.variable_balance_rhs_cm3_s.push_back(
                step.variable_balance_rhs_cm3_s);
            history.variable_balance_residual_cm3_s.push_back(
                step.variable_balance_residual_cm3_s);
            history.prescribed_nuclei_density_cm3.push_back(
                step.prescribed_nuclei_density_cm3);
            history.recycling_source_nuclei_cm3_s.push_back(
                step.recycling_source_nuclei_cm3_s);
            history.local_atom_exhaust_nuclei_cm3_s.push_back(
                step.local_atom_exhaust_nuclei_cm3_s);
            history.local_molecule_exhaust_nuclei_cm3_s.push_back(
                step.local_molecule_exhaust_nuclei_cm3_s);
            history.ion_divergence_closure_nuclei_cm3_s.push_back(
                step.ion_divergence_closure_nuclei_cm3_s);
            history.ion_balance_coefficient_s.push_back(
                step.ion_balance_coefficient_s);
        }
        const double h2plus_transport_rate = molecular_transport_loss_rate(
            boundary,
            flowM_before,
            flowM,
            dx[k]
        );
        if (k == 0 && !history.rate_diagnostics.empty()) {
            history.rate_diagnostics[0] = rate_calculator.evaluate(
                config,
                boundary,
                boundary_local,
                bg_full_boundary,
                0.0,
                plasma,
                grid,
                h2plus_transport_rate
            );
        }
        history.rate_diagnostics.push_back(
            rate_calculator.evaluate(
                config,
                boundary,
                step.local_final,
                bg_full,
                x_right,
                plasma,
                grid,
                h2plus_transport_rate
            )
        );
        if (config.io.verbose_logging) {
            log_rate_snapshot(x_right, history.rate_diagnostics.back(), levels);
        }
        if (variable_nuclei_balance_closure && config.io.verbose_logging) {
            const auto summary = summarize_subpopulations(bg_full, flowA, flowM, boundary, levels);
            const auto local_temps = evaluate_plasma_temperatures(config, x_right);
            std::cout << "[DCR_Solver][variable-progress] cell=" << cell_index
                      << "/" << (n_nodes - 1)
                      << " x=" << x_right
                      << " cm"
                      << " Te=" << local_temps.electron_eV
                      << " Ti=" << local_temps.ion_eV
                      << " iters=" << step.iterations
                      << " rel=" << step.final_rel
                      << " resid_rel=" << step.final_resid_rel
                      << " wall=" << step.elapsed_seconds
                      << " s"
                      << " H+=" << summary.H_plus
                      << " H=" << summary.H
                      << " H2=" << summary.H2
                      << " H2+=" << summary.H2_plus
                      << " H-=" << summary.H_minus
                      << " flowA=" << positive_sum(flowA)
                      << " flowM=" << positive_sum(flowM)
                      << " nuclei_total=" << summary.nuclei_total
                      << " L_I=" << step.ion_divergence_closure_nuclei_cm3_s
                      << "\n";
            std::cout.flush();
        }
    }

    if (config.numerics.adaptive_recycling_domain.enabled) {
        const auto estimate = estimate_recycling_layer_from_atomic_flow(
            history.x_cm,
            history.flowA,
            config.numerics.adaptive_recycling_domain.epsilon_A
        );
        const auto molecular_estimate = estimate_flow_depletion_layer(
            history.x_cm,
            history.flowM,
            config.numerics.adaptive_recycling_domain.epsilon_M
        );
        history.adaptive_recycling_diagnostics_enabled = true;
        history.adaptive_recycling_L1_found = estimate.found;
        history.adaptive_recycling_L1_index = estimate.index;
        history.adaptive_recycling_L1_cm = estimate.L1_cm;
        history.adaptive_atomic_flow_fraction = estimate.atomic_flow_fraction;
        history.adaptive_recycling_LM_found = molecular_estimate.found;
        history.adaptive_recycling_LM_index = molecular_estimate.index;
        history.adaptive_recycling_LM_cm = molecular_estimate.length_cm;
        history.adaptive_molecular_flow_fraction = molecular_estimate.flow_fraction;
        if (estimate.found || molecular_estimate.found) {
            std::vector<double> Te_eV;
            std::vector<double> Ti_eV;
            Te_eV.reserve(history.rate_diagnostics.size());
            for (const auto& snap : history.rate_diagnostics) {
                Te_eV.push_back(snap.electron_temperature_eV);
                Ti_eV.push_back(snap.ion_temperature_eV);
            }
            const double c_s_M = dcr::physics::calculate_thermal_speed(
                config.plasma.neutral_molecule_temperature_eV,
                boundary.molecule_mass_amu
            );
            const double L1_for_profile = estimate.found
                ? estimate.L1_cm
                : std::max(config.grid.length_cm, history.x_cm.empty() ? 0.0 : history.x_cm.back());
            const double LM_for_profile = molecular_estimate.found
                ? molecular_estimate.length_cm
                : std::max(config.grid.length_cm, history.x_cm.empty() ? 0.0 : history.x_cm.back());
            const auto profile = build_adaptive_transport_profile(
                history.x_cm,
                history.background_full,
                levels,
                boundary.ion_indices,
                boundary.atom_bg_indices,
                boundary.mol_bg_indices,
                Te_eV,
                Ti_eV,
                L1_for_profile,
                LM_for_profile,
                c_s_M,
                config.grid.spatial_exhaust_width_cm,
                config.numerics.adaptive_recycling_domain.ion_velocity_floor_fraction
            );
            history.adaptive_transport_profile_available = true;
            history.adaptive_ion_divergence_nuclei_cm3_s = profile.ion_divergence_nuclei_cm3_s;
            history.adaptive_pump_exhaust_nuclei_cm3_s = profile.pump_exhaust_nuclei_cm3_s;
            history.adaptive_neutral_remainder_nuclei_cm3_s = profile.neutral_remainder_nuclei_cm3_s;
            history.adaptive_atomic_comp_nuclei_cm3_s = profile.atomic_group_nuclei_divergence_cm3_s;
            history.adaptive_molecular_comp_nuclei_cm3_s = profile.molecular_group_nuclei_divergence_cm3_s;
        }
        if (config.io.verbose_logging) {
            std::cout << "[DCR_Solver][adaptive] passive F_A/L1 diagnostics: found="
                      << (estimate.found ? "true" : "false")
                      << " index=" << estimate.index
                      << " L1=" << estimate.L1_cm << " cm\n";
            std::cout << "[DCR_Solver][adaptive] passive F_M/LM diagnostics: found="
                      << (molecular_estimate.found ? "true" : "false")
                      << " index=" << molecular_estimate.index
                      << " LM=" << molecular_estimate.length_cm << " cm\n";
            if (history.adaptive_transport_profile_available) {
                std::cout << "[DCR_Solver][adaptive] passive transport profile available"
                          << " nodes=" << history.adaptive_ion_divergence_nuclei_cm3_s.size()
                          << "\n";
            }
        }
    }

    return history;
}

MarchingHistory run_adaptive_variable_marching(
    const dcr::io::Config& config,
    const dcr::atomic::AtomicData& atomic_data,
    const dcr::state::PlasmaState& plasma,
    const EEDFGridView& grid,
    const BoundaryPhaseResult& boundary) {

    const auto& adaptive = config.numerics.adaptive_recycling_domain;
    const int max_passes = std::max(1, adaptive.max_outer_iterations);
    const double tolerance = std::max(0.0, adaptive.L1_relative_tolerance);

    std::vector<double> L1_history_cm;
    std::vector<double> LM_history_cm;
    std::vector<double> F_A_end_history;
    std::vector<double> F_M_end_history;
    L1_history_cm.reserve(static_cast<size_t>(max_passes));
    LM_history_cm.reserve(static_cast<size_t>(max_passes));
    F_A_end_history.reserve(static_cast<size_t>(max_passes));
    F_M_end_history.reserve(static_cast<size_t>(max_passes));

    MarchingHistory history = run_full_marching(
        config, atomic_data, plasma, grid, boundary, nullptr
    );
    append_outer_history(
        config, history, L1_history_cm, LM_history_cm, F_A_end_history, F_M_end_history
    );

    bool converged = false;

    for (int pass = 2; pass <= max_passes; ++pass) {
        AdaptiveTransportProfile profile;
        if (!build_profile_from_history(config, atomic_data, boundary, history, profile)) {
            if (config.io.verbose_logging) {
                std::cout << "[DCR_Solver][adaptive-outer] stopping before pass=" << pass
                          << " reason=profile_build_failed\n";
            }
            break;
        }

        MarchingHistory next = run_full_marching(
            config, atomic_data, plasma, grid, boundary, &profile
        );
        append_outer_history(
            config, next, L1_history_cm, LM_history_cm, F_A_end_history, F_M_end_history
        );

        const double L1_rel = relative_change(
            L1_history_cm[L1_history_cm.size() - 2], L1_history_cm.back()
        );
        const double LM_rel = relative_change(
            LM_history_cm[LM_history_cm.size() - 2], LM_history_cm.back()
        );
        const double max_rel = std::max(L1_rel, LM_rel);

        history = std::move(next);
        if (max_rel <= tolerance) {
            converged = true;
            break;
        }
    }

    history.adaptive_outer_loop_enabled = true;
    history.adaptive_outer_loop_converged = converged;
    history.adaptive_outer_iterations = static_cast<int>(L1_history_cm.size());
    history.adaptive_outer_L1_history_cm = std::move(L1_history_cm);
    history.adaptive_outer_LM_history_cm = std::move(LM_history_cm);
    history.adaptive_outer_F_A_end_history = std::move(F_A_end_history);
    history.adaptive_outer_F_M_end_history = std::move(F_M_end_history);

    return history;
}

} // namespace dcr::solver
