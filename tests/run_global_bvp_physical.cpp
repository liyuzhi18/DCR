#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "../src/atomic/AtomicData.hpp"
#include "../src/io/ConfigLoader.hpp"
#include "../src/physics/Sheath.hpp"
#include "../src/physics/WallBoundary.hpp"
#include "../src/solver/boundary/BoundaryPhase.hpp"
#include "../src/solver/core/ConfigPaths.hpp"
#include "../src/solver/core/TemperatureProfile.hpp"
#include "../src/solver/marching/GlobalBVP.hpp"
#include "../src/solver/output/HDF5Output.hpp"
#include "TestDCRSetup.hpp"

namespace {

void require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

template <typename VectorAt>
dcr::base::Vector interpolate(
    const std::vector<double>& coarse_x,
    double x,
    VectorAt vector_at) {
    auto upper_iterator = std::upper_bound(coarse_x.begin(), coarse_x.end(), x);
    size_t upper = static_cast<size_t>(std::distance(coarse_x.begin(), upper_iterator));
    if (upper == 0) upper = 1;
    if (upper >= coarse_x.size()) upper = coarse_x.size() - 1;
    const size_t lower = upper - 1;
    const double span = coarse_x[upper] - coarse_x[lower];
    const double fraction = span > 0.0
        ? std::clamp((x - coarse_x[lower]) / span, 0.0, 1.0) : 0.0;
    return (1.0 - fraction) * vector_at(lower) + fraction * vector_at(upper);
}

double profile_difference(
    const dcr::solver::GlobalBVPResult& coarse,
    const dcr::solver::GlobalBVPResult& fine) {
    double difference_squared = 0.0;
    double fine_squared = 0.0;
    for (size_t k = 0; k < fine.history.x_cm.size(); ++k) {
        const double x = fine.history.x_cm[k];
        const auto p = interpolate(coarse.history.x_cm, x,
            [&](size_t i) -> const dcr::base::Vector& {
                return coarse.history.background_full[i];
            });
        const auto a = interpolate(coarse.history.x_cm, x,
            [&](size_t i) -> const dcr::base::Vector& { return coarse.history.flowA[i]; });
        const auto m = interpolate(coarse.history.x_cm, x,
            [&](size_t i) -> const dcr::base::Vector& { return coarse.history.flowM[i]; });
        difference_squared += (p - fine.history.background_full[k]).squaredNorm();
        difference_squared += (a - fine.history.flowA[k]).squaredNorm();
        difference_squared += (m - fine.history.flowM[k]).squaredNorm();
        fine_squared += fine.history.background_full[k].squaredNorm();
        fine_squared += fine.history.flowA[k].squaredNorm();
        fine_squared += fine.history.flowM[k].squaredNorm();
    }
    return std::sqrt(difference_squared / std::max(1.0, fine_squared));
}

struct StageMetrics {
    double target_ion_flux = 0.0;
    double max_atomic_neutral_density = 0.0;
    double max_nuclei_density = 0.0;
    double upstream_h_density = 0.0;
    double upstream_h_plus_density = 0.0;
    double upstream_h_minus_density = 0.0;
    double upstream_h2_density = 0.0;
    double upstream_h2_plus_density = 0.0;
    double upstream_h2_minus_density = 0.0;
    double upstream_nuclei_density = 0.0;
    double atom_tail_ratio = 0.0;
    double molecule_tail_ratio = 0.0;
    double integrated_ion_source = 0.0;
    double integrated_ion_sink = 0.0;
    double ionization_tail_fraction = 0.0;
    double recombination_tail_fraction = 0.0;
    double recombination_degree = 0.0;
    double first_dx = 0.0;
    double minimum_dx = 0.0;
    double maximum_dx = 0.0;
};

StageMetrics stage_metrics(
    const dcr::io::Config& config,
    const dcr::atomic::AtomicData& atomic_data,
    const dcr::solver::BoundaryPhaseResult& boundary,
    const dcr::solver::GlobalBVPResult& result) {
    StageMetrics out;
    const auto& levels = atomic_data.get_levels();
    const auto& x = result.history.x_cm;
    out.minimum_dx = std::numeric_limits<double>::infinity();
    double tail_ion_source = 0.0;
    double tail_ion_sink = 0.0;
    for (size_t k = 0; k + 1 < x.size(); ++k) {
        const double dx = x[k + 1] - x[k];
        if (k == 0) out.first_dx = dx;
        out.minimum_dx = std::min(out.minimum_dx, dx);
        out.maximum_dx = std::max(out.maximum_dx, dx);
        const auto& left = result.history.rate_diagnostics[k].atomic_sources;
        const auto& right = result.history.rate_diagnostics[k + 1].atomic_sources;
        out.integrated_ion_source += 0.5 * dx *
            (left.full_ion_nuclei_source_cm3_s + right.full_ion_nuclei_source_cm3_s);
        out.integrated_ion_sink += 0.5 * dx *
            (left.full_ion_nuclei_sink_cm3_s + right.full_ion_nuclei_sink_cm3_s);
        if (0.5 * (x[k] + x[k + 1]) >= 0.9 * x.back()) {
            tail_ion_source += 0.5 * dx *
                (left.full_ion_nuclei_source_cm3_s + right.full_ion_nuclei_source_cm3_s);
            tail_ion_sink += 0.5 * dx *
                (left.full_ion_nuclei_sink_cm3_s + right.full_ion_nuclei_sink_cm3_s);
        }
    }
    if (!std::isfinite(out.minimum_dx)) out.minimum_dx = 0.0;
    out.recombination_degree = out.integrated_ion_sink /
        std::max(1.0, out.integrated_ion_source);
    out.ionization_tail_fraction = tail_ion_source /
        std::max(1.0, out.integrated_ion_source);
    out.recombination_tail_fraction = tail_ion_sink /
        std::max(1.0, out.integrated_ion_sink);

    for (int gi : boundary.ion_indices) {
        out.target_ion_flux += -dcr::solver::global_ion_velocity_cm_s(
            config, levels[static_cast<size_t>(gi)], 0.0) *
            result.history.background_full.front()(gi);
    }
    for (size_t k = 0; k < x.size(); ++k) {
        double atomic_neutral = 0.0;
        double nuclei = 0.0;
        for (int gi = 0; gi < result.history.background_full[k].size(); ++gi) {
            const double density = result.history.background_full[k](gi);
            nuclei += levels[static_cast<size_t>(gi)].atomicity * density;
            if (levels[static_cast<size_t>(gi)].type == dcr::atomic::SpeciesType::Atom &&
                levels[static_cast<size_t>(gi)].charge == 0) {
                atomic_neutral += density;
            }
        }
        for (int ai = 0; ai < result.history.flowA[k].size(); ++ai) {
            const int gi = boundary.A_indices[static_cast<size_t>(ai)];
            const double density = result.history.flowA[k](ai);
            atomic_neutral += density;
            nuclei += levels[static_cast<size_t>(gi)].atomicity * density;
        }
        for (int mi = 0; mi < result.history.flowM[k].size(); ++mi) {
            const int gi = boundary.M_indices[static_cast<size_t>(mi)];
            nuclei += levels[static_cast<size_t>(gi)].atomicity *
                result.history.flowM[k](mi);
        }
        out.max_atomic_neutral_density = std::max(
            out.max_atomic_neutral_density, atomic_neutral);
        out.max_nuclei_density = std::max(out.max_nuclei_density, nuclei);
    }
    out.atom_tail_ratio = result.history.flowA.back().sum() /
        std::max(1.0, result.history.flowA.front().sum());
    out.molecule_tail_ratio = result.history.flowM.back().sum() /
        std::max(1.0, result.history.flowM.front().sum());

    const auto& upstream_background = result.history.background_full.back();
    for (int gi = 0; gi < upstream_background.size(); ++gi) {
        const auto& level = levels[static_cast<size_t>(gi)];
        const double density = upstream_background(gi);
        out.upstream_nuclei_density += level.atomicity * density;
        if (level.atomicity == 1 && level.charge == 0) out.upstream_h_density += density;
        if (level.atomicity == 1 && level.charge > 0) out.upstream_h_plus_density += density;
        if (level.atomicity == 1 && level.charge < 0) out.upstream_h_minus_density += density;
        if (level.atomicity == 2 && level.charge == 0) out.upstream_h2_density += density;
        if (level.atomicity == 2 && level.charge > 0) out.upstream_h2_plus_density += density;
        if (level.atomicity == 2 && level.charge < 0) out.upstream_h2_minus_density += density;
    }
    for (int ai = 0; ai < result.history.flowA.back().size(); ++ai) {
        const int gi = boundary.A_indices[static_cast<size_t>(ai)];
        const double density = result.history.flowA.back()(ai);
        out.upstream_h_density += density;
        out.upstream_nuclei_density += levels[static_cast<size_t>(gi)].atomicity * density;
    }
    for (int mi = 0; mi < result.history.flowM.back().size(); ++mi) {
        const int gi = boundary.M_indices[static_cast<size_t>(mi)];
        const double density = result.history.flowM.back()(mi);
        out.upstream_h2_density += density;
        out.upstream_nuclei_density += levels[static_cast<size_t>(gi)].atomicity * density;
    }
    return out;
}

void log_node_densities(
    double length_cm,
    const dcr::atomic::AtomicData& atomic_data,
    const dcr::solver::BoundaryPhaseResult& boundary,
    const dcr::solver::GlobalBVPResult& result) {
    const auto& levels = atomic_data.get_levels();
    const double target_atom_flow = std::max(1.0, result.history.flowA.front().sum());
    const double target_molecule_flow = std::max(1.0, result.history.flowM.front().sum());
    for (size_t k = 0; k < result.history.x_cm.size(); ++k) {
        double n_h = 0.0;
        double n_h_plus = 0.0;
        double n_h_minus = 0.0;
        double n_h2 = 0.0;
        double n_h2_plus = 0.0;
        double n_h2_minus = 0.0;
        double n_nuc = 0.0;
        const auto& background = result.history.background_full[k];
        for (int gi = 0; gi < background.size(); ++gi) {
            const auto& level = levels[static_cast<size_t>(gi)];
            const double density = background(gi);
            n_nuc += level.atomicity * density;
            if (level.atomicity == 1 && level.charge == 0) n_h += density;
            if (level.atomicity == 1 && level.charge > 0) n_h_plus += density;
            if (level.atomicity == 1 && level.charge < 0) n_h_minus += density;
            if (level.atomicity == 2 && level.charge == 0) n_h2 += density;
            if (level.atomicity == 2 && level.charge > 0) n_h2_plus += density;
            if (level.atomicity == 2 && level.charge < 0) n_h2_minus += density;
        }
        for (int ai = 0; ai < result.history.flowA[k].size(); ++ai) {
            const int gi = boundary.A_indices[static_cast<size_t>(ai)];
            const double density = result.history.flowA[k](ai);
            n_h += density;
            n_nuc += levels[static_cast<size_t>(gi)].atomicity * density;
        }
        for (int mi = 0; mi < result.history.flowM[k].size(); ++mi) {
            const int gi = boundary.M_indices[static_cast<size_t>(mi)];
            const double density = result.history.flowM[k](mi);
            n_h2 += density;
            n_nuc += levels[static_cast<size_t>(gi)].atomicity * density;
        }
        std::cout << "[physical-global-bvp-node] L_cm=" << length_cm
                  << " node=" << k
                  << " x_cm=" << result.history.x_cm[k]
                  << " n_H=" << n_h
                  << " n_H_plus=" << n_h_plus
                  << " n_H_minus=" << n_h_minus
                  << " n_H2=" << n_h2
                  << " n_H2_plus=" << n_h2_plus
                  << " n_H2_minus=" << n_h2_minus
                  << " n_nuc=" << n_nuc
                  << " atom_flow_ratio=" << result.history.flowA[k].sum() / target_atom_flow
                  << " molecule_flow_ratio=" << result.history.flowM[k].sum() /
                        target_molecule_flow
                  << std::endl;
    }
}

std::pair<std::string, double> largest_residual_block(
    const dcr::solver::GlobalBVPResult& result) {
    std::pair<std::string, double> largest{"ion", result.final_ion_residual};
    const std::vector<std::pair<std::string, double>> blocks{
        {"A", result.final_flow_a_residual},
        {"M", result.final_flow_m_residual},
        {"local", result.final_local_residual},
        {"boundary", result.final_boundary_residual}};
    for (const auto& block : blocks) {
        if (block.second > largest.second) largest = block;
    }
    return largest;
}

} // namespace

int main(int argc, char** argv) {
    const std::string config_path = argc > 1
        ? argv[1]
        : "config/codex_full_T1_nnuc1e16_w1_variable_L50_Ls100_n200_tol3e-6_ufloor001_TiBohm_localTi_prescribedVelocity.yaml";
    dcr::io::Config base_config = dcr::io::ConfigLoader::load(config_path);
    const double target_length_cm = argc > 2
        ? std::stod(argv[2]) : base_config.global_bvp.final_domain_length_cm;
    const std::string run_mode = argc > 3 ? argv[3] : "production";
    const bool fallback_only = run_mode == "fallback" || run_mode == "diagnose";
    const bool stop_at_target_failure = run_mode == "diagnose";
    require(target_length_cm >= 1.0, "Physical target length must be at least 1 cm.");
    dcr::solver::normalize_input_roots(base_config, config_path);
    base_config.io.output_dir += "_accepted";
    base_config.io.verbose_logging = false;
    base_config.numerics.marching_solver = "global_sparse_newton";
    base_config.numerics.marching_tolerance = 1.0e-8;

    dcr::atomic::AtomicData atomic_data(base_config);
    auto boundary_config = base_config;
    boundary_config.grid.length_cm = 0.01;
    boundary_config.grid.num_cells = 2;
    boundary_config.grid.type = "linear";
    boundary_config.grid.first_cell_cm = 0.0;
    auto boundary_plasma = test_dcr::make_plasma_state(
        boundary_config, atomic_data.get_total_states());
    const auto temperatures = dcr::solver::evaluate_plasma_temperatures(
        boundary_config, 0.0);
    boundary_plasma.init_Te().setConstant(temperatures.electron_eV);
    boundary_plasma.init_Ti().setConstant(temperatures.ion_eV);
    test_dcr::EEDFContext eedf(
        temperatures.electron_eV, test_dcr::make_eedf_config(boundary_config));
    const double ion_mass_amu = test_dcr::estimate_ion_mass_amu(boundary_config);
    const auto wall = dcr::physics::compute_wall_recycling(
        boundary_config.wall.material,
        temperatures.electron_eV,
        temperatures.ion_eV,
        ion_mass_amu,
        boundary_config.wall.sheath_potential_drop);
    const auto boundary = dcr::solver::run_boundary_phase(
        boundary_config, atomic_data, boundary_plasma, eedf.grid, ion_mass_amu, wall);
    require(boundary.converged, "Physical-case boundary phase did not converge.");

    if (run_mode == "snapshot") {
        const std::string snapshot_path = base_config.io.output_dir +
            "/global_bvp_L" +
            std::to_string(static_cast<int>(std::round(target_length_cm))) +
            "_failed_state.tsv";
        std::ifstream input(snapshot_path);
        require(input.good(), "Failed to open preserved global-BVP snapshot.");
        std::string lambda_label;
        double chemistry_fraction = 0.0;
        input >> lambda_label >> chemistry_fraction;
        std::string header;
        std::getline(input, header);
        std::getline(input, header);
        std::vector<double> values;
        int index = 0;
        double value = 0.0;
        double residual = 0.0;
        while (input >> index >> value >> residual) values.push_back(value);
        dcr::base::Vector log_state(static_cast<int>(values.size()));
        for (int i = 0; i < log_state.size(); ++i) {
            log_state(i) = values[static_cast<size_t>(i)];
        }
        auto diagnostic_config = base_config;
        diagnostic_config.grid.length_cm = target_length_cm;
        diagnostic_config.grid.num_cells = 25;
        diagnostic_config.grid.type = "log";
        diagnostic_config.grid.first_cell_cm = 0.001;
        auto diagnostic_plasma = test_dcr::make_plasma_state(
            diagnostic_config, atomic_data.get_total_states());
        diagnostic_plasma.init_Te().setConstant(temperatures.electron_eV);
        diagnostic_plasma.init_Ti().setConstant(temperatures.ion_eV);
        dcr::solver::diagnose_global_bvp_log_state(
            diagnostic_config, atomic_data, diagnostic_plasma, eedf.grid,
            boundary, wall, log_state, chemistry_fraction);
        return 0;
    }

    dcr::solver::GlobalBVPResult previous;
    bool have_previous = false;
    double previous_length = 0.0;
    StageMetrics previous_metrics;
    std::string solution_path = "short_domain_seed";
    auto solve_stage = [&](double length_cm, int nodes, double log_shift = 0.0,
                           int iteration_limit = 0) {
        auto config = base_config;
        config.grid.length_cm = length_cm;
        config.grid.num_cells = nodes;
        config.grid.type = nodes <= 5 ? "linear" : "log";
        config.grid.first_cell_cm = nodes <= 5 ? 0.0 : 0.001;
        if (iteration_limit > 0) {
            config.numerics.marching_max_iterations = iteration_limit;
        }
        auto plasma = test_dcr::make_plasma_state(config, atomic_data.get_total_states());
        plasma.init_Te().setConstant(temperatures.electron_eV);
        plasma.init_Ti().setConstant(temperatures.ion_eV);
        return dcr::solver::solve_global_target_conditioned_bvp(
            config, atomic_data, plasma, eedf.grid, boundary, wall,
            have_previous ? &previous : nullptr, log_shift);
    };
    auto accept_stage = [&](double length_cm, int nodes,
                            const dcr::solver::GlobalBVPResult& result) {
        require(result.converged, "Physical continuation stage did not converge.");
        auto metric_config = base_config;
        metric_config.grid.length_cm = length_cm;
        metric_config.grid.num_cells = nodes;
        const StageMetrics metrics = stage_metrics(
            metric_config, atomic_data, boundary, result);
        const auto largest_block = largest_residual_block(result);
        const auto& levels = atomic_data.get_levels();
        const dcr::atomic::EnergyLevel* proton = nullptr;
        for (int gi : boundary.ion_indices) {
            if (levels[static_cast<size_t>(gi)].atomicity == 1) {
                proton = &levels[static_cast<size_t>(gi)];
                break;
            }
        }
        require(proton != nullptr, "No proton state found for physical telemetry.");
        const auto upstream_temperatures = dcr::solver::evaluate_plasma_temperatures(
            metric_config, length_cm);
        const double upstream_bohm_speed = dcr::physics::calculate_Bohm_speed(
            upstream_temperatures.electron_eV,
            upstream_temperatures.ion_eV,
            proton->mass_amu);
        const double upstream_ion_velocity = dcr::solver::global_ion_velocity_cm_s(
            metric_config, *proton, length_cm);
        double change = 0.0;
        if (have_previous && previous.history.x_cm.back() == result.history.x_cm.back()) {
            change = profile_difference(previous, result);
        }
        std::cout << "[physical-global-bvp] current_domain_length_cm=" << length_cm
                  << " solution_path=" << solution_path
                  << " ion_velocity_transition_length_cm="
                  << metric_config.global_bvp.ion_velocity_transition_length_cm
                  << " u_ion_at_current_boundary_over_u_B="
                  << upstream_ion_velocity / upstream_bohm_speed
                  << " nodes=" << nodes
                  << " residual=" << result.residual_norm
                  << " chemistry=" << result.max_chemistry_nuclei_relative_error
                  << " nuclei=" << result.integrated_nuclei_balance_relative_error
                  << " ion=" << result.ion_flux_chemistry_relative_error
                  << " solved_upstream_ion_flux=" << result.upstream_proton_flux_cm2_s
                  << " upstream_ion_density="
                  << result.upstream_proton_flux_cm2_s / std::abs(upstream_ion_velocity)
                  << " delta_L=" << (have_previous ? length_cm - previous_length : 0.0)
                  << " q_change=" << (have_previous
                        ? std::abs(result.upstream_proton_flux_cm2_s -
                            previous.upstream_proton_flux_cm2_s) /
                            result.upstream_proton_flux_cm2_s : 0.0)
                  << " target_ion_flux=" << metrics.target_ion_flux
                  << " target_ion_flux_change=" << (have_previous
                        ? std::abs(metrics.target_ion_flux - previous_metrics.target_ion_flux) /
                            std::max(1.0, metrics.target_ion_flux) : 0.0)
                  << " max_nH=" << metrics.max_atomic_neutral_density
                  << " max_nnuc=" << metrics.max_nuclei_density
                  << " upstream_n_H=" << metrics.upstream_h_density
                  << " upstream_n_H_plus=" << metrics.upstream_h_plus_density
                  << " upstream_n_H_minus=" << metrics.upstream_h_minus_density
                  << " upstream_n_H2=" << metrics.upstream_h2_density
                  << " upstream_n_H2_plus=" << metrics.upstream_h2_plus_density
                  << " upstream_n_H2_minus=" << metrics.upstream_h2_minus_density
                  << " upstream_n_nuc=" << metrics.upstream_nuclei_density
                  << " atom_flow_ratio=" << metrics.atom_tail_ratio
                  << " molecule_flow_ratio=" << metrics.molecule_tail_ratio
                  << " D_rec=" << metrics.recombination_degree
                  << " ionization_tail_fraction=" << metrics.ionization_tail_fraction
                  << " recombination_tail_fraction=" << metrics.recombination_tail_fraction
                  << " delta_ion_source=" << (have_previous
                        ? metrics.integrated_ion_source - previous_metrics.integrated_ion_source : 0.0)
                  << " delta_ion_sink=" << (have_previous
                        ? metrics.integrated_ion_sink - previous_metrics.integrated_ion_sink : 0.0)
                  << " first_dx=" << metrics.first_dx
                  << " min_dx=" << metrics.minimum_dx
                  << " max_dx=" << metrics.maximum_dx
                  << " newton_iterations=" << result.iterations
                  << " min_line_search=" << result.minimum_line_search_factor
                  << " largest_block=" << largest_block.first
                  << " largest_block_residual=" << largest_block.second
                  << " same_domain_profile_change=" << change << std::endl;
        log_node_densities(length_cm, atomic_data, boundary, result);
        previous = result;
        have_previous = true;
        previous_length = length_cm;
        previous_metrics = metrics;
    };

    for (const auto& initial : std::vector<std::pair<double, int>>{
            {0.01, 2}, {0.1, 5}, {1.0, 10}}) {
        const auto result = solve_stage(initial.first, initial.second);
        accept_stage(initial.first, initial.second, result);
    }

    auto nodes_for_length = [&](double length_cm) {
        if (length_cm == target_length_cm) return 25;
        return std::max(
            static_cast<int>(previous.history.x_cm.size()) + 1,
            static_cast<int>(std::lround(
                10.0 + 15.0 * length_cm / target_length_cm)));
    };
    auto solve_and_accept_length = [&](double length_cm) {
        const int nodes = nodes_for_length(length_cm);
        const auto result = solve_stage(length_cm, nodes);
        accept_stage(length_cm, nodes, result);
    };
    auto advance_adaptively = [&](double requested_length_cm) {
        double domain_step = requested_length_cm - previous_length;
        constexpr double minimum_domain_step = 0.01;
        while (previous_length < requested_length_cm - 1.0e-12) {
            const double trial_length = std::min(
                requested_length_cm, previous_length + domain_step);
            try {
                solve_and_accept_length(trial_length);
                domain_step = std::min(
                    requested_length_cm - previous_length, 1.5 * domain_step);
            } catch (const std::runtime_error& error) {
                std::cout << "[physical-global-bvp-domain-failure]"
                          << " requested_L_cm=" << requested_length_cm
                          << " failed_L_cm=" << trial_length
                          << " domain_step_cm=" << domain_step
                          << " error='" << error.what() << "'"
                          << std::endl;
                if (stop_at_target_failure &&
                    requested_length_cm == target_length_cm) {
                    throw;
                }
                domain_step *= 0.5;
                if (domain_step < minimum_domain_step) {
                    throw std::runtime_error(
                        "Fallback domain increment collapsed below 0.01 cm.");
                }
            }
        }
    };
    auto run_fallback = [&]() {
        solution_path = "fallback_domain_continuation";
        const double first_fallback_target = std::min(target_length_cm, 2.0);
        try {
            solve_and_accept_length(first_fallback_target);
        } catch (const std::runtime_error& error) {
            std::cout << "[physical-global-bvp-domain-failure]"
                      << " requested_L_cm=" << first_fallback_target
                      << " failed_L_cm=" << first_fallback_target
                      << " error='" << error.what() << "'"
                      << " retrying_quarter_cm_stages=true" << std::endl;
            for (double staged_length : {1.25, 1.5, 1.75, 2.0}) {
                const double length_cm = std::min(target_length_cm, staged_length);
                if (length_cm <= previous_length + 1.0e-12) continue;
                advance_adaptively(length_cm);
                if (length_cm == target_length_cm) break;
            }
        }
        for (double fallback_length : {5.0, 10.0, 20.0, 50.0}) {
            const double length_cm = std::min(target_length_cm, fallback_length);
            if (length_cm <= previous_length + 1.0e-12) continue;
            advance_adaptively(length_cm);
            if (length_cm == target_length_cm) break;
        }
        require(previous_length == target_length_cm,
            "Fallback continuation did not reach the fixed physical domain.");
    };

    if (fallback_only) {
        std::cout << "[physical-global-bvp] skipping_direct_fixed_domain_attempt=true"
                  << " reason=previous_quick_attempt_failed" << std::endl;
        run_fallback();
    } else {
        try {
            solution_path = "direct_fixed_domain_correction";
            const auto result = solve_stage(target_length_cm, 25, 0.0, 150);
            accept_stage(target_length_cm, 25, result);
        } catch (const std::runtime_error& error) {
            std::cout << "[physical-global-bvp] direct_fixed_domain_failed='"
                      << error.what() << "' using_fallback_domain_continuation=true"
                      << std::endl;
            run_fallback();
        }
    }

    std::cout << "[physical-global-bvp] fixed_domain_solution_path="
              << solution_path << std::endl;

    auto final_config = base_config;
    final_config.grid.length_cm = target_length_cm;
    final_config.grid.num_cells = 200;
    final_config.grid.type = "log";
    final_config.grid.first_cell_cm = 0.001;
    auto final_plasma = test_dcr::make_plasma_state(
        final_config, atomic_data.get_total_states());
    final_plasma.init_Te().setConstant(temperatures.electron_eV);
    final_plasma.init_Ti().setConstant(temperatures.ion_eV);
    dcr::solver::GlobalBVPResult coarse_100;
    for (int nodes : {50, 100}) {
        const auto result = solve_stage(target_length_cm, nodes);
        accept_stage(target_length_cm, nodes, result);
        if (nodes == 100) coarse_100 = result;
    }
    const auto accepted_final = solve_stage(target_length_cm, 200);
    accept_stage(target_length_cm, 200, accepted_final);
    const auto shifted = dcr::solver::solve_global_target_conditioned_bvp(
        final_config, atomic_data, final_plasma, eedf.grid,
        boundary, wall, &coarse_100, std::log(0.8));
    require(shifted.converged, "Second physical 200-node solve did not converge.");
    require(std::abs(accepted_final.upstream_proton_flux_cm2_s -
        coarse_100.upstream_proton_flux_cm2_s) /
        accepted_final.upstream_proton_flux_cm2_s < 1.0e-5,
        "Physical 100-to-200-node q_up refinement did not converge.");
    require(profile_difference(coarse_100, accepted_final) < 1.0e-5,
        "Physical 100-to-200-node profile refinement did not converge.");
    require(std::abs(shifted.upstream_proton_flux_cm2_s -
        accepted_final.upstream_proton_flux_cm2_s) /
        accepted_final.upstream_proton_flux_cm2_s < 1.0e-6,
        "Physical 200-node q_up is initial-guess dependent.");
    require(profile_difference(accepted_final, shifted) < 1.0e-6,
        "Physical 200-node profile is initial-guess dependent.");
    const StageMetrics final_metrics = stage_metrics(
        final_config, atomic_data, boundary, accepted_final);
    const bool ionization_saturated = final_metrics.ionization_tail_fraction < 1.0e-3;
    const bool recombination_saturated = final_metrics.recombination_tail_fraction < 1.0e-3;
    const bool recycling_domain_complete =
        final_metrics.atom_tail_ratio <=
            base_config.numerics.adaptive_recycling_domain.epsilon_A &&
        final_metrics.molecule_tail_ratio <=
            base_config.numerics.adaptive_recycling_domain.epsilon_M;
    std::cout << "[physical-global-bvp-final] solution_path=" << solution_path
              << " atom_flow_ratio=" << final_metrics.atom_tail_ratio
              << " molecule_flow_ratio=" << final_metrics.molecule_tail_ratio
              << " ionization_saturated=" << ionization_saturated
              << " recombination_saturated=" << recombination_saturated
              << " recycling_domain_complete=" << recycling_domain_complete
              << " D_rec_valid_for_complete_domain="
              << (recycling_domain_complete && ionization_saturated && recombination_saturated)
              << std::endl;

    dcr::solver::write_hdf5_output(
        final_config, atomic_data, accepted_final.boundary, accepted_final.history);
    std::cout << "[PASS] Accepted physical 1 eV global BVP written to "
              << final_config.io.output_dir << std::endl;
    return 0;
}
