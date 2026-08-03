#include "QSSMarching.hpp"

#include "QSSCommon.hpp"

#include "../core/RateAnalysis.hpp"
#include "../core/TemperatureProfile.hpp"
#include "../marching/CellAdvance.hpp"
#include "../marching/LocalSystemAssembler.hpp"

#include "../../physics/Sheath.hpp"
#include "../../physics/WallBoundary.hpp"

#include <Eigen/LU>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <vector>

namespace dcr::solver {

struct QSSCellAttempt {
    bool converged = false;
    int iterations = 0;
    double rel = std::numeric_limits<double>::infinity();
    dcr::base::Vector retained;
    dcr::base::Vector transient;
    dcr::base::Vector flowA;
    dcr::base::Vector flowM;
};

QSSMarchingResult run_qss_marching(
    const dcr::io::Config& config,
    const dcr::atomic::AtomicData& atomic_data,
    dcr::state::PlasmaState& plasma,
    const EEDFGridView& grid,
    const QSSBoundaryResult& qss_boundary) {

    QSSMarchingResult out;

    const auto& levels = atomic_data.get_levels();
    const auto dx = build_step_sizes_cm(config);
    const int n_nodes = static_cast<int>(dx.size()) + 1;
    out.history.x_cm.assign(static_cast<size_t>(n_nodes), 0.0);
    for (int i = 1; i < n_nodes; ++i) {
        out.history.x_cm[static_cast<size_t>(i)] =
            out.history.x_cm[static_cast<size_t>(i - 1)] + dx[static_cast<size_t>(i - 1)];
    }

    out.history.background_full.reserve(static_cast<size_t>(n_nodes));
    out.history.flowA.reserve(static_cast<size_t>(n_nodes));
    out.history.flowM.reserve(static_cast<size_t>(n_nodes));
    out.history.qss_local_transient_atomic.reserve(static_cast<size_t>(n_nodes));
    out.history.qss_flow_transient_atomic.reserve(static_cast<size_t>(n_nodes));
    out.history.rate_diagnostics.reserve(static_cast<size_t>(n_nodes));
    out.history.qss_local_transient_atomic_indices = qss_boundary.layout.transient_atomic_indices;
    out.history.qss_flow_transient_atomic_indices = qss_boundary.layout.transient_atomic_donor_indices;

    dcr::base::Vector retained = qss_boundary.retained_background;
    dcr::base::Vector transient = qss_boundary.transient_atomic;
    dcr::base::Vector flowA = qss_boundary.boundary.flowA_last;
    dcr::base::Vector flowM = qss_boundary.boundary.flowM_last;

    const dcr::base::Vector retained_boundary = qss_boundary.retained_background;
    const dcr::base::Vector transient_boundary = qss_boundary.transient_atomic;
    const dcr::base::Vector flowA_initial = qss_boundary.boundary.flowA_last;
    const dcr::base::Vector flowM_initial = qss_boundary.boundary.flowM_last;

    const double tol = std::max(1e-12, config.numerics.tolerance);
    const int max_iter = std::max(1, config.numerics.max_iterations);
    const int slow_iter_threshold = std::max(1, config.numerics.marching_slow_iter_threshold);
    const bool couple_flowA_retained = config.numerics.couple_flowA_retained;
    const bool exclude_molecular_ion_from_transient_reconstruction =
        config.numerics.qss_exclude_molecular_ion_from_transient_reconstruction;
    if (config.io.verbose_logging) {
        std::cout << "[DCR_Solver][QSS] FlowA-retained coupling: "
                  << (couple_flowA_retained ? "on" : "off") << "\n";
    }
    const double omega_base = (config.numerics.relaxation > 0.0 && config.numerics.relaxation <= 1.0)
        ? config.numerics.relaxation
        : 1.0;
    const double omega_flow_base = std::min(omega_base, 0.25);
    const int retained_size = retained.size();
    const int transient_size = transient.size();
    const int flowA_size = flowA.size();
    const int flowM_size = flowM.size();
    const int x_size = retained_size + transient_size + flowA_size + flowM_size;
    const double omega_min = 0.005;
    const double omega_min_first_cell = 0.05;
    std::vector<int> flowA_atomic_ion_positions;
    std::vector<int> flowA_atomic_neutral_positions;
    std::vector<int> flowA_molecular_neutral_positions;
    std::vector<int> retained_molecular_ion_positions;
    for (int i = 0; i < retained_size; ++i) {
        const int gi = qss_boundary.boundary.P_indices[static_cast<size_t>(i)];
        if (gi < 0 || gi >= static_cast<int>(levels.size())) continue;
        const auto& level = levels[static_cast<size_t>(gi)];
        if (level.type == dcr::atomic::SpeciesType::Atom && level.charge == 0) {
            flowA_atomic_neutral_positions.push_back(i);
        } else if (level.type == dcr::atomic::SpeciesType::Molecule && level.charge == 0) {
            flowA_molecular_neutral_positions.push_back(i);
        } else if (level.charge > 0 && level.atomicity <= 1) {
            flowA_atomic_ion_positions.push_back(i);
        } else if (level.charge > 0) {
            retained_molecular_ion_positions.push_back(i);
        }
    }

    const AtomicRateCalculator rate_calculator(atomic_data);
    const dcr::base::Vector bg0 = lift_qss_background_to_full(
        atomic_data.get_total_states(), qss_boundary.boundary, qss_boundary.layout, retained, transient);
    out.history.background_full.push_back(bg0);
    out.history.flowA.push_back(flowA);
    out.history.flowM.push_back(flowM);
    const auto local0 = assemble_local_system(
        config, atomic_data, plasma, grid, qss_boundary.boundary, bg0, flowA, flowM, 0.0
    );
    const QSSLocalBlocks flow_blocks0 = build_qss_local_blocks(
        local0.R_full, qss_boundary.layout, levels
    );
    dcr::base::Vector flow_transient0 = reconstruct_qss_atomic_flow_transient(flow_blocks0, flowA);
    apply_non_negative(flow_transient0);
    out.history.rate_diagnostics.push_back(
        rate_calculator.evaluate(config, qss_boundary.boundary, local0, bg0, 0.0)
    );
    out.history.qss_local_transient_atomic.push_back(transient);
    out.history.qss_flow_transient_atomic.push_back(flow_transient0);
    if (config.io.verbose_logging) {
        log_rate_snapshot(0.0, out.history.rate_diagnostics.back());
    }

    auto pack_state = [&](const dcr::base::Vector& retained_state,
                          const dcr::base::Vector& transient_state,
                          const dcr::base::Vector& flowA_state,
                          const dcr::base::Vector& flowM_state) {
        dcr::base::Vector x = dcr::base::Vector::Zero(x_size);
        if (retained_size > 0) x.segment(0, retained_size) = retained_state;
        if (transient_size > 0) x.segment(retained_size, transient_size) = transient_state;
        if (flowA_size > 0) x.segment(retained_size + transient_size, flowA_size) = flowA_state;
        if (flowM_size > 0) x.segment(retained_size + transient_size + flowA_size, flowM_size) = flowM_state;
        return x;
    };

    auto unpack_state = [&](const dcr::base::Vector& x,
                            dcr::base::Vector& retained_state,
                            dcr::base::Vector& transient_state,
                            dcr::base::Vector& flowA_state,
                            dcr::base::Vector& flowM_state) {
        if (retained_size > 0) retained_state = x.segment(0, retained_size);
        if (transient_size > 0) transient_state = x.segment(retained_size, transient_size);
        if (flowA_size > 0) flowA_state = x.segment(retained_size + transient_size, flowA_size);
        if (flowM_size > 0) flowM_state = x.segment(retained_size + transient_size + flowA_size, flowM_size);
    };

        auto solve_cell_attempt = [&](const dcr::base::Vector& retained_seed,
                                     const dcr::base::Vector& transient_seed,
                                     const dcr::base::Vector& flowA_seed,
                                     const dcr::base::Vector& flowM_seed,
                                     const dcr::base::Vector& flowA_inflow,
                                     const dcr::base::Vector& flowM_inflow,
                                     int cell_index,
                                     double x_right,
                                     double dx_cm,
                                     QSSCellAttempt& out_attempt) {
        out_attempt.retained = retained_seed;
        out_attempt.transient = transient_seed;
        out_attempt.flowA = flowA_seed;
        out_attempt.flowM = flowM_seed;

        const auto local_temperatures = evaluate_plasma_temperatures(config, x_right);
        const double c_s_A = dcr::physics::calculate_thermal_speed(
            local_temperatures.ion_eV, qss_boundary.boundary.atom_mass_amu);
        const double w_local = std::max(config.grid.spatial_exhaust_width_cm, 1e-12);
        const double exA_over_w = c_s_A / w_local;

        out_attempt.rel = std::numeric_limits<double>::infinity();
        out_attempt.iterations = 0;
        out_attempt.converged = false;
        bool coupled_fallback_logged = false;

            const bool is_first_cell = (cell_index == 1);
            const double omega_min_cell = is_first_cell ? omega_min_first_cell : omega_min;
            double omega_bg_iter = std::clamp(omega_base, omega_min_cell, 1.0);
            double omega_flow_iter = std::clamp(omega_flow_base, omega_min_cell, 1.0);
        double prev_rel_change = std::numeric_limits<double>::infinity();
        int no_improve_count = 0;
        int rel_growth_count = 0;
        auto log_coupled_fallback = [&](const char* reason) {
            if (!config.io.verbose_logging || coupled_fallback_logged) return;
            std::cout << "[DCR_Solver][QSS] Marching cell " << cell_index
                      << " x=" << x_right
                      << " cm coupled FlowA fallback: " << reason << "\n";
            coupled_fallback_logged = true;
        };

        for (int iter = 0; iter < max_iter; ++iter) {
            const dcr::base::Vector bg_full_iter = lift_qss_background_to_full(
                atomic_data.get_total_states(), qss_boundary.boundary, qss_boundary.layout,
                out_attempt.retained, out_attempt.transient);

            const auto local_for_flow = assemble_local_system(
                config,
                atomic_data,
                plasma,
                grid,
                qss_boundary.boundary,
                bg_full_iter,
                out_attempt.flowA,
                out_attempt.flowM,
                x_right
            );
            const QSSLocalBlocks flow_blocks = build_qss_local_blocks(
                local_for_flow.R_full, qss_boundary.layout, levels
            );
            // March the recycling flows from the fixed upwind inflow for this cell.
            // The inner Picard loop updates the local cell state, not the left-boundary inflow.
            const auto flow_advanced = advance_recycling_flow_one_step(
                config,
                atomic_data,
                qss_boundary.boundary,
                local_for_flow,
                flowA_inflow,
                flowM_inflow,
                dx_cm
            );

            dcr::base::Vector flowA_candidate = out_attempt.flowA;
            if (flowA_candidate.size() == 1) {
                flowA_candidate(0) = implicit_atomic_flow_step(
                    flowA_inflow(0),
                    flow_blocks.atomic_flow_effective_rate_s,
                    exA_over_w,
                    qss_boundary.boundary.u_A,
                    dx_cm
                );
            }
            dcr::base::Vector flowM_candidate = flow_advanced.flowM_next;

            const auto local_for_bg = assemble_local_system(
                config,
                atomic_data,
                plasma,
                grid,
                qss_boundary.boundary,
                bg_full_iter,
                flowA_candidate,
                flowM_candidate,
                x_right
            );
            const QSSLocalBlocks blocks = build_qss_local_blocks(
                local_for_bg.R_full, qss_boundary.layout, levels
            );

            dcr::base::Vector source_r;
            dcr::base::Vector source_t;
            build_qss_source_vectors(blocks, flowA_candidate, flowM_candidate, source_r, source_t);
            dcr::base::Vector source_rM_only = dcr::base::Vector::Zero(blocks.R_rr.rows());
            if (blocks.source_rM.cols() == flowM_candidate.size() && flowM_candidate.size() > 0) {
                source_rM_only = blocks.source_rM * flowM_candidate;
            }

            std::vector<int> p_pos(levels.size(), -1);
            for (int i = 0; i < static_cast<int>(qss_boundary.boundary.P_indices.size()); ++i) {
                const int gi = qss_boundary.boundary.P_indices[static_cast<size_t>(i)];
                if (gi >= 0 && gi < static_cast<int>(levels.size())) p_pos[static_cast<size_t>(gi)] = i;
            }

            double n_I_weighted = 0.0;
            for (int gi : qss_boundary.boundary.ion_indices) {
                if (gi < 0 || gi >= static_cast<int>(levels.size())) continue;
                const int pi = p_pos[static_cast<size_t>(gi)];
                if (pi < 0 || pi >= out_attempt.retained.size()) continue;
                n_I_weighted += static_cast<double>(std::max(1, levels[static_cast<size_t>(gi)].atomicity))
                              * std::max(out_attempt.retained(pi), 0.0);
            }

            const double n_a = group_sum(bg_full_iter, qss_boundary.boundary.atom_bg_indices) + positive_sum(out_attempt.transient);
            const double n_m = group_sum(bg_full_iter, qss_boundary.boundary.mol_bg_indices);
            const double c_s_M = dcr::physics::calculate_thermal_speed(
                config.plasma.neutral_molecule_temperature_eV, qss_boundary.boundary.molecule_mass_amu);
            const double exM_over_w = c_s_M / w_local;
            const double gamma_ex_a = exA_over_w * std::max(0.0, n_a);
            const double gamma_ex_m = exM_over_w * std::max(0.0, n_m);
            const double source_nuclei = nuclei_from_source(qss_boundary.boundary, levels, source_r) -
                ((blocks.R_tt.rows() > 0) ? nuclei_from_source(qss_boundary.boundary, levels, blocks.transient_to_retained * source_t) : 0.0);
            // Ion-flux divergence balances recycling source against local
            // background neutral exhaust. Recycled-flow exhaust (A/M) is not
            // included here.
            const double L_I = source_nuclei - (gamma_ex_a + 2.0 * gamma_ex_m);

            dcr::base::Vector A_r_diag = build_retained_transport_diag(
                qss_boundary.boundary,
                levels,
                qss_boundary.layout,
                (n_I_weighted > 0.0) ? (L_I / n_I_weighted) : 0.0,
                exA_over_w,
                exM_over_w
            );
            dcr::base::Vector retained_rhs = A_r_diag.asDiagonal() * out_attempt.retained - source_r;
            dcr::base::Vector retained_rhs_flowM = A_r_diag.asDiagonal() * out_attempt.retained - source_rM_only;
            // Match the reduced local QSS model with T_t = 0 in the
            // eliminated transient atomic block during marching as well.
            dcr::base::Vector transient_rhs = -source_t;

            dcr::base::Matrix R_eff = build_qss_effective_retained_matrix(
                blocks, exclude_molecular_ion_from_transient_reconstruction
            );
            dcr::base::Vector rhs_eff = retained_rhs;
            dcr::base::Vector rhs_eff_flowM = retained_rhs_flowM;
            if (blocks.R_tt.rows() > 0) {
                rhs_eff -= blocks.transient_to_retained * transient_rhs;
                rhs_eff_flowM -= blocks.transient_to_retained * transient_rhs;
            }

            dcr::base::Vector stoich = dcr::base::Vector::Zero(out_attempt.retained.size());
            for (int i = 0; i < out_attempt.retained.size(); ++i) {
                const int gi = qss_boundary.boundary.P_indices[static_cast<size_t>(i)];
                stoich(i) = (gi >= 0 && gi < static_cast<int>(levels.size()))
                    ? static_cast<double>(std::max(1, levels[static_cast<size_t>(gi)].atomicity))
                    : 1.0;
            }
            const double flow_nuclei =
                nuclei_sum(flowA_candidate, qss_boundary.boundary.A_indices, levels) +
                nuclei_sum(flowM_candidate, qss_boundary.boundary.M_indices, levels);
            const double target_bg_nuclei = std::max(0.0, config.plasma.total_density - flow_nuclei);

            dcr::base::Matrix C = R_eff;
            const int conservation_row = (C.rows() > 1) ? 1 : 0;
            C.row(conservation_row) = stoich.transpose();
            rhs_eff(conservation_row) = target_bg_nuclei;
            rhs_eff_flowM(conservation_row) = target_bg_nuclei;

            dcr::base::Vector retained_candidate;
            bool used_coupled_flowA = false;
            if (couple_flowA_retained) {
                const bool valid_flow_shape = flowA_candidate.size() == 1 && blocks.source_rA.cols() == 1;
                if (!valid_flow_shape) {
                    log_coupled_fallback("invalid FlowA shape");
                } else {
                    const double flow_coef = dx_cm / std::max(qss_boundary.boundary.u_A, 1e-30);
                    const double lhs_flowA =
                        1.0 - flow_coef * flow_blocks.atomic_flow_effective_rate_s + flow_coef * exA_over_w;
                    if (!std::isfinite(lhs_flowA) || lhs_flowA <= 1e-14) {
                        log_coupled_fallback("singular FlowA transport coefficient");
                    } else {
                        // Linearize the flow-A loss rate against the dominant retained
                        // local groups so the retained<->flowA feedback enters the same
                        // linear system instead of one Picard iteration later.
                        dcr::base::Vector d_rate_dn =
                            dcr::base::Vector::Zero(out_attempt.retained.size());
                        const double base_rate_s = flow_blocks.atomic_flow_effective_rate_s;
                        auto accumulate_group_derivative = [&](const std::vector<int>& positions) {
                            if (positions.empty()) return;
                            double total_density = 0.0;
                            for (int pos : positions) {
                                if (pos < 0 || pos >= out_attempt.retained.size()) continue;
                                total_density += std::max(out_attempt.retained(pos), 0.0);
                            }
                            const double delta_total =
                                std::max(1.0e6, 1.0e-4 * std::max(total_density, 1.0));
                            dcr::base::Vector retained_perturbed = out_attempt.retained;
                            if (total_density > 0.0) {
                                for (int pos : positions) {
                                    if (pos < 0 || pos >= retained_perturbed.size()) continue;
                                    const double weight =
                                        std::max(out_attempt.retained(pos), 0.0) / total_density;
                                    retained_perturbed(pos) += delta_total * weight;
                                }
                            } else {
                                const double delta_each =
                                    delta_total / static_cast<double>(positions.size());
                                for (int pos : positions) {
                                    if (pos < 0 || pos >= retained_perturbed.size()) continue;
                                    retained_perturbed(pos) += delta_each;
                                }
                            }

                            const dcr::base::Vector bg_full_perturbed =
                                lift_qss_background_to_full(
                                    atomic_data.get_total_states(),
                                    qss_boundary.boundary,
                                    qss_boundary.layout,
                                    retained_perturbed,
                                    out_attempt.transient
                                );
                            const auto local_perturbed = assemble_local_system(
                                config,
                                atomic_data,
                                plasma,
                                grid,
                                qss_boundary.boundary,
                                bg_full_perturbed,
                                out_attempt.flowA,
                                out_attempt.flowM,
                                x_right
                            );
                            const QSSLocalBlocks flow_blocks_perturbed =
                                build_qss_local_blocks(
                                    local_perturbed.R_full,
                                    qss_boundary.layout,
                                    levels
                                );
                            const double derivative =
                                (flow_blocks_perturbed.atomic_flow_effective_rate_s - base_rate_s) /
                                delta_total;
                            if (!std::isfinite(derivative)) return;
                            for (int pos : positions) {
                                if (pos < 0 || pos >= d_rate_dn.size()) continue;
                                d_rate_dn(pos) = derivative;
                            }
                        };
                        accumulate_group_derivative(flowA_atomic_ion_positions);
                        accumulate_group_derivative(flowA_atomic_neutral_positions);
                        accumulate_group_derivative(flowA_molecular_neutral_positions);
                        accumulate_group_derivative(retained_molecular_ion_positions);

                        const int retained_unknowns = C.rows();
                        const int aug_size = retained_unknowns + 1;
                        dcr::base::Matrix aug = dcr::base::Matrix::Zero(aug_size, aug_size);
                        dcr::base::Vector aug_rhs = dcr::base::Vector::Zero(aug_size);
                        const dcr::base::Vector flow_row_coupling =
                            (-flow_coef * std::max(out_attempt.flowA(0), 0.0)) * d_rate_dn;
                        if (retained_unknowns > 0) {
                            aug.topLeftCorner(retained_unknowns, retained_unknowns) = C;
                            aug.topRightCorner(retained_unknowns, 1) = blocks.source_rA.col(0);
                            aug.bottomLeftCorner(1, retained_unknowns) =
                                flow_row_coupling.transpose();
                            aug(conservation_row, aug_size - 1) = 0.0;
                            aug_rhs.head(retained_unknowns) = rhs_eff_flowM;
                        }
                        aug(aug_size - 1, aug_size - 1) = lhs_flowA;
                        aug_rhs(aug_size - 1) =
                            flowA_inflow(0) + flow_row_coupling.dot(out_attempt.retained);

                        Eigen::FullPivLU<dcr::base::Matrix> lu(aug);
                        if (lu.rank() < aug_size) {
                            log_coupled_fallback("rank-deficient augmented system");
                        } else {
                            const dcr::base::Vector coupled_solution = lu.solve(aug_rhs);
                            if (!coupled_solution.allFinite()) {
                                log_coupled_fallback("non-finite augmented solution");
                            } else {
                                retained_candidate = coupled_solution.head(retained_unknowns);
                                flowA_candidate(0) = std::max(0.0, coupled_solution(aug_size - 1));
                                used_coupled_flowA = true;
                            }
                        }
                    }
                }
            }
            if (!used_coupled_flowA) {
                retained_candidate = solve_linear(C, rhs_eff);
            }
            apply_non_negative(retained_candidate);
            normalize_background_to_target(qss_boundary.boundary, levels, target_bg_nuclei, retained_candidate);

            dcr::base::Vector transient_candidate = solve_qss_transient_atomic(
                blocks,
                retained_candidate,
                transient_rhs,
                exclude_molecular_ion_from_transient_reconstruction
            );
            apply_non_negative(transient_candidate);

            const double rel = std::max(
                std::max(relative_change(retained_candidate, out_attempt.retained),
                         relative_change(transient_candidate, out_attempt.transient)),
                std::max(relative_change(flowA_candidate, out_attempt.flowA),
                         relative_change(flowM_candidate, out_attempt.flowM))
            );
            out_attempt.rel = rel;
            out_attempt.iterations = iter + 1;

            const dcr::base::Vector xk = pack_state(out_attempt.retained,
                                                   out_attempt.transient,
                                                   out_attempt.flowA,
                                                   out_attempt.flowM);

            const dcr::base::Vector g_candidate = pack_state(
                retained_candidate, transient_candidate, flowA_candidate, flowM_candidate);

            auto apply_damped = [&](const dcr::base::Vector& g) {
                dcr::base::Vector x = dcr::base::Vector::Zero(x_size);
                if (retained_size > 0)
                    x.segment(0, retained_size) =
                        (1.0 - omega_bg_iter) * xk.segment(0, retained_size) +
                        omega_bg_iter * g.segment(0, retained_size);
                if (transient_size > 0)
                    x.segment(retained_size, transient_size) =
                        (1.0 - omega_bg_iter) * xk.segment(retained_size, transient_size) +
                        omega_bg_iter * g.segment(retained_size, transient_size);
                if (flowA_size > 0)
                    x.segment(retained_size + transient_size, flowA_size) =
                        (1.0 - omega_flow_iter) * xk.segment(retained_size + transient_size, flowA_size) +
                        omega_flow_iter * g.segment(retained_size + transient_size, flowA_size);
                if (flowM_size > 0)
                    x.segment(retained_size + transient_size + flowA_size, flowM_size) =
                        (1.0 - omega_flow_iter) * xk.segment(retained_size + transient_size + flowA_size, flowM_size) +
                        omega_flow_iter * g.segment(retained_size + transient_size + flowA_size, flowM_size);
                return x;
            };
            dcr::base::Vector x_next = apply_damped(g_candidate);
            unpack_state(x_next, out_attempt.retained, out_attempt.transient, out_attempt.flowA, out_attempt.flowM);
            apply_non_negative(out_attempt.retained);
            apply_non_negative(out_attempt.transient);
            apply_non_negative(out_attempt.flowA);
            apply_non_negative(out_attempt.flowM);
            {
                const double flow_nuclei_next =
                    nuclei_sum(out_attempt.flowA, qss_boundary.boundary.A_indices, levels) +
                    nuclei_sum(out_attempt.flowM, qss_boundary.boundary.M_indices, levels);
                const double target_bg_nuclei_next =
                    std::max(0.0, config.plasma.total_density - flow_nuclei_next);
                normalize_background_to_target(qss_boundary.boundary, levels, target_bg_nuclei_next, out_attempt.retained);
            }
        if (config.io.verbose_logging && (cell_index == 1 || (iter + 1) % 20 == 0)) {
                const dcr::base::Vector bg_log = lift_qss_background_to_full(
                    atomic_data.get_total_states(), qss_boundary.boundary, qss_boundary.layout,
                    out_attempt.retained, out_attempt.transient);
                const auto bg = summarize_background(qss_boundary.boundary, bg_log, levels);
                std::cout << "[DCR_Solver][QSS] Marching cell " << cell_index
                          << " x=" << x_right
                          << " cm iter=" << (iter + 1)
                          << " rel=" << rel
                          << " omega{bg=" << omega_bg_iter
                          << ", flow=" << omega_flow_iter
                          << "}"
                          << " bg{H+=" << bg.H_plus
                          << ", H=" << bg.H
                          << ", H2=" << bg.H2
                          << ", H2+=" << bg.H2_plus
                          << ", H-=" << bg.H_minus
                          << "} flow{A=" << positive_sum(out_attempt.flowA)
                          << ", M=" << positive_sum(out_attempt.flowM)
                          << "}\n";
            }

            if (rel < tol) {
                out_attempt.converged = true;
                break;
            }

            if (std::isfinite(prev_rel_change) && rel > prev_rel_change * 1.001) {
                ++rel_growth_count;
            } else {
                rel_growth_count = 0;
            }

            if (rel >= prev_rel_change * 0.999) {
                ++no_improve_count;
            } else {
                no_improve_count = 0;
            }

            if (rel_growth_count >= 3 &&
                (omega_bg_iter > omega_min_cell || omega_flow_iter > omega_min_cell)) {
                const double new_omega_bg = std::max(omega_min_cell, 0.5 * omega_bg_iter);
                const double new_omega_flow = std::max(omega_min_cell, 0.5 * omega_flow_iter);
                if (new_omega_bg < omega_bg_iter || new_omega_flow < omega_flow_iter) {
                    omega_bg_iter = new_omega_bg;
                    omega_flow_iter = new_omega_flow;
                    rel_growth_count = 0;
                    no_improve_count = 0;
                }
            } else if (no_improve_count >= 5 &&
                       (omega_bg_iter > omega_min_cell || omega_flow_iter > omega_min_cell)) {
                const double new_omega_bg = std::max(omega_min_cell, 0.5 * omega_bg_iter);
                const double new_omega_flow = std::max(omega_min_cell, 0.5 * omega_flow_iter);
                if (new_omega_bg < omega_bg_iter || new_omega_flow < omega_flow_iter) {
                    omega_bg_iter = new_omega_bg;
                    omega_flow_iter = new_omega_flow;
                    no_improve_count = 0;
                }
            } else if (((iter + 1) % 50 == 0) && rel > (10.0 * tol) &&
                       (omega_bg_iter > omega_min_cell || omega_flow_iter > omega_min_cell)) {
                const double new_omega_bg = std::max(omega_min_cell, 0.5 * omega_bg_iter);
                const double new_omega_flow = std::max(omega_min_cell, 0.5 * omega_flow_iter);
                if (new_omega_bg < omega_bg_iter || new_omega_flow < omega_flow_iter) {
                    omega_bg_iter = new_omega_bg;
                    omega_flow_iter = new_omega_flow;
                }
            }

            prev_rel_change = rel;
        }

    };

    for (size_t k = 0; k < dx.size(); ++k) {
        const int cell_index = static_cast<int>(k + 1);
        const double x_right = out.history.x_cm[k + 1];
        const double dx_cm = dx[static_cast<size_t>(k)];

        const dcr::base::Vector retained_current = retained;
        const dcr::base::Vector transient_current = transient;
        const dcr::base::Vector flowA_current = flowA;
        const dcr::base::Vector flowM_current = flowM;

        QSSCellAttempt accepted;
        const bool first_cell = (cell_index == 1);
        const dcr::base::Vector retained_seed = first_cell ? retained_boundary : retained_current;
        const dcr::base::Vector transient_seed = first_cell ? transient_boundary : transient_current;
        const dcr::base::Vector flowA_seed = flowA_current;
        const dcr::base::Vector flowM_seed = flowM_current;

        solve_cell_attempt(retained_seed,
                          transient_seed,
                          flowA_seed,
                          flowM_seed,
                          flowA_current,
                          flowM_current,
                          cell_index,
                          x_right,
                          dx_cm,
                          accepted);

        retained = accepted.retained;
        transient = accepted.transient;
        flowA = accepted.flowA;
        flowM = accepted.flowM;

        const dcr::base::Vector bg_full = lift_qss_background_to_full(
            atomic_data.get_total_states(), qss_boundary.boundary, qss_boundary.layout, retained, transient);
        out.history.background_full.push_back(bg_full);
        out.history.flowA.push_back(flowA);
        out.history.flowM.push_back(flowM);
        const auto local_diag = assemble_local_system(
            config, atomic_data, plasma, grid, qss_boundary.boundary, bg_full, flowA, flowM, x_right
        );
        const QSSLocalBlocks flow_blocks_diag = build_qss_local_blocks(
            local_diag.R_full, qss_boundary.layout, levels
        );
        dcr::base::Vector flow_transient_diag =
            reconstruct_qss_atomic_flow_transient(flow_blocks_diag, flowA);
        apply_non_negative(flow_transient_diag);
        out.history.qss_local_transient_atomic.push_back(transient);
        out.history.qss_flow_transient_atomic.push_back(flow_transient_diag);
        out.history.rate_diagnostics.push_back(
            rate_calculator.evaluate(config, qss_boundary.boundary, local_diag, bg_full, x_right)
        );
        if (config.io.verbose_logging) {
            const auto bg = summarize_background(qss_boundary.boundary, bg_full, levels);
            std::cout << "[DCR_Solver][QSS] Marching cell " << cell_index
                      << " x=" << x_right
                      << " cm " << (accepted.converged ? "converged" : "max-iter")
                      << " in " << accepted.iterations
                      << " iterations (rel=" << accepted.rel << ")"
                      << " bg{H+=" << bg.H_plus
                      << ", H=" << bg.H
                      << ", H2=" << bg.H2
                      << ", H2+=" << bg.H2_plus
                      << ", H-=" << bg.H_minus
                      << "} flow{A=" << positive_sum(flowA)
                      << ", M=" << positive_sum(flowM)
                      << "}\n";
            log_rate_snapshot(x_right, out.history.rate_diagnostics.back());
        }
    }

    out.retained_background = retained;
    out.transient_atomic = transient;
    out.flowA = flowA;
    out.flowM = flowM;
    return out;
}

} // namespace dcr::solver
