#include "QSSBoundary.hpp"

#include "QSSCommon.hpp"

#include "../core/TemperatureProfile.hpp"
#include "../marching/LocalSystemAssembler.hpp"

#include "../../physics/Sheath.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <vector>

namespace dcr::solver {

namespace {

BoundaryPhaseResult build_qss_seed_boundary(
    const dcr::io::Config& config,
    const dcr::atomic::AtomicData& atomic_data,
    double ion_mass_amu,
    const dcr::physics::WallRecycling& wall) {

    BoundaryPhaseResult out;
    const auto boundary_temperatures = evaluate_plasma_temperatures(config, 0.0);
    const auto& levels = atomic_data.get_levels();
    const int total_states = atomic_data.get_total_states();

    out.population = dcr::base::Vector::Zero(total_states);

    const double u_bohm = dcr::physics::calculate_Bohm_speed(
        boundary_temperatures.electron_eV, boundary_temperatures.ion_eV, ion_mass_amu);
    const double E_ref_atom = wall.gamma_E_atom * wall.ion_impact_energy_ev;
    out.u_A = dcr::physics::calculate_speed_from_energy_ev(E_ref_atom, ion_mass_amu);

    out.atom_mass_amu = 1.0;
    out.molecule_mass_amu = 2.0;
    for (const auto& sp : config.species) {
        if (!sp.is_molecule && sp.charge == 0) {
            out.atom_mass_amu = sp.mass_amu;
        }
        if (sp.is_molecule && sp.charge == 0 && sp.mass_amu > 1.0) {
            out.molecule_mass_amu = sp.mass_amu;
        }
    }
    out.u_M = dcr::physics::calculate_thermal_speed(config.wall.temperature_eV, out.molecule_mass_amu);

    const auto& recycling_indices = atomic_data.get_recycling_indices();
    out.explicit_recycling = !recycling_indices.empty();
    out.is_recycling.assign(static_cast<size_t>(total_states), false);
    for (int idx : recycling_indices) {
        if (idx >= 0 && idx < total_states) out.is_recycling[static_cast<size_t>(idx)] = true;
    }

    for (int gi = 0; gi < total_states; ++gi) {
        if (out.is_recycling[static_cast<size_t>(gi)]) out.R_indices.push_back(gi);
        else out.P_indices.push_back(gi);
    }

    if (out.explicit_recycling) {
        for (int gi : out.R_indices) {
            if (gi < 0 || gi >= static_cast<int>(levels.size())) continue;
            const auto& lvl = levels[static_cast<size_t>(gi)];
            if (lvl.type == dcr::atomic::SpeciesType::Atom && lvl.charge == 0) out.A_indices.push_back(gi);
            if (lvl.type == dcr::atomic::SpeciesType::Molecule && lvl.charge == 0) out.M_indices.push_back(gi);
        }
    } else {
        for (int gi : out.P_indices) {
            if (gi < 0 || gi >= static_cast<int>(levels.size())) continue;
            const auto& lvl = levels[static_cast<size_t>(gi)];
            if (lvl.type == dcr::atomic::SpeciesType::Atom && lvl.charge == 0) out.A_indices.push_back(gi);
            if (lvl.type == dcr::atomic::SpeciesType::Molecule && lvl.charge == 0) out.M_indices.push_back(gi);
        }
    }

    out.atom_ground = find_ground_state(out.A_indices, levels);
    out.molecule_ground = find_ground_state(out.M_indices, levels);

    for (int gi : out.P_indices) {
        if (gi < 0 || gi >= static_cast<int>(levels.size())) continue;
        const auto& lvl = levels[static_cast<size_t>(gi)];
        if (lvl.type == dcr::atomic::SpeciesType::Ion && lvl.charge > 0) {
            out.ion_indices.push_back(gi);
        } else if (lvl.type == dcr::atomic::SpeciesType::Atom && lvl.charge == 0) {
            out.atom_bg_indices.push_back(gi);
        } else if (lvl.type == dcr::atomic::SpeciesType::Molecule && lvl.charge == 0) {
            out.mol_bg_indices.push_back(gi);
        }
    }

    for (const auto& ic : config.plasma.initial_conditions) {
        if (ic.species_index < 0 || ic.species_index >= total_states) continue;
        if (out.explicit_recycling && out.is_recycling[static_cast<size_t>(ic.species_index)]) continue;
        out.population(ic.species_index) = ic.fraction * config.plasma.total_density;
    }

    dcr::base::Vector nP = dcr::base::Vector::Zero(static_cast<int>(out.P_indices.size()));
    dcr::base::Vector stoich = dcr::base::Vector::Zero(static_cast<int>(out.P_indices.size()));
    for (int i = 0; i < nP.size(); ++i) {
        const int gi = out.P_indices[static_cast<size_t>(i)];
        if (gi >= 0 && gi < total_states) {
            nP(i) = std::max(out.population(gi), 0.0);
        }
        stoich(i) = (gi >= 0 && gi < static_cast<int>(levels.size()))
            ? static_cast<double>(std::max(1, levels[static_cast<size_t>(gi)].atomicity))
            : 1.0;
    }

    const double nuclei_target = std::max(0.0, config.plasma.total_density);
    const double weighted = stoich.dot(nP);
    if (weighted > 0.0 && nuclei_target > 0.0) {
        nP *= (nuclei_target / weighted);
    } else if (nuclei_target > 0.0) {
        const double stoich_sum = std::max(1.0, stoich.sum());
        nP.setConstant(nuclei_target / stoich_sum);
    }

    for (int i = 0; i < nP.size(); ++i) {
        const int gi = out.P_indices[static_cast<size_t>(i)];
        if (gi >= 0 && gi < total_states) {
            out.population(gi) = nP(i);
        }
    }

    out.have_flow_last = false;
    out.flowA_last = dcr::base::Vector::Zero(static_cast<int>(out.A_indices.size()));
    out.flowM_last = dcr::base::Vector::Zero(static_cast<int>(out.M_indices.size()));
    return out;
}

} // namespace

QSSBoundaryResult solve_qss_boundary(
    const dcr::io::Config& config,
    const dcr::atomic::AtomicData& atomic_data,
    dcr::state::PlasmaState& plasma,
    const EEDFGridView& grid,
    double ion_mass_amu,
    const dcr::physics::WallRecycling& wall) {

    const auto& levels = atomic_data.get_levels();
    // Build the QSS boundary problem directly from the config/state partition,
    // rather than warm-starting from a hidden full-DCR boundary solve.
    const BoundaryPhaseResult seed_boundary =
        build_qss_seed_boundary(config, atomic_data, ion_mass_amu, wall);
    const QSSLayout layout = build_qss_layout(seed_boundary, levels);
    BoundaryPhaseResult qss_boundary = make_qss_boundary_template(seed_boundary, layout);
    if (config.io.verbose_logging) {
        log_qss_layout(layout, levels);
    }

    dcr::base::Vector retained = make_qss_retained_background(qss_boundary, seed_boundary.population);
    dcr::base::Vector transient = extract_transient_seed(layout, seed_boundary.population);
    dcr::base::Vector flowA = extract_atomic_flow_seed(layout);
    dcr::base::Vector flowM = extract_molecular_flow_seed(qss_boundary);

    const auto boundary_temperatures = evaluate_plasma_temperatures(config, 0.0);
    const double u_bohm = dcr::physics::calculate_Bohm_speed(
        boundary_temperatures.electron_eV, boundary_temperatures.ion_eV, ion_mass_amu);
    const double mu_M = 2.0;
    const double c_A = (qss_boundary.u_A > 0.0) ? (wall.alpha_atom * u_bohm / qss_boundary.u_A) : 0.0;
    const double c_A_base = (qss_boundary.u_A > 0.0) ? (u_bohm / qss_boundary.u_A) : 0.0;
    const double c_M_atom = (qss_boundary.u_M > 0.0) ? (wall.alpha_molecule * u_bohm / (mu_M * qss_boundary.u_M)) : 0.0;

    std::vector<int> p_pos(levels.size(), -1);
    for (int i = 0; i < static_cast<int>(qss_boundary.P_indices.size()); ++i) {
        const int gi = qss_boundary.P_indices[static_cast<size_t>(i)];
        if (gi >= 0 && gi < static_cast<int>(levels.size())) p_pos[static_cast<size_t>(gi)] = i;
    }
    const RecyclingModel rec_model = build_recycling_model(qss_boundary, levels, p_pos, c_A, c_A_base, c_M_atom);

    const double c_s_A = dcr::physics::calculate_thermal_speed(
        boundary_temperatures.ion_eV, qss_boundary.atom_mass_amu);
    const double c_s_M = dcr::physics::calculate_thermal_speed(
        config.plasma.neutral_molecule_temperature_eV, qss_boundary.molecule_mass_amu);
    const double w_local = std::max(config.grid.boundary_poloidal_width_cm, 1e-12);
    const double exA_over_w = c_s_A / w_local;
    const double exM_over_w = c_s_M / w_local;

    const int max_iter = std::max(1, config.numerics.max_iterations);
    const double tol = std::max(1e-12, config.numerics.tolerance);
    double omega_iter = (config.numerics.relaxation > 0.0 && config.numerics.relaxation <= 1.0)
        ? config.numerics.relaxation
        : 1.0;
    const double omega_min = 0.05;
    const double omega_max = 0.95;
    const int retained_size = retained.size();
    const int transient_size = transient.size();
    const int flowA_size = flowA.size();
    const int flowM_size = flowM.size();
    const int state_size = retained_size + transient_size + flowA_size + flowM_size;
    dcr::base::Vector best_retained = retained;
    dcr::base::Vector best_transient = transient;
    dcr::base::Vector best_flowA = flowA;
    dcr::base::Vector best_flowM = flowM;
    double best_rel = std::numeric_limits<double>::infinity();
    bool converged = false;
    double prev_rel_boundary = std::numeric_limits<double>::infinity();
    int rel_growth_count_boundary = 0;

    auto pack_state = [&](const dcr::base::Vector& retained_state,
                          const dcr::base::Vector& transient_state,
                          const dcr::base::Vector& flowA_state,
                          const dcr::base::Vector& flowM_state) {
        dcr::base::Vector state = dcr::base::Vector::Zero(state_size);
        if (retained_size > 0) state.segment(0, retained_size) = retained_state;
        if (transient_size > 0) state.segment(retained_size, transient_size) = transient_state;
        if (flowA_size > 0) state.segment(retained_size + transient_size, flowA_size) = flowA_state;
        if (flowM_size > 0) state.segment(retained_size + transient_size + flowA_size, flowM_size) = flowM_state;
        return state;
    };
    auto unpack_state = [&](const dcr::base::Vector& state,
                            dcr::base::Vector& retained_state,
                            dcr::base::Vector& transient_state,
                            dcr::base::Vector& flowA_state,
                            dcr::base::Vector& flowM_state) {
        if (retained_size > 0) retained_state = state.segment(0, retained_size);
        if (transient_size > 0) transient_state = state.segment(retained_size, transient_size);
        if (flowA_size > 0) flowA_state = state.segment(retained_size + transient_size, flowA_size);
        if (flowM_size > 0) flowM_state = state.segment(retained_size + transient_size + flowA_size, flowM_size);
    };

    for (int iter = 0; iter < max_iter; ++iter) {
        const dcr::base::Vector retained_prev = retained;
        const dcr::base::Vector transient_prev = transient;
        const dcr::base::Vector flowA_prev = flowA;
        const dcr::base::Vector flowM_prev = flowM;

        dcr::base::Vector bg_full =
            lift_qss_background_to_full(atomic_data.get_total_states(), qss_boundary, layout, retained, transient);

        double n_I_weighted = 0.0;
        for (int gi : qss_boundary.ion_indices) {
            if (gi < 0 || gi >= static_cast<int>(levels.size())) continue;
            const int pi = p_pos[static_cast<size_t>(gi)];
            if (pi < 0 || pi >= retained.size()) continue;
            n_I_weighted += static_cast<double>(std::max(1, levels[static_cast<size_t>(gi)].atomicity))
                          * std::max(retained(pi), 0.0);
        }

        const double n_a = group_sum(bg_full, qss_boundary.atom_bg_indices) + positive_sum(transient);
        const double n_m = group_sum(bg_full, qss_boundary.mol_bg_indices);

        const double total_guess_nuclei =
            n_I_weighted +
            n_a +
            mu_M * n_m +
            nuclei_sum(flowA, qss_boundary.A_indices, levels) +
            nuclei_sum(flowM, qss_boundary.M_indices, levels);
        if (total_guess_nuclei > 0.0) {
            const double scale = config.plasma.total_density / total_guess_nuclei;
            retained *= scale;
            transient *= scale;
            flowA *= scale;
            flowM *= scale;
            bg_full = lift_qss_background_to_full(
                atomic_data.get_total_states(), qss_boundary, layout, retained, transient);
            n_I_weighted *= scale;
        }

        const auto local_system = assemble_local_system(
            config, atomic_data, plasma, grid, qss_boundary, bg_full, flowA, flowM, 0.0);
        const QSSLocalBlocks blocks = build_qss_local_blocks(local_system.R_full, layout, levels);

        dcr::base::Vector source_r;
        dcr::base::Vector source_t;
        build_qss_source_vectors(blocks, flowA, flowM, source_r, source_t);

        const double source_nuclei = nuclei_from_source(qss_boundary, levels, source_r) -
            ((blocks.R_tt.rows() > 0) ? nuclei_from_source(qss_boundary, levels, blocks.transient_to_retained * source_t) : 0.0);
        const double gamma_ex_a = exA_over_w * std::max(0.0, n_a);
        const double gamma_ex_m = exM_over_w * std::max(0.0, n_m);
        // Ion-flux divergence balances recycling source against local
        // background neutral exhaust. Recycled-flow exhaust (A/M) is not
        // included here.
        const double L_I = source_nuclei - (gamma_ex_a + mu_M * gamma_ex_m);

        dcr::base::Vector A_r_diag = build_retained_transport_diag(
            qss_boundary,
            levels,
            layout,
            (n_I_weighted > 0.0) ? (L_I / n_I_weighted) : 0.0,
            exA_over_w,
            exM_over_w
        );
        dcr::base::Vector retained_rhs = A_r_diag.asDiagonal() * retained - source_r;
        // User-requested QSS local model: eliminate transient atomic states
        // with T_t = 0, so no transient-transport correction appears in the
        // retained local solve.
        dcr::base::Vector transient_rhs = -source_t;

        dcr::base::Matrix R_eff = build_qss_effective_retained_matrix(
            blocks,
            config.numerics.qss_exclude_molecular_ion_from_transient_reconstruction
        );
        dcr::base::Vector rhs_eff = retained_rhs;
        if (blocks.R_tt.rows() > 0) {
            rhs_eff -= blocks.transient_to_retained * transient_rhs;
        }

        dcr::base::Vector stoich = dcr::base::Vector::Zero(retained.size());
        for (int i = 0; i < retained.size(); ++i) {
            const int gi = qss_boundary.P_indices[static_cast<size_t>(i)];
            stoich(i) = (gi >= 0 && gi < static_cast<int>(levels.size()))
                ? static_cast<double>(std::max(1, levels[static_cast<size_t>(gi)].atomicity))
                : 1.0;
        }
        const double flow_nuclei =
            nuclei_sum(flowA, qss_boundary.A_indices, levels) +
            nuclei_sum(flowM, qss_boundary.M_indices, levels);
        const double target_bg_nuclei = std::max(0.0, config.plasma.total_density - flow_nuclei);

        dcr::base::Matrix C = R_eff;
        const int conservation_row = (C.rows() > 1) ? 1 : 0;
        C.row(conservation_row) = stoich.transpose();
        rhs_eff(conservation_row) = target_bg_nuclei;
        dcr::base::Vector retained_candidate = solve_linear(C, rhs_eff);
        apply_non_negative(retained_candidate);
        normalize_background_to_target(qss_boundary, levels, target_bg_nuclei, retained_candidate);

        dcr::base::Vector transient_candidate = solve_qss_transient_atomic(
            blocks,
            retained_candidate,
            transient_rhs,
            config.numerics.qss_exclude_molecular_ion_from_transient_reconstruction
        );
        apply_non_negative(transient_candidate);

        double n_A_total_candidate = 0.0;
        double n_M_total_candidate = 0.0;
        for (size_t k = 0; k < rec_model.ion_p_cols.size(); ++k) {
            const int pi = rec_model.ion_p_cols[k];
            if (pi < 0 || pi >= retained_candidate.size()) continue;
            const double n_i = std::max(retained_candidate(pi), 0.0);
            n_A_total_candidate += rec_model.recA_coeff[k] * n_i;
            n_M_total_candidate += rec_model.recM_coeff[k] * n_i;
        }
        dcr::base::Vector flowA_candidate = flowA;
        dcr::base::Vector flowM_candidate = flowM;
        assign_qss_flow_distribution(
            qss_boundary,
            n_A_total_candidate,
            n_M_total_candidate,
            flowA_candidate,
            flowM_candidate
        );

        const dcr::base::Vector xk = pack_state(retained, transient, flowA, flowM);
        const dcr::base::Vector x_candidate =
            pack_state(retained_candidate, transient_candidate, flowA_candidate, flowM_candidate);
        const dcr::base::Vector r_candidate = x_candidate - xk;

        dcr::base::Vector x_next = xk + std::clamp(omega_iter, omega_min, omega_max) * r_candidate;

        unpack_state(x_next, retained, transient, flowA, flowM);
        apply_non_negative(retained);
        apply_non_negative(transient);
        apply_non_negative(flowA);
        apply_non_negative(flowM);
        const double mixed_flow_nuclei =
            nuclei_sum(flowA, qss_boundary.A_indices, levels) +
            nuclei_sum(flowM, qss_boundary.M_indices, levels);
        const double mixed_target_bg_nuclei =
            std::max(0.0, config.plasma.total_density - mixed_flow_nuclei);
        normalize_background_to_target(qss_boundary, levels, mixed_target_bg_nuclei, retained);

        const double rel = std::max(
            std::max(relative_change(retained, retained_prev),
                     relative_change(transient, transient_prev)),
            std::max(relative_change(flowA, flowA_prev),
                     relative_change(flowM, flowM_prev))
        );
        if (rel < best_rel) {
            best_rel = rel;
            best_retained = retained;
            best_transient = transient;
            best_flowA = flowA;
            best_flowM = flowM;
        }
        // Adaptive omega reduction: halve on 3 consecutive rel increases,
        // mirroring the divergence guard in the marching solver.
        if (std::isfinite(prev_rel_boundary) && rel > prev_rel_boundary * 1.001) {
            ++rel_growth_count_boundary;
        } else {
            rel_growth_count_boundary = 0;
        }
        if (rel_growth_count_boundary >= 3 && omega_iter > omega_min) {
            omega_iter = std::max(omega_min, 0.5 * omega_iter);
            rel_growth_count_boundary = 0;
        }
        prev_rel_boundary = rel;
        omega_iter = std::clamp(omega_iter, omega_min, omega_max);

        if (config.io.verbose_logging) {
            const dcr::base::Vector bg_log =
                lift_qss_background_to_full(atomic_data.get_total_states(), qss_boundary, layout, retained, transient);
            log_qss_boundary_iter(iter + 1, rel, flowA, flowM, bg_log, qss_boundary, levels);
        }

        if (rel < tol) {
            converged = true;
            break;
        }
    }

    // The reduced QSS boundary map can still enter a broad limit cycle after
    // the source-operator update. If strict convergence is not reached, keep
    // the best iterate encountered rather than returning the final point on
    // that cycle.
    if (!converged) {
        retained = best_retained;
        transient = best_transient;
        flowA = best_flowA;
        flowM = best_flowM;
        if (config.io.verbose_logging) {
            const dcr::base::Vector bg_best =
                lift_qss_background_to_full(atomic_data.get_total_states(), qss_boundary, layout, retained, transient);
            const auto bg = summarize_background(qss_boundary, bg_best, levels);
            std::cout << "[DCR_Solver][QSS] Boundary did not reach strict tolerance; "
                      << "using best iterate with rel=" << best_rel
                      << " bg{H+=" << bg.H_plus
                      << ", H=" << bg.H
                      << ", H2=" << bg.H2
                      << ", H2+=" << bg.H2_plus
                      << ", H-=" << bg.H_minus
                      << "} flow{A=" << positive_sum(flowA)
                      << ", M=" << positive_sum(flowM)
                      << "}\n";
        }
    }

    qss_boundary.population = lift_qss_background_to_full(
        atomic_data.get_total_states(), qss_boundary, layout, retained, transient);
    qss_boundary.flowA_last = flowA;
    qss_boundary.flowM_last = flowM;
    qss_boundary.have_flow_last = true;

    QSSBoundaryResult out;
    out.boundary = qss_boundary;
    out.layout = layout;
    out.retained_background = retained;
    out.transient_atomic = transient;
    return out;
}

} // namespace dcr::solver
