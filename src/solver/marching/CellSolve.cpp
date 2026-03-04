#include "CellSolve.hpp"

#include "../../physics/Sheath.hpp"

#include <cmath>
#include <iostream>
#include <limits>

namespace dcr::solver {

dcr::base::Vector make_background_full(const dcr::base::Vector& nP,
                                       const BoundaryPhaseResult& boundary,
                                       int total_states) {
    dcr::base::Vector full = dcr::base::Vector::Zero(total_states);
    const int Pn = static_cast<int>(boundary.P_indices.size());
    for (int i = 0; i < Pn && i < nP.size(); ++i) {
        const int gi = boundary.P_indices[static_cast<size_t>(i)];
        if (gi >= 0 && gi < total_states) {
            full(gi) = std::max(nP(i), 0.0);
        }
    }
    return full;
}

namespace {

struct BgSubgroupSums {
    double H_plus = 0.0;
    double H = 0.0;
    double H2 = 0.0;
    double H2_plus = 0.0;
    double H_minus = 0.0;
};

struct BackgroundSolveResult {
    dcr::base::Vector nP_new;
    // Diagnostics for the solved linear system C*n = rhs (with closure row).
    double linear_residual_norm = std::numeric_limits<double>::quiet_NaN();
    double linear_mse = std::numeric_limits<double>::quiet_NaN();
    double linear_mse_rel = std::numeric_limits<double>::quiet_NaN();
};

double group_sum(const dcr::base::Vector& full,
                 const std::vector<int>& indices) {
    double s = 0.0;
    for (int gi : indices) {
        if (gi >= 0 && gi < full.size()) s += std::max(full(gi), 0.0);
    }
    return s;
}

double nuclei_sum(const dcr::base::Vector& vec,
                  const std::vector<int>& global_indices,
                  const std::vector<dcr::atomic::EnergyLevel>& levels) {
    double s = 0.0;
    for (size_t i = 0; i < global_indices.size() && static_cast<int>(i) < vec.size(); ++i) {
        const int gi = global_indices[i];
        if (gi < 0 || gi >= static_cast<int>(levels.size())) continue;
        s += levels[gi].atomicity * std::max(vec(static_cast<int>(i)), 0.0);
    }
    return s;
}

double nuclei_sum_background(const dcr::base::Vector& nP,
                             const BoundaryPhaseResult& boundary,
                             const std::vector<dcr::atomic::EnergyLevel>& levels) {
    double s = 0.0;
    const int n = std::min<int>(nP.size(), static_cast<int>(boundary.P_indices.size()));
    for (int i = 0; i < n; ++i) {
        const int gi = boundary.P_indices[static_cast<size_t>(i)];
        if (gi < 0 || gi >= static_cast<int>(levels.size())) continue;
        s += levels[gi].atomicity * std::max(nP(i), 0.0);
    }
    return s;
}

// Electron density from quasi-neutrality on a full state vector.
dcr::base::Matrix extract_R_PP(const dcr::base::Matrix& R_full,
                               const std::vector<int>& P_indices) {
    const int Pn = static_cast<int>(P_indices.size());
    dcr::base::Matrix Rpp = dcr::base::Matrix::Zero(Pn, Pn);
    for (int i = 0; i < Pn; ++i) {
        const int gi = P_indices[static_cast<size_t>(i)];
        if (gi < 0 || gi >= R_full.rows()) continue;
        for (int j = 0; j < Pn; ++j) {
            const int gj = P_indices[static_cast<size_t>(j)];
            if (gj < 0 || gj >= R_full.cols()) continue;
            Rpp(i, j) = R_full(gi, gj);
        }
    }
    return Rpp;
}

void apply_nonneg(dcr::base::Vector& nP) {
    for (int i = 0; i < nP.size(); ++i) {
        if (nP(i) < 0.0) nP(i) = 0.0;
    }
}

// Solve replaced square system using explicit inverse (requested),
// with QR fallback if the matrix is rank-deficient.
dcr::base::Vector solve_replaced_system(const dcr::base::Matrix& A,
                                        const dcr::base::Vector& b) {
    if (A.rows() == A.cols()) {
        Eigen::FullPivLU<dcr::base::Matrix> lu(A);
        if (lu.rank() == A.rows()) {
            const dcr::base::Matrix A_inv = lu.inverse();
            return A_inv * b;
        }
    }
    // Fallback for rank-deficient/rectangular systems.
    return A.colPivHouseholderQr().solve(b);
}

BackgroundSolveResult solve_background_at_cell(
    const dcr::io::Config& config,
    const BoundaryPhaseResult& boundary,
    const std::vector<dcr::atomic::EnergyLevel>& levels,
    const LocalSystem& local_system,
    const dcr::base::Vector& nP_guess,
    const dcr::base::Vector& flowA_new,
    const dcr::base::Vector& flowM_new) {

    BackgroundSolveResult result;
    const int Pn = static_cast<int>(boundary.P_indices.size());
    if (Pn == 0) {
        result.nP_new = nP_guess;
        return result;
    }

    std::vector<int> p_pos(levels.size(), -1);
    for (int i = 0; i < Pn; ++i) {
        const int gi = boundary.P_indices[static_cast<size_t>(i)];
        if (gi >= 0 && gi < static_cast<int>(levels.size())) p_pos[static_cast<size_t>(gi)] = i;
    }

    const dcr::base::Vector bg_full = make_background_full(nP_guess, boundary, static_cast<int>(levels.size()));
    double n_I = 0.0;
    for (int gi : boundary.ion_indices) {
        if (gi < 0 || gi >= static_cast<int>(levels.size()) || gi >= bg_full.size()) continue;
        const double mu_i = std::max(1, levels[gi].atomicity);
        n_I += mu_i * std::max(bg_full(gi), 0.0);
    }
    const double n_a = group_sum(bg_full, boundary.atom_bg_indices);
    const double n_m = group_sum(bg_full, boundary.mol_bg_indices);

    // Derive grouped recycling source directly from the assembled S vector.
    double source_nuclei_from_S = 0.0;
    for (int pi = 0; pi < Pn; ++pi) {
        if (pi >= local_system.S_background.size()) break;
        const int gi = boundary.P_indices[static_cast<size_t>(pi)];
        const double mu_i = (gi >= 0 && gi < static_cast<int>(levels.size()))
            ? static_cast<double>(std::max(1, levels[static_cast<size_t>(gi)].atomicity))
            : 1.0;
        source_nuclei_from_S += mu_i * local_system.S_background(pi);
    }

    const double nA_flow = flowA_new.size() > 0 ? flowA_new.cwiseMax(0.0).sum() : 0.0;
    const double nM_flow = flowM_new.size() > 0 ? flowM_new.cwiseMax(0.0).sum() : 0.0;
    const double mu_M = 2.0;
    const double c_s_A = dcr::physics::calculate_thermal_speed(
        config.plasma.neutral_atom_temperature_eV, boundary.atom_mass_amu
    );
    const double c_s_M = dcr::physics::calculate_thermal_speed(
        config.plasma.neutral_molecule_temperature_eV, boundary.molecule_mass_amu
    );
    const double w_local = std::max(config.grid.poloidal_width_cm, 1e-12);
    const double Gamma_ex_a_over_w = (c_s_A / w_local) * nA_flow;
    const double Gamma_ex_m_over_w = (c_s_M / w_local) * nM_flow;

    // Ion transport/source balance.
    // Use the same assembled S used on RHS for the grouped recycling source term,
    // with local exhaust terms restored.
    const double L_I = source_nuclei_from_S - (Gamma_ex_a_over_w + mu_M * Gamma_ex_m_over_w);
    const double L_a = Gamma_ex_a_over_w;
    const double L_m = Gamma_ex_m_over_w;

    dcr::base::Matrix A = dcr::base::Matrix::Zero(Pn, Pn);
    if (n_I > 0.0) {
        for (int gi : boundary.ion_indices) {
            const int pi = (gi >= 0 && gi < static_cast<int>(p_pos.size())) ? p_pos[static_cast<size_t>(gi)] : -1;
            if (pi >= 0 && gi >= 0 && gi < static_cast<int>(levels.size())) {
                // Ion-state share uses n_i / (sum_j mu_j n_j).
                A(pi, pi) = L_I / n_I;
            }
        }
    }
    if (n_a > 0.0) {
        const double coeff = L_a / n_a;
        for (int gi : boundary.atom_bg_indices) {
            const int pi = (gi >= 0 && gi < static_cast<int>(p_pos.size())) ? p_pos[static_cast<size_t>(gi)] : -1;
            if (pi >= 0) A(pi, pi) = coeff;
        }
    }
    if (n_m > 0.0) {
        const double coeff = L_m / n_m;
        for (int gi : boundary.mol_bg_indices) {
            const int pi = (gi >= 0 && gi < static_cast<int>(p_pos.size())) ? p_pos[static_cast<size_t>(gi)] : -1;
            if (pi >= 0) A(pi, pi) = coeff;
        }
    }

    const dcr::base::Matrix Rpp = extract_R_PP(local_system.R_full, boundary.P_indices);

    // Local background solve from Eq. (1.345) with frozen coefficients in this iteration:
    //   R_PP n^{k+1} = (T - S), with T frozen from current iterate n^k.
    // Replace one row by atomicity closure so the singular chemistry system is well-posed.
    dcr::base::Matrix C = Rpp;
    dcr::base::Vector rhs = A * nP_guess - local_system.S_background;
    dcr::base::Vector stoich = dcr::base::Vector::Zero(Pn);
    for (int i = 0; i < Pn; ++i) {
        const int gi = boundary.P_indices[static_cast<size_t>(i)];
        stoich(i) = (gi >= 0 && gi < static_cast<int>(levels.size()))
            ? static_cast<double>(levels[gi].atomicity) : 1.0;
    }
    int constraint_row = 12;
    const double flow_nuclei = nuclei_sum(flowA_new, boundary.A_indices, levels)
                             + nuclei_sum(flowM_new, boundary.M_indices, levels);
    const double target_bg_nuclei = std::max(0.0, config.plasma.total_density - flow_nuclei);

    auto set_original_diag = [&](const dcr::base::Vector& n_eval) {
        // Diagnostic on original Eq. (1.345) form:
        //   R_PP n - (A n - S)
        const dcr::base::Vector r = Rpp * n_eval - (A * n_eval - local_system.S_background);
        result.linear_residual_norm = r.norm();
        result.linear_mse = r.squaredNorm() / static_cast<double>(Pn);
        const double lhs_norm = (Rpp * n_eval).norm();
        result.linear_mse_rel = result.linear_residual_norm / std::max(1.0, lhs_norm);
    };

    // Single background linear solve path: row replacement with nuclei closure.
    dcr::base::Matrix C_replaced = C;
    dcr::base::Vector rhs_replaced = rhs;
    C_replaced.row(constraint_row) = stoich.transpose();
    rhs_replaced(constraint_row) = target_bg_nuclei;
    dcr::base::Vector nP_new = solve_replaced_system(C_replaced, rhs_replaced);

    if (!nP_new.allFinite()) {
        dcr::base::Vector fallback = nP_guess;
        apply_nonneg(fallback);
        result.nP_new = fallback;
        set_original_diag(fallback);
        return result;
    }

    const double now = (stoich.array() * nP_new.array()).sum();
    if (now > 0.0) {
        nP_new *= (target_bg_nuclei / now);
        result.nP_new = nP_new;
        set_original_diag(nP_new);
        return result;
    }

    dcr::base::Vector fallback = nP_guess;
    apply_nonneg(fallback);
    result.nP_new = fallback;
    set_original_diag(fallback);
    return result;
}

double relative_change(const dcr::base::Vector& now, const dcr::base::Vector& old) {
    if (now.size() == 0 || old.size() == 0) return 0.0;
    const double diff = (now - old).cwiseAbs().maxCoeff();
    const double scale = std::max(1.0, old.cwiseAbs().maxCoeff());
    return diff / scale;
}

// Group background populations into requested hydrogen sub-groups.
BgSubgroupSums summarize_bg_subgroups(const dcr::base::Vector& nP,
                                      const BoundaryPhaseResult& boundary,
                                      const std::vector<dcr::atomic::EnergyLevel>& levels) {
    BgSubgroupSums s;
    const int Pn = std::min<int>(nP.size(), static_cast<int>(boundary.P_indices.size()));
    for (int i = 0; i < Pn; ++i) {
        const int gi = boundary.P_indices[static_cast<size_t>(i)];
        if (gi < 0 || gi >= static_cast<int>(levels.size())) continue;
        const auto& lvl = levels[gi];
        const double n = std::max(nP(i), 0.0);

        if (lvl.type == dcr::atomic::SpeciesType::Ion && lvl.charge > 0) {
            if (lvl.atomicity >= 2) s.H2_plus += n;
            else s.H_plus += n;
        } else if (lvl.type == dcr::atomic::SpeciesType::Ion && lvl.charge < 0) {
            s.H_minus += n;
        } else if (lvl.type == dcr::atomic::SpeciesType::Atom && lvl.charge == 0) {
            s.H += n;
        } else if (lvl.type == dcr::atomic::SpeciesType::Molecule && lvl.charge == 0) {
            s.H2 += n;
        }
    }
    return s;
}

double positive_sum(const dcr::base::Vector& v) {
    double s = 0.0;
    for (int i = 0; i < v.size(); ++i) s += std::max(v(i), 0.0);
    return s;
}

} // namespace

CellImplicitResult solve_cell_implicit(
    const dcr::io::Config& config,
    const dcr::atomic::AtomicData& atomic_data,
    const dcr::state::PlasmaState& plasma,
    const EEDFGridView& grid,
    const BoundaryPhaseResult& boundary,
    const std::vector<dcr::atomic::EnergyLevel>& levels,
    const dcr::base::Vector& nP_old,
    const dcr::base::Vector& flowA_old,
    const dcr::base::Vector& flowM_old,
    double dx_cm,
    double x_left_cm,
    double x_right_cm,
    int cell_index,
    bool detailed_log,
    bool emit_summary_log,
    const dcr::base::Vector* nP_init_override,
    const dcr::base::Vector* flowA_init_override,
    const dcr::base::Vector* flowM_init_override) {

    CellImplicitResult out;
    const int total_states = atomic_data.get_total_states();

    dcr::base::Vector nP_iter = nP_old;
    dcr::base::Vector flowA_iter = flowA_old;
    dcr::base::Vector flowM_iter = flowM_old;
    if (nP_init_override && nP_init_override->size() == nP_old.size()) {
        nP_iter = *nP_init_override;
    }
    if (flowA_init_override && flowA_init_override->size() == flowA_old.size()) {
        flowA_iter = *flowA_init_override;
    }
    if (flowM_init_override && flowM_init_override->size() == flowM_old.size()) {
        flowM_iter = *flowM_init_override;
    }

    const double tol = std::max(config.numerics.tolerance, 1e-12);
    const int max_iter = std::max(config.numerics.max_iterations, 1);
    const double omega_raw = config.numerics.relaxation;
    const double omega_init = (omega_raw > 0.0 && omega_raw <= 1.0) ? omega_raw : 1.0;
    double omega_iter = omega_init;
    double prev_rel_change = std::numeric_limits<double>::infinity();
    int no_improve_count = 0;

    double last_rel = std::numeric_limits<double>::infinity();
    double last_resid_rel = std::numeric_limits<double>::infinity();

    for (int iter = 0; iter < max_iter; ++iter) {
        const dcr::base::Vector bg_iter_full = make_background_full(nP_iter, boundary, total_states);

        // Step A: flow update at x_{k+1} from Eq. (1.346) with implicit block solve.
        const auto local_for_flow = assemble_local_system(
            atomic_data, plasma, grid, boundary, bg_iter_full, flowA_iter, flowM_iter
        );
        const auto flow_advanced = advance_recycling_flow_one_step(
            config, atomic_data, boundary, local_for_flow, flowA_old, flowM_old, dx_cm
        );

        // Step B: background update at x_{k+1} from grouped Eq. (1.345).
        const auto local_for_bg = assemble_local_system(
            atomic_data, plasma, grid, boundary, bg_iter_full,
            flow_advanced.flowA_next, flow_advanced.flowM_next
        );
        const auto bg_solve = solve_background_at_cell(
            config,
            boundary,
            levels,
            local_for_bg,
            nP_iter,
            flow_advanced.flowA_next,
            flow_advanced.flowM_next
        );
        dcr::base::Vector nP_new = bg_solve.nP_new;

        dcr::base::Vector flowA_new = flow_advanced.flowA_next;
        dcr::base::Vector flowM_new = flow_advanced.flowM_next;

        // Optional damping for difficult nonlinear coupling.
        if (omega_iter < 1.0) {
            nP_new = (1.0 - omega_iter) * nP_iter + omega_iter * nP_new;
            flowA_new = (1.0 - omega_iter) * flowA_iter + omega_iter * flowA_new;
            flowM_new = (1.0 - omega_iter) * flowM_iter + omega_iter * flowM_new;
        }

        const double rel_np = relative_change(nP_new, nP_iter);
        const double rel_a = relative_change(flowA_new, flowA_iter);
        const double rel_m = relative_change(flowM_new, flowM_iter);
        const double rel = std::max(rel_np, std::max(rel_a, rel_m));
        const double resid_rel = std::isfinite(bg_solve.linear_mse_rel)
            ? bg_solve.linear_mse_rel
            : std::numeric_limits<double>::infinity();
        last_rel = rel;
        last_resid_rel = resid_rel;

        if (rel >= prev_rel_change * 0.999) {
            ++no_improve_count;
        } else {
            no_improve_count = 0;
        }
        if (no_improve_count >= 5 && omega_iter > 0.1) {
            omega_iter = std::max(0.1, 0.5 * omega_iter);
            no_improve_count = 0;
            if (config.io.verbose_logging && detailed_log) {
                std::cout << "[DCR_Solver] Marching cell " << cell_index
                          << ": slow convergence, reducing relaxation to "
                          << omega_iter << "\n";
            }
        }
        if (((iter + 1) % 50 == 0) && rel > (10.0 * tol) && omega_iter > 0.1) {
            omega_iter = std::max(0.1, 0.5 * omega_iter);
            if (config.io.verbose_logging && detailed_log) {
                std::cout << "[DCR_Solver] Marching cell " << cell_index
                          << ": periodic damping, reducing relaxation to "
                          << omega_iter << "\n";
            }
        }
        prev_rel_change = rel;

        nP_iter = nP_new;
        flowA_iter = flowA_new;
        flowM_iter = flowM_new;

        if (config.io.verbose_logging && detailed_log) {
            const double flowA_sum_iter = positive_sum(flowA_new);
            const double flowM_sum_iter = positive_sum(flowM_new);
            const double flowA_nuc_iter = nuclei_sum(flowA_new, boundary.A_indices, levels);
            const double flowM_nuc_iter = nuclei_sum(flowM_new, boundary.M_indices, levels);
            std::cout << "[DCR_Solver] Marching cell " << cell_index
                      << " x=[" << x_left_cm << "," << x_right_cm << "] cm"
                      << " iter " << (iter + 1)
                      << " rel_change=" << rel
                      << " rel_np=" << rel_np
                      << " rel_a=" << rel_a
                      << " rel_m=" << rel_m
                      << " ||Rpp*n-(A*n-S)||=" << bg_solve.linear_residual_norm
                      << " resid_rel(diag)=" << resid_rel
                      << " omega=" << omega_iter
                      << " flow_sum{A=" << flowA_sum_iter
                      << ", M=" << flowM_sum_iter
                      << "} flow_nuclei{A=" << flowA_nuc_iter
                      << ", M=" << flowM_nuc_iter
                      << "}"
                      << "\n";

        }

        if (rel < tol) {
            out.iterations = iter + 1;
            out.converged = true;
            break;
        }
        if (iter == max_iter - 1) {
            out.iterations = max_iter;
            out.converged = false;
        }
    }

    out.nP_new = nP_iter;
    out.flowA_new = flowA_iter;
    out.flowM_new = flowM_iter;

    const dcr::base::Vector bg_full = make_background_full(out.nP_new, boundary, total_states);
    out.local_final = assemble_local_system(
        atomic_data, plasma, grid, boundary, bg_full, out.flowA_new, out.flowM_new
    );
    out.flow_final = advance_recycling_flow_one_step(
        config, atomic_data, boundary, out.local_final, out.flowA_new, out.flowM_new, 0.0
    );

    out.final_rel = last_rel;
    out.final_resid_rel = last_resid_rel;

    if (config.io.verbose_logging && emit_summary_log) {
        const auto bg = summarize_bg_subgroups(out.nP_new, boundary, levels);
        double bg_nuclei = 0.0;
        for (int i = 0; i < out.nP_new.size() && i < static_cast<int>(boundary.P_indices.size()); ++i) {
            const int gi = boundary.P_indices[static_cast<size_t>(i)];
            if (gi < 0 || gi >= static_cast<int>(levels.size())) continue;
            bg_nuclei += levels[gi].atomicity * std::max(out.nP_new(i), 0.0);
        }
        const double flowA_nuclei = nuclei_sum(out.flowA_new, boundary.A_indices, levels);
        const double flowM_nuclei = nuclei_sum(out.flowM_new, boundary.M_indices, levels);
        const double total_nuclei = bg_nuclei + flowA_nuclei + flowM_nuclei;
        const double flowA_total = positive_sum(out.flowA_new);
        const double flowM_total = positive_sum(out.flowM_new);
        std::cout << "[DCR_Solver] Marching cell " << cell_index
                  << " x=" << x_right_cm << " cm"
                  << (out.converged ? " converged" : " max-iter")
                  << " in " << out.iterations
                  << " iterations (rel=" << last_rel
                  << ", resid_rel(diag)=" << out.final_resid_rel << ")"
                  << " bg{H+=" << bg.H_plus
                  << ", H=" << bg.H
                  << ", H2=" << bg.H2
                  << ", H2+=" << bg.H2_plus
                  << ", H-=" << bg.H_minus
                  << "} flow{A=" << flowA_total
                  << ", M=" << flowM_total
                  << "} nuclei_total=" << total_nuclei
                  << "\n";
    }

    return out;
}

} // namespace dcr::solver
