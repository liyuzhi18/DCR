#include "CellSolve.hpp"

#include "../../physics/Sheath.hpp"
#include "../core/TemperatureProfile.hpp"

#include <cmath>
#include <iostream>
#include <limits>
#include <vector>

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
            return lu.solve(b);
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

    const double mu_M = 2.0;
    const double c_s_A = dcr::physics::calculate_thermal_speed(
        config.plasma.neutral_atom_temperature_eV, boundary.atom_mass_amu
    );
    const double c_s_M = dcr::physics::calculate_thermal_speed(
        config.plasma.neutral_molecule_temperature_eV, boundary.molecule_mass_amu
    );
    const double w_local = std::max(config.grid.poloidal_width_cm, 1e-12);
    // Local background exhaust only (no flow exhaust in local solve).
    const double Gamma_ex_a_over_w = (c_s_A / w_local) * n_a;
    const double Gamma_ex_m_over_w = (c_s_M / w_local) * n_m;
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
    //   R_PP n^{k+1} = A n^k - S.
    // Replace one row by atomicity closure so the singular chemistry system is well-posed.
    dcr::base::Matrix C = Rpp;
    dcr::base::Vector rhs = A * nP_guess - local_system.S_background;
    dcr::base::Vector stoich = dcr::base::Vector::Zero(Pn);
    for (int i = 0; i < Pn; ++i) {
        const int gi = boundary.P_indices[static_cast<size_t>(i)];
        stoich(i) = (gi >= 0 && gi < static_cast<int>(levels.size()))
            ? static_cast<double>(levels[gi].atomicity) : 1.0;
    }
    // Keep the marching closure row aligned with boundary closure row.
    int constraint_row = 1;
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
    // Practical convergence gate for the outer Picard loop:
    // if the fixed-point update is already small and the original local
    // background balance residual is also small, stop instead of waiting for
    // rel_change to crawl all the way down to the strict tolerance.
    const double practical_rel_tol = std::max(100.0 * tol, 1.0e-5);
    const double practical_resid_tol = std::max(1000.0 * tol, 1.0e-4);
    const int practical_stop_min_iter = 10;
    const double omega_raw = config.numerics.relaxation;
    const double omega_min = 0.005;
    const double omega_init = (omega_raw > 0.0 && omega_raw <= 1.0) ? omega_raw : 1.0;
    double omega_iter = std::clamp(omega_init, omega_min, 1.0);
    double prev_rel_change = std::numeric_limits<double>::infinity();
    int no_improve_count = 0;
    int rel_growth_count = 0;
    const auto cell_temperatures = evaluate_plasma_temperatures(config, x_right_cm);

    double last_rel = std::numeric_limits<double>::infinity();
    double last_resid_rel = std::numeric_limits<double>::infinity();
    bool converged_by_practical = false;

    // Anderson mixing on the outer fixed-point map x -> G(x), with x=[nP; flowA; flowM].
    const int anderson_depth = 3;
    const double anderson_reg = 1e-12;
    const double anderson_step_limit = 5.0;
    const double anderson_amp_limit = 10.0;
    std::vector<dcr::base::Vector> anderson_cand_hist;
    std::vector<dcr::base::Vector> anderson_res_hist;

    const int nP_size = nP_iter.size();
    const int nA_size = flowA_iter.size();
    const int nM_size = flowM_iter.size();
    const int x_size = nP_size + nA_size + nM_size;
    auto pack_state = [&](const dcr::base::Vector& nP,
                          const dcr::base::Vector& nA,
                          const dcr::base::Vector& nM) {
        dcr::base::Vector x = dcr::base::Vector::Zero(x_size);
        if (nP_size > 0) x.segment(0, nP_size) = nP;
        if (nA_size > 0) x.segment(nP_size, nA_size) = nA;
        if (nM_size > 0) x.segment(nP_size + nA_size, nM_size) = nM;
        return x;
    };
    auto unpack_state = [&](const dcr::base::Vector& x,
                            dcr::base::Vector& nP,
                            dcr::base::Vector& nA,
                            dcr::base::Vector& nM) {
        if (nP_size > 0) nP = x.segment(0, nP_size);
        if (nA_size > 0) nA = x.segment(nP_size, nA_size);
        if (nM_size > 0) nM = x.segment(nP_size + nA_size, nM_size);
    };

    for (int iter = 0; iter < max_iter; ++iter) {
        const dcr::base::Vector bg_iter_full = make_background_full(nP_iter, boundary, total_states);

        // Step A: flow update at x_{k+1} from Eq. (1.346) with implicit block solve.
        const auto local_for_flow = assemble_local_system(
            config, atomic_data, plasma, grid, boundary, bg_iter_full, flowA_iter, flowM_iter, x_right_cm
        );
        const auto flow_advanced = advance_recycling_flow_one_step(
            config, atomic_data, boundary, local_for_flow, flowA_old, flowM_old, dx_cm
        );

        // Step B: background update at x_{k+1} from grouped Eq. (1.345).
        const auto local_for_bg = assemble_local_system(
            config,
            atomic_data,
            plasma,
            grid,
            boundary,
            bg_iter_full,
            flow_advanced.flowA_next,
            flow_advanced.flowM_next,
            x_right_cm
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

        // Build current/fixed-point state vectors.
        const dcr::base::Vector xk = pack_state(nP_iter, flowA_iter, flowM_iter);
        const dcr::base::Vector gk = pack_state(nP_new, flowA_new, flowM_new);

        // Damped candidate (legacy relaxation path).
        dcr::base::Vector x_candidate = gk;
        if (omega_iter < 1.0) {
            x_candidate = xk + omega_iter * (gk - xk);
        }
        const dcr::base::Vector r_candidate = x_candidate - xk;

        bool used_anderson = false;
        dcr::base::Vector x_next = x_candidate;
        if (anderson_depth > 0 && x_size > 0) {
            const int hist_take =
                std::min(anderson_depth, static_cast<int>(anderson_cand_hist.size()));
            const int pool_size = hist_take + 1; // history + current candidate
            if (pool_size >= 2) {
                const int m = pool_size - 1;
                std::vector<dcr::base::Vector> cand_pool;
                std::vector<dcr::base::Vector> res_pool;
                cand_pool.reserve(static_cast<size_t>(pool_size));
                res_pool.reserve(static_cast<size_t>(pool_size));
                const int start = static_cast<int>(anderson_cand_hist.size()) - hist_take;
                for (int h = start; h < static_cast<int>(anderson_cand_hist.size()); ++h) {
                    cand_pool.push_back(anderson_cand_hist[static_cast<size_t>(h)]);
                    res_pool.push_back(anderson_res_hist[static_cast<size_t>(h)]);
                }
                cand_pool.push_back(x_candidate);
                res_pool.push_back(r_candidate);

                const dcr::base::Vector& r_ref = res_pool.back();
                dcr::base::Matrix B = dcr::base::Matrix::Zero(x_size, m);
                dcr::base::Matrix dC = dcr::base::Matrix::Zero(x_size, m);
                for (int j = 0; j < m; ++j) {
                    B.col(j) = res_pool[static_cast<size_t>(j)] - r_ref;
                    dC.col(j) = cand_pool[static_cast<size_t>(j)] - cand_pool.back();
                }
                const dcr::base::Vector rhs = -r_ref;
                dcr::base::Matrix BtB = B.transpose() * B;
                BtB.diagonal().array() += anderson_reg;
                const dcr::base::Vector beta = BtB.ldlt().solve(B.transpose() * rhs);
                if (beta.allFinite()) {
                    const dcr::base::Vector x_mix = cand_pool.back() + dC * beta;
                    if (x_mix.allFinite()) {
                        const double scale_xk = std::max(1.0, xk.cwiseAbs().maxCoeff());
                        const double rel_cand = (x_candidate - xk).cwiseAbs().maxCoeff() / scale_xk;
                        const double rel_mix = (x_mix - xk).cwiseAbs().maxCoeff() / scale_xk;
                        const double max_cand = x_candidate.cwiseAbs().maxCoeff();
                        const double max_mix = x_mix.cwiseAbs().maxCoeff();
                        const bool stable_mix =
                            rel_mix <= anderson_step_limit * std::max(rel_cand, 1e-14) &&
                            max_mix <= anderson_amp_limit * std::max(1.0, max_cand);
                        if (stable_mix) {
                            x_next = x_mix;
                            used_anderson = true;
                        } else {
                            // Bad extrapolation direction: fall back and reset memory.
                            anderson_cand_hist.clear();
                            anderson_res_hist.clear();
                        }
                    }
                }
            }
        }

        unpack_state(x_next, nP_new, flowA_new, flowM_new);
        // Physical safeguards after mixing.
        nP_new = nP_new.cwiseMax(0.0);
        flowA_new = flowA_new.cwiseMax(0.0);
        flowM_new = flowM_new.cwiseMax(0.0);
        {
            const double flow_nuclei_next =
                nuclei_sum(flowA_new, boundary.A_indices, levels) +
                nuclei_sum(flowM_new, boundary.M_indices, levels);
            const double target_bg_nuclei_next =
                std::max(0.0, config.plasma.total_density - flow_nuclei_next);
            const double bg_nuclei_now = nuclei_sum_background(nP_new, boundary, levels);
            if (target_bg_nuclei_next <= 0.0) {
                nP_new.setZero();
            } else if (bg_nuclei_now > 0.0) {
                nP_new *= (target_bg_nuclei_next / bg_nuclei_now);
            }
        }
        // Keep the map evaluations for future Anderson steps.
        if (anderson_depth > 0 && x_size > 0) {
            anderson_cand_hist.push_back(x_candidate);
            anderson_res_hist.push_back(r_candidate);
            if (static_cast<int>(anderson_cand_hist.size()) > anderson_depth) {
                anderson_cand_hist.erase(anderson_cand_hist.begin());
                anderson_res_hist.erase(anderson_res_hist.begin());
            }
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
        if (std::isfinite(prev_rel_change) && rel > prev_rel_change * 1.001) {
            ++rel_growth_count;
        } else {
            rel_growth_count = 0;
        }

        if (rel_growth_count >= 3 && omega_iter > omega_min) {
            omega_iter = std::max(omega_min, 0.5 * omega_iter);
            rel_growth_count = 0;
            no_improve_count = 0;
            if (config.io.verbose_logging && detailed_log) {
                std::cout << "[DCR_Solver] Marching cell " << cell_index
                          << ": rel_change growth detected, reducing relaxation to "
                          << omega_iter << "\n";
            }
        } else if (no_improve_count >= 5 && omega_iter > omega_min) {
            omega_iter = std::max(omega_min, 0.5 * omega_iter);
            no_improve_count = 0;
            if (config.io.verbose_logging && detailed_log) {
                std::cout << "[DCR_Solver] Marching cell " << cell_index
                          << ": slow convergence, reducing relaxation to "
                          << omega_iter << "\n";
            }
        }
        if (((iter + 1) % 50 == 0) && rel > (10.0 * tol) && omega_iter > omega_min) {
            omega_iter = std::max(omega_min, 0.5 * omega_iter);
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
                      << " T{e=" << cell_temperatures.electron_eV
                      << ", i=" << cell_temperatures.ion_eV
                      << "}"
                      << " iter " << (iter + 1)
                      << " rel_change=" << rel
                      << " rel_np=" << rel_np
                      << " rel_a=" << rel_a
                      << " rel_m=" << rel_m
                      << " ||Rpp*n-(A*n-S)||=" << bg_solve.linear_residual_norm
                      << " resid_rel(diag)=" << resid_rel
                      << " omega=" << omega_iter
                      << " anderson=" << (used_anderson ? "on" : "off")
                      << " flow_sum{A=" << flowA_sum_iter
                      << ", M=" << flowM_sum_iter
                      << "} flow_nuclei{A=" << flowA_nuc_iter
                      << ", M=" << flowM_nuc_iter
                      << "}"
                      << "\n";

            if (cell_index == 1) {
                const auto bg_iter = summarize_bg_subgroups(nP_new, boundary, levels);
                const double bg_nuc_iter = nuclei_sum_background(nP_new, boundary, levels);
                const double total_nuc_iter = bg_nuc_iter + flowA_nuc_iter + flowM_nuc_iter;
                std::cout << "[DCR_Solver] Marching cell 1 densities"
                          << " iter " << (iter + 1)
                          << " bg{H+=" << bg_iter.H_plus
                          << ", H=" << bg_iter.H
                          << ", H2=" << bg_iter.H2
                          << ", H2+=" << bg_iter.H2_plus
                          << ", H-=" << bg_iter.H_minus
                          << "} flow{A=" << flowA_sum_iter
                          << ", M=" << flowM_sum_iter
                          << "} nuclei{bg=" << bg_nuc_iter
                          << ", total=" << total_nuc_iter
                          << "}\n";
            }

        }

        const bool strict_converged = (rel < tol);
        const bool practical_converged =
            (iter + 1 >= practical_stop_min_iter) &&
            std::isfinite(resid_rel) &&
            (rel < practical_rel_tol) &&
            (resid_rel < practical_resid_tol);

        if (strict_converged || practical_converged) {
            out.iterations = iter + 1;
            out.converged = true;
            converged_by_practical = practical_converged && !strict_converged;
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
        config, atomic_data, plasma, grid, boundary, bg_full, out.flowA_new, out.flowM_new, x_right_cm
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
                  << " T{e=" << cell_temperatures.electron_eV
                  << ", i=" << cell_temperatures.ion_eV
                  << "}"
                  << (out.converged
                          ? (converged_by_practical ? " converged(practical)" : " converged")
                          : " max-iter")
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
