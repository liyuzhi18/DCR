#include "CellSolve.hpp"
#include "LogNewtonCellSolve.hpp"

#include "../../physics/Sheath.hpp"
#include "../core/TemperatureProfile.hpp"

#include <chrono>
#include <cmath>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
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
    double w_local = std::numeric_limits<double>::quiet_NaN();
    double source_nuclei_from_S = std::numeric_limits<double>::quiet_NaN();
    double gamma_ex_a_over_w = std::numeric_limits<double>::quiet_NaN();
    double gamma_ex_m_over_w = std::numeric_limits<double>::quiet_NaN();
    double L_I = std::numeric_limits<double>::quiet_NaN();
    double ion_balance_coeff = std::numeric_limits<double>::quiet_NaN();
};

struct MarchingMapEvaluation {
    dcr::base::Vector x_projected;
    dcr::base::Vector nP_iter;
    dcr::base::Vector flowA_iter;
    dcr::base::Vector flowM_iter;
    dcr::base::Vector x_image;
    dcr::base::Vector nP_image;
    dcr::base::Vector flowA_image;
    dcr::base::Vector flowM_image;
    dcr::base::Vector residual;
    double rel_np = std::numeric_limits<double>::infinity();
    double rel_a = std::numeric_limits<double>::infinity();
    double rel_m = std::numeric_limits<double>::infinity();
    double rel = std::numeric_limits<double>::infinity();
    double resid_rel = std::numeric_limits<double>::infinity();
    LocalSystem local_for_bg;
    FlowAdvanceResult flow_advanced;
    BackgroundSolveResult bg_solve;
};

struct KrylovSolveResult {
    dcr::base::Vector step;
    bool converged = false;
    int iterations = 0;
    int restarts = 0;
    double residual_norm = std::numeric_limits<double>::infinity();
};

double elapsed_seconds_since(const std::chrono::steady_clock::time_point& start) {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
}

std::string format_elapsed_seconds(double seconds) {
    std::ostringstream out;
    out << std::fixed << std::setprecision(3) << seconds;
    return out.str();
}

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
    const dcr::base::Vector& nP_floor_ref,
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
    // Ion-flux divergence balances recycling source against local background
    // neutral exhaust. Recycled-flow exhaust (A/M) is not included here.
    const double L_I = source_nuclei_from_S - (Gamma_ex_a_over_w + mu_M * Gamma_ex_m_over_w);
    const double L_a = Gamma_ex_a_over_w;
    const double L_m = Gamma_ex_m_over_w;
    result.w_local = w_local;
    result.source_nuclei_from_S = source_nuclei_from_S;
    result.gamma_ex_a_over_w = Gamma_ex_a_over_w;
    result.gamma_ex_m_over_w = Gamma_ex_m_over_w;
    result.L_I = L_I;
    result.ion_balance_coeff = (n_I > 0.0)
        ? (L_I / n_I)
        : std::numeric_limits<double>::quiet_NaN();

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

double fraction_to_boundary_alpha(const dcr::base::Vector& x,
                                  const dcr::base::Vector& delta,
                                  double safety) {
    double alpha_max = 1.0;
    const double x_floor = 1.0e-30;
    for (int i = 0; i < x.size() && i < delta.size(); ++i) {
        if (delta(i) >= 0.0) continue;
        if (x(i) <= x_floor) continue;
        alpha_max = std::min(alpha_max, safety * x(i) / (-delta(i)));
    }
    return std::clamp(alpha_max, 0.0, 1.0);
}

KrylovSolveResult gmres_solve_matrix_free(
    const std::function<dcr::base::Vector(const dcr::base::Vector&)>& apply_A,
    const dcr::base::Vector& rhs,
    int krylov_dim,
    int max_restarts,
    double rel_tol) {

    KrylovSolveResult out;
    const int n = rhs.size();
    out.step = dcr::base::Vector::Zero(n);
    if (n == 0) {
        out.converged = true;
        out.residual_norm = 0.0;
        return out;
    }

    const int m = std::max(1, krylov_dim);
    const int restarts = std::max(0, max_restarts);
    const double tol = std::max(1.0e-12, rel_tol);
    const double rhs_norm = std::max(1.0, rhs.norm());

    dcr::base::Vector x = dcr::base::Vector::Zero(n);
    for (int restart = 0; restart <= restarts; ++restart) {
        const dcr::base::Vector r = rhs - apply_A(x);
        const double beta = r.norm();
        out.residual_norm = beta;
        if (beta <= tol * rhs_norm) {
            out.step = x;
            out.converged = true;
            out.restarts = restart;
            return out;
        }

        dcr::base::Matrix V = dcr::base::Matrix::Zero(n, m + 1);
        dcr::base::Matrix H = dcr::base::Matrix::Zero(m + 1, m);
        dcr::base::Vector g = dcr::base::Vector::Zero(m + 1);
        V.col(0) = r / beta;
        g(0) = beta;

        dcr::base::Vector best_y;
        int best_cols = 0;
        double best_residual = beta;

        for (int j = 0; j < m; ++j) {
            dcr::base::Vector w = apply_A(V.col(j));
            for (int i = 0; i <= j; ++i) {
                H(i, j) = V.col(i).dot(w);
                w.noalias() -= H(i, j) * V.col(i);
            }
            H(j + 1, j) = w.norm();
            if (H(j + 1, j) > 0.0 && j + 1 < m + 1) {
                V.col(j + 1) = w / H(j + 1, j);
            }

            const dcr::base::Matrix Hsmall = H.block(0, 0, j + 2, j + 1);
            const dcr::base::Vector gsmall = g.head(j + 2);
            const dcr::base::Vector y = Hsmall.colPivHouseholderQr().solve(gsmall);
            const double residual = (gsmall - Hsmall * y).norm();
            out.iterations = restart * m + (j + 1);
            if (residual < best_residual) {
                best_residual = residual;
                best_y = y;
                best_cols = j + 1;
            }
            if (residual <= tol * rhs_norm || H(j + 1, j) <= 1.0e-14) {
                x.noalias() += V.leftCols(j + 1) * y;
                out.step = x;
                out.converged = (residual <= tol * rhs_norm);
                out.residual_norm = residual;
                out.restarts = restart;
                return out;
            }
        }

        if (best_cols > 0) {
            x.noalias() += V.leftCols(best_cols) * best_y;
            out.residual_norm = best_residual;
        } else {
            break;
        }
    }

    out.step = x;
    return out;
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
    bool emit_summary_log) {

    CellImplicitResult out;
    const auto solve_timer_start = std::chrono::steady_clock::now();
    const int total_states = atomic_data.get_total_states();

    dcr::base::Vector nP_iter = nP_old;
    dcr::base::Vector flowA_iter = flowA_old;
    dcr::base::Vector flowM_iter = flowM_old;
    const dcr::base::Vector& flowA_inflow = flowA_old;
    const dcr::base::Vector& flowM_inflow = flowM_old;

    const double marching_tol =
        (config.numerics.marching_tolerance > 0.0)
            ? config.numerics.marching_tolerance
            : config.numerics.tolerance;
    const double tol = std::max(marching_tol, 1e-12);
    const int marching_cap =
        (config.numerics.marching_max_iterations > 0)
            ? config.numerics.marching_max_iterations
            : config.numerics.max_iterations;
    const int max_iter = std::max(marching_cap, 1);
    const int slow_iter_threshold = std::max(1, config.numerics.marching_slow_iter_threshold);
    // Use a strict outer fixed-point stop for the full marching solve.
    // The QSS path now runs to the requested tolerance, and the full path
    // should obey the same rule rather than exiting on a looser practical
    // criterion.
    const double omega_raw = config.numerics.relaxation;
    // The marching fixed-point map can also develop a stable 2-cycle in stiff
    // low-Te cases. Allow substantially smaller damping so the local iteration
    // can contract instead of bouncing between two quasi-neutral states.
    const double omega_min = 1e-3;
    const double omega_init = (omega_raw > 0.0 && omega_raw <= 1.0) ? omega_raw : 1.0;
    double omega_iter = std::clamp(omega_init, omega_min, 1.0);
    double prev_rel_change = std::numeric_limits<double>::infinity();
    double prev_resid_rel = std::numeric_limits<double>::infinity();
    int no_improve_count = 0;
    int rel_growth_count = 0;
    int steady_improve_count = 0;
    int floor_contraction_count = 0;
    const double omega_recover_cap = std::min(omega_init, 1.0e-1);
    const auto cell_temperatures = evaluate_plasma_temperatures(config, x_right_cm);
    double last_rel = std::numeric_limits<double>::infinity();
    double last_resid_rel = std::numeric_limits<double>::infinity();

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

    auto signed_bg_nuclei = [&](const dcr::base::Vector& nP_state) {
        double s = 0.0;
        for (int i = 0; i < nP_state.size() &&
                        i < static_cast<int>(boundary.P_indices.size()); ++i) {
            const int gi = boundary.P_indices[static_cast<size_t>(i)];
            if (gi < 0 || gi >= static_cast<int>(levels.size())) continue;
            s += levels[gi].atomicity * nP_state(i);
        }
        return s;
    };

    auto project_state = [&](const dcr::base::Vector& x_in) {
        dcr::base::Vector nP_proj = nP_iter;
        dcr::base::Vector flowA_proj = flowA_iter;
        dcr::base::Vector flowM_proj = flowM_iter;
        unpack_state(x_in, nP_proj, flowA_proj, flowM_proj);
        flowA_proj = flowA_proj.cwiseMax(0.0);
        flowM_proj = flowM_proj.cwiseMax(0.0);

        const double flow_nuclei =
            nuclei_sum(flowA_proj, boundary.A_indices, levels) +
            nuclei_sum(flowM_proj, boundary.M_indices, levels);
        const double target_bg_nuclei =
            std::max(0.0, config.plasma.total_density - flow_nuclei);
        const double bg_nuclei_now =
            signed_bg_nuclei(nP_proj);
        if (target_bg_nuclei <= 0.0) {
            nP_proj.setZero();
        } else if (std::abs(bg_nuclei_now) > 0.0) {
            nP_proj *= (target_bg_nuclei / bg_nuclei_now);
        }
        return pack_state(nP_proj, flowA_proj, flowM_proj);
    };

    auto evaluate_map = [&](const dcr::base::Vector& x_in) {
        MarchingMapEvaluation eval;
        eval.x_projected = project_state(x_in);
        unpack_state(eval.x_projected, eval.nP_iter, eval.flowA_iter, eval.flowM_iter);

        const dcr::base::Vector bg_iter_full =
            make_background_full(eval.nP_iter, boundary, total_states);
        const auto background_rates = assemble_background_rate_matrix(
            config,
            atomic_data,
            plasma,
            grid,
            bg_iter_full,
            x_right_cm
        );
        LocalSystem local_for_flow;
        local_for_flow.population_for_rates = background_rates.population_for_rates;
        local_for_flow.R_full = background_rates.R_full;
        local_for_flow.S_background = dcr::base::Vector::Zero(static_cast<int>(boundary.P_indices.size()));
        eval.flow_advanced = advance_recycling_flow_one_step(
            config,
            atomic_data,
            boundary,
            local_for_flow,
            flowA_inflow,
            flowM_inflow,
            dx_cm
        );
        eval.local_for_bg = assemble_local_system_from_background_rate_matrix(
            config,
            atomic_data,
            plasma,
            grid,
            boundary,
            background_rates,
            eval.flow_advanced.flowA_next,
            eval.flow_advanced.flowM_next,
            x_right_cm
        );
        eval.bg_solve = solve_background_at_cell(
            config,
            boundary,
            levels,
            eval.local_for_bg,
            eval.nP_iter,
            nP_old,
            eval.flow_advanced.flowA_next,
            eval.flow_advanced.flowM_next
        );
        eval.x_image = project_state(pack_state(
            eval.bg_solve.nP_new,
            eval.flow_advanced.flowA_next,
            eval.flow_advanced.flowM_next
        ));
        unpack_state(eval.x_image, eval.nP_image, eval.flowA_image, eval.flowM_image);
        eval.residual = eval.x_projected - eval.x_image;
        eval.rel_np = relative_change(eval.nP_image, eval.nP_iter);
        eval.rel_a = relative_change(eval.flowA_image, eval.flowA_iter);
        eval.rel_m = relative_change(eval.flowM_image, eval.flowM_iter);
        eval.rel = std::max(eval.rel_np, std::max(eval.rel_a, eval.rel_m));
        eval.resid_rel = std::isfinite(eval.bg_solve.linear_mse_rel)
            ? eval.bg_solve.linear_mse_rel
            : std::numeric_limits<double>::infinity();
        return eval;
    };

    auto finalize_output = [&](CellImplicitResult& result,
                               const dcr::base::Vector& nP_final,
                               const dcr::base::Vector& flowA_final,
                               const dcr::base::Vector& flowM_final,
                               int iterations,
                               bool converged,
                               double final_rel,
                               double final_resid_rel) {
        result.nP_new = nP_final.cwiseMax(0.0);
        result.flowA_new = flowA_final.cwiseMax(0.0);
        result.flowM_new = flowM_final.cwiseMax(0.0);
        const double flow_nuclei =
            nuclei_sum(result.flowA_new, boundary.A_indices, levels) +
            nuclei_sum(result.flowM_new, boundary.M_indices, levels);
        const double target_bg_nuclei =
            std::max(0.0, config.plasma.total_density - flow_nuclei);
        const double bg_nuclei = nuclei_sum_background(result.nP_new, boundary, levels);
        if (target_bg_nuclei <= 0.0) {
            result.nP_new.setZero();
        } else if (bg_nuclei > 0.0) {
            result.nP_new *= (target_bg_nuclei / bg_nuclei);
        }
        result.iterations = iterations;
        result.converged = converged;
        const dcr::base::Vector bg_full =
            make_background_full(result.nP_new, boundary, total_states);
        result.local_final = assemble_local_system(
            config,
            atomic_data,
            plasma,
            grid,
            boundary,
            bg_full,
            result.flowA_new,
            result.flowM_new,
            x_right_cm
        );
        result.flow_final = advance_recycling_flow_one_step(
            config,
            atomic_data,
            boundary,
            result.local_final,
            result.flowA_new,
            result.flowM_new,
            0.0
        );
        result.final_rel = final_rel;
        result.final_resid_rel = final_resid_rel;
        result.elapsed_seconds = elapsed_seconds_since(solve_timer_start);

        if (config.io.verbose_logging && emit_summary_log) {
            const auto bg = summarize_bg_subgroups(result.nP_new, boundary, levels);
            double bg_nuclei = 0.0;
            for (int i = 0; i < result.nP_new.size() &&
                            i < static_cast<int>(boundary.P_indices.size()); ++i) {
                const int gi = boundary.P_indices[static_cast<size_t>(i)];
                if (gi < 0 || gi >= static_cast<int>(levels.size())) continue;
                bg_nuclei += levels[gi].atomicity * std::max(result.nP_new(i), 0.0);
            }
            const double flowA_nuclei = nuclei_sum(result.flowA_new, boundary.A_indices, levels);
            const double flowM_nuclei = nuclei_sum(result.flowM_new, boundary.M_indices, levels);
            const double total_nuclei = bg_nuclei + flowA_nuclei + flowM_nuclei;
            const double flowA_total = positive_sum(result.flowA_new);
            const double flowM_total = positive_sum(result.flowM_new);
            std::cout << "[DCR_Solver] Marching cell " << cell_index
                      << " x=" << x_right_cm << " cm"
                      << " T{e=" << cell_temperatures.electron_eV
                      << ", i=" << cell_temperatures.ion_eV
                      << "}"
                      << (result.converged ? " converged" : " max-iter")
                      << " in " << result.iterations
                      << " iterations (rel=" << final_rel
                      << ", resid_rel(diag)=" << result.final_resid_rel
                      << ", wall=" << format_elapsed_seconds(result.elapsed_seconds) << " s)"
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
    };

    const std::string marching_solver = config.numerics.marching_solver.empty()
        ? "picard" : config.numerics.marching_solver;
    if (marching_solver == "log_newton_krylov_ptc") {
        return solve_cell_implicit_log_newton(
            config,
            atomic_data,
            plasma,
            grid,
            boundary,
            levels,
            nP_old,
            flowA_old,
            flowM_old,
            dx_cm,
            x_left_cm,
            x_right_cm,
            cell_index,
            detailed_log,
            emit_summary_log,
            nullptr,
            nullptr,
            nullptr
        );
    }
    const bool use_nk_solver =
        (marching_solver == "newton_krylov_ptc") ||
        (marching_solver == "picard_then_newton_krylov_ptc");
    if (use_nk_solver) {
        double tau = std::clamp(
            config.numerics.marching_ptc_tau_init,
            1.0e-6,
            std::max(config.numerics.marching_ptc_tau_init,
                     config.numerics.marching_ptc_tau_max)
        );
        const double tau_max = std::max(tau, config.numerics.marching_ptc_tau_max);
        const double alpha_min = std::clamp(config.numerics.marching_nk_alpha_min, 1.0e-6, 0.5);
        dcr::base::Vector x_work = project_state(pack_state(nP_iter, flowA_iter, flowM_iter));
        bool converged = false;
        int iterations = max_iter;

        for (int iter = 0; iter < max_iter; ++iter) {
            const auto eval = evaluate_map(x_work);
            last_rel = eval.rel;
            last_resid_rel = eval.resid_rel;

            if (config.io.verbose_logging && detailed_log) {
                const double flowA_sum_iter = positive_sum(eval.flowA_image);
                const double flowM_sum_iter = positive_sum(eval.flowM_image);
                const double flowA_nuc_iter = nuclei_sum(eval.flowA_image, boundary.A_indices, levels);
                const double flowM_nuc_iter = nuclei_sum(eval.flowM_image, boundary.M_indices, levels);
                const auto bg_iter = summarize_bg_subgroups(eval.nP_image, boundary, levels);
                std::cout << "[DCR_Solver] Marching cell " << cell_index
                          << " x=[" << x_left_cm << "," << x_right_cm << "] cm"
                          << " T{e=" << cell_temperatures.electron_eV
                          << ", i=" << cell_temperatures.ion_eV
                          << "}"
                          << " iter " << (iter + 1)
                          << " solver=NK-PTC"
                          << " rel_change=" << eval.rel
                          << " rel_np=" << eval.rel_np
                          << " rel_a=" << eval.rel_a
                          << " rel_m=" << eval.rel_m
	                          << " ||F||=" << eval.residual.norm()
	                          << " resid_rel(diag)=" << eval.resid_rel
	                          << " tau=" << tau
	                          << " balance{w=" << eval.bg_solve.w_local
	                          << ", S_nuc=" << eval.bg_solve.source_nuclei_from_S
	                          << ", exA=" << eval.bg_solve.gamma_ex_a_over_w
	                          << ", exM=" << eval.bg_solve.gamma_ex_m_over_w
	                          << ", L_I=" << eval.bg_solve.L_I
	                          << ", ion_coeff=" << eval.bg_solve.ion_balance_coeff
	                          << "}"
	                          << " bg{H+=" << bg_iter.H_plus
                          << ", H=" << bg_iter.H
                          << ", H2=" << bg_iter.H2
                          << ", H2+=" << bg_iter.H2_plus
                          << ", H-=" << bg_iter.H_minus
                          << "}"
                          << " flow{A=" << flowA_sum_iter
                          << ", M=" << flowM_sum_iter
                          << "}"
                          << "\n";
            }

            if (eval.rel < tol) {
                x_work = eval.x_image;
                converged = true;
                iterations = iter + 1;
                break;
            }

            const double linear_tol = std::clamp(
                0.1 * std::sqrt(std::max(eval.rel, tol)),
                1.0e-4,
                5.0e-2
            );
            const auto apply_A = [&](const dcr::base::Vector& v) -> dcr::base::Vector {
                if (v.size() == 0 || v.norm() == 0.0) {
                    return dcr::base::Vector::Zero(v.size());
                }
                const double eps = std::max(
                    config.numerics.marching_nk_fd_eps,
                    config.numerics.marching_nk_fd_eps *
                        (1.0 + eval.x_projected.norm()) / std::max(1.0, v.norm())
                );
                const auto pert = evaluate_map(eval.x_projected + eps * v);
                dcr::base::Vector out_vec =
                    (1.0 / tau) * v + (pert.residual - eval.residual) / eps;
                return out_vec;
            };

            auto gmres = gmres_solve_matrix_free(
                apply_A,
                -eval.residual,
                config.numerics.marching_nk_krylov_dim,
                config.numerics.marching_nk_max_restarts,
                linear_tol
            );
            dcr::base::Vector delta = gmres.step;
            if (!delta.allFinite() || delta.norm() == 0.0) {
                delta = 0.1 * (eval.x_image - eval.x_projected);
                tau = std::max(1.0e-6, 0.5 * tau);
            }

            const double norm0 = std::max(1.0e-30, eval.residual.norm());
            const double alpha_pos = 1.0;
            double alpha = 1.0;
            bool accepted = false;
            MarchingMapEvaluation accepted_eval;
            while (alpha >= alpha_min) {
                const auto trial = evaluate_map(eval.x_projected + alpha * delta);
                const double trial_norm = trial.residual.norm();
                if (trial_norm < norm0 * (1.0 - 1.0e-4 * alpha) ||
                    trial_norm < 0.95 * norm0) {
                    accepted_eval = trial;
                    accepted = true;
                    break;
                }
                alpha *= 0.5;
            }

            if (!accepted) {
                // Fall back to a conservative Picard-like correction if the
                // Newton step does not provide sufficient decrease.
                alpha = std::min(0.1, std::max(alpha_min, std::max(1.0e-2, 0.05 * tau)));
                const auto trial = evaluate_map(
                    eval.x_projected + alpha * (eval.x_image - eval.x_projected)
                );
                accepted_eval = trial;
                accepted = true;
                tau = std::max(1.0e-6, 0.5 * tau);
                if (config.io.verbose_logging && detailed_log) {
                    std::cout << "[DCR_Solver] Marching cell " << cell_index
                              << ": NK line search failed, falling back to conservative Picard step"
                              << " alpha=" << alpha
                              << " alpha_pos=" << alpha_pos
                              << " tau=" << tau
                              << "\n";
                }
                if (!accepted) {
                    accepted_eval = eval;
                    alpha = 0.0;
                }
            } else if (alpha >= 0.75 && gmres.converged) {
                tau = std::min(tau_max, 1.5 * tau);
            } else if (alpha < 0.25 || !gmres.converged) {
                tau = std::max(1.0e-6, 0.5 * tau);
            }

            x_work = accepted_eval.x_projected;
            if (config.io.verbose_logging && detailed_log) {
                std::cout << "[DCR_Solver] Marching cell " << cell_index
                          << ": NK step"
                          << " alpha=" << alpha
                          << " gmres_iters=" << gmres.iterations
                          << " gmres_resid=" << gmres.residual_norm
                          << " alpha_pos=" << alpha_pos
                          << " tau_next=" << tau
                          << "\n";
            }

            if (iter == max_iter - 1) {
                iterations = max_iter;
            }
        }

        unpack_state(project_state(x_work), nP_iter, flowA_iter, flowM_iter);
        finalize_output(out, nP_iter, flowA_iter, flowM_iter, iterations, converged,
                        last_rel, last_resid_rel);
        return out;
    }

    dcr::base::Vector x_prev = pack_state(nP_iter, flowA_iter, flowM_iter);
    dcr::base::Vector x_prevprev = x_prev;
    bool have_prevprev = false;

    for (int iter = 0; iter < max_iter; ++iter) {
        const dcr::base::Vector bg_iter_full = make_background_full(nP_iter, boundary, total_states);
        const auto background_rates = assemble_background_rate_matrix(
            config, atomic_data, plasma, grid, bg_iter_full, x_right_cm
        );

        // Step A: flow update at x_{k+1} from Eq. (1.346) with implicit upwind march.
        // The inner Picard loop updates the local cell state, but the left-cell inflow
        // stays fixed during the solve for this spatial step.
        LocalSystem local_for_flow;
        local_for_flow.population_for_rates = background_rates.population_for_rates;
        local_for_flow.R_full = background_rates.R_full;
        local_for_flow.S_background = dcr::base::Vector::Zero(static_cast<int>(boundary.P_indices.size()));
        const auto flow_advanced = advance_recycling_flow_one_step(
            config, atomic_data, boundary, local_for_flow, flowA_inflow, flowM_inflow, dx_cm
        );

        // Step B: background update at x_{k+1} from grouped Eq. (1.345).
        const auto local_for_bg = assemble_local_system_from_background_rate_matrix(
            config,
            atomic_data,
            plasma,
            grid,
            boundary,
            background_rates,
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
            nP_old,
            flow_advanced.flowA_next,
            flow_advanced.flowM_next
        );
        dcr::base::Vector nP_new = bg_solve.nP_new;

        dcr::base::Vector flowA_new = flow_advanced.flowA_next;
        dcr::base::Vector flowM_new = flow_advanced.flowM_next;

        // Build current/fixed-point state vectors.
        const dcr::base::Vector xk = pack_state(nP_iter, flowA_iter, flowM_iter);
        const dcr::base::Vector gk = pack_state(nP_new, flowA_new, flowM_new);

        // Damped candidate with simple 2-cycle rejection. In stiff cases the
        // accepted iterate can alternate between two quasi-neutral states; if
        // the new candidate lands much closer to the state from two iterations
        // ago than to the current state, treat it as a cycle and shrink the
        // damping before accepting.
        dcr::base::Vector x_next = xk;
        double omega_used = omega_iter;
        int cycle_backtracks = 0;
        while (true) {
            dcr::base::Vector x_candidate = gk;
            if (omega_used < 1.0) {
                x_candidate = xk + omega_used * (gk - xk);
            }

            dcr::base::Vector nP_trial = nP_new;
            dcr::base::Vector flowA_trial = flowA_new;
            dcr::base::Vector flowM_trial = flowM_new;
            unpack_state(x_candidate, nP_trial, flowA_trial, flowM_trial);
            nP_trial = nP_trial.cwiseMax(0.0);
            flowA_trial = flowA_trial.cwiseMax(0.0);
            flowM_trial = flowM_trial.cwiseMax(0.0);
            {
                const double flow_nuclei_next =
                    nuclei_sum(flowA_trial, boundary.A_indices, levels) +
                    nuclei_sum(flowM_trial, boundary.M_indices, levels);
                const double target_bg_nuclei_next =
                    std::max(0.0, config.plasma.total_density - flow_nuclei_next);
                const double bg_nuclei_now =
                    nuclei_sum_background(nP_trial, boundary, levels);
                if (target_bg_nuclei_next <= 0.0) {
                    nP_trial.setZero();
                } else if (bg_nuclei_now > 0.0) {
                    nP_trial *= (target_bg_nuclei_next / bg_nuclei_now);
                }
            }

            x_candidate = pack_state(nP_trial, flowA_trial, flowM_trial);
            const double step_rel = relative_change(x_candidate, xk);
            const double cycle_rel = have_prevprev
                ? relative_change(x_candidate, x_prevprev)
                : std::numeric_limits<double>::infinity();
            const bool two_cycle =
                have_prevprev &&
                step_rel > (10.0 * tol) &&
                cycle_rel < std::max(0.25 * step_rel, 10.0 * tol);
            if (!two_cycle || omega_used <= omega_min * 1.001 || cycle_backtracks >= 8) {
                nP_new = std::move(nP_trial);
                flowA_new = std::move(flowA_trial);
                flowM_new = std::move(flowM_trial);
                x_next = std::move(x_candidate);
                break;
            }
            omega_used = std::max(omega_min, 0.5 * omega_used);
            ++cycle_backtracks;
        }
        omega_iter = omega_used;
        const double rel_np = relative_change(nP_new, nP_iter);
        const double rel_a = relative_change(flowA_new, flowA_iter);
        const double rel_m = relative_change(flowM_new, flowM_iter);
        const double rel = std::max(rel_np, std::max(rel_a, rel_m));
        const double resid_rel = std::isfinite(bg_solve.linear_mse_rel)
            ? bg_solve.linear_mse_rel
            : std::numeric_limits<double>::infinity();
        last_rel = rel;
        last_resid_rel = resid_rel;

        const bool residual_not_improving =
            std::isfinite(prev_resid_rel) && resid_rel >= prev_resid_rel * 0.999;
        if (rel >= prev_rel_change * 0.999 && residual_not_improving) {
            ++no_improve_count;
        } else {
            no_improve_count = 0;
        }
        if (std::isfinite(prev_rel_change) &&
            rel > prev_rel_change * 1.02 &&
            residual_not_improving) {
            ++rel_growth_count;
        } else {
            rel_growth_count = 0;
        }
        if (std::isfinite(prev_rel_change) &&
            rel < prev_rel_change * 0.9995 &&
            (!std::isfinite(prev_resid_rel) || resid_rel <= prev_resid_rel * 1.001) &&
            cycle_backtracks == 0) {
            ++steady_improve_count;
        } else {
            steady_improve_count = 0;
        }
        if (omega_iter <= omega_min * 1.001 &&
            std::isfinite(prev_rel_change) &&
            rel < prev_rel_change * 0.9995 &&
            (!std::isfinite(prev_resid_rel) || resid_rel <= prev_resid_rel * 1.001) &&
            cycle_backtracks == 0) {
            ++floor_contraction_count;
        } else {
            floor_contraction_count = 0;
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
        if (steady_improve_count >= 8 && omega_iter < omega_recover_cap) {
            omega_iter = std::min(omega_recover_cap, 1.5 * omega_iter);
            steady_improve_count = 0;
            if (config.io.verbose_logging && detailed_log) {
                std::cout << "[DCR_Solver] Marching cell " << cell_index
                          << ": stable contraction detected, increasing relaxation to "
                          << omega_iter << "\n";
            }
        }
        if (floor_contraction_count >= 25 && omega_iter < omega_recover_cap) {
            omega_iter = std::min(omega_recover_cap, 2.0 * omega_iter);
            floor_contraction_count = 0;
            steady_improve_count = 0;
            if (config.io.verbose_logging && detailed_log) {
                std::cout << "[DCR_Solver] Marching cell " << cell_index
                          << ": escaped damping floor under monotone contraction, increasing relaxation to "
                          << omega_iter << "\n";
            }
        }
        prev_rel_change = rel;
        prev_resid_rel = resid_rel;

        nP_iter = nP_new;
        flowA_iter = flowA_new;
        flowM_iter = flowM_new;
        x_prevprev = x_prev;
        x_prev = x_next;
        have_prevprev = true;

        if (config.io.verbose_logging && detailed_log) {
            const double flowA_sum_iter = positive_sum(flowA_new);
            const double flowM_sum_iter = positive_sum(flowM_new);
            const double flowA_nuc_iter = nuclei_sum(flowA_new, boundary.A_indices, levels);
            const double flowM_nuc_iter = nuclei_sum(flowM_new, boundary.M_indices, levels);
            const auto bg_iter = summarize_bg_subgroups(nP_new, boundary, levels);
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
	                      << " balance{w=" << bg_solve.w_local
	                      << ", S_nuc=" << bg_solve.source_nuclei_from_S
	                      << ", exA=" << bg_solve.gamma_ex_a_over_w
	                      << ", exM=" << bg_solve.gamma_ex_m_over_w
	                      << ", L_I=" << bg_solve.L_I
	                      << ", ion_coeff=" << bg_solve.ion_balance_coeff
	                      << "}"
	                      << " bg{H+=" << bg_iter.H_plus
                      << ", H=" << bg_iter.H
                      << ", H2=" << bg_iter.H2
                      << ", H2+=" << bg_iter.H2_plus
                      << ", H-=" << bg_iter.H_minus
                      << "}"
                      << " flow_sum{A=" << flowA_sum_iter
                      << ", M=" << flowM_sum_iter
                      << "} flow_nuclei{A=" << flowA_nuc_iter
                      << ", M=" << flowM_nuc_iter
                      << "}"
                      << "\n";

        }

        const bool strict_converged = (rel < tol);

        if (strict_converged) {
            out.iterations = iter + 1;
            out.converged = true;
            break;
        }
        if (iter == max_iter - 1) {
            out.iterations = max_iter;
            out.converged = false;
        }
    }

    finalize_output(out, nP_iter, flowA_iter, flowM_iter, out.iterations,
                    out.converged, last_rel, last_resid_rel);
    return out;
}

} // namespace dcr::solver
