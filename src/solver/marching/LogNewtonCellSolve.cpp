#include "LogNewtonCellSolve.hpp"

#include "../../physics/Sheath.hpp"
#include "../core/TemperatureProfile.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <vector>

namespace dcr::solver {

namespace {

constexpr double kLogStateFloor = 1.0e-60;
constexpr double kLogMin = -138.0; // exp(-138) ~ 1e-60
constexpr double kLogMax = 700.0;

struct BgSubgroupSums {
    double H_plus = 0.0;
    double H = 0.0;
    double H2 = 0.0;
    double H2_plus = 0.0;
    double H_minus = 0.0;
};

struct BackgroundSolveResult {
    dcr::base::Vector nP_new;
    double linear_residual_norm = std::numeric_limits<double>::quiet_NaN();
    double linear_mse = std::numeric_limits<double>::quiet_NaN();
    double linear_mse_rel = std::numeric_limits<double>::quiet_NaN();
};

struct LogMarchingMapEvaluation {
    dcr::base::Vector y_projected;
    dcr::base::Vector y_image;
    dcr::base::Vector x_projected;
    dcr::base::Vector x_image;
    dcr::base::Vector nP_iter;
    dcr::base::Vector flowA_iter;
    dcr::base::Vector flowM_iter;
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

dcr::base::Vector solve_replaced_system(const dcr::base::Matrix& A,
                                        const dcr::base::Vector& b) {
    if (A.rows() == A.cols()) {
        Eigen::FullPivLU<dcr::base::Matrix> lu(A);
        if (lu.rank() == A.rows()) {
            return lu.solve(b);
        }
    }
    return A.colPivHouseholderQr().solve(b);
}

double relative_change(const dcr::base::Vector& now, const dcr::base::Vector& old) {
    if (now.size() == 0 || old.size() == 0) return 0.0;
    const double diff = (now - old).cwiseAbs().maxCoeff();
    const double scale = std::max(1.0, old.cwiseAbs().maxCoeff());
    return diff / scale;
}

double positive_sum(const dcr::base::Vector& v) {
    double s = 0.0;
    for (int i = 0; i < v.size(); ++i) s += std::max(v(i), 0.0);
    return s;
}

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

dcr::base::Vector encode_positive_state(const dcr::base::Vector& x) {
    dcr::base::Vector y = x;
    for (int i = 0; i < y.size(); ++i) {
        y(i) = std::log(std::max(x(i), kLogStateFloor));
    }
    return y;
}

dcr::base::Vector decode_positive_state(const dcr::base::Vector& y) {
    dcr::base::Vector x = y;
    for (int i = 0; i < x.size(); ++i) {
        x(i) = std::exp(std::clamp(y(i), kLogMin, kLogMax));
    }
    return x;
}

BackgroundSolveResult solve_background_at_cell_log(
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
    dcr::base::Matrix C = Rpp;
    dcr::base::Vector rhs = A * nP_guess - local_system.S_background;
    dcr::base::Vector stoich = dcr::base::Vector::Zero(Pn);
    for (int i = 0; i < Pn; ++i) {
        const int gi = boundary.P_indices[static_cast<size_t>(i)];
        stoich(i) = (gi >= 0 && gi < static_cast<int>(levels.size()))
            ? static_cast<double>(levels[gi].atomicity) : 1.0;
    }

    const double flow_nuclei = nuclei_sum(flowA_new, boundary.A_indices, levels)
                             + nuclei_sum(flowM_new, boundary.M_indices, levels);
    const double target_bg_nuclei = std::max(0.0, config.plasma.total_density - flow_nuclei);

    auto set_original_diag = [&](const dcr::base::Vector& n_eval) {
        const dcr::base::Vector r = Rpp * n_eval - (A * n_eval - local_system.S_background);
        result.linear_residual_norm = r.norm();
        result.linear_mse = r.squaredNorm() / static_cast<double>(Pn);
        const double lhs_norm = (Rpp * n_eval).norm();
        result.linear_mse_rel = result.linear_residual_norm / std::max(1.0, lhs_norm);
    };

    dcr::base::Matrix C_replaced = C;
    dcr::base::Vector rhs_replaced = rhs;
    C_replaced.row(1) = stoich.transpose();
    rhs_replaced(1) = target_bg_nuclei;
    dcr::base::Vector nP_new = solve_replaced_system(C_replaced, rhs_replaced);

    if (!nP_new.allFinite()) {
        result.nP_new = nP_guess;
        set_original_diag(nP_guess);
        return result;
    }

    for (int i = 0; i < nP_new.size(); ++i) {
        nP_new(i) = std::max(nP_new(i), kLogStateFloor);
    }
    const double now = (stoich.array() * nP_new.array()).sum();
    if (now > 0.0 && target_bg_nuclei > 0.0) {
        nP_new *= (target_bg_nuclei / now);
    }
    for (int i = 0; i < nP_new.size(); ++i) {
        nP_new(i) = std::max(nP_new(i), kLogStateFloor);
    }
    result.nP_new = nP_new;
    set_original_diag(nP_new);
    return result;
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
    dcr::base::Vector r = rhs - apply_A(x);
    double beta = r.norm();
    out.residual_norm = beta;
    if (beta <= tol * rhs_norm) {
        out.converged = true;
        out.step = x;
        return out;
    }

    for (int restart = 0; restart <= restarts; ++restart) {
        dcr::base::Matrix V = dcr::base::Matrix::Zero(n, m + 1);
        dcr::base::Matrix H = dcr::base::Matrix::Zero(m + 1, m);
        dcr::base::Vector cs = dcr::base::Vector::Zero(m);
        dcr::base::Vector sn = dcr::base::Vector::Zero(m);
        dcr::base::Vector g = dcr::base::Vector::Zero(m + 1);

        beta = r.norm();
        if (beta <= tol * rhs_norm) {
            out.converged = true;
            out.step = x;
            out.residual_norm = beta;
            return out;
        }
        V.col(0) = r / beta;
        g(0) = beta;

        double best_residual = std::numeric_limits<double>::infinity();
        dcr::base::Vector best_y = dcr::base::Vector::Zero(m);
        int best_cols = 0;

        for (int j = 0; j < m; ++j) {
            dcr::base::Vector w = apply_A(V.col(j));
            for (int i = 0; i <= j; ++i) {
                H(i, j) = V.col(i).dot(w);
                w.noalias() -= H(i, j) * V.col(i);
            }
            H(j + 1, j) = w.norm();
            if (H(j + 1, j) > 0.0) {
                V.col(j + 1) = w / H(j + 1, j);
            }

            for (int i = 0; i < j; ++i) {
                const double tmp = cs(i) * H(i, j) + sn(i) * H(i + 1, j);
                H(i + 1, j) = -sn(i) * H(i, j) + cs(i) * H(i + 1, j);
                H(i, j) = tmp;
            }

            const double h0 = H(j, j);
            const double h1 = H(j + 1, j);
            const double denom = std::hypot(h0, h1);
            cs(j) = (denom > 0.0) ? (h0 / denom) : 1.0;
            sn(j) = (denom > 0.0) ? (h1 / denom) : 0.0;
            H(j, j) = cs(j) * h0 + sn(j) * h1;
            H(j + 1, j) = 0.0;

            const double gj = g(j);
            g(j) = cs(j) * gj;
            g(j + 1) = -sn(j) * gj;

            const double resid = std::abs(g(j + 1));
            if (resid < best_residual) {
                dcr::base::Matrix Hsquare = H.topLeftCorner(j + 1, j + 1);
                dcr::base::Vector gtop = g.head(j + 1);
                best_y = Hsquare.template triangularView<Eigen::Upper>().solve(gtop);
                best_residual = resid;
                best_cols = j + 1;
            }
            out.iterations += 1;

            if (resid <= tol * rhs_norm) {
                x.noalias() += V.leftCols(j + 1) * best_y.head(j + 1);
                out.converged = true;
                out.step = x;
                out.residual_norm = resid;
                out.restarts = restart;
                return out;
            }
        }

        out.restarts = restart + 1;
        if (best_cols > 0) {
            x.noalias() += V.leftCols(best_cols) * best_y.head(best_cols);
        }
        r = rhs - apply_A(x);
        out.residual_norm = r.norm();
        if (out.residual_norm <= tol * rhs_norm) {
            out.converged = true;
            out.step = x;
            return out;
        }
    }

    out.step = x;
    return out;
}

} // namespace

CellImplicitResult solve_cell_implicit_log_newton(
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
    const auto solve_timer_start = std::chrono::steady_clock::now();
    const int total_states = atomic_data.get_total_states();
    dcr::base::Vector nP_iter = nP_old;
    dcr::base::Vector flowA_iter = flowA_old;
    dcr::base::Vector flowM_iter = flowM_old;
    const dcr::base::Vector& flowA_inflow = flowA_old;
    const dcr::base::Vector& flowM_inflow = flowM_old;
    if (nP_init_override && nP_init_override->size() == nP_old.size()) {
        nP_iter = nP_init_override->cwiseMax(kLogStateFloor);
    }
    if (flowA_init_override && flowA_init_override->size() == flowA_old.size()) {
        flowA_iter = flowA_init_override->cwiseMax(kLogStateFloor);
    }
    if (flowM_init_override && flowM_init_override->size() == flowM_old.size()) {
        flowM_iter = flowM_init_override->cwiseMax(kLogStateFloor);
    }

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
    const auto cell_temperatures = evaluate_plasma_temperatures(config, x_right_cm);

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

    auto project_positive_state = [&](const dcr::base::Vector& x_in) {
        dcr::base::Vector nP_proj = dcr::base::Vector::Constant(nP_size, kLogStateFloor);
        dcr::base::Vector flowA_proj = dcr::base::Vector::Constant(nA_size, kLogStateFloor);
        dcr::base::Vector flowM_proj = dcr::base::Vector::Constant(nM_size, kLogStateFloor);
        unpack_state(x_in, nP_proj, flowA_proj, flowM_proj);
        nP_proj = nP_proj.cwiseMax(kLogStateFloor);
        flowA_proj = flowA_proj.cwiseMax(kLogStateFloor);
        flowM_proj = flowM_proj.cwiseMax(kLogStateFloor);

        const double flow_nuclei =
            nuclei_sum(flowA_proj, boundary.A_indices, levels) +
            nuclei_sum(flowM_proj, boundary.M_indices, levels);
        const double target_bg_nuclei =
            std::max(0.0, config.plasma.total_density - flow_nuclei);
        const double bg_nuclei_now =
            nuclei_sum_background(nP_proj, boundary, levels);
        if (target_bg_nuclei > 0.0 && bg_nuclei_now > 0.0) {
            nP_proj *= (target_bg_nuclei / bg_nuclei_now);
        }
        nP_proj = nP_proj.cwiseMax(kLogStateFloor);
        return pack_state(nP_proj, flowA_proj, flowM_proj);
    };

    auto evaluate_map = [&](const dcr::base::Vector& y_in) {
        LogMarchingMapEvaluation eval;
        const dcr::base::Vector x_trial = decode_positive_state(y_in);
        eval.x_projected = project_positive_state(x_trial);
        eval.y_projected = encode_positive_state(eval.x_projected);
        unpack_state(eval.x_projected, eval.nP_iter, eval.flowA_iter, eval.flowM_iter);

        const dcr::base::Vector bg_iter_full =
            make_background_full(eval.nP_iter, boundary, total_states);
        const auto local_for_flow = assemble_local_system(
            config, atomic_data, plasma, grid, boundary, bg_iter_full,
            eval.flowA_iter, eval.flowM_iter, x_right_cm
        );
        eval.flow_advanced = advance_recycling_flow_one_step(
            config, atomic_data, boundary, local_for_flow,
            flowA_inflow, flowM_inflow, dx_cm
        );
        eval.local_for_bg = assemble_local_system(
            config, atomic_data, plasma, grid, boundary, bg_iter_full,
            eval.flow_advanced.flowA_next, eval.flow_advanced.flowM_next, x_right_cm
        );
        eval.bg_solve = solve_background_at_cell_log(
            config, boundary, levels, eval.local_for_bg, eval.nP_iter,
            eval.flow_advanced.flowA_next, eval.flow_advanced.flowM_next
        );
        eval.x_image = project_positive_state(pack_state(
            eval.bg_solve.nP_new,
            eval.flow_advanced.flowA_next,
            eval.flow_advanced.flowM_next
        ));
        eval.y_image = encode_positive_state(eval.x_image);
        unpack_state(eval.x_image, eval.nP_image, eval.flowA_image, eval.flowM_image);
        eval.residual = eval.y_projected - eval.y_image;
        eval.rel_np = relative_change(eval.nP_image, eval.nP_iter);
        eval.rel_a = relative_change(eval.flowA_image, eval.flowA_iter);
        eval.rel_m = relative_change(eval.flowM_image, eval.flowM_iter);
        eval.rel = std::max(eval.rel_np, std::max(eval.rel_a, eval.rel_m));
        eval.resid_rel = std::isfinite(eval.bg_solve.linear_mse_rel)
            ? eval.bg_solve.linear_mse_rel
            : std::numeric_limits<double>::infinity();
        return eval;
    };

    auto finalize_output = [&](const dcr::base::Vector& y_final,
                               int iterations,
                               bool converged,
                               double final_rel,
                               double final_resid_rel) {
        const dcr::base::Vector x_final = project_positive_state(decode_positive_state(y_final));
        unpack_state(x_final, out.nP_new, out.flowA_new, out.flowM_new);
        out.iterations = iterations;
        out.converged = converged;
        const dcr::base::Vector bg_full =
            make_background_full(out.nP_new, boundary, total_states);
        out.local_final = assemble_local_system(
            config, atomic_data, plasma, grid, boundary, bg_full,
            out.flowA_new, out.flowM_new, x_right_cm
        );
        out.flow_final = advance_recycling_flow_one_step(
            config, atomic_data, boundary, out.local_final,
            out.flowA_new, out.flowM_new, 0.0
        );
        out.final_rel = final_rel;
        out.final_resid_rel = final_resid_rel;
        out.elapsed_seconds = elapsed_seconds_since(solve_timer_start);

        if (config.io.verbose_logging && emit_summary_log) {
            const auto bg = summarize_bg_subgroups(out.nP_new, boundary, levels);
            double bg_nuclei = 0.0;
            for (int i = 0; i < out.nP_new.size() &&
                            i < static_cast<int>(boundary.P_indices.size()); ++i) {
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
                      << (out.converged ? " converged" : " max-iter")
                      << " in " << out.iterations
                      << " iterations (rel=" << out.final_rel
                      << ", resid_rel(diag)=" << out.final_resid_rel
                      << ", wall=" << format_elapsed_seconds(out.elapsed_seconds) << " s)"
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

    double tau = std::clamp(
        config.numerics.marching_ptc_tau_init,
        1.0e-6,
        std::max(config.numerics.marching_ptc_tau_init,
                 config.numerics.marching_ptc_tau_max)
    );
    const double tau_max = std::max(tau, config.numerics.marching_ptc_tau_max);
    const double alpha_min = std::clamp(config.numerics.marching_nk_alpha_min, 1.0e-6, 0.5);
    dcr::base::Vector y_work = encode_positive_state(
        project_positive_state(pack_state(nP_iter, flowA_iter, flowM_iter))
    );
    bool converged = false;
    int iterations = max_iter;
    double last_rel = std::numeric_limits<double>::infinity();
    double last_resid_rel = std::numeric_limits<double>::infinity();

    for (int iter = 0; iter < max_iter; ++iter) {
        const auto eval = evaluate_map(y_work);
        last_rel = eval.rel;
        last_resid_rel = eval.resid_rel;

        if (config.io.verbose_logging && detailed_log) {
            const double flowA_sum_iter = positive_sum(eval.flowA_image);
            const double flowM_sum_iter = positive_sum(eval.flowM_image);
            const auto bg_iter = summarize_bg_subgroups(eval.nP_image, boundary, levels);
            std::cout << "[DCR_Solver] Marching cell " << cell_index
                      << " x=[" << x_left_cm << "," << x_right_cm << "] cm"
                      << " T{e=" << cell_temperatures.electron_eV
                      << ", i=" << cell_temperatures.ion_eV
                      << "}"
                      << " iter " << (iter + 1)
                      << " solver=LOG-NK-PTC"
                      << " rel_change=" << eval.rel
                      << " rel_np=" << eval.rel_np
                      << " rel_a=" << eval.rel_a
                      << " rel_m=" << eval.rel_m
                      << " ||F||=" << eval.residual.norm()
                      << " resid_rel(diag)=" << eval.resid_rel
                      << " tau=" << tau
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
            y_work = eval.y_image;
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
                    (1.0 + eval.y_projected.norm()) / std::max(1.0, v.norm())
            );
            const auto pert = evaluate_map(eval.y_projected + eps * v);
            return (1.0 / tau) * v + (pert.residual - eval.residual) / eps;
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
            delta = 0.1 * (eval.y_image - eval.y_projected);
            tau = std::max(1.0e-6, 0.5 * tau);
        }

        const double norm0 = std::max(1.0e-30, eval.residual.norm());
        double alpha = 1.0;
        bool accepted = false;
        LogMarchingMapEvaluation accepted_eval;
        while (alpha >= alpha_min) {
            const auto trial = evaluate_map(eval.y_projected + alpha * delta);
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
            alpha = std::min(0.1, std::max(alpha_min, 0.05 * tau));
            const auto trial = evaluate_map(
                eval.y_projected + alpha * (eval.y_image - eval.y_projected)
            );
            accepted_eval = trial;
            accepted = true;
            tau = std::max(1.0e-6, 0.5 * tau);
            if (config.io.verbose_logging && detailed_log) {
                std::cout << "[DCR_Solver] Marching cell " << cell_index
                          << ": LOG-NK line search failed, falling back to conservative log-Picard step"
                          << " alpha=" << alpha
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

        y_work = accepted_eval.y_projected;
        if (config.io.verbose_logging && detailed_log) {
            std::cout << "[DCR_Solver] Marching cell " << cell_index
                      << ": LOG-NK step"
                      << " alpha=" << alpha
                      << " gmres_iters=" << gmres.iterations
                      << " gmres_resid=" << gmres.residual_norm
                      << " tau_next=" << tau
                      << "\n";
        }
    }

    finalize_output(y_work, iterations, converged, last_rel, last_resid_rel);
    return out;
}

} // namespace dcr::solver
