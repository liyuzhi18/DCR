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
    bool variable_nuclei_balance_closure = false;
    double variable_ion_divergence_nuclei_cm3_s = std::numeric_limits<double>::quiet_NaN();
    double variable_flowA_divergence_nuclei_cm3_s = std::numeric_limits<double>::quiet_NaN();
    double variable_flowM_divergence_nuclei_cm3_s = std::numeric_limits<double>::quiet_NaN();
    double variable_neutral_exhaust_nuclei_cm3_s = std::numeric_limits<double>::quiet_NaN();
    double variable_balance_rhs = std::numeric_limits<double>::quiet_NaN();
    double variable_balance_residual = std::numeric_limits<double>::quiet_NaN();
    double recycling_source_nuclei_cm3_s = std::numeric_limits<double>::quiet_NaN();
    double local_atom_exhaust_nuclei_cm3_s = std::numeric_limits<double>::quiet_NaN();
    double local_molecule_exhaust_nuclei_cm3_s = std::numeric_limits<double>::quiet_NaN();
    double ion_divergence_closure_nuclei_cm3_s = std::numeric_limits<double>::quiet_NaN();
    double ion_balance_coefficient_s = std::numeric_limits<double>::quiet_NaN();
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

double group_sum(const dcr::base::Vector& full,
                 const std::vector<int>& indices) {
    double sum = 0.0;
    for (int gi : indices) {
        if (gi >= 0 && gi < full.size()) sum += std::max(full(gi), 0.0);
    }
    return sum;
}

double nuclei_sum_background(
    const dcr::base::Vector& nP,
    const BoundaryPhaseResult& boundary,
    const std::vector<dcr::atomic::EnergyLevel>& levels) {
    return nuclei_sum(nP, boundary.P_indices, levels);
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
    const dcr::base::Vector& nP_old,
    const dcr::base::Vector& flowA_old,
    const dcr::base::Vector& flowM_old,
    const dcr::base::Vector& flowA_new,
    const dcr::base::Vector& flowM_new,
    double dx_cm,
    double x_left_cm,
    double x_right_cm,
    const AdaptiveTransportProfile* adaptive_profile,
    int adaptive_profile_node_index) {

    BackgroundSolveResult result;
    const bool variable_nuclei_balance_closure =
        config.numerics.adaptive_recycling_domain.closure_mode == "variable_nuclei_balance";
    result.variable_nuclei_balance_closure = variable_nuclei_balance_closure;
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

    const dcr::base::Vector bg_full = make_background_full(
        nP_guess, boundary, static_cast<int>(levels.size()));
    double ion_nuclei_density = 0.0;
    for (int gi : boundary.ion_indices) {
        if (gi < 0 || gi >= static_cast<int>(levels.size()) || gi >= bg_full.size()) continue;
        ion_nuclei_density += std::max(1, levels[static_cast<size_t>(gi)].atomicity) *
            std::max(bg_full(gi), 0.0);
    }
    double source_nuclei = 0.0;
    for (int pi = 0; pi < Pn && pi < local_system.S_background.size(); ++pi) {
        const int gi = boundary.P_indices[static_cast<size_t>(pi)];
        const double atomicity = (gi >= 0 && gi < static_cast<int>(levels.size()))
            ? std::max(1, levels[static_cast<size_t>(gi)].atomicity) : 1;
        source_nuclei += atomicity * local_system.S_background(pi);
    }
    const auto local_temperatures = evaluate_plasma_temperatures(config, x_right_cm);
    const double atom_speed = dcr::physics::calculate_thermal_speed(
        local_temperatures.ion_eV, boundary.atom_mass_amu);
    const double flow_atom_speed = dcr::physics::calculate_thermal_speed(
        config.plasma.neutral_atom_temperature_eV, boundary.atom_mass_amu);
    const double molecule_speed = dcr::physics::calculate_thermal_speed(
        config.plasma.neutral_molecule_temperature_eV, boundary.molecule_mass_amu);
    const double width = std::max(config.grid.spatial_exhaust_width_cm, 1.0e-12);
    auto background_exhaust = [&](const std::vector<int>& indices, double speed) {
        double total = 0.0;
        for (int gi : indices) {
            const int pi = (gi >= 0 && gi < static_cast<int>(p_pos.size()))
                ? p_pos[static_cast<size_t>(gi)] : -1;
            if (pi < 0 || pi >= nP_guess.size() ||
                gi >= static_cast<int>(levels.size())) continue;
            total += std::max(1, levels[static_cast<size_t>(gi)].atomicity) *
                speed / width * std::max(nP_guess(pi), 0.0);
        }
        return total;
    };
    const double atom_exhaust = background_exhaust(boundary.atom_bg_indices, atom_speed);
    const double molecule_exhaust = background_exhaust(
        boundary.mol_bg_indices, molecule_speed);
    const double legacy_L_I = source_nuclei - atom_exhaust - molecule_exhaust;
    double L_I_for_matrix = legacy_L_I;
    if (config.numerics.adaptive_recycling_domain.apply_ion_closure &&
        adaptive_profile != nullptr && adaptive_profile_node_index >= 0) {
        const size_t node = static_cast<size_t>(adaptive_profile_node_index);
        if (node < adaptive_profile->ion_divergence_nuclei_cm3_s.size() &&
            std::isfinite(adaptive_profile->ion_divergence_nuclei_cm3_s[node])) {
            L_I_for_matrix = -adaptive_profile->ion_divergence_nuclei_cm3_s[node];
        }
    }
    result.recycling_source_nuclei_cm3_s = source_nuclei;
    result.local_atom_exhaust_nuclei_cm3_s = atom_exhaust;
    result.local_molecule_exhaust_nuclei_cm3_s = molecule_exhaust;
    result.ion_divergence_closure_nuclei_cm3_s = L_I_for_matrix;
    result.ion_balance_coefficient_s = ion_nuclei_density > 0.0
        ? L_I_for_matrix / ion_nuclei_density
        : std::numeric_limits<double>::quiet_NaN();

    dcr::base::Matrix loss = dcr::base::Matrix::Zero(Pn, Pn);
    if (ion_nuclei_density > 0.0) {
        for (int gi : boundary.ion_indices) {
            const int pi = (gi >= 0 && gi < static_cast<int>(p_pos.size()))
                ? p_pos[static_cast<size_t>(gi)] : -1;
            if (pi >= 0) loss(pi, pi) = L_I_for_matrix / ion_nuclei_density;
        }
    }
    for (int gi : boundary.atom_bg_indices) {
        const int pi = (gi >= 0 && gi < static_cast<int>(p_pos.size()))
            ? p_pos[static_cast<size_t>(gi)] : -1;
        if (pi >= 0) loss(pi, pi) = atom_speed / width;
    }
    for (int gi : boundary.mol_bg_indices) {
        const int pi = (gi >= 0 && gi < static_cast<int>(p_pos.size()))
            ? p_pos[static_cast<size_t>(gi)] : -1;
        if (pi >= 0) loss(pi, pi) = molecule_speed / width;
    }

    const dcr::base::Matrix Rpp = extract_R_PP(local_system.R_full, boundary.P_indices);
    dcr::base::Matrix replaced = Rpp;
    dcr::base::Vector rhs = loss * nP_guess - local_system.S_background;
    dcr::base::Vector atomicity = dcr::base::Vector::Ones(Pn);
    for (int pi = 0; pi < Pn; ++pi) {
        const int gi = boundary.P_indices[static_cast<size_t>(pi)];
        if (gi >= 0 && gi < static_cast<int>(levels.size())) {
            atomicity(pi) = std::max(1, levels[static_cast<size_t>(gi)].atomicity);
        }
    }
    auto compact_nuclei_flux = [&](const dcr::base::Vector& flow,
                                   const std::vector<int>& indices,
                                   double velocity) {
        return velocity * nuclei_sum(flow, indices, levels);
    };
    const double fixed_ion_velocity_length_cm =
        config.numerics.adaptive_recycling_domain.ion_velocity_length_cm;
    const double L1_for_ion_velocity = fixed_ion_velocity_length_cm > 0.0
        ? fixed_ion_velocity_length_cm
        : ((adaptive_profile != nullptr && adaptive_profile->L1_cm > 0.0)
            ? adaptive_profile->L1_cm
            : std::max(config.grid.length_cm, x_right_cm));
    const double LM_for_ion_velocity = fixed_ion_velocity_length_cm > 0.0
        ? fixed_ion_velocity_length_cm
        : ((adaptive_profile != nullptr && adaptive_profile->LM_cm > 0.0)
            ? adaptive_profile->LM_cm
            : std::max(config.grid.length_cm, x_right_cm));
    auto ion_velocity = [&](int gi, double x_cm) {
        if (gi < 0 || gi >= static_cast<int>(levels.size())) return 0.0;
        const double length_cm = levels[static_cast<size_t>(gi)].atomicity >= 2
            ? LM_for_ion_velocity : L1_for_ion_velocity;
        if (!(length_cm > 0.0)) return 0.0;
        const double floor_fraction = std::clamp(
            config.numerics.adaptive_recycling_domain.ion_velocity_floor_fraction,
            0.0, 1.0);
        const double shape = std::max(floor_fraction, 1.0 - x_cm / length_cm);
        const auto temperatures = evaluate_plasma_temperatures(config, x_cm);
        const double mass_amu = std::max(
            levels[static_cast<size_t>(gi)].mass_amu, 1.0e-12);
        return dcr::physics::calculate_Bohm_speed(
            temperatures.electron_eV, temperatures.ion_eV, mass_amu) * shape;
    };
    auto ion_nuclei_flux = [&](const dcr::base::Vector& state, double x_cm) {
        double total = 0.0;
        for (int gi : boundary.ion_indices) {
            const int pi = (gi >= 0 && gi < static_cast<int>(p_pos.size()))
                ? p_pos[static_cast<size_t>(gi)] : -1;
            if (pi < 0 || pi >= state.size() ||
                gi >= static_cast<int>(levels.size())) continue;
            const double mu = static_cast<double>(
                std::max(1, levels[static_cast<size_t>(gi)].atomicity));
            total += mu * (-ion_velocity(gi, x_cm)) * std::max(state(pi), 0.0);
        }
        return total;
    };
    const double flowA_divergence = dx_cm > 0.0
        ? (compact_nuclei_flux(flowA_new, boundary.A_indices, boundary.u_A) -
           compact_nuclei_flux(flowA_old, boundary.A_indices, boundary.u_A)) / dx_cm
        : 0.0;
    const double flowM_divergence = dx_cm > 0.0
        ? (compact_nuclei_flux(flowM_new, boundary.M_indices, boundary.u_M) -
           compact_nuclei_flux(flowM_old, boundary.M_indices, boundary.u_M)) / dx_cm
        : 0.0;
    const double flow_exhaust = (flow_atom_speed / width) *
        nuclei_sum(flowA_new, boundary.A_indices, levels) +
        (molecule_speed / width) * nuclei_sum(flowM_new, boundary.M_indices, levels);
    const double ion_flux_left = ion_nuclei_flux(nP_old, x_left_cm);
    dcr::base::Vector closure_row = atomicity;
    const double flow_nuclei = nuclei_sum(flowA_new, boundary.A_indices, levels) +
        nuclei_sum(flowM_new, boundary.M_indices, levels);
    double closure_rhs = std::max(0.0, config.plasma.total_density - flow_nuclei);
    if (variable_nuclei_balance_closure) {
        closure_row.setZero();
        if (dx_cm > 0.0) {
            for (int gi : boundary.ion_indices) {
                const int pi = (gi >= 0 && gi < static_cast<int>(p_pos.size()))
                    ? p_pos[static_cast<size_t>(gi)] : -1;
                if (pi < 0 || gi >= static_cast<int>(levels.size())) continue;
                closure_row(pi) += std::max(1, levels[static_cast<size_t>(gi)].atomicity) *
                    (-ion_velocity(gi, x_right_cm)) / dx_cm;
            }
        }
        for (int gi : boundary.atom_bg_indices) {
            const int pi = (gi >= 0 && gi < static_cast<int>(p_pos.size()))
                ? p_pos[static_cast<size_t>(gi)] : -1;
            if (pi < 0 || gi >= static_cast<int>(levels.size())) continue;
            closure_row(pi) += std::max(1, levels[static_cast<size_t>(gi)].atomicity) *
                atom_speed / width;
        }
        for (int gi : boundary.mol_bg_indices) {
            const int pi = (gi >= 0 && gi < static_cast<int>(p_pos.size()))
                ? p_pos[static_cast<size_t>(gi)] : -1;
            if (pi < 0 || gi >= static_cast<int>(levels.size())) continue;
            closure_row(pi) += std::max(1, levels[static_cast<size_t>(gi)].atomicity) *
                molecule_speed / width;
        }
        closure_rhs = (dx_cm > 0.0 ? ion_flux_left / dx_cm : 0.0) -
            flowA_divergence - flowM_divergence - flow_exhaust;
        result.variable_flowA_divergence_nuclei_cm3_s = flowA_divergence;
        result.variable_flowM_divergence_nuclei_cm3_s = flowM_divergence;
        result.variable_balance_rhs = closure_rhs;
    }
    const int constraint_row = std::min(1, Pn - 1);
    replaced.row(constraint_row) = closure_row.transpose();
    rhs(constraint_row) = closure_rhs;

    auto set_original_diag = [&](const dcr::base::Vector& n_eval) {
        const dcr::base::Vector r = Rpp * n_eval -
            (loss * n_eval - local_system.S_background);
        result.linear_residual_norm = r.norm();
        result.linear_mse = r.squaredNorm() / static_cast<double>(Pn);
        const double lhs_norm = (Rpp * n_eval).norm();
        result.linear_mse_rel = result.linear_residual_norm / std::max(1.0, lhs_norm);
        if (variable_nuclei_balance_closure) {
            const double ion_divergence = dx_cm > 0.0
                ? (ion_nuclei_flux(n_eval, x_right_cm) - ion_flux_left) / dx_cm
                : 0.0;
            double local_exhaust = 0.0;
            for (int gi : boundary.atom_bg_indices) {
                const int pi = (gi >= 0 && gi < static_cast<int>(p_pos.size()))
                    ? p_pos[static_cast<size_t>(gi)] : -1;
                if (pi < 0 || pi >= n_eval.size() ||
                    gi >= static_cast<int>(levels.size())) continue;
                local_exhaust += std::max(1, levels[static_cast<size_t>(gi)].atomicity) *
                    atom_speed / width * std::max(n_eval(pi), 0.0);
            }
            for (int gi : boundary.mol_bg_indices) {
                const int pi = (gi >= 0 && gi < static_cast<int>(p_pos.size()))
                    ? p_pos[static_cast<size_t>(gi)] : -1;
                if (pi < 0 || pi >= n_eval.size() ||
                    gi >= static_cast<int>(levels.size())) continue;
                local_exhaust += std::max(1, levels[static_cast<size_t>(gi)].atomicity) *
                    molecule_speed / width * std::max(n_eval(pi), 0.0);
            }
            result.variable_ion_divergence_nuclei_cm3_s = ion_divergence;
            result.variable_neutral_exhaust_nuclei_cm3_s = local_exhaust + flow_exhaust;
            result.variable_balance_residual = ion_divergence + flowA_divergence +
                flowM_divergence + local_exhaust + flow_exhaust;
        }
    };

    dcr::base::Vector nP_new = solve_replaced_system(replaced, rhs);

    if (!nP_new.allFinite()) {
        result.nP_new = nP_guess;
        set_original_diag(nP_guess);
        return result;
    }

    for (int i = 0; i < nP_new.size(); ++i) {
        nP_new(i) = std::max(nP_new(i), kLogStateFloor);
    }
    if (!variable_nuclei_balance_closure) {
        const double solved_nuclei = (atomicity.array() * nP_new.array()).sum();
        if (closure_rhs > 0.0 && solved_nuclei > 0.0) {
            nP_new *= closure_rhs / solved_nuclei;
        }
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
    const dcr::base::Vector* flowM_init_override,
    const AdaptiveTransportProfile* adaptive_profile) {

    CellImplicitResult out;
    const auto solve_timer_start = std::chrono::steady_clock::now();
    const int total_states = atomic_data.get_total_states();
    dcr::base::Vector nP_iter = nP_old;
    dcr::base::Vector flowA_iter = flowA_old;
    dcr::base::Vector flowM_iter = flowM_old;
    const dcr::base::Vector& flowA_inflow = flowA_old;
    const dcr::base::Vector& flowM_inflow = flowM_old;
    const bool variable_nuclei_balance_closure =
        config.numerics.adaptive_recycling_domain.closure_mode == "variable_nuclei_balance";
    if (nP_init_override && nP_init_override->size() == nP_old.size()) {
        nP_iter = nP_init_override->cwiseMax(kLogStateFloor);
    }
    if (flowA_init_override && flowA_init_override->size() == flowA_old.size()) {
        flowA_iter = flowA_init_override->cwiseMax(kLogStateFloor);
    }
    if (flowM_init_override && flowM_init_override->size() == flowM_old.size()) {
        flowM_iter = flowM_init_override->cwiseMax(kLogStateFloor);
    }

    const double tol = std::max(config.numerics.tolerance, 1e-12);
    const int max_iter = std::max(config.numerics.max_iterations, 1);
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
    auto impose_nuclei_closure = [&](dcr::base::Vector& nP_state,
                                     const dcr::base::Vector& flowA_state,
                                     const dcr::base::Vector& flowM_state) {
        if (variable_nuclei_balance_closure) return;
        const double flow_nuclei = nuclei_sum(flowA_state, boundary.A_indices, levels) +
            nuclei_sum(flowM_state, boundary.M_indices, levels);
        const double target_background =
            std::max(0.0, config.plasma.total_density - flow_nuclei);
        const double background_nuclei =
            nuclei_sum_background(nP_state, boundary, levels);
        if (target_background > 0.0 && background_nuclei > 0.0) {
            nP_state *= target_background / background_nuclei;
        }
        nP_state = nP_state.cwiseMax(kLogStateFloor);
    };

    auto project_positive_state = [&](const dcr::base::Vector& x_in) {
        dcr::base::Vector nP_proj = dcr::base::Vector::Constant(nP_size, kLogStateFloor);
        dcr::base::Vector flowA_proj = dcr::base::Vector::Constant(nA_size, kLogStateFloor);
        dcr::base::Vector flowM_proj = dcr::base::Vector::Constant(nM_size, kLogStateFloor);
        unpack_state(x_in, nP_proj, flowA_proj, flowM_proj);
        nP_proj = nP_proj.cwiseMax(kLogStateFloor);
        flowA_proj = flowA_proj.cwiseMax(kLogStateFloor);
        flowM_proj = flowM_proj.cwiseMax(kLogStateFloor);
        impose_nuclei_closure(nP_proj, flowA_proj, flowM_proj);
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
            config,
            atomic_data,
            plasma,
            grid,
            boundary,
            bg_iter_full,
            eval.flowA_iter,
            eval.flowM_iter,
            x_right_cm
        );
        eval.flow_advanced = advance_recycling_flow_one_step(
            config,
            atomic_data,
            boundary,
            local_for_flow,
            flowA_inflow,
            flowM_inflow,
            dx_cm
        );
        eval.local_for_bg = assemble_local_system(
            config,
            atomic_data,
            plasma,
            grid,
            boundary,
            bg_iter_full,
            eval.flow_advanced.flowA_next,
            eval.flow_advanced.flowM_next,
            x_right_cm
        );
        eval.bg_solve = solve_background_at_cell_log(
            config,
            boundary,
            levels,
            eval.local_for_bg,
            eval.nP_iter,
            nP_old,
            flowA_inflow,
            flowM_inflow,
            eval.flow_advanced.flowA_next,
            eval.flow_advanced.flowM_next,
            dx_cm,
            x_left_cm,
            x_right_cm,
            adaptive_profile,
            cell_index
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
        const dcr::base::Vector x_final =
            project_positive_state(decode_positive_state(y_final));
        unpack_state(x_final, out.nP_new, out.flowA_new, out.flowM_new);
        out.iterations = iterations;
        out.converged = converged;
        const dcr::base::Vector bg_full =
            make_background_full(out.nP_new, boundary, total_states);
        out.local_final = assemble_local_system(
            config,
            atomic_data,
            plasma,
            grid,
            boundary,
            bg_full,
            out.flowA_new,
            out.flowM_new,
            x_right_cm
        );
        out.flow_final = advance_recycling_flow_one_step(
            config,
            atomic_data,
            boundary,
            out.local_final,
            out.flowA_new,
            out.flowM_new,
            0.0
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
            const double flowA_nuclei =
                nuclei_sum(out.flowA_new, boundary.A_indices, levels);
            const double flowM_nuclei =
                nuclei_sum(out.flowM_new, boundary.M_indices, levels);
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
    BackgroundSolveResult final_bg_solve_diag;
    bool have_final_bg_solve_diag = false;
    const int slow_iter_threshold = std::max(1, config.numerics.marching_slow_iter_threshold);
    const auto should_log_iteration_record = [&](int iter) {
        return detailed_log && (config.io.verbose_logging || (iter + 1) > slow_iter_threshold);
    };

    for (int iter = 0; iter < max_iter; ++iter) {
        const auto eval = evaluate_map(y_work);
        final_bg_solve_diag = eval.bg_solve;
        have_final_bg_solve_diag = true;
        last_rel = eval.rel;
        last_resid_rel = eval.resid_rel;

        if (should_log_iteration_record(iter)) {
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
                      << "} L_I="
                      << eval.bg_solve.ion_divergence_closure_nuclei_cm3_s
                      << " source_nuc="
                      << eval.bg_solve.recycling_source_nuclei_cm3_s
                      << "\n";
        }

        const double residual_tolerance = std::max(10.0 * tol, 1.0e-8);
        if (eval.rel < tol && std::isfinite(eval.resid_rel) &&
            eval.resid_rel < residual_tolerance && eval.residual.allFinite()) {
            y_work = eval.y_image;
            converged = true;
            iterations = iter + 1;
            break;
        }
        if (!eval.residual.allFinite() || !std::isfinite(eval.resid_rel)) {
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
            const double trial_rel = trial.rel;
            if (trial_norm < norm0 * (1.0 - 1.0e-4 * alpha) ||
                trial_norm < 0.95 * norm0 ||
                (std::isfinite(trial_rel) &&
                 (trial_rel < eval.rel * (1.0 - 1.0e-4 * alpha) ||
                  trial_rel < 0.95 * eval.rel))) {
                accepted_eval = trial;
                accepted = true;
                break;
            }
            alpha *= 0.5;
        }

        if (!accepted) {
            const double picard_delta_norm =
                (eval.y_image - eval.y_projected).norm();
            const double fallback_alpha0 =
                std::min(0.1, std::max(alpha_min, std::max(1.0e-2, 0.05 * tau)));
            alpha = fallback_alpha0;
            accepted_eval = evaluate_map(
                eval.y_projected + alpha * (eval.y_image - eval.y_projected)
            );
            double best_norm = accepted_eval.residual.norm();
            double best_rel = accepted_eval.rel;
            if (picard_delta_norm > 0.0) {
                const double fallback_alphas[] = {0.1, 0.05, 0.02, 0.01};
                for (double candidate_raw : fallback_alphas) {
                    const double candidate = std::min(0.1, std::max(alpha_min, candidate_raw));
                    const auto trial = evaluate_map(
                        eval.y_projected + candidate * (eval.y_image - eval.y_projected)
                    );
                    const double trial_norm = trial.residual.norm();
                    const double trial_rel = trial.rel;
                    if (!std::isfinite(trial_norm) || !std::isfinite(trial_rel)) continue;
                    if (trial_norm <= 1.02 * norm0 || trial_rel <= 1.02 * eval.rel) {
                        accepted_eval = trial;
                        alpha = candidate;
                        best_norm = trial_norm;
                        best_rel = trial_rel;
                        break;
                    }
                    if (!std::isfinite(best_norm) || trial_norm < best_norm ||
                        (std::isfinite(best_rel) && trial_rel < best_rel)) {
                        accepted_eval = trial;
                        alpha = candidate;
                        best_norm = trial_norm;
                        best_rel = trial_rel;
                    }
                }
            }
            tau = std::max(1.0e-6, 0.5 * tau);
            if (should_log_iteration_record(iter)) {
                std::cout << "[DCR_Solver] Marching cell " << cell_index
                          << ": LOG-NK line search failed, falling back to conservative log-Picard step"
                          << " alpha=" << alpha
                          << " fallback_resid=" << best_norm
                          << " tau=" << tau
                          << "\n";
            }
        } else if (alpha >= 0.75 && gmres.converged) {
            tau = std::min(tau_max, 1.5 * tau);
        } else if (alpha < 0.25 || !gmres.converged) {
            tau = std::max(1.0e-6, 0.5 * tau);
        }

        y_work = accepted_eval.y_projected;
        final_bg_solve_diag = accepted_eval.bg_solve;
        have_final_bg_solve_diag = true;
        if (should_log_iteration_record(iter)) {
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
    if (have_final_bg_solve_diag) {
        out.variable_nuclei_balance_closure =
            final_bg_solve_diag.variable_nuclei_balance_closure;
        out.variable_ion_divergence_nuclei_cm3_s =
            final_bg_solve_diag.variable_ion_divergence_nuclei_cm3_s;
        out.variable_flowA_divergence_nuclei_cm3_s =
            final_bg_solve_diag.variable_flowA_divergence_nuclei_cm3_s;
        out.variable_flowM_divergence_nuclei_cm3_s =
            final_bg_solve_diag.variable_flowM_divergence_nuclei_cm3_s;
        out.variable_neutral_exhaust_nuclei_cm3_s =
            final_bg_solve_diag.variable_neutral_exhaust_nuclei_cm3_s;
        out.variable_balance_rhs_cm3_s = final_bg_solve_diag.variable_balance_rhs;
        out.variable_balance_residual_cm3_s =
            final_bg_solve_diag.variable_balance_residual;
        out.prescribed_nuclei_density_cm3 = config.plasma.total_density;
        out.recycling_source_nuclei_cm3_s =
            final_bg_solve_diag.recycling_source_nuclei_cm3_s;
        out.local_atom_exhaust_nuclei_cm3_s =
            final_bg_solve_diag.local_atom_exhaust_nuclei_cm3_s;
        out.local_molecule_exhaust_nuclei_cm3_s =
            final_bg_solve_diag.local_molecule_exhaust_nuclei_cm3_s;
        out.ion_divergence_closure_nuclei_cm3_s =
            final_bg_solve_diag.ion_divergence_closure_nuclei_cm3_s;
        out.ion_balance_coefficient_s =
            final_bg_solve_diag.ion_balance_coefficient_s;
    }
    return out;
}

} // namespace dcr::solver
