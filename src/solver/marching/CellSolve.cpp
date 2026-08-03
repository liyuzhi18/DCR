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
    // Diagnostics for the solved finite-volume system C*n = rhs.
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
    double prescribed_nuclei_density_cm3 = std::numeric_limits<double>::quiet_NaN();
    double recycling_source_nuclei_cm3_s = std::numeric_limits<double>::quiet_NaN();
    double local_atom_exhaust_nuclei_cm3_s = std::numeric_limits<double>::quiet_NaN();
    double local_molecule_exhaust_nuclei_cm3_s = std::numeric_limits<double>::quiet_NaN();
    double ion_divergence_closure_nuclei_cm3_s = std::numeric_limits<double>::quiet_NaN();
    double ion_balance_coefficient_s = std::numeric_limits<double>::quiet_NaN();
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
    (void)nP_floor_ref;
    const int Pn = static_cast<int>(boundary.P_indices.size());
    if (Pn == 0) {
        result.nP_new = nP_guess;
        return result;
    }

    std::vector<int> p_pos(levels.size(), -1);
    for (int i = 0; i < Pn; ++i) {
        const int gi = boundary.P_indices[static_cast<size_t>(i)];
        if (gi >= 0 && gi < static_cast<int>(levels.size())) {
            p_pos[static_cast<size_t>(gi)] = i;
        }
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
            const double mu = static_cast<double>(
                std::max(1, levels[static_cast<size_t>(gi)].atomicity));
            total += mu * (speed / width) * std::max(nP_guess(pi), 0.0);
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

    result.prescribed_nuclei_density_cm3 = config.plasma.total_density;
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
                                   double velocity_cm_s) {
        double total = 0.0;
        const int n = std::min<int>(flow.size(), static_cast<int>(indices.size()));
        for (int j = 0; j < n; ++j) {
            const int gi = indices[static_cast<size_t>(j)];
            if (gi < 0 || gi >= static_cast<int>(levels.size())) continue;
            const double mu = static_cast<double>(std::max(1, levels[static_cast<size_t>(gi)].atomicity));
            total += mu * std::max(flow(j), 0.0) * velocity_cm_s;
        }
        return total;
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
    const double flowA_div_nuclei = (dx_cm > 0.0)
        ? (compact_nuclei_flux(flowA_new, boundary.A_indices, boundary.u_A) -
           compact_nuclei_flux(flowA_old, boundary.A_indices, boundary.u_A)) / dx_cm
        : 0.0;
    const double flowM_div_nuclei = (dx_cm > 0.0)
        ? (compact_nuclei_flux(flowM_new, boundary.M_indices, boundary.u_M) -
           compact_nuclei_flux(flowM_old, boundary.M_indices, boundary.u_M)) / dx_cm
        : 0.0;
    const double flowA_exhaust_nuclei = (flow_atom_speed / width) *
        nuclei_sum(flowA_new, boundary.A_indices, levels);
    const double flowM_exhaust_nuclei = (molecule_speed / width) *
        nuclei_sum(flowM_new, boundary.M_indices, levels);
    const double flow_exhaust_nuclei = flowA_exhaust_nuclei + flowM_exhaust_nuclei;
    const double ion_flux_left_nuclei = ion_nuclei_flux(nP_old, x_left_cm);
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
                const double mu = static_cast<double>(
                    std::max(1, levels[static_cast<size_t>(gi)].atomicity));
                closure_row(pi) += mu * (-ion_velocity(gi, x_right_cm)) / dx_cm;
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
        closure_rhs = (dx_cm > 0.0 ? ion_flux_left_nuclei / dx_cm : 0.0) -
            flowA_div_nuclei - flowM_div_nuclei - flow_exhaust_nuclei;
        result.variable_flowA_divergence_nuclei_cm3_s = flowA_div_nuclei;
        result.variable_flowM_divergence_nuclei_cm3_s = flowM_div_nuclei;
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
                ? (ion_nuclei_flux(n_eval, x_right_cm) - ion_flux_left_nuclei) / dx_cm
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
            result.variable_neutral_exhaust_nuclei_cm3_s =
                local_exhaust + flow_exhaust_nuclei;
            result.variable_balance_residual = ion_divergence + flowA_div_nuclei +
                flowM_div_nuclei + local_exhaust + flow_exhaust_nuclei;
        }
    };

    dcr::base::Vector nP_new = solve_replaced_system(replaced, rhs);

    if (!nP_new.allFinite()) {
        dcr::base::Vector fallback = nP_guess;
        apply_nonneg(fallback);
        const double fallback_nuclei =
            (atomicity.array() * fallback.array()).sum();
        if (!variable_nuclei_balance_closure) {
            if (closure_rhs <= 0.0) {
                fallback.setZero();
            } else if (fallback_nuclei > 0.0) {
                fallback *= closure_rhs / fallback_nuclei;
            }
        }
        result.nP_new = fallback;
        set_original_diag(fallback);
        return result;
    }

    apply_nonneg(nP_new);
    if (!variable_nuclei_balance_closure) {
        const double solved_background_nuclei =
            (atomicity.array() * nP_new.array()).sum();
        if (closure_rhs <= 0.0) {
            nP_new.setZero();
        } else if (solved_background_nuclei > 0.0) {
            nP_new *= closure_rhs / solved_background_nuclei;
        }
    }
    result.nP_new = nP_new;
    set_original_diag(nP_new);
    return result;
}

void copy_variable_nuclei_diagnostics(CellImplicitResult& result,
                                      const BackgroundSolveResult& bg_solve) {
    result.variable_nuclei_balance_closure = bg_solve.variable_nuclei_balance_closure;
    result.variable_ion_divergence_nuclei_cm3_s = bg_solve.variable_ion_divergence_nuclei_cm3_s;
    result.variable_flowA_divergence_nuclei_cm3_s = bg_solve.variable_flowA_divergence_nuclei_cm3_s;
    result.variable_flowM_divergence_nuclei_cm3_s = bg_solve.variable_flowM_divergence_nuclei_cm3_s;
    result.variable_neutral_exhaust_nuclei_cm3_s = bg_solve.variable_neutral_exhaust_nuclei_cm3_s;
    result.variable_balance_rhs_cm3_s = bg_solve.variable_balance_rhs;
    result.variable_balance_residual_cm3_s = bg_solve.variable_balance_residual;
    result.prescribed_nuclei_density_cm3 = bg_solve.prescribed_nuclei_density_cm3;
    result.recycling_source_nuclei_cm3_s = bg_solve.recycling_source_nuclei_cm3_s;
    result.local_atom_exhaust_nuclei_cm3_s = bg_solve.local_atom_exhaust_nuclei_cm3_s;
    result.local_molecule_exhaust_nuclei_cm3_s = bg_solve.local_molecule_exhaust_nuclei_cm3_s;
    result.ion_divergence_closure_nuclei_cm3_s =
        bg_solve.ion_divergence_closure_nuclei_cm3_s;
    result.ion_balance_coefficient_s = bg_solve.ion_balance_coefficient_s;
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
    bool emit_summary_log,
    const AdaptiveTransportProfile* adaptive_profile) {

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
    const bool variable_nuclei_balance_closure =
        config.numerics.adaptive_recycling_domain.closure_mode == "variable_nuclei_balance";
    const auto should_log_iteration_record = [&](int iter) {
        return detailed_log && (config.io.verbose_logging || (iter + 1) > slow_iter_threshold);
    };
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
        if (target_background <= 0.0) {
            nP_state.setZero();
        } else if (background_nuclei > 0.0) {
            nP_state *= target_background / background_nuclei;
        }
    };

    auto project_state = [&](const dcr::base::Vector& x_in) {
        dcr::base::Vector nP_proj = nP_iter;
        dcr::base::Vector flowA_proj = flowA_iter;
        dcr::base::Vector flowM_proj = flowM_iter;
        unpack_state(x_in, nP_proj, flowA_proj, flowM_proj);
        nP_proj = nP_proj.cwiseMax(0.0);
        flowA_proj = flowA_proj.cwiseMax(0.0);
        flowM_proj = flowM_proj.cwiseMax(0.0);
        impose_nuclei_closure(nP_proj, flowA_proj, flowM_proj);
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
                               double final_resid_rel,
                               const BackgroundSolveResult* bg_solve_diag) {
        result.nP_new = nP_final.cwiseMax(0.0);
        result.flowA_new = flowA_final.cwiseMax(0.0);
        result.flowM_new = flowM_final.cwiseMax(0.0);
        impose_nuclei_closure(result.nP_new, result.flowA_new, result.flowM_new);
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
        if (bg_solve_diag != nullptr) {
            copy_variable_nuclei_diagnostics(result, *bg_solve_diag);
        }

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
            nullptr,
            adaptive_profile
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
        BackgroundSolveResult final_bg_solve_diag;
        bool have_final_bg_solve_diag = false;

        for (int iter = 0; iter < max_iter; ++iter) {
            const auto eval = evaluate_map(x_work);
            final_bg_solve_diag = eval.bg_solve;
            have_final_bg_solve_diag = true;
            last_rel = eval.rel;
            last_resid_rel = eval.resid_rel;

            if (should_log_iteration_record(iter)) {
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
                if (eval.bg_solve.variable_nuclei_balance_closure) {
                    std::cout << "[DCR_Solver][variable-nuclei] cell " << cell_index
                              << " ion_div=" << eval.bg_solve.variable_ion_divergence_nuclei_cm3_s
                              << " flowA_div=" << eval.bg_solve.variable_flowA_divergence_nuclei_cm3_s
                              << " flowM_div=" << eval.bg_solve.variable_flowM_divergence_nuclei_cm3_s
                              << " neutral_exhaust=" << eval.bg_solve.variable_neutral_exhaust_nuclei_cm3_s
                              << " rhs=" << eval.bg_solve.variable_balance_rhs
                              << " residual=" << eval.bg_solve.variable_balance_residual
                              << " L_I=" << eval.bg_solve.ion_divergence_closure_nuclei_cm3_s
                              << " source_nuc=" << eval.bg_solve.recycling_source_nuclei_cm3_s
                              << " note=backward_upwind_balance"
                              << "\n";
                }
            }

            const double residual_tolerance = std::max(10.0 * tol, 1.0e-8);
            if (eval.rel < tol && std::isfinite(eval.resid_rel) &&
                eval.resid_rel < residual_tolerance && eval.residual.allFinite()) {
                x_work = eval.x_image;
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
                // Fall back to a conservative Picard-like correction if the
                // Newton step does not provide sufficient decrease.
                const double picard_delta_norm =
                    (eval.x_image - eval.x_projected).norm();
                const double fallback_alpha0 =
                    std::min(0.1, std::max(alpha_min, std::max(1.0e-2, 0.05 * tau)));
                alpha = fallback_alpha0;
                accepted_eval = evaluate_map(
                    eval.x_projected + alpha * (eval.x_image - eval.x_projected)
                );
                double best_norm = accepted_eval.residual.norm();
                double best_rel = accepted_eval.rel;
                if (picard_delta_norm > 0.0) {
                    const double fallback_alphas[] = {0.1, 0.05, 0.02, 0.01};
                    for (double candidate_raw : fallback_alphas) {
                        const double candidate = std::min(0.1, std::max(alpha_min, candidate_raw));
                        const auto trial = evaluate_map(
                            eval.x_projected + candidate * (eval.x_image - eval.x_projected)
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
                accepted = true;
                tau = std::max(1.0e-6, 0.5 * tau);
                if (should_log_iteration_record(iter)) {
                    std::cout << "[DCR_Solver] Marching cell " << cell_index
                              << ": NK line search failed, falling back to conservative Picard step"
                              << " alpha=" << alpha
                              << " fallback_resid=" << best_norm
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
            final_bg_solve_diag = accepted_eval.bg_solve;
            have_final_bg_solve_diag = true;
            if (should_log_iteration_record(iter)) {
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
                        last_rel, last_resid_rel,
                        have_final_bg_solve_diag ? &final_bg_solve_diag : nullptr);
        return out;
    }

    dcr::base::Vector x_prev = pack_state(nP_iter, flowA_iter, flowM_iter);
    dcr::base::Vector x_prevprev = x_prev;
    bool have_prevprev = false;
    BackgroundSolveResult final_bg_solve_diag;
    bool have_final_bg_solve_diag = false;

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
            nP_old,
            flowA_inflow,
            flowM_inflow,
            flow_advanced.flowA_next,
            flow_advanced.flowM_next,
            dx_cm,
            x_left_cm,
            x_right_cm,
            adaptive_profile,
            cell_index
        );
        final_bg_solve_diag = bg_solve;
        have_final_bg_solve_diag = true;
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
            impose_nuclei_closure(nP_trial, flowA_trial, flowM_trial);

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
            if (should_log_iteration_record(iter)) {
                std::cout << "[DCR_Solver] Marching cell " << cell_index
                          << ": rel_change growth detected, reducing relaxation to "
                          << omega_iter << "\n";
            }
        } else if (no_improve_count >= 5 && omega_iter > omega_min) {
            omega_iter = std::max(omega_min, 0.5 * omega_iter);
            no_improve_count = 0;
            if (should_log_iteration_record(iter)) {
                std::cout << "[DCR_Solver] Marching cell " << cell_index
                          << ": slow convergence, reducing relaxation to "
                          << omega_iter << "\n";
            }
        }
        if (((iter + 1) % 50 == 0) && rel > (10.0 * tol) && omega_iter > omega_min) {
            omega_iter = std::max(omega_min, 0.5 * omega_iter);
            if (should_log_iteration_record(iter)) {
                std::cout << "[DCR_Solver] Marching cell " << cell_index
                          << ": periodic damping, reducing relaxation to "
                          << omega_iter << "\n";
            }
        }
        if (steady_improve_count >= 8 && omega_iter < omega_recover_cap) {
            omega_iter = std::min(omega_recover_cap, 1.5 * omega_iter);
            steady_improve_count = 0;
            if (should_log_iteration_record(iter)) {
                std::cout << "[DCR_Solver] Marching cell " << cell_index
                          << ": stable contraction detected, increasing relaxation to "
                          << omega_iter << "\n";
            }
        }
        if (floor_contraction_count >= 25 && omega_iter < omega_recover_cap) {
            omega_iter = std::min(omega_recover_cap, 2.0 * omega_iter);
            floor_contraction_count = 0;
            steady_improve_count = 0;
            if (should_log_iteration_record(iter)) {
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

        if (should_log_iteration_record(iter)) {
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
            if (bg_solve.variable_nuclei_balance_closure) {
                std::cout << "[DCR_Solver][variable-nuclei] cell " << cell_index
                          << " ion_div=" << bg_solve.variable_ion_divergence_nuclei_cm3_s
                          << " flowA_div=" << bg_solve.variable_flowA_divergence_nuclei_cm3_s
                          << " flowM_div=" << bg_solve.variable_flowM_divergence_nuclei_cm3_s
                          << " neutral_exhaust=" << bg_solve.variable_neutral_exhaust_nuclei_cm3_s
                          << " rhs=" << bg_solve.variable_balance_rhs
                          << " residual=" << bg_solve.variable_balance_residual
                          << " L_I=" << bg_solve.ion_divergence_closure_nuclei_cm3_s
                          << " source_nuc=" << bg_solve.recycling_source_nuclei_cm3_s
                          << " note=backward_upwind_balance"
                          << "\n";
            }

        }

        const double residual_tolerance = std::max(10.0 * tol, 1.0e-8);
        const bool strict_converged =
            (rel < tol) && std::isfinite(resid_rel) && resid_rel < residual_tolerance;

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
                    out.converged, last_rel, last_resid_rel,
                    have_final_bg_solve_diag ? &final_bg_solve_diag : nullptr);
    return out;
}

} // namespace dcr::solver
