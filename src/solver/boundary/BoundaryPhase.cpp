#include "BoundaryPhase.hpp"

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

struct BgSubgroupSums {
    double H_plus = 0.0;
    double H = 0.0;
    double H2 = 0.0;
    double H2_plus = 0.0;
    double H_minus = 0.0;
};

struct RecyclingModel {
    std::vector<int> ion_p_cols;
    std::vector<double> recA_coeff;
    std::vector<double> recM_coeff;
};

double quasineutral_electron_density(const dcr::base::Vector& population,
                                     const std::vector<dcr::atomic::EnergyLevel>& levels) {
    const int n = std::min<int>(population.size(), static_cast<int>(levels.size()));
    double ne = 0.0;
    for (int i = 0; i < n; ++i) {
        const int z = levels[static_cast<size_t>(i)].charge;
        if (z == 0) continue;
        ne += static_cast<double>(z) * std::max(population(i), 0.0);
    }
    return std::max(0.0, ne);
}

int find_ground_state(const std::vector<int>& indices,
                      const std::vector<dcr::atomic::EnergyLevel>& levels) {
    int best = -1;
    double e_best = std::numeric_limits<double>::infinity();
    for (int gi : indices) {
        if (gi < 0 || gi >= static_cast<int>(levels.size())) continue;
        if (levels[static_cast<size_t>(gi)].energy_eV < e_best) {
            e_best = levels[static_cast<size_t>(gi)].energy_eV;
            best = gi;
        }
    }
    return best;
}

dcr::base::Matrix extract_R_pp(const dcr::base::Matrix& R_full,
                               const std::vector<int>& p_indices) {
    const int Pn = static_cast<int>(p_indices.size());
    dcr::base::Matrix Rpp = dcr::base::Matrix::Zero(Pn, Pn);
    for (int i = 0; i < Pn; ++i) {
        const int gi = p_indices[static_cast<size_t>(i)];
        if (gi < 0 || gi >= R_full.rows()) continue;
        for (int j = 0; j < Pn; ++j) {
            const int gj = p_indices[static_cast<size_t>(j)];
            if (gj < 0 || gj >= R_full.cols()) continue;
            Rpp(i, j) = R_full(gi, gj);
        }
    }
    return Rpp;
}

dcr::base::Vector solve_linear(const dcr::base::Matrix& A,
                               const dcr::base::Vector& b) {
    if (A.rows() == A.cols()) {
        Eigen::FullPivLU<dcr::base::Matrix> lu(A);
        if (lu.rank() == A.rows()) {
            return lu.solve(b);
        }
    }
    return A.colPivHouseholderQr().solve(b);
}

BgSubgroupSums summarize_background(const BoundaryPhaseResult& boundary,
                                    const dcr::base::Vector& population,
                                    const std::vector<dcr::atomic::EnergyLevel>& levels) {
    BgSubgroupSums s;
    for (int gi : boundary.P_indices) {
        if (gi < 0 || gi >= static_cast<int>(levels.size()) || gi >= population.size()) continue;
        const auto& lvl = levels[static_cast<size_t>(gi)];
        const double n = std::max(population(gi), 0.0);
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

double positive_sum_at_indices(const dcr::base::Vector& full,
                               const std::vector<int>& indices) {
    double s = 0.0;
    for (int gi : indices) {
        if (gi >= 0 && gi < full.size()) s += std::max(full(gi), 0.0);
    }
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

double elapsed_seconds_since(const std::chrono::steady_clock::time_point& start) {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
}

std::string format_elapsed_seconds(double seconds) {
    std::ostringstream out;
    out << std::fixed << std::setprecision(3) << seconds;
    return out.str();
}

struct KrylovSolveResult {
    dcr::base::Vector step;
    bool converged = false;
    int iterations = 0;
    int restarts = 0;
    double residual_norm = std::numeric_limits<double>::infinity();
};

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
            const dcr::base::Vector resid = gsmall - Hsmall * y;
            const double resid_norm = resid.norm();

            best_y = y;
            best_cols = j + 1;
            best_residual = resid_norm;
            out.iterations += 1;

            if (resid_norm <= tol * rhs_norm) {
                x.noalias() += V.leftCols(j + 1) * y;
                out.step = x;
                out.converged = true;
                out.residual_norm = resid_norm;
                out.restarts = restart;
                return out;
            }
        }

        if (best_cols > 0) {
            x.noalias() += V.leftCols(best_cols) * best_y;
            out.residual_norm = best_residual;
        }
    }

    out.step = x;
    return out;
}

RecyclingModel build_recycling_model(
    const BoundaryPhaseResult& boundary,
    const std::vector<dcr::atomic::EnergyLevel>& levels,
    const std::vector<int>& p_pos,
    double c_A,
    double c_A_base,
    double c_M_atom) {

    RecyclingModel model;
    model.ion_p_cols.reserve(boundary.ion_indices.size());
    model.recA_coeff.reserve(boundary.ion_indices.size());
    model.recM_coeff.reserve(boundary.ion_indices.size());

    for (int gi : boundary.ion_indices) {
        if (gi < 0 || gi >= static_cast<int>(p_pos.size())) continue;
        const int pi = p_pos[static_cast<size_t>(gi)];
        if (pi < 0) continue;

        const int mu_i = (gi >= 0 && gi < static_cast<int>(levels.size()))
            ? std::max(1, levels[static_cast<size_t>(gi)].atomicity)
            : 1;
        const bool molecular_ion = mu_i > 1;

        model.ion_p_cols.push_back(pi);
        // Molecular ions are assumed to neutralize and return as atoms only:
        // H2+ -> 2H, so the returned atomic density coefficient must conserve nuclei.
        model.recA_coeff.push_back(molecular_ion ? (c_A_base * static_cast<double>(mu_i)) : c_A);
        model.recM_coeff.push_back(molecular_ion ? 0.0 : c_M_atom);
    }

    return model;
}

void assign_flow_distribution(const BoundaryPhaseResult& boundary,
                              double n_A_total,
                              double n_M_total,
                              dcr::base::Vector& nA_vec,
                              dcr::base::Vector& nM_vec,
                              dcr::base::Vector& population) {
    if (boundary.explicit_recycling) {
        for (int gi : boundary.R_indices) {
            if (gi >= 0 && gi < population.size()) population(gi) = 0.0;
        }
        if (boundary.atom_ground >= 0 && boundary.atom_ground < population.size()) {
            population(boundary.atom_ground) = std::max(0.0, n_A_total);
        }
        if (boundary.molecule_ground >= 0 && boundary.molecule_ground < population.size()) {
            population(boundary.molecule_ground) = std::max(0.0, n_M_total);
        }
        return;
    }

    nA_vec.setZero();
    nM_vec.setZero();
    for (size_t j = 0; j < boundary.A_indices.size(); ++j) {
        if (boundary.A_indices[j] == boundary.atom_ground) {
            nA_vec(static_cast<int>(j)) = std::max(0.0, n_A_total);
            break;
        }
    }
    for (size_t j = 0; j < boundary.M_indices.size(); ++j) {
        if (boundary.M_indices[j] == boundary.molecule_ground) {
            nM_vec(static_cast<int>(j)) = std::max(0.0, n_M_total);
            break;
        }
    }
}

} // namespace

BoundaryPhaseResult run_boundary_phase(
    const dcr::io::Config& config,
    const dcr::atomic::AtomicData& atomic_data,
    dcr::state::PlasmaState& plasma,
    const EEDFGridView& grid,
    double ion_mass_amu,
    const dcr::physics::WallRecycling& wall) {

    BoundaryPhaseResult out;
    const auto boundary_timer_start = std::chrono::steady_clock::now();
    const auto boundary_temperatures = evaluate_plasma_temperatures(config, 0.0);

    const int total_states = atomic_data.get_total_states();
    const auto& levels = atomic_data.get_levels();
    out.population = dcr::base::Vector::Zero(total_states);

    // Speeds for recycling closure.
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
        // Config loader defaults mass_amu to 1.0 when unspecified.
        // Keep physical H2 default (2.0) unless user explicitly sets a larger value.
        if (sp.is_molecule && sp.charge == 0 && sp.mass_amu > 1.0) {
            out.molecule_mass_amu = sp.mass_amu;
        }
    }
    out.u_M = dcr::physics::calculate_thermal_speed(config.wall.temperature_eV, out.molecule_mass_amu);

    // Partition states.
    const auto& recycling_indices = atomic_data.get_recycling_indices();
    out.explicit_recycling = !recycling_indices.empty();
    out.is_recycling.assign(total_states, false);
    for (int idx : recycling_indices) {
        if (idx >= 0 && idx < total_states) out.is_recycling[static_cast<size_t>(idx)] = true;
    }

    out.P_indices.clear();
    out.R_indices.clear();
    for (int gi = 0; gi < total_states; ++gi) {
        if (out.is_recycling[static_cast<size_t>(gi)]) out.R_indices.push_back(gi);
        else out.P_indices.push_back(gi);
    }

    out.A_indices.clear();
    out.M_indices.clear();
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

    out.ion_indices.clear();
    out.atom_bg_indices.clear();
    out.mol_bg_indices.clear();
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

    const int Pn = static_cast<int>(out.P_indices.size());
    if (Pn == 0) {
        out.converged = true;
        out.iterations = 0;
        out.final_rel_change = 0.0;
        out.final_residual_rel = 0.0;
        out.elapsed_seconds = elapsed_seconds_since(boundary_timer_start);
        if (config.io.verbose_logging) {
            std::cout << "[DCR_Solver] Boundary skipped: no P states"
                      << " (wall=" << format_elapsed_seconds(out.elapsed_seconds) << " s).\n";
        }
        return out;
    }

    // Initial background from config IC.
    for (const auto& ic : config.plasma.initial_conditions) {
        if (ic.species_index < 0 || ic.species_index >= total_states) continue;
        if (out.explicit_recycling && out.is_recycling[static_cast<size_t>(ic.species_index)]) continue;
        out.population(ic.species_index) = ic.fraction * config.plasma.total_density;
    }

    dcr::base::Vector nP = dcr::base::Vector::Zero(Pn);
    dcr::base::Vector stoich = dcr::base::Vector::Zero(Pn);
    std::vector<int> p_pos(static_cast<size_t>(total_states), -1);
    for (int i = 0; i < Pn; ++i) {
        const int gi = out.P_indices[static_cast<size_t>(i)];
        p_pos[static_cast<size_t>(gi)] = i;
        stoich(i) = (gi >= 0 && gi < static_cast<int>(levels.size()))
            ? static_cast<double>(std::max(1, levels[static_cast<size_t>(gi)].atomicity))
            : 1.0;
        nP(i) = std::max(out.population(gi), 0.0);
    }

    const double n_nuclei0 = std::max(0.0, config.plasma.total_density);
    const double init_weighted = stoich.dot(nP);
    if (init_weighted > 0.0 && n_nuclei0 > 0.0) {
        nP *= (n_nuclei0 / init_weighted);
    } else if (n_nuclei0 > 0.0) {
        const double stoich_sum = std::max(1.0, stoich.sum());
        nP.setConstant(n_nuclei0 / stoich_sum);
    }

    const double mu_M = 2.0;
    const double c_A = (out.u_A > 0.0) ? (wall.alpha_atom * u_bohm / out.u_A) : 0.0;
    const double c_A_base = (out.u_A > 0.0) ? (u_bohm / out.u_A) : 0.0;
    const double c_M_atom = (out.u_M > 0.0) ? (wall.alpha_molecule * u_bohm / (mu_M * out.u_M)) : 0.0;
    const RecyclingModel rec_model = build_recycling_model(out, levels, p_pos, c_A, c_A_base, c_M_atom);

    const double c_s_A = dcr::physics::calculate_thermal_speed(
        config.plasma.neutral_atom_temperature_eV, out.atom_mass_amu);
    const double c_s_M = dcr::physics::calculate_thermal_speed(
        config.plasma.neutral_molecule_temperature_eV, out.molecule_mass_amu);
    const double w_local = std::max(config.grid.poloidal_width_cm, 1e-12);
    const double exA_coef = c_s_A / w_local;
    const double exM_coef = c_s_M / w_local;

    const int boundary_cap =
        (config.numerics.boundary_max_iterations > 0)
            ? config.numerics.boundary_max_iterations
            : config.numerics.max_iterations;
    const int max_iter = std::max(1, boundary_cap);
    const double boundary_tol =
        (config.numerics.boundary_tolerance > 0.0)
            ? config.numerics.boundary_tolerance
            : config.numerics.tolerance;
    const double tol = std::max(1e-14, boundary_tol);
    const double omega_raw = config.numerics.relaxation;
    double omega_iter = (omega_raw > 0.0 && omega_raw <= 1.0) ? omega_raw : 1.0;
    // The boundary fixed-point map can be much stiffer than the marching-cell
    // solve at low-Te operating points. A floor of 0.05 is large enough to
    // sustain a stable 2-cycle for those cases, so allow substantially smaller
    // damping before the line search gives up.
    const double omega_min = 1e-3;
    const double omega_max = 0.95;
    double prev_metric = std::numeric_limits<double>::infinity();
    double prev_rel_change = std::numeric_limits<double>::infinity();
    double prev_residual_rel = std::numeric_limits<double>::infinity();
    int rel_plateau_count = 0;
    int stagnation_count = 0;
    int rel_floor_count = 0;
    int tiny_step_count = 0;
    // Match the QSS path and require the strict tolerance for an accepted
    // boundary solve. Keep the diagnostics that detect stalls/floors, but do
    // not terminate early on them.
    const bool strict_boundary_only = true;

    dcr::base::Vector nA_vec = dcr::base::Vector::Zero(static_cast<int>(out.A_indices.size()));
    dcr::base::Vector nM_vec = dcr::base::Vector::Zero(static_cast<int>(out.M_indices.size()));

    // Boundary uses only Eq. (1.345) fixed-point form; Eq. (1.346) is marching-only.
    constexpr int constraint_row = 0;
    bool converged = false;
    int boundary_iterations = 0;
    double final_rel_change = std::numeric_limits<double>::infinity();
    double final_residual_rel = std::numeric_limits<double>::infinity();
    const std::string boundary_solver = config.numerics.boundary_solver.empty()
        ? "picard" : config.numerics.boundary_solver;

    if (boundary_solver == "newton_krylov_ptc") {
        struct BoundaryMapEvaluation {
            dcr::base::Vector x_projected;
            dcr::base::Vector x_image;
            dcr::base::Vector residual;
            dcr::base::Vector rhs_frozen;
            dcr::base::Matrix Rpp;
            double rel = std::numeric_limits<double>::infinity();
            double resid_rel = std::numeric_limits<double>::infinity();
            double flowA_total = 0.0;
            double flowM_total = 0.0;
            double target_bg_nuclei = 0.0;
        };

        auto compute_grouped = [&](const dcr::base::Vector& nP_state,
                                   double& n_I_weighted,
                                   double& n_a,
                                   double& n_m,
                                   double& n_A_total,
                                   double& n_M_total) {
            n_I_weighted = 0.0;
            for (int gi : out.ion_indices) {
                if (gi < 0 || gi >= static_cast<int>(levels.size())) continue;
                const int pi = p_pos[static_cast<size_t>(gi)];
                if (pi < 0) continue;
                const double mu_i = static_cast<double>(std::max(1, levels[static_cast<size_t>(gi)].atomicity));
                n_I_weighted += mu_i * std::max(nP_state(pi), 0.0);
            }
            n_a = 0.0;
            for (int gi : out.atom_bg_indices) {
                const int pi = (gi >= 0 && gi < static_cast<int>(p_pos.size())) ? p_pos[static_cast<size_t>(gi)] : -1;
                if (pi >= 0) n_a += std::max(nP_state(pi), 0.0);
            }
            n_m = 0.0;
            for (int gi : out.mol_bg_indices) {
                const int pi = (gi >= 0 && gi < static_cast<int>(p_pos.size())) ? p_pos[static_cast<size_t>(gi)] : -1;
                if (pi >= 0) n_m += std::max(nP_state(pi), 0.0);
            }
            n_A_total = 0.0;
            n_M_total = 0.0;
            for (size_t k = 0; k < rec_model.ion_p_cols.size(); ++k) {
                const int pi = rec_model.ion_p_cols[k];
                if (pi < 0 || pi >= nP_state.size()) continue;
                const double n_i = std::max(nP_state(pi), 0.0);
                n_A_total += rec_model.recA_coeff[k] * n_i;
                n_M_total += rec_model.recM_coeff[k] * n_i;
            }
            n_A_total = std::max(0.0, n_A_total);
            n_M_total = std::max(0.0, n_M_total);
        };

        auto project_state = [&](const dcr::base::Vector& x_in) {
            dcr::base::Vector x_proj = x_in.cwiseMax(0.0);
            double n_I_weighted = 0.0;
            double n_a = 0.0;
            double n_m = 0.0;
            double n_A_total = 0.0;
            double n_M_total = 0.0;
            compute_grouped(x_proj, n_I_weighted, n_a, n_m, n_A_total, n_M_total);
            const double total_guess_nuclei =
                std::max(0.0, n_I_weighted) +
                std::max(0.0, n_a) +
                mu_M * std::max(0.0, n_m) +
                std::max(0.0, n_A_total) +
                mu_M * std::max(0.0, n_M_total);
            if (n_nuclei0 > 0.0 && total_guess_nuclei > 0.0) {
                x_proj *= (n_nuclei0 / total_guess_nuclei);
            }
            return x_proj;
        };

        auto project_to_target_bg = [&](const dcr::base::Vector& x_in, double target_bg_nuclei) {
            dcr::base::Vector x_proj = x_in.cwiseMax(0.0);
            if (target_bg_nuclei <= 0.0) {
                x_proj.setZero();
                return x_proj;
            }
            const double constrained_now = stoich.dot(x_proj);
            if (constrained_now > 0.0) {
                x_proj *= (target_bg_nuclei / constrained_now);
            } else {
                const double stoich_sum = std::max(1.0, stoich.sum());
                x_proj.setConstant(target_bg_nuclei / stoich_sum);
            }
            return x_proj;
        };

        auto evaluate_map = [&](const dcr::base::Vector& x_in) {
            BoundaryMapEvaluation eval;
            eval.x_projected = project_state(x_in);

            for (int i = 0; i < Pn; ++i) {
                const int gi = out.P_indices[static_cast<size_t>(i)];
                out.population(gi) = eval.x_projected(i);
            }

            double n_I_weighted = 0.0;
            double n_a = 0.0;
            double n_m = 0.0;
            double n_A_total = 0.0;
            double n_M_total = 0.0;
            compute_grouped(eval.x_projected, n_I_weighted, n_a, n_m, n_A_total, n_M_total);
            eval.flowA_total = n_A_total;
            eval.flowM_total = n_M_total;

            dcr::base::Vector nA_tmp = dcr::base::Vector::Zero(static_cast<int>(out.A_indices.size()));
            dcr::base::Vector nM_tmp = dcr::base::Vector::Zero(static_cast<int>(out.M_indices.size()));
            assign_flow_distribution(out, n_A_total, n_M_total, nA_tmp, nM_tmp, out.population);

            dcr::base::Vector population_for_R = dcr::base::Vector::Zero(total_states);
            for (int i = 0; i < Pn; ++i) {
                const int gi = out.P_indices[static_cast<size_t>(i)];
                if (gi >= 0 && gi < total_states) {
                    population_for_R(gi) = std::max(eval.x_projected(i), 0.0);
                }
            }

            dcr::base::Matrix R_full = dcr::base::Matrix::Zero(total_states, total_states);
            const double ne_local = quasineutral_electron_density(population_for_R, levels);
            const LocalKineticContext plasma_local(
                plasma,
                grid,
                boundary_temperatures.electron_eV,
                boundary_temperatures.ion_eV,
                ne_local
            );
            for (const auto& proc : atomic_data.get_processes()) {
                if (!proc) continue;
                proc->apply(plasma_local.plasma(), plasma_local.grid(), population_for_R, R_full, nullptr);
            }

            eval.Rpp = extract_R_pp(R_full, out.P_indices);

            dcr::base::Vector S_bg = dcr::base::Vector::Zero(Pn);
            dcr::base::Matrix A = dcr::base::Matrix::Zero(Pn, Pn);
            for (int pi = 0; pi < Pn; ++pi) {
                const int gi = out.P_indices[static_cast<size_t>(pi)];
                if (gi < 0 || gi >= static_cast<int>(levels.size())) continue;

                const auto& row_lvl = levels[static_cast<size_t>(gi)];
                bool use_A_source = false;
                bool use_M_source = false;
                if (row_lvl.type == dcr::atomic::SpeciesType::Ion && row_lvl.charge > 0) {
                    use_A_source = true;
                    use_M_source = true;
                } else if (row_lvl.type == dcr::atomic::SpeciesType::Atom && row_lvl.charge == 0) {
                    use_A_source = false;
                    use_M_source = true;
                }

                double Si = 0.0;
                if (use_A_source) {
                    for (size_t j = 0; j < out.A_indices.size(); ++j) {
                        const int gj = out.A_indices[j];
                        if (gj < 0 || gj >= total_states) continue;
                        const double nA = out.explicit_recycling
                            ? std::max(out.population(gj), 0.0)
                            : ((static_cast<int>(j) < nA_tmp.size()) ? std::max(nA_tmp(static_cast<int>(j)), 0.0) : 0.0);
                        Si += R_full(gi, gj) * nA;
                    }
                }
                if (use_M_source) {
                    for (size_t j = 0; j < out.M_indices.size(); ++j) {
                        const int gj = out.M_indices[j];
                        if (gj < 0 || gj >= total_states) continue;
                        const double nM = out.explicit_recycling
                            ? std::max(out.population(gj), 0.0)
                            : ((static_cast<int>(j) < nM_tmp.size()) ? std::max(nM_tmp(static_cast<int>(j)), 0.0) : 0.0);
                        Si += R_full(gi, gj) * nM;
                    }
                }
                S_bg(pi) = Si;
            }

            const double source_nuclei_from_S = stoich.dot(S_bg);
            const double Gamma_ex_a_over_w = exA_coef * std::max(0.0, n_a);
            const double Gamma_ex_m_over_w = exM_coef * std::max(0.0, n_m);
            const double L_I = source_nuclei_from_S - (Gamma_ex_a_over_w + mu_M * Gamma_ex_m_over_w);
            const double L_a = Gamma_ex_a_over_w;
            const double L_m = Gamma_ex_m_over_w;

            if (n_I_weighted > 0.0) {
                for (int gi : out.ion_indices) {
                    if (gi < 0 || gi >= static_cast<int>(p_pos.size())) continue;
                    const int pi = p_pos[static_cast<size_t>(gi)];
                    if (pi >= 0) A(pi, pi) = L_I / n_I_weighted;
                }
            }
            if (n_a > 0.0) {
                const double coeff = L_a / n_a;
                for (int gi : out.atom_bg_indices) {
                    if (gi < 0 || gi >= static_cast<int>(p_pos.size())) continue;
                    const int pi = p_pos[static_cast<size_t>(gi)];
                    if (pi >= 0) A(pi, pi) = coeff;
                }
            }
            if (n_m > 0.0) {
                const double coeff = L_m / n_m;
                for (int gi : out.mol_bg_indices) {
                    if (gi < 0 || gi >= static_cast<int>(p_pos.size())) continue;
                    const int pi = p_pos[static_cast<size_t>(gi)];
                    if (pi >= 0) A(pi, pi) = coeff;
                }
            }

            eval.rhs_frozen = A * eval.x_projected - S_bg;
            const double flow_nuclei = std::max(0.0, n_A_total) + mu_M * std::max(0.0, n_M_total);
            eval.target_bg_nuclei = std::max(0.0, n_nuclei0 - flow_nuclei);

            dcr::base::Matrix C = eval.Rpp;
            dcr::base::Vector rhs = eval.rhs_frozen;
            C.row(constraint_row) = stoich.transpose();
            rhs(constraint_row) = eval.target_bg_nuclei;
            dcr::base::Vector x_candidate = solve_linear(C, rhs);
            if (!x_candidate.allFinite()) {
                x_candidate = eval.x_projected;
            }

            eval.x_image = project_to_target_bg(x_candidate, eval.target_bg_nuclei);
            eval.residual = eval.x_projected - eval.x_image;
            const double diff = (eval.x_image - eval.x_projected).cwiseAbs().maxCoeff();
            const double scale = std::max(1.0, eval.x_projected.cwiseAbs().maxCoeff());
            eval.rel = diff / scale;
            const dcr::base::Vector lhs = eval.Rpp * eval.x_image;
            const dcr::base::Vector residual_linear = lhs - eval.rhs_frozen;
            eval.resid_rel = residual_linear.norm() / std::max(1.0, lhs.norm());
            return eval;
        };

        double tau = std::clamp(
            config.numerics.boundary_ptc_tau_init,
            1.0e-6,
            std::max(config.numerics.boundary_ptc_tau_init,
                     config.numerics.boundary_ptc_tau_max)
        );
        const double tau_max = std::max(tau, config.numerics.boundary_ptc_tau_max);
        const double alpha_min = std::clamp(config.numerics.boundary_nk_alpha_min, 1.0e-6, 0.5);
        dcr::base::Vector x_work = project_state(nP);

        for (int iter = 0; iter < max_iter; ++iter) {
            const auto eval = evaluate_map(x_work);
            if (config.io.verbose_logging) {
                BgSubgroupSums bg_iter;
                for (int pi = 0; pi < Pn; ++pi) {
                    const int gi = out.P_indices[static_cast<size_t>(pi)];
                    if (gi < 0 || gi >= static_cast<int>(levels.size())) continue;
                    const auto& lvl = levels[static_cast<size_t>(gi)];
                    const double n = std::max(eval.x_image(pi), 0.0);
                    if (lvl.type == dcr::atomic::SpeciesType::Ion && lvl.charge > 0) {
                        if (lvl.atomicity >= 2) bg_iter.H2_plus += n;
                        else bg_iter.H_plus += n;
                    } else if (lvl.type == dcr::atomic::SpeciesType::Ion && lvl.charge < 0) {
                        bg_iter.H_minus += n;
                    } else if (lvl.type == dcr::atomic::SpeciesType::Atom && lvl.charge == 0) {
                        bg_iter.H += n;
                    } else if (lvl.type == dcr::atomic::SpeciesType::Molecule && lvl.charge == 0) {
                        bg_iter.H2 += n;
                    }
                }
                std::cout << "[DCR_Solver] Boundary iter " << (iter + 1)
                          << " T{e=" << boundary_temperatures.electron_eV
                          << ", i=" << boundary_temperatures.ion_eV
                          << "} solver=NK-PTC"
                          << " rel_change=" << eval.rel
                          << " ||F||=" << eval.residual.norm()
                          << " resid_rel(diag)=" << eval.resid_rel
                          << " tau=" << tau
                          << " bg{H+=" << bg_iter.H_plus
                          << ", H=" << bg_iter.H
                          << ", H2=" << bg_iter.H2
                          << ", H2+=" << bg_iter.H2_plus
                          << ", H-=" << bg_iter.H_minus
                          << "} flow{A=" << eval.flowA_total
                          << ", M=" << eval.flowM_total
                          << "}\n";
            }

            final_rel_change = eval.rel;
            final_residual_rel = eval.resid_rel;
            boundary_iterations = iter + 1;

            if (eval.rel < tol) {
                nP = eval.x_image;
                converged = true;
                if (config.io.verbose_logging) {
                    std::cout << "[DCR_Solver] Boundary converged in " << (iter + 1)
                              << " iterations"
                              << " (wall=" << format_elapsed_seconds(elapsed_seconds_since(boundary_timer_start))
                              << " s).\n";
                }
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
                    config.numerics.boundary_nk_fd_eps,
                    config.numerics.boundary_nk_fd_eps *
                        (1.0 + eval.x_projected.norm()) / std::max(1.0, v.norm())
                );
                const auto pert = evaluate_map(eval.x_projected + eps * v);
                return (1.0 / tau) * v + (pert.residual - eval.residual) / eps;
            };

            auto gmres = gmres_solve_matrix_free(
                apply_A,
                -eval.residual,
                config.numerics.boundary_nk_krylov_dim,
                config.numerics.boundary_nk_max_restarts,
                linear_tol
            );

            dcr::base::Vector delta = gmres.step;
            if (!delta.allFinite() || delta.norm() == 0.0) {
                delta = 0.1 * (eval.x_image - eval.x_projected);
                tau = std::max(1.0e-6, 0.5 * tau);
            }

            const double norm0 = std::max(1.0e-30, eval.residual.norm());
            const double alpha_pos = fraction_to_boundary_alpha(eval.x_projected, delta, 0.99);
            double alpha = std::min(1.0, alpha_pos);
            bool accepted = false;
            BoundaryMapEvaluation accepted_eval;
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
                alpha = std::min(0.1, std::max(alpha_min, 0.05 * tau));
                accepted_eval = evaluate_map(
                    eval.x_projected + alpha * (eval.x_image - eval.x_projected)
                );
                tau = std::max(1.0e-6, 0.5 * tau);
                if (config.io.verbose_logging) {
                    std::cout << "[DCR_Solver] Boundary: NK line search failed, falling back to conservative Picard step"
                              << " alpha=" << alpha
                              << " alpha_pos=" << alpha_pos
                              << " tau=" << tau
                              << "\n";
                }
            } else if (alpha >= 0.75 && gmres.converged) {
                tau = std::min(tau_max, 1.5 * tau);
            } else if (alpha < 0.25 || !gmres.converged) {
                tau = std::max(1.0e-6, 0.5 * tau);
            }

            x_work = accepted_eval.x_projected;
            if (config.io.verbose_logging) {
                std::cout << "[DCR_Solver] Boundary: NK step"
                          << " alpha=" << alpha
                          << " gmres_iters=" << gmres.iterations
                          << " gmres_resid=" << gmres.residual_norm
                          << " alpha_pos=" << alpha_pos
                          << " tau_next=" << tau
                          << "\n";
            }

            if (iter == max_iter - 1 && config.io.verbose_logging) {
                std::cout << "[DCR_Solver] Boundary reached max iterations without convergence"
                          << " (wall=" << format_elapsed_seconds(elapsed_seconds_since(boundary_timer_start))
                          << " s).\n";
            }
        }

        if (!converged) {
            nP = project_state(x_work);
        }
    } else for (int iter = 0; iter < max_iter; ++iter) {
        // Write current background iterate into full population vector.
        for (int i = 0; i < Pn; ++i) {
            const int gi = out.P_indices[static_cast<size_t>(i)];
            out.population(gi) = nP(i);
        }

        // Grouped background totals.
        double n_I_weighted = 0.0;
        for (int gi : out.ion_indices) {
            if (gi < 0 || gi >= static_cast<int>(levels.size())) continue;
            const int pi = p_pos[static_cast<size_t>(gi)];
            if (pi < 0) continue;
            const double mu_i = static_cast<double>(std::max(1, levels[static_cast<size_t>(gi)].atomicity));
            n_I_weighted += mu_i * std::max(nP(pi), 0.0);
        }
        double n_a = 0.0;
        for (int gi : out.atom_bg_indices) {
            const int pi = (gi >= 0 && gi < static_cast<int>(p_pos.size())) ? p_pos[static_cast<size_t>(gi)] : -1;
            if (pi >= 0) n_a += std::max(nP(pi), 0.0);
        }
        double n_m = 0.0;
        for (int gi : out.mol_bg_indices) {
            const int pi = (gi >= 0 && gi < static_cast<int>(p_pos.size())) ? p_pos[static_cast<size_t>(gi)] : -1;
            if (pi >= 0) n_m += std::max(nP(pi), 0.0);
        }

        // Recycling flow closure from current ion populations (not Eq. 1.346).
        double n_A_total = 0.0;
        double n_M_total = 0.0;
        for (size_t k = 0; k < rec_model.ion_p_cols.size(); ++k) {
            const int pi = rec_model.ion_p_cols[k];
            if (pi < 0 || pi >= nP.size()) continue;
            const double n_i = std::max(nP(pi), 0.0);
            n_A_total += rec_model.recA_coeff[k] * n_i;
            n_M_total += rec_model.recM_coeff[k] * n_i;
        }
        n_A_total = std::max(0.0, n_A_total);
        n_M_total = std::max(0.0, n_M_total);

        // Consistent per-iteration normalization:
        // scale background and flow together so (bg + flow) nuclei stays at n_nuclei0.
        if (n_nuclei0 > 0.0) {
            const double total_guess_nuclei =
                std::max(0.0, n_I_weighted) +
                std::max(0.0, n_a) +
                mu_M * std::max(0.0, n_m) +
                std::max(0.0, n_A_total) +
                mu_M * std::max(0.0, n_M_total);
            if (total_guess_nuclei > 0.0) {
                const double scale_state = n_nuclei0 / total_guess_nuclei;
                if (std::isfinite(scale_state) && scale_state > 0.0 && std::abs(scale_state - 1.0) > 1e-14) {
                    nP *= scale_state;
                    n_I_weighted *= scale_state;
                    n_a *= scale_state;
                    n_m *= scale_state;
                    n_A_total *= scale_state;
                    n_M_total *= scale_state;
                    for (int i = 0; i < Pn; ++i) {
                        const int gi = out.P_indices[static_cast<size_t>(i)];
                        out.population(gi) = nP(i);
                    }
                }
            }
        }

        assign_flow_distribution(out, n_A_total, n_M_total, nA_vec, nM_vec, out.population);

        // Build population used for process rates with background-only density.
        // Recycling-flow populations are excluded from R-assembly by request.
        dcr::base::Vector population_for_R = dcr::base::Vector::Zero(total_states);
        for (int i = 0; i < Pn; ++i) {
            const int gi = out.P_indices[static_cast<size_t>(i)];
            if (gi >= 0 && gi < total_states) {
                population_for_R(gi) = std::max(nP(i), 0.0);
            }
        }

        // Assemble full CR matrix R(population).
        dcr::base::Matrix R_full = dcr::base::Matrix::Zero(total_states, total_states);
        const double ne_local = quasineutral_electron_density(population_for_R, levels);
        const LocalKineticContext plasma_local(
            plasma,
            grid,
            boundary_temperatures.electron_eV,
            boundary_temperatures.ion_eV,
            ne_local
        );
        for (const auto& proc : atomic_data.get_processes()) {
            if (!proc) continue;
            proc->apply(plasma_local.plasma(), plasma_local.grid(), population_for_R, R_full, nullptr);
        }

        const dcr::base::Matrix Rpp = extract_R_pp(R_full, out.P_indices);

        // Test mode requested: comment out all LHS(A) and S terms for boundary.
        // This makes the boundary linear solve:
        //   R_PP n^{k+1} = 0, with one row replaced by the nuclei closure.
        const bool disable_boundary_lhs_and_s = false;

        dcr::base::Vector S_bg = dcr::base::Vector::Zero(Pn);
        dcr::base::Matrix A = dcr::base::Matrix::Zero(Pn, Pn);
        if (!disable_boundary_lhs_and_s) {
            // Recycling source S on background rows.
            for (int pi = 0; pi < Pn; ++pi) {
                const int gi = out.P_indices[static_cast<size_t>(pi)];
                if (gi < 0 || gi >= static_cast<int>(levels.size())) continue;

                const auto& row_lvl = levels[static_cast<size_t>(gi)];
                bool use_A_source = false;
                bool use_M_source = false;
                if (row_lvl.type == dcr::atomic::SpeciesType::Ion && row_lvl.charge > 0) {
                    use_A_source = true;
                    use_M_source = true;
                } else if (row_lvl.type == dcr::atomic::SpeciesType::Atom && row_lvl.charge == 0) {
                    use_A_source = false;
                    use_M_source = true;
                } else if (row_lvl.type == dcr::atomic::SpeciesType::Molecule && row_lvl.charge == 0) {
                    use_A_source = false;
                    use_M_source = false;
                }

                double Si = 0.0;
                if (use_A_source) {
                    for (size_t j = 0; j < out.A_indices.size(); ++j) {
                        const int gj = out.A_indices[j];
                        if (gj < 0 || gj >= total_states) continue;
                        const double nA = out.explicit_recycling
                            ? std::max(out.population(gj), 0.0)
                            : ((static_cast<int>(j) < nA_vec.size()) ? std::max(nA_vec(static_cast<int>(j)), 0.0) : 0.0);
                        Si += R_full(gi, gj) * nA;
                    }
                }
                if (use_M_source) {
                    for (size_t j = 0; j < out.M_indices.size(); ++j) {
                        const int gj = out.M_indices[j];
                        if (gj < 0 || gj >= total_states) continue;
                        const double nM = out.explicit_recycling
                            ? std::max(out.population(gj), 0.0)
                            : ((static_cast<int>(j) < nM_vec.size()) ? std::max(nM_vec(static_cast<int>(j)), 0.0) : 0.0);
                        Si += R_full(gi, gj) * nM;
                    }
                }
                S_bg(pi) = Si;
            }

            // Eq. (1.345) grouped transport coefficients.
            // Keep only local exhaust terms (a,m). Do not include recycling-flow exhaust (A,M).
            const double source_nuclei_from_S = stoich.dot(S_bg);
            const double Gamma_ex_a_over_w = exA_coef * std::max(0.0, n_a);
            const double Gamma_ex_m_over_w = exM_coef * std::max(0.0, n_m);
            const double L_I = source_nuclei_from_S - (Gamma_ex_a_over_w + mu_M * Gamma_ex_m_over_w);
            const double L_a = Gamma_ex_a_over_w;
            const double L_m = Gamma_ex_m_over_w;

            if (n_I_weighted > 0.0) {
                for (int gi : out.ion_indices) {
                    if (gi < 0 || gi >= static_cast<int>(p_pos.size())) continue;
                    const int pi = p_pos[static_cast<size_t>(gi)];
                    if (pi >= 0) A(pi, pi) = L_I / n_I_weighted;
                }
            }
            if (n_a > 0.0) {
                const double coeff = L_a / n_a;
                for (int gi : out.atom_bg_indices) {
                    if (gi < 0 || gi >= static_cast<int>(p_pos.size())) continue;
                    const int pi = p_pos[static_cast<size_t>(gi)];
                    if (pi >= 0) A(pi, pi) = coeff;
                }
            }
            if (n_m > 0.0) {
                const double coeff = L_m / n_m;
                for (int gi : out.mol_bg_indices) {
                    if (gi < 0 || gi >= static_cast<int>(p_pos.size())) continue;
                    const int pi = p_pos[static_cast<size_t>(gi)];
                    if (pi >= 0) A(pi, pi) = coeff;
                }
            }
        }

        // Boundary fixed-point linear solve in R n = T - S form:
        //   R_PP n^{k+1} = A(n^k) n^k - S(n^k),
        // with A and S frozen from the current iterate.
        const dcr::base::Vector rhs_frozen = A * nP - S_bg;
        dcr::base::Matrix C = Rpp;
        dcr::base::Vector rhs = rhs_frozen;

        const double flow_nuclei = std::max(0.0, n_A_total) + mu_M * std::max(0.0, n_M_total);
        const double target_bg_nuclei = std::max(0.0, n_nuclei0 - flow_nuclei);

        C.row(constraint_row) = stoich.transpose();
        rhs(constraint_row) = target_bg_nuclei;

        dcr::base::Vector nP_candidate = solve_linear(C, rhs);
        if (!nP_candidate.allFinite()) {
            if (config.io.verbose_logging) {
                std::cout << "[DCR_Solver] Boundary linear solve produced non-finite values at iter "
                          << (iter + 1) << ".\n";
            }
            break;
        }

        const dcr::base::Vector xk = nP;
        const dcr::base::Vector nP_target = nP_candidate;
        dcr::base::Vector nP_new = nP;
        dcr::base::Vector ax_minus_s = dcr::base::Vector::Zero(Pn);
        double rel_change = std::numeric_limits<double>::infinity();
        double residual_rel = std::numeric_limits<double>::infinity();
        double constraint_err = std::numeric_limits<double>::infinity();
        double omega_used = std::clamp(omega_iter, omega_min, omega_max);
        double accepted_metric = std::numeric_limits<double>::infinity();
        bool had_negative_any = false;
        int accepted_ls = -1;

        // Adaptive damping: try larger step when smooth, back off on stalls/oscillation.
        int ls = 0;
        while (ls < 8) {
            dcr::base::Vector nP_trial = nP + omega_used * (nP_target - nP);

            // Positivity floor on trial update:
            // if any component crosses below zero, replace that trial component
            // by half of the previous iterate value.
            bool had_negative_component = false;
            for (int i = 0; i < nP_trial.size(); ++i) {
                if (nP_trial(i) < 0.0) {
                    nP_trial(i) = 0.5 * std::max(nP(i), 0.0);
                    had_negative_component = true;
                }
            }
            had_negative_any = had_negative_any || had_negative_component;

            // Enforce the nuclei closure exactly after relaxation.
            if (target_bg_nuclei <= 0.0) {
                nP_trial.setZero();
            } else {
                const double constrained_now = stoich.dot(nP_trial);
                if (constrained_now > 0.0) {
                    nP_trial *= (target_bg_nuclei / constrained_now);
                }
            }

            const double diff_trial = (nP_trial - nP).cwiseAbs().maxCoeff();
            const double scale_trial = std::max(1.0, nP.cwiseAbs().maxCoeff());
            const double rel_change_trial = diff_trial / scale_trial;

            const dcr::base::Vector ax_minus_s_trial = rhs_frozen;
            const dcr::base::Vector lhs_trial = Rpp * nP_trial;
            const dcr::base::Vector residual_trial = lhs_trial - rhs_frozen;
            const double residual_rel_trial = residual_trial.norm() / std::max(1.0, lhs_trial.norm());
            const double constraint_err_trial = std::abs(stoich.dot(nP_trial) - target_bg_nuclei);
            const double metric_trial = std::max(rel_change_trial, residual_rel_trial);

            const bool accept_trial =
                !std::isfinite(prev_metric) ||
                metric_trial <= prev_metric * 1.02 ||
                omega_used <= omega_min * 1.001 ||
                ls == 7;

            if (accept_trial) {
                nP_new = nP_trial;
                ax_minus_s = ax_minus_s_trial;
                rel_change = rel_change_trial;
                residual_rel = residual_rel_trial;
                constraint_err = constraint_err_trial;
                accepted_metric = metric_trial;
                accepted_ls = ls;
                break;
            }

            if (had_negative_component) {
                omega_used = std::max(omega_min, 0.5 * omega_used);
            } else {
                omega_used = std::max(omega_min, 0.5 * omega_used);
            }
            ++ls;
        }

        if (std::isfinite(prev_metric)) {
            if (accepted_metric < 0.7 * prev_metric) {
                omega_iter = std::min(omega_max, omega_used * 1.2);
            } else if (accepted_metric > 1.05 * prev_metric) {
                omega_iter = std::max(omega_min, omega_used * 0.7);
            } else {
                omega_iter = std::clamp(omega_used, omega_min, omega_max);
            }
        } else {
            omega_iter = std::min(omega_max, omega_used * 1.1);
        }
        // If step became too small and trial was clean, grow it back adaptively.
        // This avoids getting stuck at tiny damping after temporary negativity.
        if (!had_negative_any && accepted_ls == 0 &&
            rel_change < std::max(1e-6, 10.0 * tol)) {
            omega_iter = std::min(omega_max, std::max(omega_iter, 1.25 * omega_used));
        }
        if (!had_negative_any && omega_iter <= (1.05 * omega_min) &&
            rel_change < std::max(1e-6, 50.0 * tol)) {
            omega_iter = std::min(omega_max, 1.5 * omega_iter);
        }
        prev_metric = accepted_metric;

        // If rel_change plateaus, nudge omega upward to escape very slow damping.
        if (std::isfinite(prev_rel_change)) {
            const double rel_band = 0.02 * std::max(1e-16, prev_rel_change);
            if (std::abs(rel_change - prev_rel_change) <= rel_band) {
                ++rel_plateau_count;
            } else {
                rel_plateau_count = 0;
            }
        }
        if (rel_plateau_count >= 6) {
            // Plateau at high damping often indicates a near-2-cycle/floor:
            // back off omega to re-damp instead of pushing higher.
            if (omega_iter > 0.2) {
                omega_iter = std::max(omega_min, omega_iter * 0.6);
            } else {
                omega_iter = std::min(omega_max, omega_iter * 1.15);
            }
            rel_plateau_count = 0;
        }
        const bool rel_stalled =
            std::isfinite(prev_rel_change) &&
            std::abs(rel_change - prev_rel_change) <= 0.01 * std::max(1e-16, prev_rel_change);
        const bool resid_stalled =
            std::isfinite(prev_residual_rel) &&
            std::abs(residual_rel - prev_residual_rel) <= 0.01 * std::max(1e-16, prev_residual_rel);
        if (rel_stalled && resid_stalled) {
            ++stagnation_count;
        } else {
            stagnation_count = 0;
        }
        // Additional plateau stop: if rel_change sits near a floor for many iterations,
        // stop and emit the final boundary summary.
        if (std::isfinite(prev_rel_change)) {
            const double rel_ref = std::max(1e-16, std::max(rel_change, prev_rel_change));
            const double rel_delta = std::abs(rel_change - prev_rel_change);
            const bool near_floor = rel_change <= std::max(20.0 * tol, 1e-10);
            const bool almost_constant = rel_delta <= 0.005 * rel_ref; // <=0.5% drift
            if (near_floor && almost_constant) {
                ++rel_floor_count;
            } else {
                rel_floor_count = 0;
            }
        } else {
            rel_floor_count = 0;
        }
        prev_rel_change = rel_change;
        prev_residual_rel = residual_rel;
        final_rel_change = rel_change;
        final_residual_rel = residual_rel;
        boundary_iterations = iter + 1;

        if (config.io.verbose_logging) {
            const double bg_nuclei_iter = stoich.dot(nP_new);
            const double flow_nuclei_iter =
                std::max(0.0, n_A_total) + mu_M * std::max(0.0, n_M_total);
            const double nuclei_total_iter = bg_nuclei_iter + flow_nuclei_iter;
            const double cons_rel_err =
                std::abs(nuclei_total_iter - n_nuclei0) / std::max(1.0, n_nuclei0);
            const double ls_constraint_rel =
                constraint_err / std::max(1.0, n_nuclei0);
            BgSubgroupSums bg_iter;
            for (int pi = 0; pi < Pn; ++pi) {
                const int gi = out.P_indices[static_cast<size_t>(pi)];
                if (gi < 0 || gi >= static_cast<int>(levels.size())) continue;
                const auto& lvl = levels[static_cast<size_t>(gi)];
                const double n = std::max(nP_new(pi), 0.0);
                if (lvl.type == dcr::atomic::SpeciesType::Ion && lvl.charge > 0) {
                    if (lvl.atomicity >= 2) bg_iter.H2_plus += n;
                    else bg_iter.H_plus += n;
                } else if (lvl.type == dcr::atomic::SpeciesType::Ion && lvl.charge < 0) {
                    bg_iter.H_minus += n;
                } else if (lvl.type == dcr::atomic::SpeciesType::Atom && lvl.charge == 0) {
                    bg_iter.H += n;
                } else if (lvl.type == dcr::atomic::SpeciesType::Molecule && lvl.charge == 0) {
                    bg_iter.H2 += n;
                }
            }

            std::cout << "[DCR_Solver] Boundary iter " << (iter + 1)
                      << " T{e=" << boundary_temperatures.electron_eV
                      << ", i=" << boundary_temperatures.ion_eV
                      << "}"
                      << " rel_change=" << rel_change
                      << " omega=" << omega_used
                      << " omega_next=" << omega_iter
                      << " ls_constraint_err=" << constraint_err
                      << " ls_constraint_rel=" << ls_constraint_rel
                      << " constrained_val=" << (target_bg_nuclei > 0.0 ? stoich.dot(nP_new) / target_bg_nuclei : 0.0)
                      << " wsum(Ax-S)=" << stoich.dot(ax_minus_s)
                      << " wsumabs(Ax-S)=" << (stoich.array() * ax_minus_s.cwiseAbs().array()).sum()
                      << " residual_rel(diag)=" << residual_rel
                      << " cons_rel_err=" << cons_rel_err
                      << " bg{H+=" << bg_iter.H_plus
                      << ", H=" << bg_iter.H
                      << ", H2=" << bg_iter.H2
                      << ", H2+=" << bg_iter.H2_plus
                      << ", H-=" << bg_iter.H_minus
                      << "} flow{A=" << n_A_total
                      << ", M=" << n_M_total
                      << "} nuclei{bg=" << bg_nuclei_iter
                      << ", flow=" << flow_nuclei_iter
                      << ", total=" << nuclei_total_iter
                      << ", target=" << n_nuclei0
                      << "}\n";
        }

        nP = nP_new;
        if (rel_change < tol) {
            converged = true;
            if (config.io.verbose_logging) {
                std::cout << "[DCR_Solver] Boundary converged in " << (iter + 1)
                          << " iterations"
                          << " (wall=" << format_elapsed_seconds(elapsed_seconds_since(boundary_timer_start))
                          << " s).\n";
            }
            break;
        }
        if (!strict_boundary_only && stagnation_count >= 40) {
            converged = true;
            if (config.io.verbose_logging) {
                std::cout << "[DCR_Solver] Boundary stagnated near residual floor; stopping at iter "
                          << (iter + 1)
                          << " (rel_change=" << rel_change
                          << ", residual_rel=" << residual_rel
                          << ", wall=" << format_elapsed_seconds(elapsed_seconds_since(boundary_timer_start))
                          << " s).\n";
            }
            break;
        }
        if (!strict_boundary_only && rel_floor_count >= 80) {
            converged = true;
            if (config.io.verbose_logging) {
                std::cout << "[DCR_Solver] Boundary rel_change plateau; stopping at iter "
                          << (iter + 1)
                          << " (rel_change=" << rel_change
                          << ", residual_rel=" << residual_rel
                          << ", wall=" << format_elapsed_seconds(elapsed_seconds_since(boundary_timer_start))
                          << " s).\n";
            }
            break;
        }
        const double tiny_step_tol = std::max(2e-7, 20.0 * tol);
        const double tiny_resid_tol = std::max(1e-4, 1000.0 * tol);
        if (rel_change <= tiny_step_tol && residual_rel <= tiny_resid_tol) {
            ++tiny_step_count;
        } else {
            tiny_step_count = 0;
        }
        if (!strict_boundary_only && tiny_step_count >= 40) {
            converged = true;
            if (config.io.verbose_logging) {
                std::cout << "[DCR_Solver] Boundary tiny-step floor reached; stopping at iter "
                          << (iter + 1)
                          << " (rel_change=" << rel_change
                          << ", residual_rel=" << residual_rel
                          << ", wall=" << format_elapsed_seconds(elapsed_seconds_since(boundary_timer_start))
                          << " s).\n";
            }
            break;
        }

        if (iter == max_iter - 1 && config.io.verbose_logging) {
            std::cout << "[DCR_Solver] Boundary reached max iterations without convergence"
                      << " (wall=" << format_elapsed_seconds(elapsed_seconds_since(boundary_timer_start))
                      << " s).\n";
        }
    }

    out.converged = converged;
    out.iterations = boundary_iterations;
    out.final_rel_change = final_rel_change;
    out.final_residual_rel = final_residual_rel;
    out.elapsed_seconds = elapsed_seconds_since(boundary_timer_start);

    // Final output projection.
    for (int gi : out.R_indices) {
        if (gi >= 0 && gi < out.population.size()) out.population(gi) = 0.0;
    }
    for (int i = 0; i < Pn; ++i) {
        const int gi = out.P_indices[static_cast<size_t>(i)];
        out.population(gi) = std::max(0.0, nP(i));
    }

    double n_A_total_final = 0.0;
    double n_M_total_final = 0.0;
    for (size_t k = 0; k < rec_model.ion_p_cols.size(); ++k) {
        const int pi = rec_model.ion_p_cols[k];
        if (pi < 0 || pi >= nP.size()) continue;
        const double n_i = std::max(nP(pi), 0.0);
        n_A_total_final += rec_model.recA_coeff[k] * n_i;
        n_M_total_final += rec_model.recM_coeff[k] * n_i;
    }
    n_A_total_final = std::max(0.0, n_A_total_final);
    n_M_total_final = std::max(0.0, n_M_total_final);

    if (out.explicit_recycling) {
        if (out.atom_ground >= 0 && out.atom_ground < out.population.size()) {
            out.population(out.atom_ground) = n_A_total_final;
        }
        if (out.molecule_ground >= 0 && out.molecule_ground < out.population.size()) {
            out.population(out.molecule_ground) = n_M_total_final;
        }
        out.have_flow_last = false;
    } else {
        out.flowA_last = dcr::base::Vector::Zero(static_cast<int>(out.A_indices.size()));
        out.flowM_last = dcr::base::Vector::Zero(static_cast<int>(out.M_indices.size()));
        for (size_t i = 0; i < out.A_indices.size(); ++i) {
            if (out.A_indices[i] == out.atom_ground) {
                out.flowA_last(static_cast<int>(i)) = n_A_total_final;
                break;
            }
        }
        for (size_t i = 0; i < out.M_indices.size(); ++i) {
            if (out.M_indices[i] == out.molecule_ground) {
                out.flowM_last(static_cast<int>(i)) = n_M_total_final;
                break;
            }
        }
        out.have_flow_last = true;
    }

    if (config.io.verbose_logging) {
        const auto bg = summarize_background(out, out.population, levels);
        const double flowA_total = out.explicit_recycling
            ? positive_sum_at_indices(out.population, out.A_indices)
            : positive_sum(out.flowA_last);
        const double flowM_total = out.explicit_recycling
            ? positive_sum_at_indices(out.population, out.M_indices)
            : positive_sum(out.flowM_last);

        std::cout << "[DCR_Solver] Boundary x=0 cm"
                  << " T{e=" << boundary_temperatures.electron_eV
                  << ", i=" << boundary_temperatures.ion_eV
                  << "}"
                  << " bg{H+=" << bg.H_plus
                  << ", H=" << bg.H
                  << ", H2=" << bg.H2
                  << ", H2+=" << bg.H2_plus
                  << ", H-=" << bg.H_minus
                  << "} flow{A=" << flowA_total
                  << ", M=" << flowM_total
                  << "}\n";
    }

    return out;
}

} // namespace dcr::solver
