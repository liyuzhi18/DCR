#include "BoundaryPhase.hpp"
#include "../../physics/Sheath.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <limits>

namespace dcr::solver {

namespace {

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

struct BgSubgroupSums {
    double H_plus = 0.0;
    double H = 0.0;
    double H2 = 0.0;
    double H2_plus = 0.0;
    double H_minus = 0.0;
};

// Group the background states into compact hydrogen buckets for log readability.
BgSubgroupSums summarize_boundary_background(const BoundaryPhaseResult& boundary,
                                            const dcr::base::Vector& population,
                                            const std::vector<dcr::atomic::EnergyLevel>& levels) {
    BgSubgroupSums s;
    for (int gi : boundary.P_indices) {
        if (gi < 0 || gi >= static_cast<int>(levels.size()) || gi >= population.size()) continue;
        const auto& lvl = levels[gi];
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

struct NewtonPolishResult {
    dcr::base::Vector x;
    double residual_rel = std::numeric_limits<double>::infinity();
    double constraint_rel = std::numeric_limits<double>::infinity();
    bool improved = false;
    int iterations = 0;
};

NewtonPolishResult polish_with_newton_soft_constraint(
    const dcr::base::Matrix& M,
    const dcr::base::Vector& rhs,
    const dcr::base::Vector& constraint,
    double target,
    const dcr::base::Vector& x_seed,
    double constraint_rel_max) {

    NewtonPolishResult out;
    out.x = x_seed;
    if (M.rows() == 0 || M.cols() == 0 || rhs.size() == 0 || constraint.size() == 0) {
        return out;
    }

    auto evaluate = [&](const dcr::base::Vector& x_eval) {
        const double r_norm = (M * x_eval - rhs).norm();
        const double lhs_norm = (M * x_eval).norm();
        const double r_rel = r_norm / std::max(1.0, lhs_norm);
        const double c_rel = std::abs(constraint.dot(x_eval) - target) / std::max(1.0, std::abs(target));
        return std::pair<double, double>{r_rel, c_rel};
    };

    const auto base_eval = evaluate(x_seed);
    double best_resid_rel = base_eval.first;
    double best_constraint_rel = base_eval.second;
    out.residual_rel = best_resid_rel;
    out.constraint_rel = best_constraint_rel;

    const dcr::base::Matrix MtM = M.transpose() * M;
    const double mtm_scale = std::max(1.0, MtM.diagonal().cwiseAbs().maxCoeff());
    const dcr::base::Matrix ssT = constraint * constraint.transpose();
    const double target_resid_rel = 1e-5;
    int total_newton_iters = 0;

    // Post-convergence Newton polish with progressively relaxed closure penalty.
    const std::array<double, 10> lambda_factors = {1e8, 1e7, 1e6, 1e5, 1e4, 1e3, 1e2, 1e1, 1.0, 1e-1};
    for (double fac : lambda_factors) {
        if (best_resid_rel <= target_resid_rel && best_constraint_rel <= constraint_rel_max) break;
        const double lambda = fac * mtm_scale;
        const dcr::base::Matrix H = MtM + lambda * ssT;
        dcr::base::Vector x_it = out.x;
        for (int it = 0; it < 30; ++it) {
            const dcr::base::Vector r = M * x_it - rhs;
            const double c_err = constraint.dot(x_it) - target;
            const dcr::base::Vector grad = M.transpose() * r + lambda * c_err * constraint;
            dcr::base::Vector delta = solve_replaced_system(H, grad);
            if (!delta.allFinite()) break;

            const double obj0 = 0.5 * (r.squaredNorm() + lambda * c_err * c_err);
            double alpha = 1.0;
            bool accepted = false;
            for (int ls = 0; ls < 20; ++ls) {
                const dcr::base::Vector x_trial = x_it - alpha * delta;
                if (!x_trial.allFinite()) {
                    alpha *= 0.5;
                    continue;
                }
                const dcr::base::Vector r_trial = M * x_trial - rhs;
                const double c_trial = constraint.dot(x_trial) - target;
                const double obj_trial = 0.5 * (r_trial.squaredNorm() + lambda * c_trial * c_trial);
                if (obj_trial < obj0) {
                    x_it = x_trial;
                    accepted = true;
                    break;
                }
                alpha *= 0.5;
            }
            if (!accepted) break;
            ++total_newton_iters;
            if (delta.norm() <= 1e-12 * (1.0 + x_it.norm())) break;
        }

        const auto eval = evaluate(x_it);
        const double cand_resid_rel = eval.first;
        const double cand_constraint_rel = eval.second;
        if (cand_constraint_rel <= constraint_rel_max && cand_resid_rel < best_resid_rel) {
            best_resid_rel = cand_resid_rel;
            best_constraint_rel = cand_constraint_rel;
            out.x = x_it;
            out.residual_rel = cand_resid_rel;
            out.constraint_rel = cand_constraint_rel;
            out.improved = true;
            out.iterations = total_newton_iters;
        }
    }

    return out;
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

    const int total_states = atomic_data.get_total_states();
    const auto& levels = atomic_data.get_levels();

    const double u_bohm = dcr::physics::calculate_Bohm_speed(
        config.plasma.Te_eV, config.plasma.Ti_eV, ion_mass_amu
    );
    const double E_ref_atom = wall.gamma_E_atom * wall.ion_impact_energy_ev;
    out.u_A = dcr::physics::calculate_speed_from_energy_ev(E_ref_atom, ion_mass_amu);

    // Keep a physical default for hydrogen molecules unless config explicitly provides one.
    out.molecule_mass_amu = 2.0;
    for (const auto& sp : config.species) {
        if (sp.is_molecule && sp.charge == 0) {
            // `mass_amu` defaults to 1.0 in YAML loader; do not let that override molecular default.
            if (sp.mass_amu > 1.0) {
                out.molecule_mass_amu = sp.mass_amu;
            }
            break;
        }
    }
    out.u_M = dcr::physics::calculate_thermal_speed(
        config.wall.temperature_eV, out.molecule_mass_amu
    );

    const auto& recycling_indices_cfg = atomic_data.get_recycling_indices();
    out.explicit_recycling = !recycling_indices_cfg.empty();
    if (config.io.verbose_logging) {
        std::cout << "[DCR_Solver] recycling_indices from AtomicData: "
                  << recycling_indices_cfg.size() << "\n";
        if (!recycling_indices_cfg.empty()) {
            std::cout << "[DCR_Solver] recycling indices:";
            for (auto idx : recycling_indices_cfg) {
                std::cout << " " << idx;
            }
            std::cout << "\n";
        } else {
            std::cout << "[DCR_Solver] recycling indices are empty -> implicit recycling mode.\n";
        }
    }

    // Find ground states for recycled neutral atom and molecule.
    out.atom_ground = -1;
    out.molecule_ground = -1;
    double min_atom_E = 1.0e99;
    double min_mol_E = 1.0e99;
    for (const auto& lvl : levels) {
        if (lvl.charge != 0) continue;
        if (out.explicit_recycling && !lvl.is_recycling) continue;
        if (lvl.type == dcr::atomic::SpeciesType::Atom && lvl.energy_eV < min_atom_E) {
            min_atom_E = lvl.energy_eV;
            out.atom_ground = lvl.global_index;
        } else if (lvl.type == dcr::atomic::SpeciesType::Molecule && lvl.energy_eV < min_mol_E) {
            min_mol_E = lvl.energy_eV;
            out.molecule_ground = lvl.global_index;
        }
    }

    // Build index sets: recycling (R) includes all neutral recycled states.
    out.is_recycling.assign(total_states, false);
    for (auto idx : recycling_indices_cfg) {
        if (idx >= 0 && idx < total_states) out.is_recycling[idx] = true;
    }
    out.R_indices.clear();
    out.P_indices.clear();
    out.R_indices.reserve(total_states);
    out.P_indices.reserve(total_states);
    for (int i = 0; i < total_states; ++i) {
        if (out.is_recycling[i]) out.R_indices.push_back(i);
        else out.P_indices.push_back(i);
    }
    if (config.io.verbose_logging) {
        std::cout << "[DCR_Solver] Partition sizes: P=" << out.P_indices.size()
                  << " R=" << out.R_indices.size() << "\n";
    }

    // Split recycling indices into atom/molecule groups for RA/RM blocks.
    out.A_indices.clear();
    out.M_indices.clear();
    if (out.explicit_recycling) {
        out.A_indices.reserve(out.R_indices.size());
        out.M_indices.reserve(out.R_indices.size());
        for (int idx : out.R_indices) {
            if (idx < 0 || idx >= static_cast<int>(levels.size())) continue;
            const auto& lvl = levels[idx];
            if (lvl.type == dcr::atomic::SpeciesType::Atom && lvl.charge == 0) {
                out.A_indices.push_back(idx);
            } else if (lvl.type == dcr::atomic::SpeciesType::Molecule && lvl.charge == 0) {
                out.M_indices.push_back(idx);
            }
        }
    } else {
        // Implicit recycling: reuse background neutral indices for R^A/R^M coefficients.
        out.A_indices.reserve(out.P_indices.size());
        out.M_indices.reserve(out.P_indices.size());
        for (int idx : out.P_indices) {
            if (idx < 0 || idx >= static_cast<int>(levels.size())) continue;
            const auto& lvl = levels[idx];
            if (lvl.type == dcr::atomic::SpeciesType::Atom && lvl.charge == 0) {
                out.A_indices.push_back(idx);
            } else if (lvl.type == dcr::atomic::SpeciesType::Molecule && lvl.charge == 0) {
                out.M_indices.push_back(idx);
            }
        }
    }
    if (!out.explicit_recycling && config.io.verbose_logging) {
        std::cout << "[DCR_Solver] Implicit recycling states: "
                  << out.A_indices.size() << " atom, " << out.M_indices.size() << " molecule.\n";
    }

    // Background group indices (within P).
    out.ion_indices.clear();
    out.atom_bg_indices.clear();
    out.mol_bg_indices.clear();
    out.ion_indices.reserve(out.P_indices.size());
    out.atom_bg_indices.reserve(out.P_indices.size());
    out.mol_bg_indices.reserve(out.P_indices.size());
    for (int idx : out.P_indices) {
        if (idx < 0 || idx >= static_cast<int>(levels.size())) continue;
        const auto& lvl = levels[idx];
        // "I" block is positive ions only (exclude negative ions such as H-).
        if (lvl.type == dcr::atomic::SpeciesType::Ion && lvl.charge > 0) {
            out.ion_indices.push_back(idx);
        } else if (lvl.type == dcr::atomic::SpeciesType::Atom && lvl.charge == 0) {
            out.atom_bg_indices.push_back(idx);
        } else if (lvl.type == dcr::atomic::SpeciesType::Molecule && lvl.charge == 0) {
            out.mol_bg_indices.push_back(idx);
        }
    }

    out.atom_mass_amu = 1.0;
    for (const auto& sp : config.species) {
        if (!sp.is_molecule && sp.charge == 0) {
            out.atom_mass_amu = sp.mass_amu;
            break;
        }
    }

    const double mu_M = 2.0;
    const double n_nuclei0 = config.plasma.total_density;
    const double c_A = (out.u_A > 0.0) ? (wall.alpha_atom * u_bohm / out.u_A) : 0.0;
    const double c_M_atom = (out.u_M > 0.0) ? (wall.alpha_molecule * u_bohm / (mu_M * out.u_M)) : 0.0;
    const double c_M_base = (out.u_M > 0.0) ? (u_bohm / (mu_M * out.u_M)) : 0.0;
    const double c_s_A = dcr::physics::calculate_thermal_speed(
        config.plasma.neutral_atom_temperature_eV, out.atom_mass_amu
    );
    const double c_s_M = dcr::physics::calculate_thermal_speed(
        config.plasma.neutral_molecule_temperature_eV, out.molecule_mass_amu
    );
    const double w_local = std::max(config.grid.poloidal_width_cm, 1e-12);
    const double exA_coef = c_s_A / w_local;
    const double exM_coef = c_s_M / w_local;
    const int max_iter = config.numerics.max_iterations;
    const double tol = config.numerics.tolerance;
    const double omega_raw = config.numerics.relaxation;
    const double omega = (omega_raw > 0.0 && omega_raw <= 1.0) ? omega_raw : 1.0;
    // --- Build population vector from initial conditions (background only) ---
    out.population = dcr::base::Vector::Zero(total_states);
    for (const auto& ic : config.plasma.initial_conditions) {
        if (ic.species_index >= 0 && ic.species_index < total_states) {
            if (!out.is_recycling[ic.species_index]) {
                out.population(ic.species_index) = ic.fraction * config.plasma.total_density;
            }
        }
    }

    const int Pn = static_cast<int>(out.P_indices.size());
    if (Pn > 0) {
        dcr::base::Vector stoich = dcr::base::Vector::Zero(Pn);
        dcr::base::Vector x = dcr::base::Vector::Zero(Pn);

        // Recycling distributions (implicit mode uses separate vectors).
        dcr::base::Vector nA_vec = dcr::base::Vector::Zero(static_cast<int>(out.A_indices.size()));
        dcr::base::Vector nM_vec = dcr::base::Vector::Zero(static_cast<int>(out.M_indices.size()));
        int atom_ground_pos = -1;
        int molecule_ground_pos = -1;
        if (!out.explicit_recycling) {
            for (size_t i = 0; i < out.A_indices.size(); ++i) {
                if (out.A_indices[i] == out.atom_ground) {
                    atom_ground_pos = static_cast<int>(i);
                    break;
                }
            }
            for (size_t i = 0; i < out.M_indices.size(); ++i) {
                if (out.M_indices[i] == out.molecule_ground) {
                    molecule_ground_pos = static_cast<int>(i);
                    break;
                }
            }
        }

        double bg_init_sum = 0.0;
        for (int i = 0; i < Pn; ++i) {
            const int gi = out.P_indices[i];
            const int nu = (gi >= 0 && gi < static_cast<int>(levels.size())) ? levels[gi].atomicity : 1;
            stoich(i) = static_cast<double>(nu);
            bg_init_sum += std::max(out.population(gi), 0.0);
        }
        if (bg_init_sum > 0.0) {
            // Keep the user-provided IC fractions as the baseline background fraction.
            for (int i = 0; i < Pn; ++i) {
                const int gi = out.P_indices[i];
                x(i) = std::max(out.population(gi), 0.0) / bg_init_sum;
            }
        } else {
            // Fallback: uniform fraction across background states.
            x.setConstant(1.0 / static_cast<double>(Pn));
        }

        // Precompute mapping from global index to P position.
        std::vector<int> p_pos(total_states, -1);
        for (int i = 0; i < Pn; ++i) {
            p_pos[out.P_indices[i]] = i;
        }
        std::vector<int> ion_cols;
        std::vector<double> recA_coeff;
        std::vector<double> recM_coeff;
        ion_cols.reserve(out.ion_indices.size());
        recA_coeff.reserve(out.ion_indices.size());
        recM_coeff.reserve(out.ion_indices.size());
        for (int gi : out.ion_indices) {
            const int pos = (gi >= 0 && gi < total_states) ? p_pos[gi] : -1;
            if (pos < 0) continue;
            const int mu_I = (gi >= 0 && gi < static_cast<int>(levels.size())) ? levels[gi].atomicity : 1;
            const bool is_molecular_ion = (mu_I > 1);
            ion_cols.push_back(pos);
            // Molecular ions recycle to molecule channel; only atomic ions feed atomic channel.
            recA_coeff.push_back(is_molecular_ion ? 0.0 : c_A);
            // For atomic-ion -> molecular recycling, c_M_atom already includes the 1/mu_M factor.
            // Do not apply an extra atomicity multiplier there.
            recM_coeff.push_back(is_molecular_ion ? (c_M_base * mu_I) : c_M_atom);
        }

        dcr::base::Vector solve_constraint = stoich;
        double solve_constraint_target = 1.0;
        // Row replacement location.
        int constraint_row = 1;

        dcr::base::Vector prev_x = x;
        double omega_iter = omega;
        double prev_rel_change = std::numeric_limits<double>::infinity();
        int no_improve_count = 0;

        for (int iter = 0; iter < max_iter; ++iter) {
            // Build absolute densities from fractions.
            for (int i = 0; i < Pn; ++i) {
                out.population(out.P_indices[i]) = x(i) * n_nuclei0;
            }

            // Group totals from current population.
            double n_I_weighted = 0.0;
            for (int idx : out.ion_indices) {
                if (idx < 0 || idx >= static_cast<int>(levels.size()) || idx >= out.population.size()) continue;
                const double mu_i = std::max(1, levels[idx].atomicity);
                const double n_i = std::max(out.population(idx), 0.0);
                n_I_weighted += mu_i * n_i;
            }
            double n_a = 0.0;
            for (int idx : out.atom_bg_indices) n_a += std::max(out.population(idx), 0.0);
            double n_m = 0.0;
            for (int idx : out.mol_bg_indices) n_m += std::max(out.population(idx), 0.0);
            double n_A_total = 0.0;
            double n_M_total = 0.0;
            for (size_t k = 0; k < out.ion_indices.size() && k < recA_coeff.size(); ++k) {
                const int gi = out.ion_indices[k];
                const double n_i = std::max(out.population(gi), 0.0);
                n_A_total += recA_coeff[k] * n_i;
                n_M_total += recM_coeff[k] * n_i;
            }

            // Initial coupled guess scaling: normalize the combined
            // (background + recycling-flow) nuclei budget to n_nuclei0.
            if (iter == 0 && n_nuclei0 > 0.0) {
                const double total_guess_nuclei =
                    std::max(0.0, n_I_weighted) +
                    std::max(0.0, n_a) +
                    mu_M * std::max(0.0, n_m) +
                    std::max(0.0, n_A_total) +
                    mu_M * std::max(0.0, n_M_total);
                if (total_guess_nuclei > 0.0) {
                    const double scale0 = n_nuclei0 / total_guess_nuclei;
                    for (int i = 0; i < Pn; ++i) {
                        out.population(out.P_indices[i]) *= scale0;
                    }
                    n_I_weighted *= scale0;
                    n_a *= scale0;
                    n_m *= scale0;
                    n_A_total *= scale0;
                    n_M_total *= scale0;
                    // Keep iterate state consistent with the rescaled initial background.
                    for (int i = 0; i < Pn; ++i) {
                        x(i) = out.population(out.P_indices[i]) / n_nuclei0;
                    }
                    prev_x = x;
                }

                // Extra guard for iter-0: do not let recycling flow exceed
                // the remaining nuclei budget after the background.
                const double bg_nuclei_iter0 =
                    std::max(0.0, n_I_weighted) +
                    std::max(0.0, n_a) +
                    mu_M * std::max(0.0, n_m);
                const double flow_nuclei_iter0 =
                    std::max(0.0, n_A_total) +
                    mu_M * std::max(0.0, n_M_total);
                const double flow_budget_iter0 = std::max(0.0, n_nuclei0 - bg_nuclei_iter0);
                if (flow_nuclei_iter0 > flow_budget_iter0 && flow_nuclei_iter0 > 0.0) {
                    const double flow_scale0 = flow_budget_iter0 / flow_nuclei_iter0;
                    n_A_total *= flow_scale0;
                    n_M_total *= flow_scale0;
                }
            }

            double n_A_iter = n_A_total;
            double n_M_iter = n_M_total;

            // Final iter-0 consistency pass: the exact bg+flow state used to build R
            // is normalized to the target nuclei density.
            if (iter == 0 && n_nuclei0 > 0.0) {
                const double total0 =
                    std::max(0.0, n_I_weighted) +
                    std::max(0.0, n_a) +
                    mu_M * std::max(0.0, n_m) +
                    std::max(0.0, n_A_iter) +
                    mu_M * std::max(0.0, n_M_iter);
                if (total0 > 0.0) {
                    const double s0 = n_nuclei0 / total0;
                    if (std::abs(s0 - 1.0) > 1e-14) {
                        for (int i = 0; i < Pn; ++i) {
                            out.population(out.P_indices[i]) *= s0;
                        }
                        n_I_weighted *= s0;
                        n_a *= s0;
                        n_m *= s0;
                        n_A_iter *= s0;
                        n_M_iter *= s0;
                        n_A_total = n_A_iter;
                        n_M_total = n_M_iter;
                        for (int i = 0; i < Pn; ++i) {
                            x(i) = out.population(out.P_indices[i]) / n_nuclei0;
                        }
                        prev_x = x;
                    }
                }
            }

            if (out.explicit_recycling) {
                // Reset recycling populations to ground states.
                for (int idx : out.R_indices) {
                    out.population(idx) = 0.0;
                }
                if (out.atom_ground >= 0 && out.atom_ground < out.population.size()) {
                    out.population(out.atom_ground) = n_A_iter;
                }
                if (out.molecule_ground >= 0 && out.molecule_ground < out.population.size()) {
                    out.population(out.molecule_ground) = n_M_iter;
                }
            } else {
                // Implicit recycling: store flow densities in separate vectors.
                nA_vec.setZero();
                nM_vec.setZero();
                if (atom_ground_pos >= 0) {
                    nA_vec(atom_ground_pos) = n_A_iter;
                }
                if (molecule_ground_pos >= 0) {
                    nM_vec(molecule_ground_pos) = n_M_iter;
                }
            }

            // Local plasma density constraint:
            // n_local = n_total - n_recycling_flow (nuclei units), normalized by n_total.
            double solve_constraint_target_iter = solve_constraint_target;
            if (n_nuclei0 > 0.0) {
                const double flow_nuclei = std::max(0.0, n_A_iter + mu_M * n_M_iter);
                solve_constraint_target_iter = std::clamp((n_nuclei0 - flow_nuclei) / n_nuclei0, 0.0, 1.0);
            }

            // Assemble rate matrix using total population (background + recycling flow).
            dcr::base::Vector population_for_R = out.population;
            if (!out.explicit_recycling) {
                if (out.atom_ground >= 0 && out.atom_ground < population_for_R.size()) {
                    population_for_R(out.atom_ground) += n_A_iter;
                }
                if (out.molecule_ground >= 0 && out.molecule_ground < population_for_R.size()) {
                    population_for_R(out.molecule_ground) += n_M_iter;
                }
            }
            dcr::base::Matrix R = dcr::base::Matrix::Zero(total_states, total_states);
            const double ne_local = quasineutral_electron_density(population_for_R, levels);
            plasma.init_ne().setConstant(ne_local);
            for (const auto& proc : atomic_data.get_processes()) {
                if (!proc) continue;
                proc->apply(plasma, grid, population_for_R, R, nullptr);
            }

            // Build local CR matrix on background block R_PP.
            dcr::base::Matrix Rpp = dcr::base::Matrix::Zero(Pn, Pn);
            for (int i = 0; i < Pn; ++i) {
                const int gi = out.P_indices[i];
                for (int j = 0; j < Pn; ++j) {
                    const int gj = out.P_indices[j];
                    Rpp(i, j) = R(gi, gj);
                }
            }

            // Build recycling source S on background rows (explicit source form).
            dcr::base::Vector S_bg = dcr::base::Vector::Zero(Pn);
            for (int i = 0; i < Pn; ++i) {
                const int gi = out.P_indices[i];
                const auto& row_lvl = levels[gi];
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
                        const double nA = out.explicit_recycling
                            ? std::max(out.population(gj), 0.0)
                            : ((j < static_cast<size_t>(nA_vec.size()))
                                   ? std::max(nA_vec(static_cast<int>(j)), 0.0)
                                   : 0.0);
                        Si += R(gi, gj) * nA;
                    }
                }
                if (use_M_source) {
                    for (size_t j = 0; j < out.M_indices.size(); ++j) {
                        const int gj = out.M_indices[j];
                        const double nM = out.explicit_recycling
                            ? std::max(out.population(gj), 0.0)
                            : ((j < static_cast<size_t>(nM_vec.size()))
                                   ? std::max(nM_vec(static_cast<int>(j)), 0.0)
                                   : 0.0);
                        Si += R(gi, gj) * nM;
                    }
                }
                S_bg(i) = Si;
            }

            // Use the same S used on the RHS to derive the grouped recycling source in L_I.
            // This keeps the scalar source term and the vector source term algebraically aligned.
            const double source_nuclei_from_S = stoich.dot(S_bg);
            const double Gamma_ex_a_over_w = exA_coef * std::max(0.0, n_A_iter);
            const double Gamma_ex_m_over_w = exM_coef * std::max(0.0, n_M_iter);

            // Restore local exhaust terms in Eq. (1.345) grouped transport:
            //   L_I = source - (Gamma_ex_a + mu_M * Gamma_ex_m)/w
            //   L_a = Gamma_ex_a/w, L_m = Gamma_ex_m/w
            const double L_I = source_nuclei_from_S - (Gamma_ex_a_over_w + mu_M * Gamma_ex_m_over_w);
            const double L_a = Gamma_ex_a_over_w;
            const double L_m = Gamma_ex_m_over_w;

            // Build transport matrix A (grouped LHS operator on x).
            dcr::base::Matrix A = dcr::base::Matrix::Zero(Pn, Pn);

            if (n_I_weighted > 0.0) {
                for (int gi : out.ion_indices) {
                    const int pi = p_pos[gi];
                    if (pi >= 0 && gi >= 0 && gi < static_cast<int>(levels.size())) {
                        // Ion-state share uses n_i / (sum_j mu_j n_j).
                        A(pi, pi) = L_I / n_I_weighted;
                    }
                }
            }
            if (n_a > 0.0) {
                const double coeff = L_a / n_a;
                for (int gi : out.atom_bg_indices) {
                    const int pi = p_pos[gi];
                    if (pi >= 0) A(pi, pi) = coeff;
                }
            }
            if (n_m > 0.0) {
                const double coeff = L_m / n_m;
                for (int gi : out.mol_bg_indices) {
                    const int pi = p_pos[gi];
                    if (pi >= 0) A(pi, pi) = coeff;
                }
            }

            const double n_scale = std::max(1.0, n_nuclei0);
            const dcr::base::Vector S_bg_scaled = S_bg / n_scale;

            // Frozen fixed-point boundary solve with row replacement:
            //   R_PP x^{k+1} = (T - S), with T frozen from current iterate x.
            const dcr::base::Matrix M = Rpp;
            const dcr::base::Vector rhs = A * x - S_bg_scaled;
            dcr::base::Vector x_new = x;
            double ls_constraint_err = std::numeric_limits<double>::quiet_NaN();
            const double ls_tol = 1e-8;
            dcr::base::Matrix C = M;
            dcr::base::Vector rhs_replaced = rhs;
            C.row(constraint_row) = solve_constraint.transpose();
            rhs_replaced(constraint_row) = solve_constraint_target_iter;
            x_new = solve_replaced_system(C, rhs_replaced);
            ls_constraint_err = std::abs(solve_constraint.dot(x_new) - solve_constraint_target_iter);
            const double constrained_sum = solve_constraint.dot(x_new);
            if (constrained_sum > 0.0) {
                x_new *= (solve_constraint_target_iter / constrained_sum);
            } else {
                if (solve_constraint_target_iter <= 0.0) {
                    x_new.setZero();
                } else {
                    x_new = x;
                }
            }

            // Under-relaxation to avoid limit cycles.
            if (omega_iter < 1.0) {
                x_new = (1.0 - omega_iter) * x + omega_iter * x_new;
            }

            const double diff = (x_new - prev_x).cwiseAbs().maxCoeff();
            const double scale = std::max(1.0, x_new.cwiseAbs().maxCoeff());
            const double rel_change = diff / scale;

            const dcr::base::Vector residual_vec = Rpp * x_new - (A * x_new - S_bg_scaled);
            const double residual = residual_vec.norm();
            const double lhs_norm = (Rpp * x_new).norm();
            const double residual_rel = residual / std::max(1.0, lhs_norm);
            const dcr::base::Vector ax_minus_s = A * x_new - S_bg_scaled;
            const double ax_minus_s_sum = ax_minus_s.sum();
            const double ax_minus_s_abs_sum = ax_minus_s.cwiseAbs().sum();
            const double ax_minus_s_weighted_sum = stoich.dot(ax_minus_s);
            const double ax_minus_s_weighted_abs_sum =
                (stoich.array() * ax_minus_s.cwiseAbs().array()).sum();
            if (rel_change >= prev_rel_change * 0.999) {
                ++no_improve_count;
            } else {
                no_improve_count = 0;
            }

            if (no_improve_count >= 5 && omega_iter > 0.1) {
                omega_iter = std::max(0.1, 0.5 * omega_iter);
                no_improve_count = 0;
                if (config.io.verbose_logging) {
                    std::cout << "[DCR_Solver] Boundary: slow convergence, reducing relaxation to "
                              << omega_iter << "\n";
                }
            }
            // Periodic damping if we're stuck above tolerance.
            if (((iter + 1) % 50 == 0) && rel_change > (10.0 * tol) && omega_iter > 0.1) {
                omega_iter = std::max(0.1, 0.5 * omega_iter);
                if (config.io.verbose_logging) {
                    std::cout << "[DCR_Solver] Boundary: periodic damping, reducing relaxation to "
                              << omega_iter << "\n";
                }
            }

            prev_x = x_new;
            x = x_new;
            prev_rel_change = rel_change;

            if (config.io.verbose_logging) {
                const double constrained_check = solve_constraint.dot(x_new);
                BgSubgroupSums bg_iter;
                for (int pi = 0; pi < Pn; ++pi) {
                    const int gi = out.P_indices[pi];
                    if (gi < 0 || gi >= static_cast<int>(levels.size())) continue;
                    const auto& lvl = levels[gi];
                    const double n = std::max(x_new(pi) * n_nuclei0, 0.0);
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
                          << " rel_change=" << rel_change
                          << " ls_constraint_err=" << ls_constraint_err
                          << " constrained_val=" << constrained_check
                          << " sum(Ax-S)=" << ax_minus_s_sum
                          << " sumabs(Ax-S)=" << ax_minus_s_abs_sum
                          << " wsum(Ax-S)=" << ax_minus_s_weighted_sum
                          << " wsumabs(Ax-S)=" << ax_minus_s_weighted_abs_sum
                          << " residual_rel(diag)=" << residual_rel
                          << " bg{H+=" << bg_iter.H_plus
                          << ", H=" << bg_iter.H
                          << ", H2=" << bg_iter.H2
                          << ", H2+=" << bg_iter.H2_plus
                          << ", H-=" << bg_iter.H_minus
                          << "}\n";
            }

            if (rel_change < tol &&
                std::isfinite(ls_constraint_err) && ls_constraint_err < ls_tol) {
                // Post-convergence Newton polish on frozen linearized system.
                const double polish_constraint_rel_max = 1e-5;
                const auto polish = polish_with_newton_soft_constraint(
                    M, rhs, solve_constraint, solve_constraint_target_iter, x_new, polish_constraint_rel_max);
                if (polish.improved) {
                    x_new = polish.x;
                    x = x_new;
                    prev_x = x_new;
                    if (config.io.verbose_logging) {
                        std::cout << "[DCR_Solver] Boundary Newton polish applied: resid_rel="
                                  << polish.residual_rel
                                  << " constraint_rel=" << polish.constraint_rel
                                  << " iters=" << polish.iterations
                                  << "\n";
                    }
                }
                if (config.io.verbose_logging) {
                    std::cout << "[DCR_Solver] Boundary converged in " << (iter + 1) << " iterations.\n";
                }
                break;
            }
            if (iter == max_iter - 1 && config.io.verbose_logging) {
                std::cout << "[DCR_Solver] Boundary reached max iterations without convergence.\n";
            }
        }

        // Update population and flow totals using the converged x.
        for (int i = 0; i < Pn; ++i) {
            out.population(out.P_indices[i]) = x(i) * n_nuclei0;
        }

        double n_A_total_final = 0.0;
        double n_M_total_final = 0.0;
        for (size_t k = 0; k < ion_cols.size(); ++k) {
            const int pos = ion_cols[k];
            if (pos >= 0 && pos < x.size()) {
                const double n_i = x(pos) * n_nuclei0;
                n_A_total_final += recA_coeff[k] * n_i;
                n_M_total_final += recM_coeff[k] * n_i;
            }
        }

        // Final physical projection: enforce nonnegative background populations and
        // renormalize background nuclei so bg + flow exactly matches n_nuclei0.
        n_A_total_final = std::max(0.0, n_A_total_final);
        n_M_total_final = std::max(0.0, n_M_total_final);
        const double flow_total_pos_final = n_A_total_final + mu_M * n_M_total_final;
        const double target_bg_total_final = std::max(0.0, n_nuclei0 - flow_total_pos_final);

        double bg_total_pos_final = 0.0;
        for (int i = 0; i < Pn; ++i) {
            const int gi = out.P_indices[i];
            if (gi < 0 || gi >= out.population.size() || gi >= static_cast<int>(levels.size())) continue;
            out.population(gi) = std::max(out.population(gi), 0.0);
            bg_total_pos_final += static_cast<double>(levels[gi].atomicity) * out.population(gi);
        }
        if (target_bg_total_final > 0.0) {
            if (bg_total_pos_final > 0.0) {
                const double scale_bg = target_bg_total_final / bg_total_pos_final;
                for (int i = 0; i < Pn; ++i) {
                    const int gi = out.P_indices[i];
                    if (gi >= 0 && gi < out.population.size()) out.population(gi) *= scale_bg;
                }
            } else if (Pn > 0) {
                const int gi0 = out.P_indices[0];
                const double nu0 = (gi0 >= 0 && gi0 < static_cast<int>(levels.size()))
                    ? std::max(1, levels[gi0].atomicity) : 1;
                if (gi0 >= 0 && gi0 < out.population.size()) {
                    out.population(gi0) = target_bg_total_final / static_cast<double>(nu0);
                }
            }
        } else {
            for (int i = 0; i < Pn; ++i) {
                const int gi = out.P_indices[i];
                if (gi >= 0 && gi < out.population.size()) out.population(gi) = 0.0;
            }
        }

        if (out.explicit_recycling) {
            for (int idx : out.R_indices) {
                out.population(idx) = 0.0;
            }
            if (out.atom_ground >= 0 && out.atom_ground < out.population.size()) {
                out.population(out.atom_ground) = n_A_total_final;
            }
            if (out.molecule_ground >= 0 && out.molecule_ground < out.population.size()) {
                out.population(out.molecule_ground) = n_M_total_final;
            }
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
    }

    if (config.io.verbose_logging) {
        const auto bg = summarize_boundary_background(out, out.population, levels);
        const double flowA_total = out.explicit_recycling
            ? positive_sum_at_indices(out.population, out.A_indices)
            : positive_sum(out.flowA_last);
        const double flowM_total = out.explicit_recycling
            ? positive_sum_at_indices(out.population, out.M_indices)
            : positive_sum(out.flowM_last);
        std::cout << "[DCR_Solver] Boundary x=0 cm"
                  << " bg{H+=" << bg.H_plus
                  << ", H=" << bg.H
                  << ", H2=" << bg.H2
                  << ", H2+=" << bg.H2_plus
                  << ", H-=" << bg.H_minus
                  << "} flow{A=" << flowA_total
                  << ", M=" << flowM_total
                  << "}\n";

        double bg_total = 0.0;
        for (int idx : out.P_indices) {
            if (idx >= 0 && idx < static_cast<int>(levels.size())) {
                bg_total += levels[idx].atomicity * out.population(idx);
            }
        }

        double flow_total = 0.0;
        if (out.explicit_recycling) {
            for (int idx : out.R_indices) {
                if (idx >= 0 && idx < static_cast<int>(levels.size())) {
                    flow_total += levels[idx].atomicity * out.population(idx);
                }
            }
        } else if (out.have_flow_last) {
            for (size_t i = 0; i < out.A_indices.size(); ++i) {
                const int gi = out.A_indices[i];
                if (gi >= 0 && gi < static_cast<int>(levels.size())) {
                    flow_total += levels[gi].atomicity * out.flowA_last(static_cast<int>(i));
                }
            }
            for (size_t i = 0; i < out.M_indices.size(); ++i) {
                const int gi = out.M_indices[i];
                if (gi >= 0 && gi < static_cast<int>(levels.size())) {
                    flow_total += levels[gi].atomicity * out.flowM_last(static_cast<int>(i));
                }
            }
        }

        std::cout << "[DCR_Solver] Boundary nuclei totals: background=" << bg_total
                  << " flow=" << flow_total
                  << " total=" << (bg_total + flow_total)
                  << " target_total=" << n_nuclei0 << "\n";
        std::cout << "[DCR_Solver] Boundary init: u_A=" << out.u_A << " u_M=" << out.u_M << "\n";
    }

    return out;
}

} // namespace dcr::solver
