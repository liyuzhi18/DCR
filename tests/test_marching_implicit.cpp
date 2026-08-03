#include <algorithm>
#include <cassert>
#include <cmath>
#include <iostream>
#include <stdexcept>

#include "../src/atomic/AtomicData.hpp"
#include "../src/physics/WallBoundary.hpp"
#include "../src/solver/boundary/BoundaryPhase.hpp"
#include "../src/solver/marching/CellAdvance.hpp"
#include "../src/solver/marching/CellSolve.hpp"
#include "../src/solver/marching/LocalSystemAssembler.hpp"
#include "TestDCRSetup.hpp"

namespace {

void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

dcr::base::Matrix extract_block(const dcr::base::Matrix& R, const std::vector<int>& idx) {
    const int n = static_cast<int>(idx.size());
    dcr::base::Matrix out = dcr::base::Matrix::Zero(n, n);
    for (int i = 0; i < n; ++i) {
        const int gi = idx[static_cast<size_t>(i)];
        if (gi < 0 || gi >= R.rows()) continue;
        for (int j = 0; j < n; ++j) {
            const int gj = idx[static_cast<size_t>(j)];
            if (gj < 0 || gj >= R.cols()) continue;
            out(i, j) = R(gi, gj);
        }
    }
    return out;
}

double implicit_residual_norm(const dcr::base::Matrix& R_rr,
                              double c_s,
                              double w_cm,
                              double u,
                              double dx_cm,
                              const dcr::base::Vector& n_old,
                              const dcr::base::Vector& n_new) {
    if (n_old.size() == 0 || n_new.size() == 0 || dx_cm <= 0.0 || u <= 0.0) return 0.0;

    const double coef = dx_cm / u;
    dcr::base::Matrix lhs = dcr::base::Matrix::Identity(n_old.size(), n_old.size()) - coef * R_rr;
    lhs.diagonal().array() += coef * (c_s / std::max(w_cm, 1e-12));

    // If the implicit system is singular, production code falls back to explicit.
    // Skip strict residual checks in that case.
    Eigen::FullPivLU<dcr::base::Matrix> lu(lhs);
    if (lu.rank() < lhs.rows()) return 0.0;

    const dcr::base::Vector res = lhs * n_new - n_old;
    return res.norm() / (1.0 + n_old.norm());
}

} // namespace

int main() {
    std::cout << "--- Testing Marching Implicit Step ---\n";

    auto cfg = test_dcr::load_reference_config(/*num_cells=*/2, /*verbose_logging=*/false);
    cfg.grid.length_cm = 0.2;
    cfg.grid.type = "linear";
    cfg.grid.first_cell_cm = 0.0;

    dcr::atomic::AtomicData atomic_data(cfg);
    auto plasma = test_dcr::make_plasma_state(cfg, atomic_data.get_total_states());
    const auto boundary_temperatures = test_dcr::plasma_temperatures_at(cfg, 0.0);
    test_dcr::EEDFContext eedf(boundary_temperatures.electron_eV);

    const double ion_mass_amu = test_dcr::estimate_ion_mass_amu(cfg);
    const auto wall = dcr::physics::compute_wall_recycling(
        cfg.wall.material,
        boundary_temperatures.electron_eV,
        boundary_temperatures.ion_eV,
        ion_mass_amu,
        cfg.wall.sheath_potential_drop
    );

    const auto boundary = dcr::solver::run_boundary_phase(
        cfg, atomic_data, plasma, eedf.grid, ion_mass_amu, wall
    );

    const dcr::base::Vector nP = test_dcr::compact_background_from_boundary(boundary);
    const dcr::base::Vector background_full = test_dcr::full_background_from_compact(
        nP, boundary, atomic_data.get_total_states()
    );
    const dcr::base::Vector flowA_old = test_dcr::flowA_from_boundary(boundary);
    const dcr::base::Vector flowM_old = test_dcr::flowM_from_boundary(boundary);

    test_dcr::assert_all_finite_nonnegative(background_full);
    test_dcr::assert_all_finite_nonnegative(flowA_old);
    test_dcr::assert_all_finite_nonnegative(flowM_old);

    const auto local = dcr::solver::assemble_local_system(
        cfg, atomic_data, plasma, eedf.grid, boundary, background_full, flowA_old, flowM_old, cfg.grid.length_cm
    );
    const auto advanced = dcr::solver::advance_recycling_flow_one_step(
        cfg, atomic_data, boundary, local, flowA_old, flowM_old, cfg.grid.length_cm
    );

    assert(advanced.flowA_next.size() == flowA_old.size());
    assert(advanced.flowM_next.size() == flowM_old.size());
    test_dcr::assert_all_finite_nonnegative(advanced.flowA_next);
    test_dcr::assert_all_finite_nonnegative(advanced.flowM_next);

    const dcr::base::Matrix R_AA = extract_block(local.R_full, boundary.A_indices);
    const dcr::base::Matrix R_MM = extract_block(local.R_full, boundary.M_indices);
    const double resA = implicit_residual_norm(
        R_AA,
        advanced.c_s_A,
        cfg.grid.spatial_exhaust_width_cm,
        boundary.u_A,
        cfg.grid.length_cm,
        flowA_old,
        advanced.flowA_next
    );
    const double resM = implicit_residual_norm(
        R_MM,
        advanced.c_s_M,
        cfg.grid.spatial_exhaust_width_cm,
        boundary.u_M,
        cfg.grid.length_cm,
        flowM_old,
        advanced.flowM_next
    );

    // For non-singular implicit systems, residual should be near solver tolerance.
    assert(resA < 1e-6);
    assert(resM < 1e-6);

    const auto step = dcr::solver::solve_cell_implicit(
        cfg,
        atomic_data,
        plasma,
        eedf.grid,
        boundary,
        atomic_data.get_levels(),
        nP,
        flowA_old,
        flowM_old,
        cfg.grid.length_cm,
        0.0,
        cfg.grid.length_cm,
        1,
        false,
        false
    );
    require(step.converged, "Local marching cell did not converge");
    test_dcr::assert_all_finite_nonnegative(step.nP_new);

    const auto& levels = atomic_data.get_levels();
    const double total_nuclei =
        test_dcr::nuclei_total_compact(step.nP_new, boundary.P_indices, levels) +
        test_dcr::nuclei_total_compact(step.flowA_new, boundary.A_indices, levels) +
        test_dcr::nuclei_total_compact(step.flowM_new, boundary.M_indices, levels);
    const double density_error = std::abs(total_nuclei - cfg.plasma.total_density) /
        std::max(1.0, cfg.plasma.total_density);
    require(density_error < 1e-12, "Local marcher did not enforce prescribed nuclei density");

    const double expected_LI = step.recycling_source_nuclei_cm3_s -
        step.local_atom_exhaust_nuclei_cm3_s -
        step.local_molecule_exhaust_nuclei_cm3_s;
    const double LI_scale = std::max({1.0, std::abs(expected_LI),
                                      std::abs(step.ion_divergence_closure_nuclei_cm3_s)});
    require(std::isfinite(step.ion_divergence_closure_nuclei_cm3_s),
            "Local marcher did not report a finite L_I");
    require(std::abs(step.ion_divergence_closure_nuclei_cm3_s - expected_LI) /
                LI_scale < 1e-12,
            "Local marcher L_I does not satisfy nuclei conservation");

    auto cfg_changed_density = cfg;
    cfg_changed_density.plasma.total_density *= 7.0;
    const auto step_changed_density = dcr::solver::solve_cell_implicit(
        cfg_changed_density,
        atomic_data,
        plasma,
        eedf.grid,
        boundary,
        levels,
        nP,
        flowA_old,
        flowM_old,
        cfg.grid.length_cm,
        0.0,
        cfg.grid.length_cm,
        1,
        false,
        false
    );
    const double changed_total_nuclei =
        test_dcr::nuclei_total_compact(step_changed_density.nP_new, boundary.P_indices, levels) +
        test_dcr::nuclei_total_compact(step_changed_density.flowA_new, boundary.A_indices, levels) +
        test_dcr::nuclei_total_compact(step_changed_density.flowM_new, boundary.M_indices, levels);
    const double changed_density_error =
        std::abs(changed_total_nuclei - cfg_changed_density.plasma.total_density) /
        std::max(1.0, cfg_changed_density.plasma.total_density);
    require(changed_density_error < 1e-12,
            "Local marcher ignored the changed prescribed nuclei density");

    auto cfg_ion_override = cfg;
    cfg_ion_override.numerics.adaptive_recycling_domain.apply_ion_closure = true;
    dcr::solver::AdaptiveTransportProfile ion_override_profile;
    ion_override_profile.ion_divergence_nuclei_cm3_s = {0.0, 2.5e19};
    const auto ion_override_step = dcr::solver::solve_cell_implicit(
        cfg_ion_override,
        atomic_data,
        plasma,
        eedf.grid,
        boundary,
        levels,
        nP,
        flowA_old,
        flowM_old,
        cfg.grid.length_cm,
        0.0,
        cfg.grid.length_cm,
        1,
        false,
        false,
        &ion_override_profile
    );
    require(std::abs(ion_override_step.ion_divergence_closure_nuclei_cm3_s + 2.5e19) /
                2.5e19 < 1e-12,
            "Adaptive ion-divergence override was not applied with the required sign");

    for (const char* marching_solver : {"picard"}) {
        auto cfg_variable = cfg;
        cfg_variable.numerics.marching_solver = marching_solver;
        cfg_variable.numerics.adaptive_recycling_domain.closure_mode =
            "variable_nuclei_balance";
        cfg_variable.numerics.adaptive_recycling_domain.ion_velocity_length_cm = 2.0;
        const auto variable_step = dcr::solver::solve_cell_implicit(
            cfg_variable,
            atomic_data,
            plasma,
            eedf.grid,
            boundary,
            levels,
            nP,
            flowA_old,
            flowM_old,
            1.0e-3,
            0.0,
            1.0e-3,
            1,
            false,
            false
        );
        require(variable_step.variable_nuclei_balance_closure,
                "Variable nuclei closure was not selected");
        require(variable_step.converged,
                "Variable nuclei closure did not converge");
        test_dcr::assert_all_finite_nonnegative(variable_step.nP_new);
        const double variable_total_nuclei =
            test_dcr::nuclei_total_compact(variable_step.nP_new, boundary.P_indices, levels) +
            test_dcr::nuclei_total_compact(variable_step.flowA_new, boundary.A_indices, levels) +
            test_dcr::nuclei_total_compact(variable_step.flowM_new, boundary.M_indices, levels);
        const double variable_density_error =
            std::abs(variable_total_nuclei - cfg_variable.plasma.total_density) /
            std::max(1.0, cfg_variable.plasma.total_density);
        require(variable_density_error > 1e-6,
                "Variable nuclei closure was projected to the prescribed density");
        const double variable_balance_scale = std::max({
            1.0,
            std::abs(variable_step.variable_ion_divergence_nuclei_cm3_s),
            std::abs(variable_step.variable_flowA_divergence_nuclei_cm3_s),
            std::abs(variable_step.variable_flowM_divergence_nuclei_cm3_s),
            std::abs(variable_step.variable_neutral_exhaust_nuclei_cm3_s)
        });
        require(std::abs(variable_step.variable_balance_residual_cm3_s) /
                    variable_balance_scale < 1e-10,
                "Variable nuclei closure did not satisfy the local balance row");
    }

    std::cout << "[PASS] Marching implicit one-step checks.\n";
    return 0;
}
