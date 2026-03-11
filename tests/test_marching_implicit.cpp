#include <algorithm>
#include <cassert>
#include <cmath>
#include <iostream>

#include "../src/atomic/AtomicData.hpp"
#include "../src/physics/WallBoundary.hpp"
#include "../src/solver/boundary/BoundaryPhase.hpp"
#include "../src/solver/marching/CellAdvance.hpp"
#include "../src/solver/marching/LocalSystemAssembler.hpp"
#include "TestDCRSetup.hpp"

namespace {

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
        cfg.grid.poloidal_width_cm,
        boundary.u_A,
        cfg.grid.length_cm,
        flowA_old,
        advanced.flowA_next
    );
    const double resM = implicit_residual_norm(
        R_MM,
        advanced.c_s_M,
        cfg.grid.poloidal_width_cm,
        boundary.u_M,
        cfg.grid.length_cm,
        flowM_old,
        advanced.flowM_next
    );

    // For non-singular implicit systems, residual should be near solver tolerance.
    assert(resA < 1e-6);
    assert(resM < 1e-6);

    std::cout << "[PASS] Marching implicit one-step checks.\n";
    return 0;
}
