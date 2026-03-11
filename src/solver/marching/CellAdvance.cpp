#include "CellAdvance.hpp"

#include "../../physics/Sheath.hpp"

#include <algorithm>

namespace dcr::solver {

namespace {

// Extract a dense sub-block R(indices, indices) from the full CR matrix.
dcr::base::Matrix build_block_matrix(const dcr::base::Matrix& R_full,
                                     const std::vector<int>& indices) {
    const int n = static_cast<int>(indices.size());
    dcr::base::Matrix block = dcr::base::Matrix::Zero(n, n);
    for (int i = 0; i < n; ++i) {
        const int gi = indices[static_cast<size_t>(i)];
        if (gi < 0 || gi >= R_full.rows()) continue;
        for (int j = 0; j < n; ++j) {
            const int gj = indices[static_cast<size_t>(j)];
            if (gj < 0 || gj >= R_full.cols()) continue;
            block(i, j) = R_full(gi, gj);
        }
    }
    return block;
}

// First-order explicit fallback used only if implicit matrix solve is singular.
dcr::base::Vector explicit_fallback_step(const dcr::base::Vector& n_old,
                                         const dcr::base::Matrix& R_block,
                                         double ex_over_w,
                                         double u,
                                         double dx_cm) {
    if (u <= 0.0 || dx_cm <= 0.0 || n_old.size() == 0) return n_old;
    const double coef = dx_cm / u;
    dcr::base::Vector rhs = R_block * n_old - ex_over_w * n_old;
    dcr::base::Vector n_new = n_old + coef * rhs;
    for (int i = 0; i < n_new.size(); ++i) {
        if (n_new(i) < 0.0) n_new(i) = 0.0;
    }
    return n_new;
}

// Backward-Euler solve for one recycling block (A or M):
// (I - dt*R + dt*ex*I) n_new = n_old, with dt = dx/u.
dcr::base::Vector implicit_step(const dcr::base::Vector& n_old,
                                const dcr::base::Matrix& R_block,
                                double ex_over_w,
                                double u,
                                double dx_cm) {
    if (u <= 0.0 || dx_cm <= 0.0 || n_old.size() == 0) return n_old;

    const int n = n_old.size();
    const double coef = dx_cm / u;
    dcr::base::Matrix lhs = dcr::base::Matrix::Identity(n, n) - coef * R_block;
    lhs.diagonal().array() += coef * ex_over_w;

    Eigen::FullPivLU<dcr::base::Matrix> lu(lhs);
    if (lu.rank() < n) {
        // If implicit matrix is rank-deficient, keep marching robust with explicit fallback.
        return explicit_fallback_step(n_old, R_block, ex_over_w, u, dx_cm);
    }

    dcr::base::Vector n_new = lu.solve(n_old);
    for (int i = 0; i < n_new.size(); ++i) {
        if (n_new(i) < 0.0) n_new(i) = 0.0;
    }
    return n_new;
}

} // namespace

FlowAdvanceResult advance_recycling_flow_one_step(
    const dcr::io::Config& config,
    const dcr::atomic::AtomicData& atomic_data,
    const BoundaryPhaseResult& boundary,
    const LocalSystem& local_system,
    const dcr::base::Vector& flowA,
    const dcr::base::Vector& flowM,
    double dx_cm) {

    (void)atomic_data;

    FlowAdvanceResult out;
    out.flowA_next = flowA;
    out.flowM_next = flowM;
    out.rhsA = dcr::base::Vector::Zero(flowA.size());
    out.rhsM = dcr::base::Vector::Zero(flowM.size());

    out.c_s_A = dcr::physics::calculate_thermal_speed(
        config.plasma.neutral_atom_temperature_eV, boundary.atom_mass_amu
    );
    out.c_s_M = dcr::physics::calculate_thermal_speed(
        config.plasma.neutral_molecule_temperature_eV, boundary.molecule_mass_amu
    );
    const double w = std::max(config.grid.poloidal_width_cm, 1e-12);
    // Flow exhaust terms in Eq. (1.346) marching.
    const double exA_over_w = out.c_s_A / w;
    const double exM_over_w = out.c_s_M / w;

    // Implicit backward-Euler step for Eq. (1.346) block form:
    // (I - dx/u * R_AA + dx/u * ex_A I) * n_A^{k+1} = n_A^k
    // (I - dx/u * R_MM + dx/u * ex_M I) * n_M^{k+1} = n_M^k
    const dcr::base::Matrix R_AA = build_block_matrix(local_system.R_full, boundary.A_indices);
    const dcr::base::Matrix R_MM = build_block_matrix(local_system.R_full, boundary.M_indices);

    out.flowA_next = implicit_step(flowA, R_AA, exA_over_w, boundary.u_A, dx_cm);
    out.flowM_next = implicit_step(flowM, R_MM, exM_over_w, boundary.u_M, dx_cm);

    if (out.flowA_next.size() == flowA.size() && R_AA.rows() == out.flowA_next.size()) {
        out.rhsA = R_AA * out.flowA_next - exA_over_w * out.flowA_next;
    }
    if (out.flowM_next.size() == flowM.size() && R_MM.rows() == out.flowM_next.size()) {
        out.rhsM = R_MM * out.flowM_next - exM_over_w * out.flowM_next;
    }

    return out;
}

} // namespace dcr::solver
