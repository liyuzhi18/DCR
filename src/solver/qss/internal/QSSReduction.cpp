#include "QSSReduction.hpp"

#include "QSSLinearSolve.hpp"

namespace dcr::solver {
using namespace qss_model_detail;

dcr::base::Matrix build_qss_effective_retained_matrix(
    const QSSLocalBlocks& blocks,
    bool exclude_molecular_ion_from_transient_reconstruction) {
    const dcr::base::Matrix& R_rr_used =
        exclude_molecular_ion_from_transient_reconstruction
        ? blocks.R_rr_with_molecular_ion_transient_source_redirected_to_ground
        : blocks.R_rr;
    if (blocks.R_tt.rows() == 0) return R_rr_used;
    const dcr::base::Matrix& R_tr_used =
        exclude_molecular_ion_from_transient_reconstruction
        ? blocks.R_tr_excluding_molecular_ion
        : blocks.R_tr;
    return R_rr_used - blocks.transient_to_retained * R_tr_used;
}

dcr::base::Vector solve_qss_transient_atomic(const QSSLocalBlocks& blocks,
                                             const dcr::base::Vector& retained_background,
                                             const dcr::base::Vector& transient_rhs,
                                             bool exclude_molecular_ion_from_transient_reconstruction) {
    if (blocks.R_tt.rows() == 0) return dcr::base::Vector();
    const dcr::base::Matrix& R_tr_used =
        exclude_molecular_ion_from_transient_reconstruction
        ? blocks.R_tr_excluding_molecular_ion
        : blocks.R_tr;
    return solve_linear(blocks.R_tt, transient_rhs - R_tr_used * retained_background);
}

dcr::base::Vector reconstruct_qss_atomic_flow_transient(
    const QSSLocalBlocks& blocks,
    const dcr::base::Vector& flow_atomic_ground) {
    if (blocks.atomic_flow_transient_from_ground.rows() == 0) return dcr::base::Vector();
    return -blocks.atomic_flow_transient_from_ground * flow_atomic_ground;
}

} // namespace dcr::solver
