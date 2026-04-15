#include "QSSAtomicDonorReduction.hpp"

#include "QSSLinearSolve.hpp"
#include "QSSMatrixBlocks.hpp"

namespace dcr::solver {
using namespace qss_model_detail;

void populate_atomic_donor_reduction(QSSLocalBlocks& out,
                                     const dcr::base::Matrix& R_full,
                                     const QSSLayout& layout,
                                     const std::vector<dcr::atomic::EnergyLevel>& levels) {
    if (layout.retained_atomic_donor_ground < 0) return;
    std::vector<int> donor_transient;
    for (int gi : layout.full_atomic_donor_indices) if (gi != layout.retained_atomic_donor_ground) donor_transient.push_back(gi);
    const std::vector<int> donor_ground = {layout.retained_atomic_donor_ground};
    const dcr::base::Matrix S_rG = extract_source_block(R_full, layout.retained_p_indices, donor_ground, levels, true, false, false);
    const dcr::base::Matrix R_GG = extract_block(R_full, donor_ground, donor_ground);
    if (donor_transient.empty()) {
        if (R_GG.rows() == 1 && R_GG.cols() == 1) out.atomic_flow_effective_rate_s = R_GG(0, 0);
        out.source_rA = S_rG;
        return;
    }
    const dcr::base::Matrix S_rT = extract_source_block(R_full, layout.retained_p_indices, donor_transient, levels, true, false, false);
    const dcr::base::Matrix R_GT = extract_block(R_full, donor_ground, donor_transient);
    const dcr::base::Matrix R_TG = extract_block(R_full, donor_transient, donor_ground);
    const dcr::base::Matrix R_TT = extract_block(R_full, donor_transient, donor_transient);
    const dcr::base::Matrix solved = solve_linear_matrix(R_TT, R_TG);
    out.atomic_flow_transient_from_ground = solved;
    out.atomic_flow_effective_rate_s = (R_GG - R_GT * solved)(0, 0);
    out.source_rA = S_rG - S_rT * solved;
}

} // namespace dcr::solver
