#include "QSSLocalBlocks.hpp"

#include "QSSAtomicDonorReduction.hpp"
#include "QSSLinearSolve.hpp"
#include "QSSMatrixBlocks.hpp"

namespace dcr::solver {
using namespace qss_model_detail;

QSSLocalBlocks build_qss_local_blocks(const dcr::base::Matrix& R_full,
                                      const QSSLayout& layout,
                                      const std::vector<dcr::atomic::EnergyLevel>& levels) {
    QSSLocalBlocks out;
    out.R_rr = extract_block(R_full, layout.retained_p_indices, layout.retained_p_indices);
    out.R_rr_with_molecular_ion_transient_source_redirected_to_ground = out.R_rr;
    out.R_rt = extract_block(R_full, layout.retained_p_indices, layout.transient_atomic_indices);
    out.R_tr = extract_block(R_full, layout.transient_atomic_indices, layout.retained_p_indices);
    out.R_tr_excluding_molecular_ion = out.R_tr;
    for (int pos : layout.retained_molecular_ion_positions) {
        if (pos >= 0 && pos < out.R_tr_excluding_molecular_ion.cols()) {
            out.R_tr_excluding_molecular_ion.col(pos).setZero();
        }
    }
    if (!layout.retained_atomic_ground_indices.empty()) {
        const int ground_global = layout.retained_atomic_ground_indices.front();
        if (ground_global >= 0 &&
            ground_global < static_cast<int>(layout.full_to_retained_p.size())) {
            const int ground_row = layout.full_to_retained_p[static_cast<size_t>(ground_global)];
            if (ground_row >= 0 &&
                ground_row < out.R_rr_with_molecular_ion_transient_source_redirected_to_ground.rows()) {
                for (int pos : layout.retained_molecular_ion_positions) {
                    if (pos < 0 || pos >= out.R_tr.cols()) continue;
                    out.R_rr_with_molecular_ion_transient_source_redirected_to_ground(ground_row, pos) +=
                        out.R_tr.col(pos).sum();
                }
            }
        }
    }
    out.R_tt = extract_block(R_full, layout.transient_atomic_indices, layout.transient_atomic_indices);
    if (out.R_tt.rows() > 0) {
        const dcr::base::Matrix identity = dcr::base::Matrix::Identity(out.R_tt.rows(), out.R_tt.cols());
        out.transient_to_retained = out.R_rt * solve_linear_matrix(out.R_tt, identity);
    }
    populate_atomic_donor_reduction(out, R_full, layout, levels);
    out.source_rM = extract_source_block(R_full, layout.retained_p_indices, layout.retained_molecular_donor_indices, levels, true, true, true);
    return out;
}

} // namespace dcr::solver
