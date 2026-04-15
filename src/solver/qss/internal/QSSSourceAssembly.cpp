#include "QSSSourceAssembly.hpp"

namespace dcr::solver {

void build_qss_source_vectors(const QSSLocalBlocks& blocks,
                              const dcr::base::Vector& flowA,
                              const dcr::base::Vector& flowM,
                              dcr::base::Vector& source_r,
                              dcr::base::Vector& source_t) {
    source_r = dcr::base::Vector::Zero(blocks.R_rr.rows());
    source_t = dcr::base::Vector::Zero(blocks.R_tt.rows());
    if (blocks.source_rA.cols() == 1 && flowA.size() == 1) source_r += blocks.source_rA.col(0) * flowA(0);
    if (blocks.source_rM.cols() == flowM.size() && flowM.size() > 0) source_r += blocks.source_rM * flowM;
}

} // namespace dcr::solver
