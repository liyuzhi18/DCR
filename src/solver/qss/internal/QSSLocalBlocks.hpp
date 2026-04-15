#pragma once

#include "QSSLayout.hpp"

#include "../../../base/Types.hpp"

namespace dcr::solver {

struct QSSLocalBlocks {
    dcr::base::Matrix R_rr;
    dcr::base::Matrix R_rr_with_molecular_ion_transient_source_redirected_to_ground;
    dcr::base::Matrix R_rt;
    dcr::base::Matrix R_tr;
    dcr::base::Matrix R_tr_excluding_molecular_ion;
    dcr::base::Matrix R_tt;
    dcr::base::Matrix source_rA;
    dcr::base::Matrix source_rM;
    dcr::base::Matrix transient_to_retained;
    dcr::base::Matrix atomic_flow_transient_from_ground;
    double atomic_flow_effective_rate_s = 0.0;
};

QSSLocalBlocks build_qss_local_blocks(
    const dcr::base::Matrix& R_full,
    const QSSLayout& layout,
    const std::vector<dcr::atomic::EnergyLevel>& levels);

} // namespace dcr::solver
