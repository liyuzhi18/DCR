#pragma once

#include "QSSLocalBlocks.hpp"

namespace dcr::solver {

dcr::base::Matrix build_qss_effective_retained_matrix(
    const QSSLocalBlocks& blocks,
    bool exclude_molecular_ion_from_transient_reconstruction);

dcr::base::Vector solve_qss_transient_atomic(
    const QSSLocalBlocks& blocks,
    const dcr::base::Vector& retained_background,
    const dcr::base::Vector& transient_rhs,
    bool exclude_molecular_ion_from_transient_reconstruction);

dcr::base::Vector reconstruct_qss_atomic_flow_transient(
    const QSSLocalBlocks& blocks,
    const dcr::base::Vector& flow_atomic_ground);

} // namespace dcr::solver
