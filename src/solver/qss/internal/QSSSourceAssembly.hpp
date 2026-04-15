#pragma once

#include "QSSLocalBlocks.hpp"

namespace dcr::solver {

void build_qss_source_vectors(const QSSLocalBlocks& blocks,
                              const dcr::base::Vector& flowA,
                              const dcr::base::Vector& flowM,
                              dcr::base::Vector& source_r,
                              dcr::base::Vector& source_t);

} // namespace dcr::solver
