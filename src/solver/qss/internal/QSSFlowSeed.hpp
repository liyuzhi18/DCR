#pragma once

#include "QSSLayout.hpp"

namespace dcr::solver {

void assign_qss_flow_distribution(const BoundaryPhaseResult& boundary,
                                  double n_A_total,
                                  double n_M_total,
                                  dcr::base::Vector& flowA,
                                  dcr::base::Vector& flowM);
dcr::base::Vector extract_transient_seed(const QSSLayout& layout,
                                         const dcr::base::Vector& full_population);
dcr::base::Vector extract_atomic_flow_seed(const QSSLayout& layout);
dcr::base::Vector extract_molecular_flow_seed(const BoundaryPhaseResult& qss_boundary);

} // namespace dcr::solver
