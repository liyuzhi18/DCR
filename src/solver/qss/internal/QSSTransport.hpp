#pragma once

#include "QSSLayout.hpp"

namespace dcr::solver {

dcr::base::Vector build_retained_transport_diag(const BoundaryPhaseResult& boundary,
                                                const std::vector<dcr::atomic::EnergyLevel>& levels,
                                                const QSSLayout& layout,
                                                double ion_coeff,
                                                double atom_coeff,
                                                double molecule_coeff);
double implicit_atomic_flow_step(double old_value,
                                 double rate_s,
                                 double exhaust_over_w,
                                 double speed_cm_s,
                                 double dx_cm);

} // namespace dcr::solver
