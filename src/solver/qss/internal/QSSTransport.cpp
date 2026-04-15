#include "QSSTransport.hpp"

#include <cmath>

namespace dcr::solver {

dcr::base::Vector build_retained_transport_diag(const BoundaryPhaseResult& boundary,
                                                const std::vector<dcr::atomic::EnergyLevel>& levels,
                                                const QSSLayout& layout,
                                                double ion_coeff,
                                                double atom_coeff,
                                                double molecule_coeff) {
    dcr::base::Vector diag = dcr::base::Vector::Zero(static_cast<int>(boundary.P_indices.size()));
    for (int i = 0; i < diag.size(); ++i) {
        const int gi = boundary.P_indices[static_cast<size_t>(i)];
        if (gi < 0 || gi >= static_cast<int>(levels.size())) continue;
        const auto& level = levels[static_cast<size_t>(gi)];
        if (level.type == dcr::atomic::SpeciesType::Ion && level.charge > 0) diag(i) = ion_coeff;
        else if (level.type == dcr::atomic::SpeciesType::Atom && level.charge == 0) diag(i) = atom_coeff;
        else if (level.type == dcr::atomic::SpeciesType::Molecule && level.charge == 0) diag(i) = molecule_coeff;
    }
    (void)layout;
    return diag;
}

double implicit_atomic_flow_step(double old_value, double rate_s, double exhaust_over_w,
                                 double speed_cm_s, double dx_cm) {
    if (speed_cm_s <= 0.0 || dx_cm <= 0.0) return std::max(0.0, old_value);
    const double lhs = 1.0 - (dx_cm / speed_cm_s) * rate_s + (dx_cm / speed_cm_s) * exhaust_over_w;
    // lhs <= 0 means net absorption rate exceeds u/dx: flow is fully absorbed
    // within the cell. Return 0 rather than old_value to avoid stalling.
    if (!std::isfinite(lhs) || lhs <= 1e-14) return 0.0;
    return std::max(0.0, old_value / lhs);
}

} // namespace dcr::solver
