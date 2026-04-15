#include "QSSFlowSeed.hpp"

namespace dcr::solver {

void assign_qss_flow_distribution(const BoundaryPhaseResult& boundary, double n_A_total, double n_M_total,
                                  dcr::base::Vector& flowA, dcr::base::Vector& flowM) {
    flowA.setZero(); flowM.setZero();
    if (flowA.size() > 0) flowA(0) = std::max(0.0, n_A_total);
    for (size_t j = 0; j < boundary.M_indices.size(); ++j)
        if (boundary.M_indices[j] == boundary.molecule_ground && static_cast<int>(j) < flowM.size())
            { flowM(static_cast<int>(j)) = std::max(0.0, n_M_total); break; }
}

dcr::base::Vector extract_transient_seed(const QSSLayout& layout, const dcr::base::Vector&) {
    return dcr::base::Vector::Zero(static_cast<int>(layout.transient_atomic_indices.size()));
}

dcr::base::Vector extract_atomic_flow_seed(const QSSLayout& layout) {
    return dcr::base::Vector::Zero(layout.retained_atomic_donor_ground >= 0 ? 1 : 0);
}

dcr::base::Vector extract_molecular_flow_seed(const BoundaryPhaseResult& qss_boundary) {
    return dcr::base::Vector::Zero(static_cast<int>(qss_boundary.M_indices.size()));
}

} // namespace dcr::solver
