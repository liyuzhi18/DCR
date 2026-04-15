#include "QSSBoundaryTemplate.hpp"

namespace dcr::solver {

BoundaryPhaseResult make_qss_boundary_template(const BoundaryPhaseResult& full_boundary,
                                               const QSSLayout& layout) {
    BoundaryPhaseResult qss = full_boundary;
    qss.P_indices = layout.retained_p_indices;
    qss.A_indices.clear();
    if (layout.retained_atomic_donor_ground >= 0) qss.A_indices.push_back(layout.retained_atomic_donor_ground);
    qss.M_indices = layout.retained_molecular_donor_indices;
    qss.atom_bg_indices = layout.retained_atomic_ground_indices;
    qss.mol_bg_indices = layout.retained_molecule_indices;
    qss.ion_indices = layout.retained_positive_ion_indices;
    qss.atom_ground = layout.retained_atomic_ground_indices.empty() ? -1 : layout.retained_atomic_ground_indices.front();
    qss.have_flow_last = false;
    qss.flowA_last = dcr::base::Vector::Zero(static_cast<int>(qss.A_indices.size()));
    qss.flowM_last = dcr::base::Vector::Zero(static_cast<int>(qss.M_indices.size()));
    return qss;
}

} // namespace dcr::solver
