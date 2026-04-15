#include "QSSPopulationLift.hpp"

namespace dcr::solver {

dcr::base::Vector make_qss_retained_background(const BoundaryPhaseResult& qss_boundary,
                                               const dcr::base::Vector& full_population) {
    dcr::base::Vector retained = dcr::base::Vector::Zero(static_cast<int>(qss_boundary.P_indices.size()));
    for (int i = 0; i < retained.size(); ++i) {
        const int gi = qss_boundary.P_indices[static_cast<size_t>(i)];
        if (gi >= 0 && gi < full_population.size()) retained(i) = std::max(full_population(gi), 0.0);
    }
    return retained;
}

dcr::base::Vector lift_qss_background_to_full(int total_states,
                                              const BoundaryPhaseResult& qss_boundary,
                                              const QSSLayout& layout,
                                              const dcr::base::Vector& retained_background,
                                              const dcr::base::Vector& transient_atomic) {
    dcr::base::Vector full = dcr::base::Vector::Zero(total_states);
    for (int i = 0; i < retained_background.size() && i < static_cast<int>(qss_boundary.P_indices.size()); ++i) {
        const int gi = qss_boundary.P_indices[static_cast<size_t>(i)];
        if (gi >= 0 && gi < total_states) full(gi) = std::max(retained_background(i), 0.0);
    }
    for (int i = 0; i < transient_atomic.size() && i < static_cast<int>(layout.transient_atomic_indices.size()); ++i) {
        const int gi = layout.transient_atomic_indices[static_cast<size_t>(i)];
        if (gi >= 0 && gi < total_states) full(gi) = std::max(transient_atomic(i), 0.0);
    }
    return full;
}

} // namespace dcr::solver
