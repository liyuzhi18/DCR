#include "QSSRecyclingModel.hpp"

namespace dcr::solver {

RecyclingModel build_recycling_model(const BoundaryPhaseResult& boundary,
                                     const std::vector<dcr::atomic::EnergyLevel>& levels,
                                     const std::vector<int>& p_pos,
                                     double c_A,
                                     double c_A_base,
                                     double c_M_atom) {
    RecyclingModel model;
    for (int gi : boundary.ion_indices) {
        if (gi < 0 || gi >= static_cast<int>(p_pos.size()) || p_pos[static_cast<size_t>(gi)] < 0) continue;
        const int mu_i = (gi >= 0 && gi < static_cast<int>(levels.size())) ? std::max(1, levels[static_cast<size_t>(gi)].atomicity) : 1;
        const bool molecular_ion = mu_i > 1;
        model.ion_p_cols.push_back(p_pos[static_cast<size_t>(gi)]);
        model.recA_coeff.push_back(molecular_ion ? (c_A_base * static_cast<double>(mu_i)) : c_A);
        model.recM_coeff.push_back(molecular_ion ? 0.0 : c_M_atom);
    }
    return model;
}

} // namespace dcr::solver
