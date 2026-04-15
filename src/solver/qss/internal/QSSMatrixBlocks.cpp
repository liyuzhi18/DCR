#include "QSSMatrixBlocks.hpp"

namespace dcr::solver::qss_model_detail {

dcr::base::Matrix extract_block(const dcr::base::Matrix& full,
                                const std::vector<int>& rows,
                                const std::vector<int>& cols) {
    dcr::base::Matrix block = dcr::base::Matrix::Zero(static_cast<int>(rows.size()), static_cast<int>(cols.size()));
    for (int i = 0; i < static_cast<int>(rows.size()); ++i) {
        const int gi = rows[static_cast<size_t>(i)];
        if (gi < 0 || gi >= full.rows()) continue;
        for (int j = 0; j < static_cast<int>(cols.size()); ++j) {
            const int gj = cols[static_cast<size_t>(j)];
            if (gj >= 0 && gj < full.cols()) block(i, j) = full(gi, gj);
        }
    }
    return block;
}

dcr::base::Matrix extract_source_block(const dcr::base::Matrix& full,
                                       const std::vector<int>& retained_rows,
                                       const std::vector<int>& donor_cols,
                                       const std::vector<dcr::atomic::EnergyLevel>& levels,
                                       bool allow_atomic_ion_rows,
                                       bool allow_atomic_ground_rows,
                                       bool allow_molecular_ion_rows) {
    dcr::base::Matrix block = dcr::base::Matrix::Zero(static_cast<int>(retained_rows.size()), static_cast<int>(donor_cols.size()));
    for (int i = 0; i < static_cast<int>(retained_rows.size()); ++i) {
        const int gi = retained_rows[static_cast<size_t>(i)];
        if (gi < 0 || gi >= full.rows() || gi >= static_cast<int>(levels.size())) continue;
        const auto& level = levels[static_cast<size_t>(gi)];
        const bool enabled =
            (allow_atomic_ion_rows && level.type == dcr::atomic::SpeciesType::Ion && level.charge > 0 && level.atomicity == 1) ||
            (allow_atomic_ground_rows && level.type == dcr::atomic::SpeciesType::Atom && level.charge == 0 && level.atomicity == 1) ||
            (allow_molecular_ion_rows && level.type == dcr::atomic::SpeciesType::Ion && level.charge > 0 && level.atomicity > 1);
        if (!enabled) continue;
        for (int j = 0; j < static_cast<int>(donor_cols.size()); ++j) {
            const int gj = donor_cols[static_cast<size_t>(j)];
            if (gj >= 0 && gj < full.cols()) block(i, j) = full(gi, gj);
        }
    }
    return block;
}

} // namespace dcr::solver::qss_model_detail
