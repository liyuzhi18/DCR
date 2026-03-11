#pragma once

#include "../../atomic/AtomicData.hpp"
#include "../../base/Types.hpp"
#include "../../io/ConfigStructs.hpp"
#include "../boundary/BoundaryPhase.hpp"
#include "../marching/LocalSystemAssembler.hpp"

#include <vector>

namespace dcr::solver {

struct AtomicEffectiveRates {
    bool valid = false;
    int atom_ground_index = -1;
    int ion_ground_index = -1;
    double scd_cm3_s = 0.0;
    double acd_cm3_s = 0.0;
};

struct AtomicQSSDiagnostics {
    bool valid = false;
    std::vector<int> excited_indices;
    double transport_frequency_s = 0.0;
    dcr::base::Vector transport_rate_cm3_s;
    dcr::base::Vector local_source_rate_cm3_s;
    dcr::base::Vector local_loss_rate_cm3_s;
    dcr::base::Vector local_loss_frequency_s;
    dcr::base::Vector transport_to_local_ratio;
    dcr::base::Vector transport_to_loss_frequency_ratio;
    double max_transport_to_local_ratio = 0.0;
    double max_transport_to_loss_frequency_ratio = 0.0;
};

struct RateDiagnosticSnapshot {
    double electron_temperature_eV = 0.0;
    double ion_temperature_eV = 0.0;
    double electron_density_cm3 = 0.0;
    AtomicEffectiveRates atomic_effective;
    AtomicQSSDiagnostics atomic_qss;
};

class AtomicRateCalculator {
public:
    explicit AtomicRateCalculator(const dcr::atomic::AtomicData& atomic_data);

    RateDiagnosticSnapshot evaluate(
        const dcr::io::Config& config,
        const BoundaryPhaseResult& boundary,
        const LocalSystem& local_system,
        const dcr::base::Vector& background_population,
        double x_cm) const;

private:
    const std::vector<dcr::atomic::EnergyLevel>& levels_;
    std::vector<int> atomic_subspace_indices_;
    std::vector<int> qss_excited_indices_;
    std::vector<int> atomic_global_to_subspace_;
    int atom_ground_index_ = -1;
    int ion_ground_index_ = -1;
};

} // namespace dcr::solver
