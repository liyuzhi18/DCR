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
    int first_excited_index = -1;
    double transport_frequency_s = 0.0;
    double first_excited_local_loss_frequency_s = 0.0;
    double relaxation_length_cm = 0.0;
    dcr::base::Vector transport_rate_cm3_s;
    dcr::base::Vector local_source_rate_cm3_s;
    dcr::base::Vector local_loss_rate_cm3_s;
    dcr::base::Vector local_loss_frequency_s;
    dcr::base::Vector transport_to_local_ratio;
    dcr::base::Vector transport_to_loss_frequency_ratio;
    double max_transport_to_local_ratio = 0.0;
    double max_transport_to_loss_frequency_ratio = 0.0;
};

struct AtomicSourceDiagnostics {
    double effective_eir_rate_cm3_s = 0.0;
    double mar_h_source_rate_cm3_s = 0.0;
    double flow_h_source_rate_cm3_s = 0.0;
    double molecular_flow_ionization_rate_cm3_s = 0.0;
    double molecular_flow_charge_exchange_rate_cm3_s = 0.0;
    double molecular_dissociation_rate_cm3_s = 0.0;
    double full_ion_nuclei_source_cm3_s = 0.0;
    double full_ion_nuclei_sink_cm3_s = 0.0;
};

struct PowerLossDiagnostics {
    double atomic_ionization_W_cm3 = 0.0;
    double molecular_ionization_W_cm3 = 0.0;
    double molecular_dissociation_W_cm3 = 0.0;
    double total_electron_inelastic_W_cm3 = 0.0;
};

struct GroupedSourceDiagnostics {
    double H_plus_cm3_s = 0.0;
    double H_cm3_s = 0.0;
    double H2_cm3_s = 0.0;
    double H_minus_cm3_s = 0.0;
    double H2_plus_cm3_s = 0.0;
};

struct FlowToLocalSourceDiagnostics {
    double H_plus_cm3_s = 0.0;
    double H_cm3_s = 0.0;
    double H2_cm3_s = 0.0;
    double H_minus_cm3_s = 0.0;
    double H2_plus_cm3_s = 0.0;
};

struct MolecularIonTransportDiagnostics {
    std::vector<int> state_indices;
    dcr::base::Vector mcx_production_cm3_s;
    dcr::base::Vector mi_production_cm3_s;
    dcr::base::Vector dr_h_source_frequency_s;
    dcr::base::Vector target_velocity_cm_s;
    dcr::base::Matrix generator_s;
    double simple_branching_dr_mar_h_source_cm3_s = 0.0;
};

struct RateDiagnosticSnapshot {
    double electron_temperature_eV = 0.0;
    double ion_temperature_eV = 0.0;
    double electron_density_cm3 = 0.0;
    AtomicEffectiveRates atomic_effective;
    AtomicQSSDiagnostics atomic_qss;
    AtomicSourceDiagnostics atomic_sources;
    PowerLossDiagnostics power_loss;
    GroupedSourceDiagnostics grouped_sources;
    FlowToLocalSourceDiagnostics flow_to_local_sources;
    MolecularIonTransportDiagnostics molecular_ion_transport;
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

    RateDiagnosticSnapshot evaluate(
        const dcr::io::Config& config,
        const BoundaryPhaseResult& boundary,
        const LocalSystem& local_system,
        const dcr::base::Vector& background_population,
        double x_cm,
        const dcr::state::PlasmaState& plasma,
        const EEDFGridView& grid,
        double h2plus_transport_rate_cm3_s = 0.0) const;

private:
    const dcr::atomic::AtomicData& atomic_data_;
    const std::vector<dcr::atomic::EnergyLevel>& levels_;
    std::vector<int> atomic_subspace_indices_;
    std::vector<int> qss_excited_indices_;
    std::vector<int> molecular_ion_indices_;
    std::vector<int> atomic_global_to_subspace_;
    std::vector<int> molecular_ion_global_to_subspace_;
    int atom_ground_index_ = -1;
    int ion_ground_index_ = -1;
    int first_excited_index_ = -1;
};

} // namespace dcr::solver
