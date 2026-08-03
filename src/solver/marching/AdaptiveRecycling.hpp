#pragma once

#include "../../atomic/EnergyLevel.hpp"
#include "../../base/Types.hpp"

#include <vector>

namespace dcr::solver {

struct RecyclingLayerEstimate {
    bool found = false;
    int index = -1;
    double L1_cm = 0.0;
    std::vector<double> atomic_flow_fraction;
};

struct FlowDepletionEstimate {
    bool found = false;
    int index = -1;
    double length_cm = 0.0;
    std::vector<double> flow_fraction;
};

struct NeutralTransportCompensation {
    std::vector<double> remainder_nuclei_cm3_s;
    std::vector<double> atomic_group_nuclei_divergence_cm3_s;
    std::vector<double> molecular_group_nuclei_divergence_cm3_s;
    std::vector<dcr::base::Vector> atomic_state_divergence_cm3_s;
    std::vector<dcr::base::Vector> molecular_state_divergence_cm3_s;
};

struct AdaptiveTransportProfile {
    double L1_cm = 0.0;
    double LM_cm = 0.0;
    std::vector<double> ion_divergence_nuclei_cm3_s;
    std::vector<double> pump_exhaust_nuclei_cm3_s;
    std::vector<double> neutral_remainder_nuclei_cm3_s;
    std::vector<double> atomic_group_nuclei_divergence_cm3_s;
    std::vector<double> molecular_group_nuclei_divergence_cm3_s;
    std::vector<dcr::base::Vector> atomic_state_divergence_cm3_s;
    std::vector<dcr::base::Vector> molecular_state_divergence_cm3_s;
};

std::vector<double> normalized_atomic_flow_fraction(
    const std::vector<dcr::base::Vector>& flowA);

std::vector<double> normalized_flow_fraction(
    const std::vector<dcr::base::Vector>& flow);

FlowDepletionEstimate estimate_flow_depletion_layer(
    const std::vector<double>& x_cm,
    const std::vector<dcr::base::Vector>& flow,
    double epsilon);

RecyclingLayerEstimate estimate_recycling_layer_from_atomic_flow(
    const std::vector<double>& x_cm,
    const std::vector<dcr::base::Vector>& flowA,
    double epsilon_A);

std::vector<double> linear_presheath_velocity_profile_cm_s(
    const std::vector<double>& x_cm,
    double L1_cm,
    double bohm_speed_cm_s,
    double floor_fraction = 1e-2);

std::vector<double> finite_difference_derivative(
    const std::vector<double>& x_cm,
    const std::vector<double>& y);

std::vector<double> ion_flux_divergence_cm3_s(
    const std::vector<double>& x_cm,
    const std::vector<double>& ion_density_cm3,
    const std::vector<double>& ion_velocity_cm_s);

std::vector<double> ion_bohm_speeds_cm_s(
    const std::vector<dcr::atomic::EnergyLevel>& levels,
    const std::vector<int>& ion_indices,
    double Te_eV,
    double Ti_eV = 0.0);

std::vector<double> nuclei_weighted_ion_flux_divergence_cm3_s(
    const std::vector<double>& x_cm,
    const std::vector<dcr::base::Vector>& background_full,
    const std::vector<dcr::atomic::EnergyLevel>& levels,
    const std::vector<int>& ion_indices,
    const std::vector<double>& bohm_speeds_cm_s,
    double L1_cm,
    double floor_fraction = 1e-2);

std::vector<double> neutral_pump_exhaust_nuclei_rate_cm3_s(
    const std::vector<dcr::base::Vector>& background_full,
    const std::vector<dcr::atomic::EnergyLevel>& levels,
    const std::vector<int>& atom_bg_indices,
    const std::vector<int>& mol_bg_indices,
    const std::vector<double>& atomic_exhaust_speed_cm_s,
    double molecular_exhaust_speed_cm_s,
    double width_cm);

std::vector<double> neutral_pump_exhaust_nuclei_rate_cm3_s(
    const std::vector<dcr::base::Vector>& background_full,
    const std::vector<dcr::atomic::EnergyLevel>& levels,
    const std::vector<int>& atom_bg_indices,
    const std::vector<int>& mol_bg_indices,
    double atomic_exhaust_speed_cm_s,
    double molecular_exhaust_speed_cm_s,
    double width_cm);

NeutralTransportCompensation build_neutral_transport_compensation(
    const std::vector<dcr::base::Vector>& background_full,
    const std::vector<dcr::atomic::EnergyLevel>& levels,
    const std::vector<int>& atom_bg_indices,
    const std::vector<int>& mol_bg_indices,
    const std::vector<double>& pump_exhaust_nuclei_cm3_s,
    const std::vector<double>& ion_divergence_nuclei_cm3_s);

AdaptiveTransportProfile build_adaptive_transport_profile(
    const std::vector<double>& x_cm,
    const std::vector<dcr::base::Vector>& background_full,
    const std::vector<dcr::atomic::EnergyLevel>& levels,
    const std::vector<int>& ion_indices,
    const std::vector<int>& atom_bg_indices,
    const std::vector<int>& mol_bg_indices,
    const std::vector<double>& Te_eV,
    const std::vector<double>& Ti_eV,
    double L1_cm,
    double LM_cm,
    double molecular_exhaust_speed_cm_s,
    double width_cm,
    double ion_velocity_floor_fraction = 1e-2);

AdaptiveTransportProfile build_adaptive_transport_profile(
    const std::vector<double>& x_cm,
    const std::vector<dcr::base::Vector>& background_full,
    const std::vector<dcr::atomic::EnergyLevel>& levels,
    const std::vector<int>& ion_indices,
    const std::vector<int>& atom_bg_indices,
    const std::vector<int>& mol_bg_indices,
    const std::vector<double>& Te_eV,
    double L1_cm,
    double LM_cm,
    double atomic_exhaust_speed_cm_s,
    double molecular_exhaust_speed_cm_s,
    double width_cm,
    double ion_velocity_floor_fraction = 1e-2);

} // namespace dcr::solver
