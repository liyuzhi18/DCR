#include "AdaptiveRecycling.hpp"

#include "../../physics/Sheath.hpp"

#include <algorithm>

namespace dcr::solver {

namespace {

double positive_sum(const dcr::base::Vector& v) {
    double total = 0.0;
    for (int i = 0; i < v.size(); ++i) total += std::max(v(i), 0.0);
    return total;
}

double group_nuclei_density(const dcr::base::Vector& population,
                            const std::vector<dcr::atomic::EnergyLevel>& levels,
                            const std::vector<int>& indices) {
    double total = 0.0;
    for (int gi : indices) {
        if (gi < 0 || gi >= population.size() || gi >= static_cast<int>(levels.size())) continue;
        const double mu = static_cast<double>(std::max(1, levels[static_cast<size_t>(gi)].atomicity));
        total += mu * std::max(population(gi), 0.0);
    }
    return total;
}

dcr::base::Vector state_partition_from_nuclei_share(
    const dcr::base::Vector& population,
    const std::vector<dcr::atomic::EnergyLevel>& levels,
    const std::vector<int>& indices,
    double group_nuclei_divergence_cm3_s) {

    dcr::base::Vector out = dcr::base::Vector::Zero(static_cast<int>(indices.size()));
    double nuclei_density = 0.0;
    for (int gi : indices) {
        if (gi < 0 || gi >= population.size() || gi >= static_cast<int>(levels.size())) continue;
        const double mu = static_cast<double>(std::max(1, levels[static_cast<size_t>(gi)].atomicity));
        nuclei_density += mu * std::max(population(gi), 0.0);
    }
    if (!(nuclei_density > 0.0)) return out;

    for (size_t j = 0; j < indices.size(); ++j) {
        const int gi = indices[j];
        if (gi < 0 || gi >= population.size() || gi >= static_cast<int>(levels.size())) continue;
        const double mu = static_cast<double>(std::max(1, levels[static_cast<size_t>(gi)].atomicity));
        const double state_nuclei = mu * std::max(population(gi), 0.0);
        out(static_cast<int>(j)) = (state_nuclei / nuclei_density) * group_nuclei_divergence_cm3_s / mu;
    }
    return out;
}

double presheath_velocity_shape(double x_cm, double length_cm, double floor_fraction) {
    if (!(length_cm > 0.0) || x_cm > length_cm) return 0.0;
    return std::max(std::clamp(floor_fraction, 0.0, 1.0), 1.0 - x_cm / length_cm);
}

} // namespace

std::vector<double> normalized_atomic_flow_fraction(
    const std::vector<dcr::base::Vector>& flowA) {

    return normalized_flow_fraction(flowA);
}

std::vector<double> normalized_flow_fraction(
    const std::vector<dcr::base::Vector>& flow) {

    std::vector<double> out(flow.size(), 0.0);
    if (flow.empty()) return out;

    const double initial = positive_sum(flow.front());
    if (!(initial > 0.0)) return out;

    for (size_t k = 0; k < flow.size(); ++k) {
        out[k] = positive_sum(flow[k]) / initial;
    }
    return out;
}

FlowDepletionEstimate estimate_flow_depletion_layer(
    const std::vector<double>& x_cm,
    const std::vector<dcr::base::Vector>& flow,
    double epsilon) {

    FlowDepletionEstimate estimate;
    estimate.flow_fraction = normalized_flow_fraction(flow);
    if (!(epsilon > 0.0)) return estimate;

    const size_t n = std::min(x_cm.size(), estimate.flow_fraction.size());
    for (size_t k = 0; k < n; ++k) {
        if (estimate.flow_fraction[k] < epsilon) {
            estimate.found = true;
            estimate.index = static_cast<int>(k);
            estimate.length_cm = x_cm[k];
            break;
        }
    }
    return estimate;
}

RecyclingLayerEstimate estimate_recycling_layer_from_atomic_flow(
    const std::vector<double>& x_cm,
    const std::vector<dcr::base::Vector>& flowA,
    double epsilon_A) {

    RecyclingLayerEstimate estimate;
    const auto generic = estimate_flow_depletion_layer(x_cm, flowA, epsilon_A);
    estimate.found = generic.found;
    estimate.index = generic.index;
    estimate.L1_cm = generic.length_cm;
    estimate.atomic_flow_fraction = generic.flow_fraction;
    return estimate;
}

std::vector<double> linear_presheath_velocity_profile_cm_s(
    const std::vector<double>& x_cm,
    double L1_cm,
    double bohm_speed_cm_s,
    double floor_fraction) {

    std::vector<double> out(x_cm.size(), 0.0);
    if (!(L1_cm > 0.0) || !(bohm_speed_cm_s > 0.0)) return out;

    for (size_t k = 0; k < x_cm.size(); ++k) {
        out[k] = bohm_speed_cm_s * presheath_velocity_shape(x_cm[k], L1_cm, floor_fraction);
    }
    return out;
}

std::vector<double> finite_difference_derivative(
    const std::vector<double>& x_cm,
    const std::vector<double>& y) {

    std::vector<double> out(y.size(), 0.0);
    const size_t n = std::min(x_cm.size(), y.size());
    if (n < 2) return out;

    auto slope = [&](size_t left, size_t right) {
        const double dx = x_cm[right] - x_cm[left];
        if (!(dx > 0.0)) return 0.0;
        return (y[right] - y[left]) / dx;
    };

    out[0] = slope(0, 1);
    for (size_t k = 1; k + 1 < n; ++k) {
        out[k] = slope(k - 1, k + 1);
    }
    out[n - 1] = slope(n - 2, n - 1);
    return out;
}

std::vector<double> ion_flux_divergence_cm3_s(
    const std::vector<double>& x_cm,
    const std::vector<double>& ion_density_cm3,
    const std::vector<double>& ion_velocity_cm_s) {

    const size_t n = std::min(ion_density_cm3.size(), ion_velocity_cm_s.size());
    std::vector<double> flux(n, 0.0);
    for (size_t k = 0; k < n; ++k) {
        flux[k] = std::max(ion_density_cm3[k], 0.0) * ion_velocity_cm_s[k];
    }
    return finite_difference_derivative(x_cm, flux);
}

std::vector<double> ion_bohm_speeds_cm_s(
    const std::vector<dcr::atomic::EnergyLevel>& levels,
    const std::vector<int>& ion_indices,
    double Te_eV,
    double Ti_eV) {

    std::vector<double> speeds(ion_indices.size(), 0.0);
    if (!(Te_eV > 0.0)) return speeds;

    for (size_t j = 0; j < ion_indices.size(); ++j) {
        const int gi = ion_indices[j];
        if (gi < 0 || gi >= static_cast<int>(levels.size())) continue;
        const double mass_amu = std::max(levels[static_cast<size_t>(gi)].mass_amu, 1.0e-12);
        speeds[j] = dcr::physics::calculate_Bohm_speed(Te_eV, Ti_eV, mass_amu);
    }
    return speeds;
}

std::vector<double> nuclei_weighted_ion_flux_divergence_cm3_s(
    const std::vector<double>& x_cm,
    const std::vector<dcr::base::Vector>& background_full,
    const std::vector<dcr::atomic::EnergyLevel>& levels,
    const std::vector<int>& ion_indices,
    const std::vector<double>& bohm_speeds_cm_s,
    double L1_cm,
    double floor_fraction) {

    std::vector<double> total(x_cm.size(), 0.0);
    const size_t n_ions = std::min(ion_indices.size(), bohm_speeds_cm_s.size());
    for (size_t j = 0; j < n_ions; ++j) {
        const int gi = ion_indices[j];
        if (gi < 0 || gi >= static_cast<int>(levels.size())) continue;

        std::vector<double> density(x_cm.size(), 0.0);
        for (size_t k = 0; k < x_cm.size() && k < background_full.size(); ++k) {
            if (gi < background_full[k].size()) {
                density[k] = std::max(background_full[k](gi), 0.0);
            }
        }

        const auto velocity = linear_presheath_velocity_profile_cm_s(
            x_cm, L1_cm, bohm_speeds_cm_s[j], floor_fraction
        );
        const auto divergence = ion_flux_divergence_cm3_s(x_cm, density, velocity);
        const double mu = static_cast<double>(std::max(1, levels[static_cast<size_t>(gi)].atomicity));
        for (size_t k = 0; k < total.size() && k < divergence.size(); ++k) {
            if (x_cm[k] > L1_cm) continue;
            total[k] += mu * divergence[k];
        }
    }
    return total;
}

std::vector<double> neutral_pump_exhaust_nuclei_rate_cm3_s(
    const std::vector<dcr::base::Vector>& background_full,
    const std::vector<dcr::atomic::EnergyLevel>& levels,
    const std::vector<int>& atom_bg_indices,
    const std::vector<int>& mol_bg_indices,
    const std::vector<double>& atomic_exhaust_speed_cm_s,
    double molecular_exhaust_speed_cm_s,
    double width_cm) {

    std::vector<double> out(background_full.size(), 0.0);
    if (!(width_cm > 0.0)) return out;

    const double m_coeff = std::max(molecular_exhaust_speed_cm_s, 0.0) / width_cm;
    for (size_t k = 0; k < background_full.size(); ++k) {
        const double a_speed = (k < atomic_exhaust_speed_cm_s.size())
            ? atomic_exhaust_speed_cm_s[k]
            : 0.0;
        const double a_coeff = std::max(a_speed, 0.0) / width_cm;
        double rate = 0.0;
        for (int gi : atom_bg_indices) {
            if (gi < 0 || gi >= background_full[k].size() || gi >= static_cast<int>(levels.size())) continue;
            const double mu = static_cast<double>(std::max(1, levels[static_cast<size_t>(gi)].atomicity));
            rate += mu * a_coeff * std::max(background_full[k](gi), 0.0);
        }
        for (int gi : mol_bg_indices) {
            if (gi < 0 || gi >= background_full[k].size() || gi >= static_cast<int>(levels.size())) continue;
            const double mu = static_cast<double>(std::max(1, levels[static_cast<size_t>(gi)].atomicity));
            rate += mu * m_coeff * std::max(background_full[k](gi), 0.0);
        }
        out[k] = rate;
    }
    return out;
}

std::vector<double> neutral_pump_exhaust_nuclei_rate_cm3_s(
    const std::vector<dcr::base::Vector>& background_full,
    const std::vector<dcr::atomic::EnergyLevel>& levels,
    const std::vector<int>& atom_bg_indices,
    const std::vector<int>& mol_bg_indices,
    double atomic_exhaust_speed_cm_s,
    double molecular_exhaust_speed_cm_s,
    double width_cm) {

    return neutral_pump_exhaust_nuclei_rate_cm3_s(
        background_full,
        levels,
        atom_bg_indices,
        mol_bg_indices,
        std::vector<double>(background_full.size(), atomic_exhaust_speed_cm_s),
        molecular_exhaust_speed_cm_s,
        width_cm
    );
}

NeutralTransportCompensation build_neutral_transport_compensation(
    const std::vector<dcr::base::Vector>& background_full,
    const std::vector<dcr::atomic::EnergyLevel>& levels,
    const std::vector<int>& atom_bg_indices,
    const std::vector<int>& mol_bg_indices,
    const std::vector<double>& pump_exhaust_nuclei_cm3_s,
    const std::vector<double>& ion_divergence_nuclei_cm3_s) {

    const size_t n = std::min({
        background_full.size(),
        pump_exhaust_nuclei_cm3_s.size(),
        ion_divergence_nuclei_cm3_s.size()
    });

    NeutralTransportCompensation out;
    out.remainder_nuclei_cm3_s.assign(n, 0.0);
    out.atomic_group_nuclei_divergence_cm3_s.assign(n, 0.0);
    out.molecular_group_nuclei_divergence_cm3_s.assign(n, 0.0);
    out.atomic_state_divergence_cm3_s.reserve(n);
    out.molecular_state_divergence_cm3_s.reserve(n);

    for (size_t k = 0; k < n; ++k) {
        const double remainder = pump_exhaust_nuclei_cm3_s[k] - ion_divergence_nuclei_cm3_s[k];
        out.remainder_nuclei_cm3_s[k] = remainder;

        const double nuc_a = group_nuclei_density(background_full[k], levels, atom_bg_indices);
        const double nuc_m = group_nuclei_density(background_full[k], levels, mol_bg_indices);
        const double den = nuc_a + nuc_m;
        double atomic_nuclei_div = 0.0;
        double molecular_nuclei_div = 0.0;
        if (den > 0.0) {
            atomic_nuclei_div = (nuc_a / den) * remainder;
            molecular_nuclei_div = (nuc_m / den) * remainder;
        }
        out.atomic_group_nuclei_divergence_cm3_s[k] = atomic_nuclei_div;
        out.molecular_group_nuclei_divergence_cm3_s[k] = molecular_nuclei_div;
        out.atomic_state_divergence_cm3_s.push_back(
            state_partition_from_nuclei_share(background_full[k], levels, atom_bg_indices, atomic_nuclei_div)
        );
        out.molecular_state_divergence_cm3_s.push_back(
            state_partition_from_nuclei_share(background_full[k], levels, mol_bg_indices, molecular_nuclei_div)
        );
    }
    return out;
}

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
    double ion_velocity_floor_fraction) {

    AdaptiveTransportProfile profile;
    profile.L1_cm = L1_cm;
    profile.LM_cm = LM_cm;
    profile.ion_divergence_nuclei_cm3_s.assign(x_cm.size(), 0.0);

    for (int gi : ion_indices) {
        if (gi < 0 || gi >= static_cast<int>(levels.size())) continue;
        const auto& level = levels[static_cast<size_t>(gi)];
        const double mu = static_cast<double>(std::max(1, level.atomicity));
        const double mass_amu = std::max(level.mass_amu, 1.0e-12);
        const double Ls_cm = (level.atomicity >= 2 && LM_cm > 0.0) ? LM_cm : L1_cm;

        std::vector<double> flux(x_cm.size(), 0.0);
        for (size_t k = 0; k < x_cm.size() && k < background_full.size(); ++k) {
            if (gi >= background_full[k].size() || k >= Te_eV.size() || k >= Ti_eV.size()) continue;
            const double shape = (Ls_cm > 0.0)
                ? presheath_velocity_shape(x_cm[k], Ls_cm, ion_velocity_floor_fraction)
                : 0.0;
            const double u = dcr::physics::calculate_Bohm_speed(Te_eV[k], Ti_eV[k], mass_amu) * shape;
            flux[k] = std::max(background_full[k](gi), 0.0) * u;
        }

        const auto div = finite_difference_derivative(x_cm, flux);
        for (size_t k = 0; k < profile.ion_divergence_nuclei_cm3_s.size() && k < div.size(); ++k) {
            if (x_cm[k] > Ls_cm) continue;
            profile.ion_divergence_nuclei_cm3_s[k] += mu * div[k];
        }
    }

    double atom_mass_amu = 1.0;
    if (!atom_bg_indices.empty()) {
        const int gi = atom_bg_indices.front();
        if (gi >= 0 && gi < static_cast<int>(levels.size())) {
            atom_mass_amu = std::max(levels[static_cast<size_t>(gi)].mass_amu, 1.0e-12);
        }
    }

    std::vector<double> atomic_exhaust_speed_cm_s(x_cm.size(), 0.0);
    for (size_t k = 0; k < atomic_exhaust_speed_cm_s.size() && k < Ti_eV.size(); ++k) {
        atomic_exhaust_speed_cm_s[k] = dcr::physics::calculate_thermal_speed(Ti_eV[k], atom_mass_amu);
    }

    profile.pump_exhaust_nuclei_cm3_s = neutral_pump_exhaust_nuclei_rate_cm3_s(
        background_full,
        levels,
        atom_bg_indices,
        mol_bg_indices,
        atomic_exhaust_speed_cm_s,
        molecular_exhaust_speed_cm_s,
        width_cm
    );

    const auto neutral = build_neutral_transport_compensation(
        background_full,
        levels,
        atom_bg_indices,
        mol_bg_indices,
        profile.pump_exhaust_nuclei_cm3_s,
        profile.ion_divergence_nuclei_cm3_s
    );
    profile.neutral_remainder_nuclei_cm3_s = neutral.remainder_nuclei_cm3_s;
    profile.atomic_group_nuclei_divergence_cm3_s = neutral.atomic_group_nuclei_divergence_cm3_s;
    profile.molecular_group_nuclei_divergence_cm3_s = neutral.molecular_group_nuclei_divergence_cm3_s;
    profile.atomic_state_divergence_cm3_s = neutral.atomic_state_divergence_cm3_s;
    profile.molecular_state_divergence_cm3_s = neutral.molecular_state_divergence_cm3_s;
    return profile;
}

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
    double ion_velocity_floor_fraction) {

    AdaptiveTransportProfile profile;
    profile.L1_cm = L1_cm;
    profile.LM_cm = LM_cm;
    profile.ion_divergence_nuclei_cm3_s.assign(x_cm.size(), 0.0);

    for (int gi : ion_indices) {
        if (gi < 0 || gi >= static_cast<int>(levels.size())) continue;
        const auto& level = levels[static_cast<size_t>(gi)];
        const double mu = static_cast<double>(std::max(1, level.atomicity));
        const double mass_amu = std::max(level.mass_amu, 1.0e-12);
        const double Ls_cm = (level.atomicity >= 2 && LM_cm > 0.0) ? LM_cm : L1_cm;

        std::vector<double> flux(x_cm.size(), 0.0);
        for (size_t k = 0; k < x_cm.size() && k < background_full.size(); ++k) {
            if (gi >= background_full[k].size() || k >= Te_eV.size()) continue;
            const double shape = (Ls_cm > 0.0)
                ? presheath_velocity_shape(x_cm[k], Ls_cm, ion_velocity_floor_fraction)
                : 0.0;
            const double u = dcr::physics::calculate_Bohm_speed(Te_eV[k], 0.0, mass_amu) * shape;
            flux[k] = std::max(background_full[k](gi), 0.0) * u;
        }

        const auto div = finite_difference_derivative(x_cm, flux);
        for (size_t k = 0; k < profile.ion_divergence_nuclei_cm3_s.size() && k < div.size(); ++k) {
            if (x_cm[k] > Ls_cm) continue;
            profile.ion_divergence_nuclei_cm3_s[k] += mu * div[k];
        }
    }

    profile.pump_exhaust_nuclei_cm3_s = neutral_pump_exhaust_nuclei_rate_cm3_s(
        background_full,
        levels,
        atom_bg_indices,
        mol_bg_indices,
        atomic_exhaust_speed_cm_s,
        molecular_exhaust_speed_cm_s,
        width_cm
    );

    const auto neutral = build_neutral_transport_compensation(
        background_full,
        levels,
        atom_bg_indices,
        mol_bg_indices,
        profile.pump_exhaust_nuclei_cm3_s,
        profile.ion_divergence_nuclei_cm3_s
    );
    profile.neutral_remainder_nuclei_cm3_s = neutral.remainder_nuclei_cm3_s;
    profile.atomic_group_nuclei_divergence_cm3_s = neutral.atomic_group_nuclei_divergence_cm3_s;
    profile.molecular_group_nuclei_divergence_cm3_s = neutral.molecular_group_nuclei_divergence_cm3_s;
    profile.atomic_state_divergence_cm3_s = neutral.atomic_state_divergence_cm3_s;
    profile.molecular_state_divergence_cm3_s = neutral.molecular_state_divergence_cm3_s;
    return profile;
}

} // namespace dcr::solver
