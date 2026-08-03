#pragma once
#include "../base/Types.hpp"
#include "../base/Constants.hpp"
#include <cmath>
#include <algorithm>

namespace dcr::physics {

inline base::Real calculate_Bohm_speed(
    base::Real Te_eV,
    base::Real Ti_eV,
    base::Real mass_amu,
    base::Real gamma_i = 0.0
){
    using namespace dcr::base::constants;
    (void)gamma_i;

    const base::Real sheath_sound_energy_erg = (Te_eV + 3.0 * Ti_eV) * eV_to_erg;
    const base::Real mass_g = mass_amu * amu_g;

    const base::Real u_bohm = std::sqrt(std::max(0.0, sheath_sound_energy_erg) / mass_g);
    return u_bohm;

}

inline base::Real calculate_thermal_speed(
    base::Real T_eV,
    base::Real mass_amu
){
    using namespace dcr::base::constants;
    
    const base::Real T_erg = T_eV * eV_to_erg;
    const base::Real mass_g = mass_amu * amu_g;

    // Requested form: v_th = sqrt(T/m)
    const base::Real v_th = std::sqrt(T_erg / mass_g);
    return v_th;
}

// Speed from kinetic energy (eV): v = sqrt(2 * E / m)
inline base::Real calculate_speed_from_energy_ev(
    base::Real energy_eV,
    base::Real mass_amu
){
    using namespace dcr::base::constants;

    if (energy_eV <= 0.0 || mass_amu <= 0.0) return 0.0;
    const base::Real energy_erg = energy_eV * eV_to_erg;
    const base::Real mass_g = mass_amu * amu_g;
    const base::Real v = std::sqrt(2.0 * energy_erg / mass_g);
    return v;
}

// Ion impact energy at the sheath entrance (eV):
// E0 = (1/2) m_i u_B^2 + (sheath_potential_drop * Te)
// with u_B based on Te + 3 Ti.
inline base::Real calculate_ion_impact_energy_ev(
    base::Real Te_eV,
    base::Real Ti_eV,
    base::Real mass_amu,
    base::Real sheath_potential_drop,
    base::Real gamma_i = 0.0
){
    using namespace dcr::base::constants;

    const base::Real u_bohm = calculate_Bohm_speed(Te_eV, Ti_eV, mass_amu, gamma_i);
    const base::Real mass_g = mass_amu * amu_g;
    const base::Real ke_erg = 0.5 * mass_g * u_bohm * u_bohm;
    const base::Real ke_ev = ke_erg / eV_to_erg;
    const base::Real sheath_ev = sheath_potential_drop * Te_eV;
    return ke_ev + sheath_ev;
}

}
