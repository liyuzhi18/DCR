#pragma once

#include "Sheath.hpp"
#include "WallReflection.hpp"

namespace dcr::physics {

struct WallRecycling {
    double ion_impact_energy_ev = 0.0;
    double alpha_atom = 0.0;      // atomic recycling fraction (from reflection)
    double gamma_E_atom = 0.0;    // energy reflection factor for atoms
    double alpha_molecule = 0.0;  // molecular recycling fraction (desorption)
    double gamma_E_molecule = 0.0;
};

inline WallRecycling compute_wall_recycling(const std::string& material,
                                            double Te_eV,
                                            double Ti_eV,
                                            double ion_mass_amu,
                                            double sheath_potential_drop,
                                            double gamma_i = 0.0) {
    WallRecycling out{};
    out.ion_impact_energy_ev = calculate_ion_impact_energy_ev(
        Te_eV, Ti_eV, ion_mass_amu, sheath_potential_drop, gamma_i
    );

    const auto coeffs = reflection_coefficients(material, out.ion_impact_energy_ev);
    out.alpha_atom = coeffs.alpha;
    out.gamma_E_atom = coeffs.gamma_E;

    // Molecules: 100% desorption (no reflection).
    // We treat the remaining fraction of recycled particles as molecular return.
    out.alpha_molecule = std::max(0.0, 1.0 - out.alpha_atom);
    out.gamma_E_molecule = 0.0;
    return out;
}

} // namespace dcr::physics
