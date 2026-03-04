#pragma once
#include "Types.hpp"

namespace dcr::base::constants {

    // --- Math ---
    constexpr Real pi = 3.14159265358979323846;

    // --- CGS Fundamental Constants ---
    
    // Mass (Grams)
    constexpr Real amu_g = 1.6726219e-24;     // Proton mass 
    constexpr Real m_e = 9.10938356e-28;    // Electron mass [g]
    
    // Boltzmann (Ergs per Kelvin)
    constexpr Real k_B = 1.380649e-16;      // [erg/K] (1e7 diff from SI)

    // Speed of Light (cm/s)
    constexpr Real c_0 = 2.99792458e10;     // [cm/s]

    // Charge (Gaussian Units - StatCoulombs / ESU)
    constexpr Real q_e_esu = 4.80320425e-10; // Elementary charge [statC]
    
    // --- Conversions ---
    // 1 eV = 1.602e-12 erg
    constexpr Real eV_to_erg = 1.60217663e-12; 
    
    // If you need to convert back to SI for output
    constexpr Real cm_to_m = 1.0e-2;
    constexpr Real erg_to_J = 1.0e-7;
    constexpr Real g_to_kg = 1.0e-3;

} // namespace dcr::base::constants