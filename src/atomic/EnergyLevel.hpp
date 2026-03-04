#pragma once
#include "../base/Types.hpp"
#include <string>

namespace dcr::atomic {

    enum class SpeciesType {Atom, Molecule, Ion};

    struct EnergyLevel {
        base::Index global_index;
        SpeciesType type;
        std::string label;

        // Physics properties
        base::Real energy_eV; 
        base::Real degeneracy;

        int charge;       // Z
        int species_id;   // ID from file
        int internal_id;  // n or v quantum number
        int block_index;  // Block ID
        int atomicity = 1; // number of nuclei (1 for atoms/ions, 2 for H2/H2+)
        base::Real mass_amu = 1.0; // species mass used for collision reduced-mass estimates

        // Flags for solver
        bool is_recycling;      
        bool is_background; 

    };
}
