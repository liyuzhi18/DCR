#pragma once
#include <string>
#include <vector>
#include "../base/Types.hpp"

namespace dcr::io {

   struct IOConfig {
        std::string output_dir = "output";
        std::string atomic_data_root = "";
        std::string data_tables_root = "data_tables";
        bool verbose_logging = true;
    };

    struct SpeciesConfig {
        std::string name;
        std::string filename; // e.g. "sclit.01"
        bool is_molecule = false;
        bool is_recycling = false;
        int charge = 0;
        bool charge_set = false;
        base::Real mass_amu = 1.0;
    };

    struct GridConfig {
        base::Real length_cm = 0.0;
        int num_cells = 0;
        std::string type = "uniform";     // "log" or "linear"
        base::Real first_cell_cm = 0.0;
        base::Real poloidal_width_cm = 1.0; // wall-parallel width w
    };

    struct InitialCondition {
        int species_index = 0;
        base::Real fraction = 0.0;
    };

    struct PlasmaConfig {
        base::Real total_density = 0.0; // cm^-3
        base::Real Te_eV = 0.0;
        base::Real Ti_eV = 0.0;
        base::Real neutral_atom_temperature_eV = 0.1;
        base::Real neutral_molecule_temperature_eV = 0.1;
        std::vector<InitialCondition> initial_conditions;
    };

    struct WallConfig {
        std::string material = "W";
        base::Real sheath_potential_drop = 3.0; 
        base::Real temperature_eV = 0.1; // wall temperature for desorption
    };

    struct NumericsConfig {
        base::Real tolerance = 1e-6;
        int max_iterations = 1000;
        base::Real relaxation = 1.0; // under-relaxation factor for fixed-point iterations
        // Marching fallback controls at fixed dx.
        int marching_slow_iter_threshold = 100;
        int marching_guess_retries = 3;
    };

    // --- The Main Config Object ---
    struct Config {
        IOConfig io;
        std::vector<SpeciesConfig> species;
        GridConfig grid;
        PlasmaConfig plasma;
        WallConfig wall;
        NumericsConfig numerics;
    };
    
}
