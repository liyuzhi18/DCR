#pragma once
#include <string>
#include <vector>
#include "../base/Types.hpp"

namespace dcr::io {

    struct SolverConfig {
        // Keep the legacy solver as the default so existing configs are unchanged.
        // Supported values:
        // - "full_dcr": current fully resolved model
        // - "qss_dcr": reduced model with atomic transient states eliminated
        std::string mode = "full_dcr";
    };

    struct TemperatureProfileConfig {
        bool enabled = false;
        std::string type = "constant"; // "constant" or "linear_x"
        base::Real value_eV = 0.0;
        base::Real x_start_cm = 0.0;
        base::Real x_end_cm = 0.0;
        base::Real value_start_eV = 0.0;
        base::Real value_end_eV = 0.0;
    };

   struct IOConfig {
        std::string output_dir = "output";
        std::string atomic_data_root = "";
        std::string data_tables_root = "data_tables";
        bool verbose_logging = true;
        // Optional marching restart seed. When set to a previous dcr_results.h5,
        // the full marching solver interpolates background/flow states from that
        // profile to build better per-cell initial guesses. Default empty keeps
        // the original behavior.
        std::string restart_profile_h5 = "";
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
        struct EEDFOptions {
            std::string type = "maxwellian";
            base::Real power_p = 1.0;
            base::Real hot_fraction = 0.0;
            base::Real hot_temperature_factor = 2.5;
        };

        base::Real total_density = 0.0; // cm^-3
        base::Real Te_eV = 0.0;
        base::Real Ti_eV = 0.0;
        EEDFOptions eedf;
        TemperatureProfileConfig electron_temperature_profile;
        TemperatureProfileConfig ion_temperature_profile;
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
        // Optional boundary-only convergence controls. Non-positive values fall
        // back to the legacy shared fields above.
        base::Real boundary_tolerance = 0.0;
        int boundary_max_iterations = 0;
        // Optional marching-only iteration cap. Non-positive means "reuse
        // max_iterations" so older configs behave exactly as before.
        base::Real marching_tolerance = 0.0;
        int marching_max_iterations = 0;
        base::Real relaxation = 1.0; // under-relaxation factor for fixed-point iterations
        // Boundary nonlinear solver selection.
        // - "picard": current damped fixed-point boundary solver
        // - "newton_krylov_ptc": Newton-Krylov with pseudo-transient continuation
        std::string boundary_solver = "picard";
        // Marching nonlinear solver selection.
        // - "picard": current damped fixed-point marching solver
        // - "newton_krylov_ptc": Newton-Krylov with pseudo-transient continuation
        // - "log_newton_krylov_ptc": log-density Newton-Krylov marching solve
        std::string marching_solver = "picard";
        // Marching fallback controls at fixed dx.
        int marching_slow_iter_threshold = 100;
        int marching_guess_retries = 3;
        // If true, abort the whole run when a marched cell reaches the maximum
        // iteration count without convergence.
        bool abort_on_marching_nonconvergence = false;
        // Newton-Krylov + pseudo-transient continuation controls.
        base::Real boundary_ptc_tau_init = 1.0;
        base::Real boundary_ptc_tau_max = 100.0;
        int boundary_nk_krylov_dim = 8;
        int boundary_nk_max_restarts = 2;
        base::Real boundary_nk_fd_eps = 1e-6;
        base::Real boundary_nk_alpha_min = 1e-4;
        base::Real marching_ptc_tau_init = 1.0;
        base::Real marching_ptc_tau_max = 100.0;
        int marching_nk_krylov_dim = 8;
        int marching_nk_max_restarts = 2;
        base::Real marching_nk_fd_eps = 1e-6;
        base::Real marching_nk_alpha_min = 1e-4;
        // Minimum neutral-H2 background retained in marched cells, expressed as
        // a fraction of the previous cell's neutral molecular background total.
        // Zero disables the floor.
        base::Real marching_h2_floor_fraction = 0.0;
        bool couple_flowA_retained = false;
        bool qss_exclude_molecular_ion_from_transient_reconstruction = false;
        // Zero out H2+(bg) -> H(n>=2) coupling in R_full to mimic ADAS-like behavior
        // (no excited-H production from DR channel e + H2+ -> H(n=2,3)).
        bool disable_h2plus_dr = false;
        // Use raw MCCC H2 dissociation cross sections with runtime EEDF integration
        // for neutral molecular "de" channels. If false, prefer the reconstructed
        // dissociation fit CSV when present, otherwise keep the original fitted de law.
        bool use_mccc_h2_dissociation = true;
    };

    // --- The Main Config Object ---
    struct Config {
        SolverConfig solver;
        IOConfig io;
        std::vector<SpeciesConfig> species;
        GridConfig grid;
        PlasmaConfig plasma;
        WallConfig wall;
        NumericsConfig numerics;
    };
    
}
