#include <cassert>
#include <fstream>
#include <iostream>
#include <filesystem>

#include "../src/io/ConfigLoader.hpp"

namespace fs = std::filesystem;

static fs::path write_temp_config(const std::string& name, const std::string& contents) {
    fs::path path = fs::temp_directory_path() / name;
    std::ofstream out(path);
    out << contents;
    out.close();
    return path;
}

int main() {
    std::cout << "--- Testing ConfigLoader ---\n";

    const std::string yaml_full =
        "solver:\n"
        "  mode: qss_dcr\n"
        "io:\n"
        "  output_dir: output/test_run\n"
        "  atomic_data_root: atomic_data/\n"
        "  verbose_logging: false\n"
        "species:\n"
        "  - name: H\n"
        "    filename: sclit.01\n"
        "    is_molecule: false\n"
        "    is_recycling: true\n"
        "    charge: 1\n"
        "    mass_amu: 1.0\n"
        "grid:\n"
        "  length_cm: 1.0\n"
        "  num_cells: 10\n"
        "  type: linear\n"
        "  first_cell_cm: 0.01\n"
        "  poloidal_width_cm: 2.0\n"
        "  boundary_poloidal_width_cm: 3.0\n"
        "  spatial_exhaust_width_cm: 4.0\n"
        "global_bvp:\n"
        "  ion_velocity_transition_length_cm: 40.0\n"
        "  ion_upstream_speed_fraction: 0.2\n"
        "  final_domain_length_cm: 60.0\n"
        "plasma:\n"
        "  total_density_cm3: 1e14\n"
        "  Te_eV: 5.0\n"
        "  Ti_eV: 2.0\n"
        "  eedf:\n"
        "    type: generalized_power\n"
        "    power_p: 0.9\n"
        "    hot_fraction: 0.05\n"
        "    hot_temperature_factor: 2.5\n"
        "  electron_temperature_profile:\n"
        "    type: linear_x\n"
        "    x_start_cm: 0.0\n"
        "    x_end_cm: 2.0\n"
        "    value_start_eV: 1.0\n"
        "    value_end_eV: 5.0\n"
        "  ion_temperature_profile:\n"
        "    type: constant_heat_flux\n"
        "    x_start_cm: 0.0\n"
        "    x_end_cm: 2.0\n"
        "    value_start_eV: 2.0\n"
        "    value_end_eV: 5.0\n"
        "  neutral_atom_temperature_eV: 0.2\n"
        "  neutral_molecule_temperature_eV: 0.15\n"
        "  initial_conditions:\n"
        "    - index: 0\n"
        "      fraction: 0.5\n"
        "wall:\n"
        "  material: W\n"
        "  sheath_potential_drop: 4.0\n"
        "  temperature_eV: 0.2\n"
        "numerics:\n"
        "  tolerance: 1e-8\n"
        "  max_iterations: 100\n"
        "  relaxation: 0.5\n"
        "  boundary_neutral_exhaust: false\n"
        "  adaptive_recycling_domain:\n"
        "    enabled: true\n"
        "    initial_L_box_cm: 3.0\n"
        "    max_L_box_cm: 30.0\n"
        "    epsilon_A: 5e-3\n"
        "    epsilon_M: 7e-3\n"
        "    max_outer_iterations: 7\n"
        "    L1_relative_tolerance: 2e-2\n"
        "    neutral_partition_mode: atomic_only\n"
        "    apply_ion_closure: true\n"
        "    closure_mode: variable_nuclei_balance\n";

    fs::path full_path = write_temp_config("dcr_test_full.yaml", yaml_full);
    auto cfg = dcr::io::ConfigLoader::load(full_path.string());

    assert(cfg.solver.mode == "qss_dcr");
    assert(cfg.io.output_dir == "output/test_run");
    assert(cfg.io.atomic_data_root == "atomic_data/");
    assert(cfg.io.verbose_logging == false);

    assert(cfg.species.size() == 1);
    assert(cfg.species[0].name == "H");
    assert(cfg.species[0].filename == "sclit.01");
    assert(cfg.species[0].is_molecule == false);
    assert(cfg.species[0].is_recycling == true);
    assert(cfg.species[0].charge == 1);
    assert(cfg.species[0].charge_set == true);
    assert(cfg.species[0].mass_amu == 1.0);

    assert(cfg.grid.length_cm == 1.0);
    assert(cfg.grid.num_cells == 10);
    assert(cfg.grid.type == "linear");
    assert(cfg.grid.first_cell_cm == 0.01);
    assert(cfg.grid.boundary_poloidal_width_cm == 3.0);
    assert(cfg.grid.spatial_exhaust_width_cm == 4.0);
    assert(cfg.global_bvp.ion_velocity_transition_length_cm == 40.0);
    assert(cfg.global_bvp.ion_upstream_speed_fraction == 0.2);
    assert(cfg.global_bvp.final_domain_length_cm == 60.0);

    assert(cfg.plasma.total_density == 1e14);
    assert(cfg.plasma.Te_eV == 5.0);
    assert(cfg.plasma.Ti_eV == 2.0);
    assert(cfg.plasma.eedf.type == "generalized_power");
    assert(cfg.plasma.eedf.power_p == 0.9);
    assert(cfg.plasma.eedf.hot_fraction == 0.05);
    assert(cfg.plasma.eedf.hot_temperature_factor == 2.5);
    assert(cfg.plasma.electron_temperature_profile.enabled == true);
    assert(cfg.plasma.electron_temperature_profile.type == "linear_x");
    assert(cfg.plasma.electron_temperature_profile.x_start_cm == 0.0);
    assert(cfg.plasma.electron_temperature_profile.x_end_cm == 2.0);
    assert(cfg.plasma.electron_temperature_profile.value_start_eV == 1.0);
    assert(cfg.plasma.electron_temperature_profile.value_end_eV == 5.0);
    assert(cfg.plasma.ion_temperature_profile.enabled == true);
    assert(cfg.plasma.ion_temperature_profile.type == "constant_heat_flux");
    assert(cfg.plasma.ion_temperature_profile.x_start_cm == 0.0);
    assert(cfg.plasma.ion_temperature_profile.x_end_cm == 2.0);
    assert(cfg.plasma.ion_temperature_profile.value_start_eV == 2.0);
    assert(cfg.plasma.ion_temperature_profile.value_end_eV == 5.0);
    assert(cfg.plasma.neutral_atom_temperature_eV == 0.2);
    assert(cfg.plasma.neutral_molecule_temperature_eV == 0.15);
    assert(cfg.plasma.initial_conditions.size() == 1);
    assert(cfg.plasma.initial_conditions[0].species_index == 0);
    assert(cfg.plasma.initial_conditions[0].fraction == 0.5);

    assert(cfg.wall.material == "W");
    assert(cfg.wall.sheath_potential_drop == 4.0);
    assert(cfg.wall.temperature_eV == 0.2);

    assert(cfg.numerics.tolerance == 1e-8);
    assert(cfg.numerics.max_iterations == 100);
    assert(cfg.numerics.relaxation == 0.5);
    assert(cfg.numerics.boundary_neutral_exhaust == false);
    assert(cfg.numerics.adaptive_recycling_domain.enabled == true);
    assert(cfg.numerics.adaptive_recycling_domain.initial_L_box_cm == 3.0);
    assert(cfg.numerics.adaptive_recycling_domain.max_L_box_cm == 30.0);
    assert(cfg.numerics.adaptive_recycling_domain.epsilon_A == 5e-3);
    assert(cfg.numerics.adaptive_recycling_domain.epsilon_M == 7e-3);
    assert(cfg.numerics.adaptive_recycling_domain.max_outer_iterations == 7);
    assert(cfg.numerics.adaptive_recycling_domain.L1_relative_tolerance == 2e-2);
    assert(cfg.numerics.adaptive_recycling_domain.neutral_partition_mode == "atomic_only");
    assert(cfg.numerics.adaptive_recycling_domain.apply_ion_closure == true);
    assert(cfg.numerics.adaptive_recycling_domain.closure_mode == "variable_nuclei_balance");

    const std::string yaml_defaults =
        "io:\n"
        "  output_dir: output/defaults\n"
        "grid:\n"
        "  length_cm: 2.0\n"
        "  num_cells: 5\n"
        "plasma:\n"
        "  total_density_cm3: 2e14\n"
        "  Te_eV: 3.0\n"
        "  Ti_eV: 1.0\n";

    fs::path defaults_path = write_temp_config("dcr_test_defaults.yaml", yaml_defaults);
    auto cfg_defaults = dcr::io::ConfigLoader::load(defaults_path.string());

    assert(cfg_defaults.solver.mode == "full_dcr");
    assert(cfg_defaults.io.output_dir == "output/defaults");
    assert(cfg_defaults.io.verbose_logging == true);
    assert(cfg_defaults.io.atomic_data_root == "");

    assert(cfg_defaults.grid.type == "uniform");
    assert(cfg_defaults.grid.first_cell_cm == 0.0);
    assert(cfg_defaults.grid.boundary_poloidal_width_cm == 1.0);
    assert(cfg_defaults.grid.spatial_exhaust_width_cm == 1.0);
    assert(cfg_defaults.global_bvp.ion_velocity_transition_length_cm == 50.0);
    assert(cfg_defaults.global_bvp.ion_upstream_speed_fraction == 0.1);
    assert(cfg_defaults.global_bvp.final_domain_length_cm == 50.0);

    assert(cfg_defaults.species.empty());
    assert(cfg_defaults.wall.material == "W");
    assert(cfg_defaults.wall.sheath_potential_drop == 3.0);
    assert(cfg_defaults.wall.temperature_eV == 0.1);
    assert(cfg_defaults.numerics.tolerance == 1e-6);
    assert(cfg_defaults.numerics.max_iterations == 1000);
    assert(cfg_defaults.numerics.relaxation == 1.0);
    assert(cfg_defaults.numerics.boundary_neutral_exhaust == true);
    assert(cfg_defaults.numerics.adaptive_recycling_domain.enabled == false);
    assert(cfg_defaults.numerics.adaptive_recycling_domain.initial_L_box_cm == 2.0);
    assert(cfg_defaults.numerics.adaptive_recycling_domain.max_L_box_cm == 20.0);
    assert(cfg_defaults.numerics.adaptive_recycling_domain.epsilon_A == 1e-2);
    assert(cfg_defaults.numerics.adaptive_recycling_domain.epsilon_M == 1e-2);
    assert(cfg_defaults.numerics.adaptive_recycling_domain.max_outer_iterations == 5);
    assert(cfg_defaults.numerics.adaptive_recycling_domain.L1_relative_tolerance == 5e-2);
    assert(cfg_defaults.numerics.adaptive_recycling_domain.neutral_partition_mode == "nuclei_fraction");
    assert(cfg_defaults.numerics.adaptive_recycling_domain.apply_ion_closure == false);
    assert(cfg_defaults.numerics.adaptive_recycling_domain.closure_mode == "fixed_density");

    assert(cfg_defaults.plasma.electron_temperature_profile.enabled == false);
    assert(cfg_defaults.plasma.ion_temperature_profile.enabled == false);
    assert(cfg_defaults.plasma.eedf.type == "maxwellian");
    assert(cfg_defaults.plasma.eedf.power_p == 1.0);
    assert(cfg_defaults.plasma.eedf.hot_fraction == 0.0);
    assert(cfg_defaults.plasma.eedf.hot_temperature_factor == 2.5);
    assert(cfg_defaults.plasma.neutral_atom_temperature_eV == 0.1);
    assert(cfg_defaults.plasma.neutral_molecule_temperature_eV == 0.1);

    bool threw = false;
    try {
        dcr::io::ConfigLoader::load("/tmp/does_not_exist.yaml");
    } catch (const std::exception&) {
        threw = true;
    }
    assert(threw && "Expected load() to throw on missing file");

    std::cout << "[PASS] ConfigLoader checks.\n";
    return 0;
}
