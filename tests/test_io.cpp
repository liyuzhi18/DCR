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
        "plasma:\n"
        "  total_density_cm3: 1e14\n"
        "  Te_eV: 5.0\n"
        "  Ti_eV: 2.0\n"
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
        "  relaxation: 0.5\n";

    fs::path full_path = write_temp_config("dcr_test_full.yaml", yaml_full);
    auto cfg = dcr::io::ConfigLoader::load(full_path.string());

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
    assert(cfg.grid.poloidal_width_cm == 2.0);

    assert(cfg.plasma.total_density == 1e14);
    assert(cfg.plasma.Te_eV == 5.0);
    assert(cfg.plasma.Ti_eV == 2.0);
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

    assert(cfg_defaults.io.output_dir == "output/defaults");
    assert(cfg_defaults.io.verbose_logging == true);
    assert(cfg_defaults.io.atomic_data_root == "");

    assert(cfg_defaults.grid.type == "uniform");
    assert(cfg_defaults.grid.first_cell_cm == 0.0);
    assert(cfg_defaults.grid.poloidal_width_cm == 1.0);

    assert(cfg_defaults.species.empty());
    assert(cfg_defaults.wall.material == "W");
    assert(cfg_defaults.wall.sheath_potential_drop == 3.0);
    assert(cfg_defaults.wall.temperature_eV == 0.1);
    assert(cfg_defaults.numerics.tolerance == 1e-6);
    assert(cfg_defaults.numerics.max_iterations == 1000);
    assert(cfg_defaults.numerics.relaxation == 1.0);

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
