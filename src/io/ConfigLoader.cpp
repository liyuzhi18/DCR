#include "ConfigLoader.hpp"
#include <yaml-cpp/yaml.h>
#include <iostream>
#include <stdexcept>

namespace dcr::io {
    Config ConfigLoader::load(const std::string& filepath) {

        Config config;

        YAML::Node root = YAML::LoadFile(filepath);   
        if (root.IsNull()) {
            throw std::runtime_error("Failed to load config file: " + filepath);
        }

        // 2. Parse IO Section
        if (root["io"]) {
            auto io = root["io"];
            config.io.output_dir = io["output_dir"].as<std::string>("output");
            config.io.atomic_data_root = io["atomic_data_root"].as<std::string>("");
            config.io.data_tables_root = io["data_tables_root"].as<std::string>("data_tables");
            config.io.verbose_logging = io["verbose_logging"].as<bool>(true);
        }

        // 3. Parse Species List
        if (root["species"]) {
            for (const auto& node : root["species"]) {
                SpeciesConfig sp;
                sp.name = node["name"].as<std::string>();
                sp.filename = node["filename"].as<std::string>();
                sp.is_molecule = node["is_molecule"].as<bool>(false);
                sp.is_recycling = node["is_recycling"].as<bool>(false);
                if (node["charge"]) {
                    sp.charge = node["charge"].as<int>(0);
                    sp.charge_set = true;
                }
                sp.mass_amu = node["mass_amu"].as<base::Real>(1.0);
                config.species.push_back(sp);
            }
        }

        // 4. Parse Grid Config
        if (root["grid"]) {
            auto g = root["grid"];
            config.grid.length_cm = g["length_cm"].as<base::Real>();
            config.grid.num_cells = g["num_cells"].as<int>();
            config.grid.type = g["type"].as<std::string>("uniform");
            config.grid.first_cell_cm = g["first_cell_cm"].as<base::Real>(0.0);
            config.grid.poloidal_width_cm = g["poloidal_width_cm"].as<base::Real>(1.0);
        }
        // 5. Parse Plasma & Initial Conditions
        if (root["plasma"]) {
            auto p = root["plasma"];
            config.plasma.total_density = p["total_density_cm3"].as<base::Real>();
            config.plasma.Te_eV = p["Te_eV"].as<base::Real>();
            config.plasma.Ti_eV = p["Ti_eV"].as<base::Real>();
            config.plasma.neutral_atom_temperature_eV = p["neutral_atom_temperature_eV"].as<base::Real>(0.1);
            config.plasma.neutral_molecule_temperature_eV = p["neutral_molecule_temperature_eV"].as<base::Real>(0.1);

            if (p["initial_conditions"]) {
                for (const auto& ic_node : p["initial_conditions"]) {
                    InitialCondition ic;
                    ic.species_index = ic_node["index"].as<int>();
                    ic.fraction = ic_node["fraction"].as<base::Real>();
                    config.plasma.initial_conditions.push_back(ic);
                }
            }
        }
        // 6. Parse Wall
        if (root["wall"]) {
            config.wall.material = root["wall"]["material"].as<std::string>("W");
            config.wall.sheath_potential_drop = root["wall"]["sheath_potential_drop"].as<base::Real>(3.0);
            if (root["wall"]["temperature_eV"]) {
                config.wall.temperature_eV = root["wall"]["temperature_eV"].as<base::Real>(0.1);
            } else if (root["wall"]["desorbed_energy_molecule_eV"]) {
                // Backward compatibility with older configs.
                config.wall.temperature_eV = root["wall"]["desorbed_energy_molecule_eV"].as<base::Real>(0.1);
            }
        }

        // 7. Parse Numerics
        if (root["numerics"]) {
            config.numerics.tolerance = root["numerics"]["tolerance"].as<base::Real>(1e-6);
            config.numerics.max_iterations = root["numerics"]["max_iterations"].as<int>(1000);
            config.numerics.relaxation = root["numerics"]["relaxation"].as<base::Real>(1.0);
            config.numerics.marching_slow_iter_threshold =
                root["numerics"]["marching_slow_iter_threshold"].as<int>(100);
            config.numerics.marching_guess_retries =
                root["numerics"]["marching_guess_retries"].as<int>(3);
        }

        return config;
    }
    
}
