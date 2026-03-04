#include "SpeciesFileReader.hpp"

#include <algorithm>
#include <cctype>
#include <fstream>
#include <sstream>
#include <stdexcept>

namespace dcr::atomic {

namespace {
using LevelKey = std::pair<int, int>;

inline LevelKey make_key(int species, int local) {
    return LevelKey{species, local};
}

// Resolve file-local (species, local) to contiguous index; returns -1 if missing.
inline int resolve_local_index(const std::map<LevelKey, int>& lookup,
                               int species,
                               int local) {
    auto it = lookup.find(make_key(species, local));
    if (it == lookup.end()) return -1;
    return it->second;
}
} // namespace

void SpeciesFileReader::skip_blank_and_comment_lines(std::istream& in) {
    std::streampos pos;
    std::string line;
    while (true) {
        pos = in.tellg();
        if (!std::getline(in, line)) break;
        if (line.empty()) continue;
        std::istringstream iss(line);
        std::string first;
        iss >> first;
        if (!first.empty() && first[0] == '#') continue;
        in.seekg(pos);
        break;
    }
}

// Load atomic data file: parse level blocks then transition block.
AtomicSpecies SpeciesFileReader::load_atomic(const std::string& file_path, int Z) {
    std::ifstream file(file_path);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open atomic data file: " + file_path);
    }

    AtomicSpecies out;
    out.Z = Z;
    out.name = "Z" + std::to_string(Z);

    // Read level blocks: header "n_bound_electrons n_states max_n_levels iz_eng".
    // Each block corresponds to a charge state (charge = Z - n_bound_electrons).
    while (true) {
        skip_blank_and_comment_lines(file);
        int n_bound_electrons = 0, n_states = 0, max_n_levels = 0;
        double iz_eng = 0.0;

        std::streampos pos = file.tellg();
        if (!(file >> n_bound_electrons >> n_states >> max_n_levels >> iz_eng)) {
            file.clear();
            file.seekg(pos);
            break; // transitions next
        }
        std::string dummy;
        std::getline(file, dummy);

        int charge_num_block = Z - n_bound_electrons;
        out.ionization_energy_ev_by_charge[charge_num_block] = iz_eng;

        parse_atomic_level_block(file, n_states, n_bound_electrons, Z, out);
    }

    skip_blank_and_comment_lines(file);
    int num_transitions = 0;
    if (file >> num_transitions) {
        std::string dummy;
        std::getline(file, dummy);
        parse_atomic_transition_block(file, num_transitions, out);
    }

    return out;
}

// Parse one atomic charge-state block of levels.
void SpeciesFileReader::parse_atomic_level_block(std::istream& in,
                                                 int num_states,
                                                 int n_bound_electrons_block,
                                                 int Z,
                                                 AtomicSpecies& out) {
    std::string line;
    for (int i = 0; i < num_states; ++i) {
        if (!std::getline(in, line)) {
            throw std::runtime_error("Unexpected EOF while reading atomic levels");
        }
        if (line.empty()) { --i; continue; }

        std::istringstream iss(line);
        AtomicLevel lvl;
        int n_bound_e_line = 0;

        iss >> n_bound_e_line >> lvl.local_index >> lvl.global_index >> lvl.label
            >> lvl.excitation_energy_ev >> lvl.statistical_weight
            >> lvl.rad_params[0] >> lvl.rad_params[1] >> lvl.rad_params[2]
            >> lvl.e_ex >> lvl.density_eff;

        for (int j = 0; j < 10; ++j) iss >> lvl.occupation[j];

        lvl.charge_number = Z - n_bound_electrons_block;
        lvl.file_block_index = n_bound_electrons_block;

        out.levels.push_back(lvl);
    }
}

// Find an atomic level by (file block = n_bound_electrons, local index).
static const AtomicLevel* find_atomic_level_by_block_local(const AtomicSpecies& out,
                                                           int n_bound_electrons,
                                                           int local_index) {
    if (local_index <= 0) return nullptr;
    const int desired_charge = out.Z - n_bound_electrons;
    for (const auto& lv : out.levels) {
        if (lv.charge_number == desired_charge && lv.local_index == local_index) {
            return &lv;
        }
    }
    return nullptr;
}

// Parse atomic transition list and compute per-transition threshold energies.
void SpeciesFileReader::parse_atomic_transition_block(std::istream& in,
                                                      int num_transitions,
                                                      AtomicSpecies& out) {
    std::string line;
    for (int i = 0; i < num_transitions; ++i) {
        if (!std::getline(in, line)) break;
        if (line.empty()) { --i; continue; }

        std::istringstream iss(line);
        AtomicTransition tr;
        iss >> tr.type;
        if (tr.type.empty() || !std::isalpha(static_cast<unsigned char>(tr.type[0]))) {
            --i;
            continue;
        }

        if (tr.type == "b") {
            iss >> tr.state_params[0] >> tr.state_params[1]
                >> tr.state_params[2] >> tr.state_params[3]
                >> tr.flag
                >> tr.params[0] >> tr.params[1] >> tr.params[2]
                >> tr.params[3] >> tr.params[4] >> tr.params[5]
                >> tr.oscillator_strength;
        } else if (tr.type == "f" || tr.type == "p") {
            iss >> tr.state_params[0] >> tr.state_params[1]
                >> tr.state_params[2] >> tr.state_params[3]
                >> tr.flag;
            if (tr.flag == 99) {
                iss >> tr.params[0] >> tr.params[1];
            } else if (tr.flag == 98) {
                for (int p = 0; p < 8; ++p) iss >> tr.params[p];
            }
        } else if (tr.type == "a") {
            iss >> tr.state_params[0] >> tr.state_params[1]
                >> tr.state_params[2] >> tr.state_params[3]
                >> tr.params[0];
        } else {
            continue;
        }

        const AtomicLevel* Lf = find_atomic_level_by_block_local(out,
                                                                 tr.state_params[0],
                                                                 tr.state_params[1]);
        const AtomicLevel* Lt = find_atomic_level_by_block_local(out,
                                                                 tr.state_params[2],
                                                                 tr.state_params[3]);

        double threshold_ev = 0.0;
        if (Lf && Lt) {
            const int charge_from = Lf->charge_number;
            const int charge_to = Lt->charge_number;
            auto it_iz = out.ionization_energy_ev_by_charge.find(charge_from);
            const double iz = (it_iz != out.ionization_energy_ev_by_charge.end()) ? it_iz->second : 0.0;

            const double dE_boundbound = Lt->excitation_energy_ev - Lf->excitation_energy_ev;

            if (charge_to > charge_from) {
                threshold_ev = std::max(0.0, iz - Lf->excitation_energy_ev + Lt->excitation_energy_ev);
            } else {
                threshold_ev = std::max(0.0, dE_boundbound);
            }
        }
        tr.threshold_ev = threshold_ev;

        out.transitions.push_back(tr);
    }
}

// Load molecular data file: parse level blocks then transition block.
MolecularSpecies SpeciesFileReader::load_molecular(const std::string& file_path, int id) {
    std::ifstream file(file_path);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open molecular data file: " + file_path);
    }

    MolecularSpecies out;
    out.id = id;
    out.name = "Mol" + std::to_string(id);

    while (true) {
        skip_blank_and_comment_lines(file);
        int n_bound_electrons = 0, n_states = 0, n_protons = 0;
        double iz_eng = 0.0;

        std::streampos pos = file.tellg();
        if (!(file >> n_bound_electrons >> n_states >> n_protons >> iz_eng)) {
            file.clear();
            file.seekg(pos);
            break; // transitions next
        }
        std::string dummy;
        std::getline(file, dummy);

        parse_molecular_level_block(file, n_states, n_bound_electrons, n_protons, out);
    }

    // Assign contiguous local_index across the entire file so transitions can
    // resolve (species_index, file_local_index) to a single index space.
    for (size_t i = 0; i < out.levels.size(); ++i) {
        out.levels[i].local_index = static_cast<int>(i) + 1;
    }

    // Build lookup table from file-local identifiers to contiguous indices.
    std::map<LevelKey, int> level_lookup;
    for (const auto& lvl : out.levels) {
        level_lookup[make_key(lvl.file_species_index, lvl.file_local_index)] = lvl.local_index;
    }

    skip_blank_and_comment_lines(file);
    int num_transitions = 0;
    if (file >> num_transitions) {
        std::string dummy;
        std::getline(file, dummy);
        parse_molecular_transition_block(file, num_transitions, level_lookup, out);
    }

    return out;
}

// Parse one molecular block of levels (file-local identifiers).
void SpeciesFileReader::parse_molecular_level_block(std::istream& in,
                                                    int num_states,
                                                    int n_bound_electrons,
                                                    int n_protons,
                                                    MolecularSpecies& out) {
    std::string line;
    for (int i = 0; i < num_states; ++i) {
        if (!std::getline(in, line)) {
            throw std::runtime_error("Unexpected EOF while reading molecular levels");
        }
        if (line.empty()) { --i; continue; }

        std::istringstream iss(line);
        MolecularLevel lvl;
        int species_index = 0;
        int level_num_tag = 0;

        iss >> species_index >> level_num_tag >> lvl.global_index
            >> lvl.label >> lvl.excitation_energy_ev >> lvl.dissociation_energy_ev;

        lvl.file_species_index = species_index;
        lvl.file_local_index = level_num_tag;
        lvl.charge_number = n_protons - n_bound_electrons;
        lvl.proton_count = n_protons;
        lvl.local_index = 0;
        out.levels.push_back(lvl);
    }
}

// Parse molecular transition list and resolve reactant/product indices.
void SpeciesFileReader::parse_molecular_transition_block(std::istream& in,
                                                         int num_transitions,
                                                         const std::map<std::pair<int, int>, int>& level_lookup,
                                                         MolecularSpecies& out) {
    std::string line;
    for (int i = 0; i < num_transitions; ++i) {
        if (!std::getline(in, line)) break;
        if (line.empty()) { --i; continue; }

        std::istringstream iss(line);
        MolecularTransition tr;
        iss >> tr.type;
        if (tr.type.empty()) { --i; continue; }

        auto read_pair = [&](MolecularStateRef& ref) -> bool {
            int species = 0;
            int local = 0;
            if (!(iss >> species >> local)) return false;
            ref.species_index = species;
            ref.file_local_index = local;
            // Resolve file-local indices into contiguous indices used in memory.
            ref.local_index = resolve_local_index(level_lookup, species, local);
            return ref.local_index > 0;
        };

        auto read_pair_allow_atomic_ref = [&](MolecularStateRef& ref) -> bool {
            int species = 0;
            int local = 0;
            if (!(iss >> species >> local)) return false;
            ref.species_index = species;
            ref.file_local_index = local;
            // Keep unresolved refs for atomic partners (species 0/1); they are
            // resolved later in AtomicData::build_molecular_processes.
            ref.local_index = resolve_local_index(level_lookup, species, local);
            if (ref.local_index > 0) return true;
            return (ref.species_index == 0 || ref.species_index == 1) && ref.file_local_index > 0;
        };

        auto read_pair_allow_unresolved = [&](MolecularStateRef& ref) -> bool {
            int species = 0;
            int local = 0;
            if (!(iss >> species >> local)) return false;
            ref.species_index = species;
            ref.file_local_index = local;
            // Allow unresolved indices (e.g., MCX partner species from atomic files).
            ref.local_index = resolve_local_index(level_lookup, species, local);
            return true;
        };

        auto fill_params_from_stream = [&](int count) {
            for (int p = 0; p < count && iss; ++p) iss >> tr.params[p];
        };

        if (tr.type == "ve") {
            // VE uses external tables; skip for now.
            continue;
        } else if (tr.type == "ev") {
            tr.reactant_count = 1;
            tr.product_count = 1;
            iss >> tr.flag;
            if (!read_pair(tr.reactants[0]) || !read_pair(tr.products[0])) continue;
            iss >> tr.threshold_ev;
            fill_params_from_stream(6);
        } else if (tr.type == "mi") {
            tr.reactant_count = 1;
            tr.product_count = 1;
            iss >> tr.flag;
            if (!read_pair(tr.reactants[0]) || !read_pair(tr.products[0])) continue;
            iss >> tr.threshold_ev;
            fill_params_from_stream(8);
        } else if (tr.type == "di") {
            continue;
        } else if (tr.type == "mide") {
            tr.reactant_count = 1;
            tr.product_count = 2;
            iss >> tr.flag;
            if (!read_pair_allow_atomic_ref(tr.reactants[0]) ||
                !read_pair_allow_atomic_ref(tr.products[0]) ||
                !read_pair_allow_atomic_ref(tr.products[1])) {
                continue;
            }
            tr.threshold_ev = 0.0;
            fill_params_from_stream(3);
        } else if (tr.type == "mcx") {
            tr.reactant_count = 2;
            tr.product_count = 2;
            iss >> tr.flag;
            if (!read_pair_allow_unresolved(tr.reactants[0]) || !read_pair_allow_unresolved(tr.reactants[1]) ||
                !read_pair_allow_unresolved(tr.products[0]) || !read_pair_allow_unresolved(tr.products[1])) {
                continue;
            }
            iss >> tr.threshold_ev;
            if (tr.flag == 99) {
                fill_params_from_stream(6);
            } else if (tr.flag == 98) {
                fill_params_from_stream(11);
            } else if (tr.flag == 90) {
                // Analytic high-v fit, no additional parameters.
            }
        } else if (tr.type == "ra" || tr.type == "da" || tr.type == "dr" || tr.type == "ed" || tr.type == "de") {
            if (!(iss >> tr.flag)) throw std::runtime_error("Missing flag in molecular transition entry");

            auto read_pairs = [&](MolecularStateRef* refs, int count) -> bool {
                for (int j = 0; j < count; ++j) {
                    if (!read_pair_allow_atomic_ref(refs[j])) {
                        return false;
                    }
                }
                return true;
            };

            auto read_params = [&]() {
                int idx = 0;
                double value = 0.0;
                while (idx < static_cast<int>(tr.params.size()) && (iss >> value)) {
                    tr.params[idx++] = value;
                }
            };

            if (tr.type == "ra") {
                tr.reactant_count = 1;
                tr.product_count = 1;
            } else if (tr.type == "da" || tr.type == "dr" || tr.type == "de") {
                tr.reactant_count = 1;
                tr.product_count = 2;
            } else { // "ed"
                switch (tr.flag) {
                    case 99: tr.reactant_count = 1; tr.product_count = 1; break;
                    case 98: tr.reactant_count = 2; tr.product_count = 2; break;
                    case 97: tr.reactant_count = 2; tr.product_count = 1; break;
                    case 96: tr.reactant_count = 2; tr.product_count = 1; break;
                    case 95: tr.reactant_count = 2; tr.product_count = 2; break;
                    default:
                        throw std::runtime_error("Unhandled ED flag in transition entry");
                }
            }

            if (tr.reactant_count > 4 || tr.product_count > 4) {
                throw std::runtime_error("Too many reactants/products in molecular transition entry");
            }

            if (!read_pairs(tr.reactants.data(), tr.reactant_count)) continue;
            if (!read_pairs(tr.products.data(), tr.product_count)) continue;
            read_params();
        } else {
            continue;
        }

        out.transitions.push_back(tr);
    }
}

} // namespace dcr::atomic
