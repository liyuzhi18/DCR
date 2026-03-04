#pragma once

#include <array>
#include <iosfwd>
#include <map>
#include <string>
#include <vector>

namespace dcr::atomic {

// Atomic level metadata parsed from sclit.* files.
struct AtomicLevel {
    int local_index = 0;
    int global_index = 0;
    std::string label;
    double excitation_energy_ev = 0.0;
    double statistical_weight = 0.0;
    std::array<double, 3> rad_params{};
    double e_ex = 0.0;
    double density_eff = 0.0;
    std::array<int, 10> occupation{};
    int charge_number = 0;
    int file_block_index = 0; // n_bound_electrons block in file
};

// Atomic transition entry (bound-bound, ionization, photo-ionization, etc.).
struct AtomicTransition {
    std::string type;
    std::array<int, 4> state_params{};
    int flag = 0;
    std::array<double, 8> params{};
    double oscillator_strength = 0.0;
    double threshold_ev = 0.0;
};

// Aggregated atomic species data from a single file.
struct AtomicSpecies {
    int Z = 0;
    std::string name;
    std::vector<AtomicLevel> levels;
    std::vector<AtomicTransition> transitions;
    std::map<int, double> ionization_energy_ev_by_charge;
};

// Molecular level metadata parsed from sclit_M.* files.
struct MolecularLevel {
    int file_species_index = 0;
    int file_local_index = 0;
    int local_index = 0; // contiguous across file
    int global_index = 0;
    std::string label;
    double excitation_energy_ev = 0.0;
    double dissociation_energy_ev = 0.0;
    int charge_number = 0;
    int proton_count = 0;
};

// Reference to a molecular level (file-local and resolved contiguous index).
struct MolecularStateRef {
    int species_index = 0;
    int local_index = 0; // resolved contiguous index
    int file_local_index = 0; // original local index in file
};

// Molecular transition entry with resolved reactant/product indices.
struct MolecularTransition {
    std::string type;
    int flag = 0;
    int reactant_count = 0;
    int product_count = 0;
    std::array<MolecularStateRef, 4> reactants{};
    std::array<MolecularStateRef, 4> products{};
    std::array<double, 11> params{};
    double threshold_ev = 0.0;
    std::string data_file;
};

// Aggregated molecular species data from a single file.
struct MolecularSpecies {
    int id = 0;
    std::string name;
    std::vector<MolecularLevel> levels;
    std::vector<MolecularTransition> transitions;
};

class SpeciesFileReader {
public:
    static AtomicSpecies load_atomic(const std::string& file_path, int Z);
    static MolecularSpecies load_molecular(const std::string& file_path, int id);

private:
    static void skip_blank_and_comment_lines(std::istream& in);

    static void parse_atomic_level_block(std::istream& in,
                                         int num_states,
                                         int n_bound_electrons_block,
                                         int Z,
                                         AtomicSpecies& out);

    static void parse_atomic_transition_block(std::istream& in,
                                              int num_transitions,
                                              AtomicSpecies& out);

    static void parse_molecular_level_block(std::istream& in,
                                            int num_states,
                                            int n_bound_electrons,
                                            int n_protons,
                                            MolecularSpecies& out);

    static void parse_molecular_transition_block(std::istream& in,
                                                 int num_transitions,
                                                 const std::map<std::pair<int, int>, int>& level_lookup,
                                                 MolecularSpecies& out);
};

} // namespace dcr::atomic
