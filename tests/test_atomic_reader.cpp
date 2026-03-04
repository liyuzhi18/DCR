#include <cassert>
#include <cstdlib>
#include <filesystem>
#include <iostream>

#include "../src/atomic/SpeciesFileReader.hpp"

namespace fs = std::filesystem;

static fs::path source_root() {
#ifdef DCR_SOURCE_DIR
    return fs::path(DCR_SOURCE_DIR);
#else
    return fs::current_path();
#endif
}

int main() {
    std::cout << "--- Testing SpeciesFileReader ---\n";
    const bool dump = (std::getenv("DCR_TEST_DUMP") != nullptr);

    const fs::path root = source_root();
    const fs::path atomic_path = root / "atomic_data" / "sclit.01";
    const fs::path molecular_path = root / "atomic_data" / "sclit_M.01";

    auto atomic = dcr::atomic::SpeciesFileReader::load_atomic(atomic_path.string(), 1);
    assert(!atomic.levels.empty());
    assert(!atomic.transitions.empty());
    assert(!atomic.ionization_energy_ev_by_charge.empty());

    if (dump) {
        std::cout << "[Atomic] Z=" << atomic.Z
                  << " levels=" << atomic.levels.size()
                  << " transitions=" << atomic.transitions.size() << "\n";
        const size_t max_levels = std::min<size_t>(3, atomic.levels.size());
        for (size_t i = 0; i < max_levels; ++i) {
            const auto& lvl = atomic.levels[i];
            std::cout << "  lvl[" << i << "] charge=" << lvl.charge_number
                      << " local=" << lvl.local_index
                      << " global=" << lvl.global_index
                      << " label=" << lvl.label
                      << " Ex_eV=" << lvl.excitation_energy_ev
                      << " g=" << lvl.statistical_weight << "\n";
        }
        const size_t max_tr = std::min<size_t>(3, atomic.transitions.size());
        for (size_t i = 0; i < max_tr; ++i) {
            const auto& tr = atomic.transitions[i];
            std::cout << "  tr[" << i << "] type=" << tr.type
                      << " flag=" << tr.flag
                      << " state=(" << tr.state_params[0] << "," << tr.state_params[1]
                      << " -> " << tr.state_params[2] << "," << tr.state_params[3] << ")"
                      << " Eth=" << tr.threshold_ev << "\n";
        }
    }

    bool has_neutral = false;
    for (const auto& lvl : atomic.levels) {
        if (lvl.charge_number == 0) {
            has_neutral = true;
            break;
        }
    }
    assert(has_neutral && "Expected at least one neutral atomic level");

    auto molecular = dcr::atomic::SpeciesFileReader::load_molecular(molecular_path.string(), 1);
    assert(!molecular.levels.empty());
    assert(!molecular.transitions.empty());

    if (dump) {
        std::cout << "[Molecular] id=" << molecular.id
                  << " levels=" << molecular.levels.size()
                  << " transitions=" << molecular.transitions.size() << "\n";
        const size_t max_levels = std::min<size_t>(3, molecular.levels.size());
        for (size_t i = 0; i < max_levels; ++i) {
            const auto& lvl = molecular.levels[i];
            std::cout << "  lvl[" << i << "] species=" << lvl.file_species_index
                      << " file_local=" << lvl.file_local_index
                      << " local=" << lvl.local_index
                      << " label=" << lvl.label
                      << " Ex_eV=" << lvl.excitation_energy_ev
                      << " Diss_eV=" << lvl.dissociation_energy_ev << "\n";
        }
        const size_t max_tr = std::min<size_t>(3, molecular.transitions.size());
        for (size_t i = 0; i < max_tr; ++i) {
            const auto& tr = molecular.transitions[i];
            std::cout << "  tr[" << i << "] type=" << tr.type
                      << " flag=" << tr.flag
                      << " Eth=" << tr.threshold_ev
                      << " reactants=" << tr.reactant_count
                      << " products=" << tr.product_count << "\n";
        }
    }

    const int level_count = static_cast<int>(molecular.levels.size());
    for (const auto& tr : molecular.transitions) {
        const bool allow_unresolved =
            (tr.type == "mcx" || tr.type == "ra" || tr.type == "da" ||
             tr.type == "dr" || tr.type == "ed" || tr.type == "de" ||
             tr.type == "mide");
        for (int i = 0; i < tr.reactant_count; ++i) {
            const auto& ref = tr.reactants[i];
            if (allow_unresolved && ref.local_index <= 0) {
                assert(ref.file_local_index > 0);
                assert(ref.species_index == 0 || ref.species_index == 1);
                continue;
            }
            assert(ref.local_index > 0 && ref.local_index <= level_count);
        }
        for (int i = 0; i < tr.product_count; ++i) {
            const auto& ref = tr.products[i];
            if (allow_unresolved && ref.local_index <= 0) {
                assert(ref.file_local_index > 0);
                assert(ref.species_index == 0 || ref.species_index == 1);
                continue;
            }
            assert(ref.local_index > 0 && ref.local_index <= level_count);
        }
    }

    std::cout << "[PASS] SpeciesFileReader checks.\n";
    return 0;
}
