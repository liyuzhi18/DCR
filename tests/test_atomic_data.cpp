#include <cassert>
#include <filesystem>
#include <iostream>

#include "../src/atomic/AtomicData.hpp"
#include "../src/io/ConfigStructs.hpp"

namespace fs = std::filesystem;

static fs::path source_root() {
#ifdef DCR_SOURCE_DIR
    return fs::path(DCR_SOURCE_DIR);
#else
    return fs::current_path();
#endif
}

int main() {
    std::cout << "--- Testing AtomicData ---\n";

    dcr::io::Config cfg;
    cfg.io.atomic_data_root = (source_root() / "atomic_data").string() + "/";

    dcr::io::SpeciesConfig h;
    h.name = "H_Atom";
    h.filename = "sclit.01";
    h.is_molecule = false;
    h.charge = 0;
    h.charge_set = true;
    h.mass_amu = 1.0;

    dcr::io::SpeciesConfig h2;
    h2.name = "H2_Molecule";
    h2.filename = "sclit_M.01";
    h2.is_molecule = true;
    h2.charge = 0;
    h2.charge_set = true;
    h2.mass_amu = 2.0;

    cfg.species.push_back(h);
    cfg.species.push_back(h2);

    dcr::atomic::AtomicData data(cfg);
    assert(data.get_total_states() > 0);
    assert(!data.get_processes().empty());

    std::cout << "[PASS] AtomicData checks.\n";
    return 0;
}
