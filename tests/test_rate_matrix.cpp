#include <cassert>
#include <filesystem>
#include <iostream>
#include <vector>

#include "../src/atomic/AtomicData.hpp"
#include "../src/io/ConfigStructs.hpp"
#include "../src/physics/EEDF.hpp"
#include "../src/state/PlasmaState.hpp"

namespace fs = std::filesystem;

static fs::path source_root() {
#ifdef DCR_SOURCE_DIR
    return fs::path(DCR_SOURCE_DIR);
#else
    return fs::current_path();
#endif
}

int main() {
    std::cout << "--- Testing Rate Matrix Assembly ---\n";

    dcr::io::Config cfg;
    cfg.io.atomic_data_root = (source_root() / "atomic_data").string() + "/";
    cfg.plasma.total_density = 1.0e12;
    cfg.plasma.Te_eV = 10.0;
    cfg.plasma.Ti_eV = 5.0;
    cfg.grid.num_cells = 1;

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

    // Initial population guess (arbitrary fractions).
    cfg.plasma.initial_conditions.push_back({0, 0.5});
    cfg.plasma.initial_conditions.push_back({1, 0.4});
    cfg.plasma.initial_conditions.push_back({2, 0.1});

    dcr::atomic::AtomicData data(cfg);
    const int total_states = data.get_total_states();
    assert(total_states > 0);

    // Build EEDF grid.
    std::vector<double> energies;
    std::vector<double> weights;
    const double emax = 200.0;
    const double dE = 1.0;
    for (double E = dE; E <= emax; E += dE) {
        energies.push_back(E);
        weights.push_back(dE);
    }

    EEDFConfig eedf_cfg;
    EEDF eedf(cfg.plasma.Te_eV, eedf_cfg);
    eedf.normalize_on_grid(energies, weights);
    EEDFGridView grid(energies, weights, &eedf);

    dcr::state::PlasmaState plasma(cfg.grid.num_cells, total_states);
    plasma.init_Te().setConstant(cfg.plasma.Te_eV);
    plasma.init_Ti().setConstant(cfg.plasma.Ti_eV);
    plasma.init_ne().setConstant(cfg.plasma.total_density);

    dcr::base::Vector population = dcr::base::Vector::Zero(total_states);
    for (const auto& ic : cfg.plasma.initial_conditions) {
        if (ic.species_index >= 0 && ic.species_index < total_states) {
            population(ic.species_index) = ic.fraction * cfg.plasma.total_density;
        }
    }

    dcr::base::Matrix R = dcr::base::Matrix::Zero(total_states, total_states);
    for (const auto& proc : data.get_processes()) {
        proc->apply(plasma, grid, population, R, nullptr);
    }

    const double sum = R.cwiseAbs().sum();
    std::cout << "|R|_sum = " << sum << "\n";
    assert(sum > 0.0);

    std::cout << "[PASS] Rate matrix assembly.\n";
    return 0;
}
