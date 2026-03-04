#include <cassert>
#include <filesystem>
#include <iostream>
#include <vector>

#include "../src/atomic/AtomicData.hpp"
#include "../src/base/Types.hpp"
#include "../src/physics/EEDF.hpp"
#include "../src/processes/MolecularProcess.hpp"
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
    std::cout << "--- Testing MCX IO Integration ---\n";

    dcr::io::Config cfg;
    cfg.io.atomic_data_root = (source_root() / "atomic_data").string() + "/";
    cfg.io.data_tables_root = (source_root() / "data_tables").string() + "/";
    cfg.plasma.total_density = 1.0e12;
    cfg.plasma.Te_eV = 5.0;
    cfg.plasma.Ti_eV = 5.0;
    cfg.grid.num_cells = 1;

    dcr::io::SpeciesConfig h_atom;
    h_atom.name = "H_Atom";
    h_atom.filename = "sclit.01";
    h_atom.is_molecule = false;
    h_atom.charge = 0;
    h_atom.charge_set = true;
    h_atom.mass_amu = 1.0;

    dcr::io::SpeciesConfig h_ion = h_atom;
    h_ion.name = "H_Ion";
    h_ion.charge = 1;
    h_ion.charge_set = true;

    dcr::io::SpeciesConfig h2_mol;
    h2_mol.name = "H2_Molecule";
    h2_mol.filename = "sclit_M.01";
    h2_mol.is_molecule = true;
    h2_mol.charge = 0;
    h2_mol.charge_set = true;
    h2_mol.mass_amu = 2.0;

    dcr::io::SpeciesConfig h2_ion = h2_mol;
    h2_ion.name = "H2_Ion";
    h2_ion.charge = 1;
    h2_ion.charge_set = true;

    cfg.species.push_back(h_atom);
    cfg.species.push_back(h_ion);
    cfg.species.push_back(h2_mol);
    cfg.species.push_back(h2_ion);

    dcr::atomic::AtomicData data(cfg);
    const int total_states = data.get_total_states();
    assert(total_states > 0);

    // Build a simple EEDF grid (MCX does not use it, but the API requires it).
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

    // Provide non-zero populations for all states to ensure MCX reactants are present.
    dcr::base::Vector population = dcr::base::Vector::Constant(total_states, cfg.plasma.total_density * 1.0e-2);

    dcr::base::Matrix R = dcr::base::Matrix::Zero(total_states, total_states);
    double mcx_sum = 0.0;
    int mcx_count = 0;
    for (const auto& proc : data.get_processes()) {
        auto* mcx = dynamic_cast<crm_detail::MolecularMCXProcess*>(proc.get());
        if (!mcx) continue;
        ++mcx_count;
        dcr::base::Matrix before = R;
        mcx->apply(plasma, grid, population, R, nullptr);
        mcx_sum += (R - before).cwiseAbs().sum();
    }

    std::cout << "MCX processes: " << mcx_count << " |sum|=" << mcx_sum << "\n";
    assert(mcx_count > 0);
    assert(mcx_sum > 0.0);

    std::cout << "[PASS] MCX integration.\n";
    return 0;
}
