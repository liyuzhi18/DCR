#include <cassert>
#include <filesystem>
#include <iostream>
#include <string>

#include <H5Cpp.h>

#include "../src/atomic/AtomicData.hpp"
#include "../src/physics/WallBoundary.hpp"
#include "../src/solver/boundary/BoundaryPhase.hpp"
#include "../src/solver/marching/MarchingDriver.hpp"
#include "../src/solver/output/HDF5Output.hpp"
#include "TestDCRSetup.hpp"

namespace fs = std::filesystem;

namespace {

std::vector<hsize_t> dataset_dims(const H5::DataSet& ds) {
    H5::DataSpace sp = ds.getSpace();
    const int rank = sp.getSimpleExtentNdims();
    std::vector<hsize_t> dims(static_cast<size_t>(rank), 0);
    if (rank > 0) sp.getSimpleExtentDims(dims.data());
    return dims;
}

bool dataset_exists(const H5::H5File& file, const std::string& path) {
    return H5Lexists(file.getId(), path.c_str(), H5P_DEFAULT) > 0;
}

} // namespace

int main() {
    std::cout << "--- Testing HDF5Output ---\n";

    auto cfg = test_dcr::load_reference_config(/*num_cells=*/4, /*verbose_logging=*/false);
    cfg.numerics.max_iterations = 80;
    const fs::path outdir = fs::temp_directory_path() / "dcr_hdf5_unit_test";
    fs::create_directories(outdir);
    cfg.io.output_dir = outdir.string();

    dcr::atomic::AtomicData atomic_data(cfg);
    auto plasma = test_dcr::make_plasma_state(cfg, atomic_data.get_total_states());
    test_dcr::EEDFContext eedf(cfg.plasma.Te_eV);

    const double ion_mass_amu = test_dcr::estimate_ion_mass_amu(cfg);
    const auto wall = dcr::physics::compute_wall_recycling(
        cfg.wall.material,
        cfg.plasma.Te_eV,
        cfg.plasma.Ti_eV,
        ion_mass_amu,
        cfg.wall.sheath_potential_drop
    );
    const auto boundary = dcr::solver::run_boundary_phase(
        cfg, atomic_data, plasma, eedf.grid, ion_mass_amu, wall
    );
    const auto history = dcr::solver::run_full_marching(cfg, atomic_data, plasma, eedf.grid, boundary);

    dcr::solver::write_hdf5_output(cfg, atomic_data, boundary, history);

    const fs::path h5_path = outdir / "dcr_results.h5";
    assert(fs::exists(h5_path));

    H5::H5File file(h5_path.string(), H5F_ACC_RDONLY);
    assert(dataset_exists(file, "/grid/x_cm"));
    assert(dataset_exists(file, "/grid/n_nuclei_cm3"));
    assert(dataset_exists(file, "/states/labels"));
    assert(dataset_exists(file, "/states/type_id"));
    assert(dataset_exists(file, "/states/charge"));
    assert(dataset_exists(file, "/states/atomicity"));
    assert(dataset_exists(file, "/states/internal_id"));
    assert(dataset_exists(file, "/states/P_indices"));
    assert(dataset_exists(file, "/states/A_indices"));
    assert(dataset_exists(file, "/states/M_indices"));
    assert(dataset_exists(file, "/population/background_full"));
    assert(dataset_exists(file, "/population/flowA"));
    assert(dataset_exists(file, "/population/flowM"));
    assert(dataset_exists(file, "/population/total_full"));

    const auto d_x = dataset_dims(file.openDataSet("/grid/x_cm"));
    const auto d_n = dataset_dims(file.openDataSet("/grid/n_nuclei_cm3"));
    const auto d_labels = dataset_dims(file.openDataSet("/states/labels"));
    const auto d_bg = dataset_dims(file.openDataSet("/population/background_full"));
    const auto d_a = dataset_dims(file.openDataSet("/population/flowA"));
    const auto d_m = dataset_dims(file.openDataSet("/population/flowM"));
    const auto d_tot = dataset_dims(file.openDataSet("/population/total_full"));

    const hsize_t n_nodes = static_cast<hsize_t>(cfg.grid.num_cells);
    const hsize_t total_states = static_cast<hsize_t>(atomic_data.get_total_states());
    const hsize_t nA = static_cast<hsize_t>(boundary.A_indices.size());
    const hsize_t nM = static_cast<hsize_t>(boundary.M_indices.size());

    assert(d_x.size() == 1 && d_x[0] == n_nodes);
    assert(d_n.size() == 1 && d_n[0] == n_nodes);
    assert(d_labels.size() == 1 && d_labels[0] == total_states);
    assert(d_bg.size() == 2 && d_bg[0] == n_nodes && d_bg[1] == total_states);
    assert(d_a.size() == 2 && d_a[0] == n_nodes && d_a[1] == nA);
    assert(d_m.size() == 2 && d_m[0] == n_nodes && d_m[1] == nM);
    assert(d_tot.size() == 2 && d_tot[0] == n_nodes && d_tot[1] == total_states);

    std::cout << "[PASS] HDF5Output checks.\n";
    return 0;
}
