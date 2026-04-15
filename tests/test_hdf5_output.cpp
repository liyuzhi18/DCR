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
    const auto boundary_temperatures = test_dcr::plasma_temperatures_at(cfg, 0.0);
    test_dcr::EEDFContext eedf(boundary_temperatures.electron_eV);

    const double ion_mass_amu = test_dcr::estimate_ion_mass_amu(cfg);
    const auto wall = dcr::physics::compute_wall_recycling(
        cfg.wall.material,
        boundary_temperatures.electron_eV,
        boundary_temperatures.ion_eV,
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
    assert(dataset_exists(file, "/timing/boundary_elapsed_seconds"));
    assert(dataset_exists(file, "/timing/boundary_iterations"));
    assert(dataset_exists(file, "/timing/marching_cell_elapsed_seconds"));
    assert(dataset_exists(file, "/timing/marching_cell_iterations"));
    assert(dataset_exists(file, "/timing/marching_cell_converged"));
    assert(dataset_exists(file, "/rates/Te_eV"));
    assert(dataset_exists(file, "/rates/Ti_eV"));
    assert(dataset_exists(file, "/rates/ne_cm3"));
    assert(dataset_exists(file, "/rates/atomic_scd_cm3_s"));
    assert(dataset_exists(file, "/rates/atomic_acd_cm3_s"));
    assert(dataset_exists(file, "/rates/atomic_qss_transport_frequency_s"));
    assert(dataset_exists(file, "/rates/atomic_qss_max_transport_to_local_ratio"));
    assert(dataset_exists(file, "/rates/atomic_qss_max_transport_to_loss_frequency_ratio"));
    assert(dataset_exists(file, "/rates/atomic_excited_indices"));
    assert(dataset_exists(file, "/rates/atomic_excited_labels"));
    assert(dataset_exists(file, "/rates/atomic_excited_transport_rate_cm3_s"));
    assert(dataset_exists(file, "/rates/atomic_excited_local_source_rate_cm3_s"));
    assert(dataset_exists(file, "/rates/atomic_excited_local_loss_rate_cm3_s"));
    assert(dataset_exists(file, "/rates/atomic_excited_local_loss_frequency_s"));
    assert(dataset_exists(file, "/rates/atomic_excited_transport_to_local_ratio"));
    assert(dataset_exists(file, "/rates/atomic_excited_transport_to_loss_frequency_ratio"));

    const auto d_x = dataset_dims(file.openDataSet("/grid/x_cm"));
    const auto d_n = dataset_dims(file.openDataSet("/grid/n_nuclei_cm3"));
    const auto d_labels = dataset_dims(file.openDataSet("/states/labels"));
    const auto d_bg = dataset_dims(file.openDataSet("/population/background_full"));
    const auto d_a = dataset_dims(file.openDataSet("/population/flowA"));
    const auto d_m = dataset_dims(file.openDataSet("/population/flowM"));
    const auto d_tot = dataset_dims(file.openDataSet("/population/total_full"));
    const auto d_boundary_wall = dataset_dims(file.openDataSet("/timing/boundary_elapsed_seconds"));
    const auto d_boundary_iters = dataset_dims(file.openDataSet("/timing/boundary_iterations"));
    const auto d_cell_wall = dataset_dims(file.openDataSet("/timing/marching_cell_elapsed_seconds"));
    const auto d_cell_iters = dataset_dims(file.openDataSet("/timing/marching_cell_iterations"));
    const auto d_cell_conv = dataset_dims(file.openDataSet("/timing/marching_cell_converged"));
    const auto d_te = dataset_dims(file.openDataSet("/rates/Te_eV"));
    const auto d_ti = dataset_dims(file.openDataSet("/rates/Ti_eV"));
    const auto d_ne = dataset_dims(file.openDataSet("/rates/ne_cm3"));
    const auto d_scd = dataset_dims(file.openDataSet("/rates/atomic_scd_cm3_s"));
    const auto d_acd = dataset_dims(file.openDataSet("/rates/atomic_acd_cm3_s"));
    const auto d_qss_transport = dataset_dims(file.openDataSet("/rates/atomic_qss_transport_frequency_s"));
    const auto d_excited = dataset_dims(file.openDataSet("/rates/atomic_excited_indices"));
    const auto d_excited_transport = dataset_dims(file.openDataSet("/rates/atomic_excited_transport_rate_cm3_s"));

    const hsize_t n_nodes = static_cast<hsize_t>(cfg.grid.num_cells);
    const hsize_t total_states = static_cast<hsize_t>(atomic_data.get_total_states());
    const hsize_t nA = static_cast<hsize_t>(boundary.A_indices.size());
    const hsize_t nM = static_cast<hsize_t>(boundary.M_indices.size());
    const hsize_t n_excited = d_excited.empty() ? 0 : d_excited[0];

    assert(d_x.size() == 1 && d_x[0] == n_nodes);
    assert(d_n.size() == 1 && d_n[0] == n_nodes);
    assert(d_labels.size() == 1 && d_labels[0] == total_states);
    assert(d_bg.size() == 2 && d_bg[0] == n_nodes && d_bg[1] == total_states);
    assert(d_a.size() == 2 && d_a[0] == n_nodes && d_a[1] == nA);
    assert(d_m.size() == 2 && d_m[0] == n_nodes && d_m[1] == nM);
    assert(d_tot.size() == 2 && d_tot[0] == n_nodes && d_tot[1] == total_states);
    assert(d_boundary_wall.size() == 1 && d_boundary_wall[0] == 1);
    assert(d_boundary_iters.size() == 1 && d_boundary_iters[0] == 1);
    assert(d_cell_wall.size() == 1 && d_cell_wall[0] == n_nodes - 1);
    assert(d_cell_iters.size() == 1 && d_cell_iters[0] == n_nodes - 1);
    assert(d_cell_conv.size() == 1 && d_cell_conv[0] == n_nodes - 1);
    assert(d_te.size() == 1 && d_te[0] == n_nodes);
    assert(d_ti.size() == 1 && d_ti[0] == n_nodes);
    assert(d_ne.size() == 1 && d_ne[0] == n_nodes);
    assert(d_scd.size() == 1 && d_scd[0] == n_nodes);
    assert(d_acd.size() == 1 && d_acd[0] == n_nodes);
    assert(d_qss_transport.size() == 1 && d_qss_transport[0] == n_nodes);
    assert(d_excited.size() == 1 && n_excited > 0);
    assert(d_excited_transport.size() == 2 &&
           d_excited_transport[0] == n_nodes &&
           d_excited_transport[1] == n_excited);

    std::cout << "[PASS] HDF5Output checks.\n";
    return 0;
}
