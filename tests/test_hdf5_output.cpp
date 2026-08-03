#include <cassert>
#include <cmath>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

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

int read_scalar_int(const H5::H5File& file, const std::string& path) {
    int value = 0;
    file.openDataSet(path).read(&value, H5::PredType::NATIVE_INT);
    return value;
}

double read_scalar_double(const H5::H5File& file, const std::string& path) {
    double value = 0.0;
    file.openDataSet(path).read(&value, H5::PredType::NATIVE_DOUBLE);
    return value;
}

std::vector<double> read_vector_double(const H5::H5File& file, const std::string& path) {
    const auto ds = file.openDataSet(path);
    const auto dims = dataset_dims(ds);
    std::vector<double> values(dims.empty() ? 0 : static_cast<size_t>(dims[0]), 0.0);
    if (!values.empty()) ds.read(values.data(), H5::PredType::NATIVE_DOUBLE);
    return values;
}

} // namespace

int main() {
    std::cout << "--- Testing HDF5Output ---\n";

    auto cfg = test_dcr::load_reference_config(/*num_cells=*/4, /*verbose_logging=*/false);
    cfg.numerics.max_iterations = 80;
    cfg.numerics.adaptive_recycling_domain.enabled = true;
    cfg.numerics.adaptive_recycling_domain.epsilon_A = 0.99;
    cfg.numerics.adaptive_recycling_domain.epsilon_M = 0.5;
    cfg.numerics.adaptive_recycling_domain.closure_mode = "variable_nuclei_balance";
    cfg.numerics.adaptive_recycling_domain.max_outer_iterations = 2;
    cfg.numerics.adaptive_recycling_domain.L1_relative_tolerance = 0.0;
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
    const auto history = dcr::solver::run_adaptive_variable_marching(
        cfg, atomic_data, plasma, eedf.grid, boundary
    );

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
    assert(dataset_exists(file, "/states/H2_plus_indices"));
    assert(dataset_exists(file, "/population/background_full"));
    assert(dataset_exists(file, "/population/flowA"));
    assert(dataset_exists(file, "/population/flowM"));
    assert(dataset_exists(file, "/population/total_full"));
    assert(dataset_exists(file, "/timing/boundary_elapsed_seconds"));
    assert(dataset_exists(file, "/timing/boundary_iterations"));
    assert(dataset_exists(file, "/timing/marching_cell_elapsed_seconds"));
    assert(dataset_exists(file, "/timing/marching_cell_iterations"));
    assert(dataset_exists(file, "/timing/marching_cell_converged"));
    assert(dataset_exists(file, "/adaptive/enabled"));
    assert(dataset_exists(file, "/adaptive/L1_found"));
    assert(dataset_exists(file, "/adaptive/L1_index"));
    assert(dataset_exists(file, "/adaptive/L1_cm"));
    assert(dataset_exists(file, "/adaptive/atomic_flow_fraction"));
    assert(dataset_exists(file, "/adaptive/LM_found"));
    assert(dataset_exists(file, "/adaptive/LM_index"));
    assert(dataset_exists(file, "/adaptive/LM_cm"));
    assert(dataset_exists(file, "/adaptive/molecular_flow_fraction"));
    assert(dataset_exists(file, "/adaptive/transport_profile_available"));
    assert(dataset_exists(file, "/adaptive/outer_loop_enabled"));
    assert(dataset_exists(file, "/adaptive/outer_converged"));
    assert(dataset_exists(file, "/adaptive/outer_iterations"));
    assert(dataset_exists(file, "/adaptive/L1_history_cm"));
    assert(dataset_exists(file, "/adaptive/LM_history_cm"));
    assert(dataset_exists(file, "/adaptive/F_A_end_history"));
    assert(dataset_exists(file, "/adaptive/F_M_end_history"));
    assert(dataset_exists(file, "/adaptive/ion_divergence_nuclei_cm3_s"));
    assert(dataset_exists(file, "/adaptive/pump_exhaust_nuclei_cm3_s"));
    assert(dataset_exists(file, "/adaptive/neutral_remainder_nuclei_cm3_s"));
    assert(dataset_exists(file, "/adaptive/atomic_comp_nuclei_cm3_s"));
    assert(dataset_exists(file, "/adaptive/molecular_comp_nuclei_cm3_s"));
    assert(dataset_exists(file, "/variable_nuclei/enabled"));
    assert(dataset_exists(file, "/variable_nuclei/background_nuclei_cm3"));
    assert(dataset_exists(file, "/variable_nuclei/flowA_nuclei_cm3"));
    assert(dataset_exists(file, "/variable_nuclei/flowM_nuclei_cm3"));
    assert(dataset_exists(file, "/variable_nuclei/total_nuclei_cm3"));
    assert(dataset_exists(file, "/variable_nuclei/ion_divergence_nuclei_cm3_s"));
    assert(dataset_exists(file, "/variable_nuclei/flowA_divergence_nuclei_cm3_s"));
    assert(dataset_exists(file, "/variable_nuclei/flowM_divergence_nuclei_cm3_s"));
    assert(dataset_exists(file, "/variable_nuclei/neutral_exhaust_nuclei_cm3_s"));
    assert(dataset_exists(file, "/variable_nuclei/balance_rhs_cm3_s"));
    assert(dataset_exists(file, "/variable_nuclei/balance_residual_cm3_s"));
    assert(dataset_exists(file, "/rates/Te_eV"));
    assert(dataset_exists(file, "/rates/Ti_eV"));
    assert(dataset_exists(file, "/rates/ne_cm3"));
    assert(dataset_exists(file, "/rates/atomic_scd_cm3_s"));
    assert(dataset_exists(file, "/rates/atomic_acd_cm3_s"));
    assert(dataset_exists(file, "/rates/molecular_dissociation_rate_cm3_s"));
    assert(dataset_exists(file, "/rates/full_ion_nuclei_source_cm3_s"));
    assert(dataset_exists(file, "/rates/full_ion_nuclei_sink_cm3_s"));
    assert(dataset_exists(file, "/rates/H2_plus_mcx_production_cm3_s"));
    assert(dataset_exists(file, "/rates/H2_plus_mi_production_cm3_s"));
    assert(dataset_exists(file, "/rates/H2_plus_dr_h_source_frequency_s"));
    assert(dataset_exists(file, "/rates/H2_plus_target_velocity_cm_s"));
    assert(dataset_exists(file, "/rates/H2_plus_generator_s"));
    assert(dataset_exists(file, "/rates/mar_simple_branching_dr_h_source_cm3_s"));
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
    const auto d_adaptive_enabled = dataset_dims(file.openDataSet("/adaptive/enabled"));
    const auto d_adaptive_l1_found = dataset_dims(file.openDataSet("/adaptive/L1_found"));
    const auto d_adaptive_l1_index = dataset_dims(file.openDataSet("/adaptive/L1_index"));
    const auto d_adaptive_l1_cm = dataset_dims(file.openDataSet("/adaptive/L1_cm"));
    const auto d_adaptive_fraction = dataset_dims(file.openDataSet("/adaptive/atomic_flow_fraction"));
    const auto d_adaptive_lm_found = dataset_dims(file.openDataSet("/adaptive/LM_found"));
    const auto d_adaptive_lm_index = dataset_dims(file.openDataSet("/adaptive/LM_index"));
    const auto d_adaptive_lm_cm = dataset_dims(file.openDataSet("/adaptive/LM_cm"));
    const auto d_adaptive_m_fraction = dataset_dims(file.openDataSet("/adaptive/molecular_flow_fraction"));
    const auto d_profile_available = dataset_dims(file.openDataSet("/adaptive/transport_profile_available"));
    const auto d_outer_enabled = dataset_dims(file.openDataSet("/adaptive/outer_loop_enabled"));
    const auto d_outer_converged = dataset_dims(file.openDataSet("/adaptive/outer_converged"));
    const auto d_outer_iterations = dataset_dims(file.openDataSet("/adaptive/outer_iterations"));
    const auto d_l1_history = dataset_dims(file.openDataSet("/adaptive/L1_history_cm"));
    const auto d_lm_history = dataset_dims(file.openDataSet("/adaptive/LM_history_cm"));
    const auto d_fa_end_history = dataset_dims(file.openDataSet("/adaptive/F_A_end_history"));
    const auto d_fm_end_history = dataset_dims(file.openDataSet("/adaptive/F_M_end_history"));
    const auto d_ion_div = dataset_dims(file.openDataSet("/adaptive/ion_divergence_nuclei_cm3_s"));
    const auto d_pump = dataset_dims(file.openDataSet("/adaptive/pump_exhaust_nuclei_cm3_s"));
    const auto d_rem = dataset_dims(file.openDataSet("/adaptive/neutral_remainder_nuclei_cm3_s"));
    const auto d_atom_comp = dataset_dims(file.openDataSet("/adaptive/atomic_comp_nuclei_cm3_s"));
    const auto d_mol_comp = dataset_dims(file.openDataSet("/adaptive/molecular_comp_nuclei_cm3_s"));
    const auto d_variable_enabled = dataset_dims(file.openDataSet("/variable_nuclei/enabled"));
    const auto d_variable_total = dataset_dims(file.openDataSet("/variable_nuclei/total_nuclei_cm3"));
    const auto d_variable_balance = dataset_dims(file.openDataSet("/variable_nuclei/balance_residual_cm3_s"));
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
    assert(d_adaptive_enabled.size() == 1 && d_adaptive_enabled[0] == 1);
    assert(d_adaptive_l1_found.size() == 1 && d_adaptive_l1_found[0] == 1);
    assert(d_adaptive_l1_index.size() == 1 && d_adaptive_l1_index[0] == 1);
    assert(d_adaptive_l1_cm.size() == 1 && d_adaptive_l1_cm[0] == 1);
    assert(d_adaptive_fraction.size() == 1 && d_adaptive_fraction[0] == n_nodes);
    assert(d_adaptive_lm_found.size() == 1 && d_adaptive_lm_found[0] == 1);
    assert(d_adaptive_lm_index.size() == 1 && d_adaptive_lm_index[0] == 1);
    assert(d_adaptive_lm_cm.size() == 1 && d_adaptive_lm_cm[0] == 1);
    assert(d_adaptive_m_fraction.size() == 1 && d_adaptive_m_fraction[0] == n_nodes);
    assert(d_profile_available.size() == 1 && d_profile_available[0] == 1);
    assert(d_outer_enabled.size() == 1 && d_outer_enabled[0] == 1);
    assert(d_outer_converged.size() == 1 && d_outer_converged[0] == 1);
    assert(d_outer_iterations.size() == 1 && d_outer_iterations[0] == 1);
    assert(d_l1_history.size() == 1 && d_l1_history[0] == 2);
    assert(d_lm_history.size() == 1 && d_lm_history[0] == 2);
    assert(d_fa_end_history.size() == 1 && d_fa_end_history[0] == 2);
    assert(d_fm_end_history.size() == 1 && d_fm_end_history[0] == 2);
    assert(d_ion_div.size() == 1 && d_ion_div[0] == n_nodes);
    assert(d_pump.size() == 1 && d_pump[0] == n_nodes);
    assert(d_rem.size() == 1 && d_rem[0] == n_nodes);
    assert(d_atom_comp.size() == 1 && d_atom_comp[0] == n_nodes);
    assert(d_mol_comp.size() == 1 && d_mol_comp[0] == n_nodes);
    assert(d_variable_enabled.size() == 1 && d_variable_enabled[0] == 1);
    assert(d_variable_total.size() == 1 && d_variable_total[0] == n_nodes);
    assert(d_variable_balance.size() == 1 && d_variable_balance[0] == n_nodes);
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

    const int adaptive_enabled = read_scalar_int(file, "/adaptive/enabled");
    const int adaptive_l1_found = read_scalar_int(file, "/adaptive/L1_found");
    const int adaptive_l1_index = read_scalar_int(file, "/adaptive/L1_index");
    const double adaptive_l1_cm = read_scalar_double(file, "/adaptive/L1_cm");
    const int adaptive_lm_found = read_scalar_int(file, "/adaptive/LM_found");
    const int adaptive_lm_index = read_scalar_int(file, "/adaptive/LM_index");
    const double adaptive_lm_cm = read_scalar_double(file, "/adaptive/LM_cm");
    const int profile_available = read_scalar_int(file, "/adaptive/transport_profile_available");
    const int outer_enabled = read_scalar_int(file, "/adaptive/outer_loop_enabled");
    const int outer_converged = read_scalar_int(file, "/adaptive/outer_converged");
    const int outer_iterations = read_scalar_int(file, "/adaptive/outer_iterations");
    const auto adaptive_fraction = read_vector_double(file, "/adaptive/atomic_flow_fraction");
    const auto adaptive_m_fraction = read_vector_double(file, "/adaptive/molecular_flow_fraction");
    const auto l1_history = read_vector_double(file, "/adaptive/L1_history_cm");
    const auto lm_history = read_vector_double(file, "/adaptive/LM_history_cm");
    const auto fa_end_history = read_vector_double(file, "/adaptive/F_A_end_history");
    const auto fm_end_history = read_vector_double(file, "/adaptive/F_M_end_history");
    const auto ion_div = read_vector_double(file, "/adaptive/ion_divergence_nuclei_cm3_s");
    const auto pump = read_vector_double(file, "/adaptive/pump_exhaust_nuclei_cm3_s");
    const auto rem = read_vector_double(file, "/adaptive/neutral_remainder_nuclei_cm3_s");
    const auto atom_comp = read_vector_double(file, "/adaptive/atomic_comp_nuclei_cm3_s");
    const auto mol_comp = read_vector_double(file, "/adaptive/molecular_comp_nuclei_cm3_s");
    const int variable_enabled = read_scalar_int(file, "/variable_nuclei/enabled");
    const auto variable_bg = read_vector_double(file, "/variable_nuclei/background_nuclei_cm3");
    const auto variable_flowA = read_vector_double(file, "/variable_nuclei/flowA_nuclei_cm3");
    const auto variable_flowM = read_vector_double(file, "/variable_nuclei/flowM_nuclei_cm3");
    const auto variable_total = read_vector_double(file, "/variable_nuclei/total_nuclei_cm3");
    const auto variable_ion = read_vector_double(file, "/variable_nuclei/ion_divergence_nuclei_cm3_s");
    const auto variable_flowA_div = read_vector_double(file, "/variable_nuclei/flowA_divergence_nuclei_cm3_s");
    const auto variable_flowM_div = read_vector_double(file, "/variable_nuclei/flowM_divergence_nuclei_cm3_s");
    const auto variable_exhaust = read_vector_double(file, "/variable_nuclei/neutral_exhaust_nuclei_cm3_s");
    const auto variable_rhs = read_vector_double(file, "/variable_nuclei/balance_rhs_cm3_s");
    const auto variable_residual = read_vector_double(file, "/variable_nuclei/balance_residual_cm3_s");
    std::cout << "[HDF5][adaptive] enabled=" << adaptive_enabled
              << " L1_found=" << adaptive_l1_found
              << " L1_index=" << adaptive_l1_index
              << " L1_cm=" << adaptive_l1_cm
              << " LM_found=" << adaptive_lm_found
              << " LM_index=" << adaptive_lm_index
              << " LM_cm=" << adaptive_lm_cm
              << " F_A[0]=" << adaptive_fraction.front()
              << " F_A[end]=" << adaptive_fraction.back()
              << " F_M[end]=" << adaptive_m_fraction.back()
              << " profile_available=" << profile_available
              << " outer_enabled=" << outer_enabled
              << " outer_converged=" << outer_converged
              << " outer_iterations=" << outer_iterations
              << " L1_outer_first=" << l1_history.front()
              << " L1_outer_last=" << l1_history.back()
              << " LM_outer_first=" << lm_history.front()
              << " LM_outer_last=" << lm_history.back()
              << "\n";
    assert(adaptive_enabled == 1);
    assert(profile_available == 1);
    assert(outer_enabled == 1);
    assert(outer_iterations == history.adaptive_outer_iterations);
    assert(outer_iterations == 2);
    assert(outer_converged == (history.adaptive_outer_loop_converged ? 1 : 0));
    assert(adaptive_l1_found == (history.adaptive_recycling_L1_found ? 1 : 0));
    assert(adaptive_l1_index == history.adaptive_recycling_L1_index);
    assert(std::abs(adaptive_l1_cm - history.adaptive_recycling_L1_cm) < 1e-14);
    assert(adaptive_lm_found == (history.adaptive_recycling_LM_found ? 1 : 0));
    assert(adaptive_lm_found == 1);
    assert(adaptive_lm_index == history.adaptive_recycling_LM_index);
    assert(std::abs(adaptive_lm_cm - history.adaptive_recycling_LM_cm) < 1e-14);
    assert(adaptive_fraction.size() == history.adaptive_atomic_flow_fraction.size());
    assert(adaptive_m_fraction.size() == history.adaptive_molecular_flow_fraction.size());
    assert(l1_history.size() == history.adaptive_outer_L1_history_cm.size());
    assert(lm_history.size() == history.adaptive_outer_LM_history_cm.size());
    assert(fa_end_history.size() == history.adaptive_outer_F_A_end_history.size());
    assert(fm_end_history.size() == history.adaptive_outer_F_M_end_history.size());
    for (size_t k = 0; k < l1_history.size(); ++k) {
        assert(std::abs(l1_history[k] - history.adaptive_outer_L1_history_cm[k]) < 1e-14);
        assert(std::abs(lm_history[k] - history.adaptive_outer_LM_history_cm[k]) < 1e-14);
        assert(std::abs(fa_end_history[k] - history.adaptive_outer_F_A_end_history[k]) < 1e-14);
        assert(std::abs(fm_end_history[k] - history.adaptive_outer_F_M_end_history[k]) < 1e-14);
    }
    for (size_t k = 0; k < adaptive_fraction.size(); ++k) {
        assert(std::abs(adaptive_fraction[k] - history.adaptive_atomic_flow_fraction[k]) < 1e-14);
        assert(std::abs(adaptive_m_fraction[k] - history.adaptive_molecular_flow_fraction[k]) < 1e-14);
        assert(std::abs(ion_div[k] - history.adaptive_ion_divergence_nuclei_cm3_s[k])
               <= 1e-12 * std::max(1.0, std::abs(ion_div[k])));
        assert(std::abs(pump[k] - history.adaptive_pump_exhaust_nuclei_cm3_s[k])
               <= 1e-12 * std::max(1.0, std::abs(pump[k])));
        assert(std::abs(rem[k] - history.adaptive_neutral_remainder_nuclei_cm3_s[k])
               <= 1e-12 * std::max(1.0, std::abs(rem[k])));
        assert(std::abs(atom_comp[k] - history.adaptive_atomic_comp_nuclei_cm3_s[k])
               <= 1e-12 * std::max(1.0, std::abs(atom_comp[k])));
        assert(std::abs(mol_comp[k] - history.adaptive_molecular_comp_nuclei_cm3_s[k])
               <= 1e-12 * std::max(1.0, std::abs(mol_comp[k])));
    }
    std::cout << "[HDF5][variable-nuclei] enabled=" << variable_enabled
              << " total[0]=" << variable_total.front()
              << " total[end]=" << variable_total.back()
              << " residual[end]=" << variable_residual.back() << "\n";
    assert(variable_enabled == 1);
    assert(history.variable_nuclei_balance_enabled == true);
    assert(variable_bg.size() == static_cast<size_t>(n_nodes));
    assert(variable_flowA.size() == static_cast<size_t>(n_nodes));
    assert(variable_flowM.size() == static_cast<size_t>(n_nodes));
    assert(variable_total.size() == static_cast<size_t>(n_nodes));
    assert(variable_ion.size() == history.variable_ion_divergence_nuclei_cm3_s.size());
    for (size_t k = 0; k < variable_total.size(); ++k) {
        const double component_total = variable_bg[k] + variable_flowA[k] + variable_flowM[k];
        assert(std::abs(variable_total[k] - component_total)
               <= 1e-12 * std::max(1.0, std::abs(variable_total[k])));
        if (k == 0) {
            assert(std::isnan(variable_residual[k]));
            continue;
        }
        assert(std::isfinite(variable_ion[k]));
        assert(std::isfinite(variable_flowA_div[k]));
        assert(std::isfinite(variable_flowM_div[k]));
        assert(std::isfinite(variable_exhaust[k]));
        assert(std::isfinite(variable_rhs[k]));
        assert(std::isfinite(variable_residual[k]));
        assert(std::abs(variable_ion[k] - history.variable_ion_divergence_nuclei_cm3_s[k])
               <= 1e-12 * std::max(1.0, std::abs(variable_ion[k])));
        assert(std::abs(variable_flowA_div[k] - history.variable_flowA_divergence_nuclei_cm3_s[k])
               <= 1e-12 * std::max(1.0, std::abs(variable_flowA_div[k])));
        assert(std::abs(variable_flowM_div[k] - history.variable_flowM_divergence_nuclei_cm3_s[k])
               <= 1e-12 * std::max(1.0, std::abs(variable_flowM_div[k])));
        assert(std::abs(variable_exhaust[k] - history.variable_neutral_exhaust_nuclei_cm3_s[k])
               <= 1e-12 * std::max(1.0, std::abs(variable_exhaust[k])));
        assert(std::abs(variable_rhs[k] - history.variable_balance_rhs_cm3_s[k])
               <= 1e-12 * std::max(1.0, std::abs(variable_rhs[k])));
        assert(std::abs(variable_residual[k] - history.variable_balance_residual_cm3_s[k])
               <= 1e-12 * std::max(1.0, std::abs(variable_residual[k])));
    }
    std::cout << "[HDF5][adaptive-profile] k0 ion=" << ion_div.front()
              << " pump=" << pump.front()
              << " rem=" << rem.front()
              << " atom=" << atom_comp.front()
              << " mol=" << mol_comp.front() << "\n";
    std::cout << "[HDF5][adaptive-profile] kend ion=" << ion_div.back()
              << " pump=" << pump.back()
              << " rem=" << rem.back()
              << " atom=" << atom_comp.back()
              << " mol=" << mol_comp.back() << "\n";

    std::cout << "[PASS] HDF5Output checks.\n";
    return 0;
}
