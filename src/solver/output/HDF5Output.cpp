#include "HDF5Output.hpp"
#include "../marching/CellSolve.hpp"

#include <H5Cpp.h>

#include <algorithm>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace dcr::solver {

namespace {

void write_scalar_double(H5::H5File& file, const std::string& name, double value) {
    const hsize_t dims[1] = {1};
    H5::DataSpace space(1, dims);
    H5::DataSet ds = file.createDataSet(name, H5::PredType::NATIVE_DOUBLE, space);
    ds.write(&value, H5::PredType::NATIVE_DOUBLE);
}

void write_scalar_int(H5::H5File& file, const std::string& name, int value) {
    const hsize_t dims[1] = {1};
    H5::DataSpace space(1, dims);
    H5::DataSet ds = file.createDataSet(name, H5::PredType::NATIVE_INT, space);
    ds.write(&value, H5::PredType::NATIVE_INT);
}

// Write a 1D double dataset.
void write_vector_double(H5::H5File& file, const std::string& name, const std::vector<double>& data) {
    const hsize_t dims[1] = {static_cast<hsize_t>(data.size())};
    H5::DataSpace space(1, dims);
    H5::DataSet ds = file.createDataSet(name, H5::PredType::NATIVE_DOUBLE, space);
    if (!data.empty()) ds.write(data.data(), H5::PredType::NATIVE_DOUBLE);
}

// Write a 1D integer dataset.
void write_vector_int(H5::H5File& file, const std::string& name, const std::vector<int>& data) {
    const hsize_t dims[1] = {static_cast<hsize_t>(data.size())};
    H5::DataSpace space(1, dims);
    H5::DataSet ds = file.createDataSet(name, H5::PredType::NATIVE_INT, space);
    if (!data.empty()) ds.write(data.data(), H5::PredType::NATIVE_INT);
}

// Write variable-length UTF-8-like C-strings (HDF5 variable string type).
void write_string_vector(H5::H5File& file, const std::string& name, const std::vector<std::string>& data) {
    const hsize_t dims[1] = {static_cast<hsize_t>(data.size())};
    H5::DataSpace space(1, dims);
    H5::StrType str_t(H5::PredType::C_S1, H5T_VARIABLE);
    H5::DataSet ds = file.createDataSet(name, str_t, space);
    if (data.empty()) return;
    std::vector<const char*> ptrs;
    ptrs.reserve(data.size());
    for (const auto& s : data) ptrs.push_back(s.c_str());
    ds.write(ptrs.data(), str_t);
}

std::vector<std::string> build_level_labels(const std::vector<int>& indices,
                                            const std::vector<dcr::atomic::EnergyLevel>& levels) {
    std::vector<std::string> labels;
    labels.reserve(indices.size());
    for (int gi : indices) {
        if (gi >= 0 && gi < static_cast<int>(levels.size())) labels.push_back(levels[static_cast<size_t>(gi)].label);
        else labels.emplace_back("unknown");
    }
    return labels;
}

// Write a row-major dense matrix represented as vector<state_vector>.
void write_matrix(H5::H5File& file,
                  const std::string& name,
                  const std::vector<dcr::base::Vector>& rows,
                  int cols) {
    if (rows.empty() || cols <= 0) return;
    const hsize_t dims[2] = {
        static_cast<hsize_t>(rows.size()),
        static_cast<hsize_t>(cols)
    };
    H5::DataSpace space(2, dims);
    H5::DataSet ds = file.createDataSet(name, H5::PredType::NATIVE_DOUBLE, space);

    std::vector<double> flat(static_cast<size_t>(rows.size()) * static_cast<size_t>(cols), 0.0);
    for (size_t i = 0; i < rows.size(); ++i) {
        const auto& r = rows[i];
        const int cmax = std::min<int>(cols, r.size());
        for (int j = 0; j < cmax; ++j) {
            flat[i * static_cast<size_t>(cols) + static_cast<size_t>(j)] = r(j);
        }
    }
    ds.write(flat.data(), H5::PredType::NATIVE_DOUBLE);
}

void write_matrix_stack(H5::H5File& file,
                        const std::string& name,
                        const std::vector<dcr::base::Matrix>& matrices,
                        int rows,
                        int cols) {
    if (matrices.empty() || rows <= 0 || cols <= 0) return;
    const hsize_t dims[3] = {
        static_cast<hsize_t>(matrices.size()),
        static_cast<hsize_t>(rows),
        static_cast<hsize_t>(cols)
    };
    H5::DataSpace space(3, dims);
    H5::DataSet ds = file.createDataSet(name, H5::PredType::NATIVE_DOUBLE, space);
    std::vector<double> flat(
        matrices.size() * static_cast<size_t>(rows) * static_cast<size_t>(cols), 0.0);
    for (size_t k = 0; k < matrices.size(); ++k) {
        const auto& matrix = matrices[k];
        const int rmax = std::min<int>(rows, matrix.rows());
        const int cmax = std::min<int>(cols, matrix.cols());
        for (int i = 0; i < rmax; ++i) {
            for (int j = 0; j < cmax; ++j) {
                const size_t offset =
                    (k * static_cast<size_t>(rows) + static_cast<size_t>(i)) *
                    static_cast<size_t>(cols) + static_cast<size_t>(j);
                flat[offset] = matrix(i, j);
            }
        }
    }
    ds.write(flat.data(), H5::PredType::NATIVE_DOUBLE);
}

// Build n_total = n_background + n_flow(A) + n_flow(M) in full-state indexing.
std::vector<dcr::base::Vector> build_total_population(
    const dcr::atomic::AtomicData& atomic_data,
    const BoundaryPhaseResult& boundary,
    const MarchingHistory& history) {

    const int total_states = atomic_data.get_total_states();
    const size_t n_nodes = history.background_full.size();
    std::vector<dcr::base::Vector> total(n_nodes, dcr::base::Vector::Zero(total_states));

    for (size_t k = 0; k < n_nodes; ++k) {
        if (k < history.background_full.size()) {
            const auto& bg = history.background_full[k];
            const int cmax = std::min<int>(total_states, bg.size());
            for (int i = 0; i < cmax; ++i) total[k](i) = bg(i);
        }
        if (k < history.flowA.size()) {
            for (size_t j = 0; j < boundary.A_indices.size(); ++j) {
                const int gi = boundary.A_indices[j];
                if (gi < 0 || gi >= total_states) continue;
                if (static_cast<int>(j) >= history.flowA[k].size()) continue;
                total[k](gi) += history.flowA[k](static_cast<int>(j));
            }
        }
        if (k < history.flowM.size()) {
            for (size_t j = 0; j < boundary.M_indices.size(); ++j) {
                const int gi = boundary.M_indices[j];
                if (gi < 0 || gi >= total_states) continue;
                if (static_cast<int>(j) >= history.flowM[k].size()) continue;
                total[k](gi) += history.flowM[k](static_cast<int>(j));
            }
        }
    }
    return total;
}

// Build total nuclei profile from total population vectors.
std::vector<double> build_total_nuclei_profile(
    const std::vector<dcr::base::Vector>& total_population,
    const std::vector<dcr::atomic::EnergyLevel>& levels) {

    std::vector<double> out(total_population.size(), 0.0);
    for (size_t k = 0; k < total_population.size(); ++k) {
        const auto& n = total_population[k];
        const int nmax = std::min<int>(n.size(), static_cast<int>(levels.size()));
        double s = 0.0;
        for (int i = 0; i < nmax; ++i) {
            s += levels[static_cast<size_t>(i)].atomicity * std::max(n(i), 0.0);
        }
        out[k] = s;
    }
    return out;
}

std::vector<double> build_background_nuclei_profile(
    const std::vector<dcr::base::Vector>& background_population,
    const std::vector<dcr::atomic::EnergyLevel>& levels) {

    return build_total_nuclei_profile(background_population, levels);
}

std::vector<double> build_compact_nuclei_profile(
    const std::vector<dcr::base::Vector>& compact_population,
    const std::vector<int>& global_indices,
    const std::vector<dcr::atomic::EnergyLevel>& levels) {

    std::vector<double> out(compact_population.size(), 0.0);
    for (size_t k = 0; k < compact_population.size(); ++k) {
        const auto& n = compact_population[k];
        const int nmax = std::min<int>(n.size(), static_cast<int>(global_indices.size()));
        double s = 0.0;
        for (int i = 0; i < nmax; ++i) {
            const int gi = global_indices[static_cast<size_t>(i)];
            if (gi < 0 || gi >= static_cast<int>(levels.size())) continue;
            s += levels[static_cast<size_t>(gi)].atomicity * std::max(n(i), 0.0);
        }
        out[k] = s;
    }
    return out;
}

std::vector<double> build_molecular_flow_lhs_transport_rate(
    const BoundaryPhaseResult& boundary,
    const MarchingHistory& history) {

    std::vector<double> out(history.x_cm.size(), 0.0);
    if (history.x_cm.size() < 2 || history.flowM.size() < 2 || boundary.u_M <= 0.0) return out;

    auto positive_sum = [](const dcr::base::Vector& v) {
        double total = 0.0;
        for (int i = 0; i < v.size(); ++i) total += std::max(v(i), 0.0);
        return total;
    };

    for (size_t k = 0; k < out.size(); ++k) {
        size_t left = 0;
        size_t right = 0;
        if (k == 0) {
            left = 0;
            right = 1;
        } else {
            left = k - 1;
            right = k;
        }
        if (right >= history.x_cm.size() || right >= history.flowM.size() || left >= history.flowM.size()) continue;
        const double dx = history.x_cm[right] - history.x_cm[left];
        if (!(dx > 0.0)) continue;
        const double lhs = boundary.u_M * (positive_sum(history.flowM[right]) - positive_sum(history.flowM[left])) / dx;
        out[k] = std::max(0.0, -lhs);
    }
    return out;
}

std::vector<double> collect_rate_scalar(
    const std::vector<RateDiagnosticSnapshot>& snapshots,
    double RateDiagnosticSnapshot::* member) {
    std::vector<double> out;
    out.reserve(snapshots.size());
    for (const auto& snap : snapshots) out.push_back(snap.*member);
    return out;
}

std::vector<double> collect_atomic_effective_scalar(
    const std::vector<RateDiagnosticSnapshot>& snapshots,
    double AtomicEffectiveRates::* member) {
    std::vector<double> out;
    out.reserve(snapshots.size());
    for (const auto& snap : snapshots) out.push_back(snap.atomic_effective.*member);
    return out;
}

std::vector<double> collect_atomic_qss_scalar(
    const std::vector<RateDiagnosticSnapshot>& snapshots,
    double AtomicQSSDiagnostics::* member) {
    std::vector<double> out;
    out.reserve(snapshots.size());
    for (const auto& snap : snapshots) out.push_back(snap.atomic_qss.*member);
    return out;
}

std::vector<double> collect_atomic_source_scalar(
    const std::vector<RateDiagnosticSnapshot>& snapshots,
    double AtomicSourceDiagnostics::* member) {
    std::vector<double> out;
    out.reserve(snapshots.size());
    for (const auto& snap : snapshots) out.push_back(snap.atomic_sources.*member);
    return out;
}

std::vector<dcr::base::Vector> collect_molecular_ion_vector(
    const std::vector<RateDiagnosticSnapshot>& snapshots,
    dcr::base::Vector MolecularIonTransportDiagnostics::* member) {
    std::vector<dcr::base::Vector> out;
    out.reserve(snapshots.size());
    for (const auto& snap : snapshots) out.push_back(snap.molecular_ion_transport.*member);
    return out;
}

std::vector<dcr::base::Matrix> collect_molecular_ion_matrix(
    const std::vector<RateDiagnosticSnapshot>& snapshots,
    dcr::base::Matrix MolecularIonTransportDiagnostics::* member) {
    std::vector<dcr::base::Matrix> out;
    out.reserve(snapshots.size());
    for (const auto& snap : snapshots) out.push_back(snap.molecular_ion_transport.*member);
    return out;
}

std::vector<double> collect_molecular_ion_scalar(
    const std::vector<RateDiagnosticSnapshot>& snapshots,
    double MolecularIonTransportDiagnostics::* member) {
    std::vector<double> out;
    out.reserve(snapshots.size());
    for (const auto& snap : snapshots) out.push_back(snap.molecular_ion_transport.*member);
    return out;
}

std::vector<double> collect_power_loss_scalar(
    const std::vector<RateDiagnosticSnapshot>& snapshots,
    double PowerLossDiagnostics::* member) {
    std::vector<double> out;
    out.reserve(snapshots.size());
    for (const auto& snap : snapshots) out.push_back(snap.power_loss.*member);
    return out;
}

std::vector<double> collect_grouped_source_scalar(
    const std::vector<RateDiagnosticSnapshot>& snapshots,
    double GroupedSourceDiagnostics::* member) {
    std::vector<double> out;
    out.reserve(snapshots.size());
    for (const auto& snap : snapshots) out.push_back(snap.grouped_sources.*member);
    return out;
}

std::vector<double> collect_flow_to_local_source_scalar(
    const std::vector<RateDiagnosticSnapshot>& snapshots,
    double FlowToLocalSourceDiagnostics::* member) {
    std::vector<double> out;
    out.reserve(snapshots.size());
    for (const auto& snap : snapshots) out.push_back(snap.flow_to_local_sources.*member);
    return out;
}

std::vector<int> collect_atomic_qss_int(
    const std::vector<RateDiagnosticSnapshot>& snapshots,
    int AtomicQSSDiagnostics::* member) {
    std::vector<int> out;
    out.reserve(snapshots.size());
    for (const auto& snap : snapshots) out.push_back(snap.atomic_qss.*member);
    return out;
}

std::vector<dcr::base::Vector> collect_atomic_qss_vector(
    const std::vector<RateDiagnosticSnapshot>& snapshots,
    dcr::base::Vector AtomicQSSDiagnostics::* member) {
    std::vector<dcr::base::Vector> out;
    out.reserve(snapshots.size());
    for (const auto& snap : snapshots) out.push_back(snap.atomic_qss.*member);
    return out;
}

} // namespace

void write_hdf5_output(
    const dcr::io::Config& config,
    const dcr::atomic::AtomicData& atomic_data,
    const BoundaryPhaseResult& boundary,
    const MarchingHistory& history) {

    // We expect at least one node (boundary node x=0).
    if (history.x_cm.empty()) {
        throw std::runtime_error("Cannot write HDF5 output: marching history is empty.");
    }

    std::filesystem::path out_dir = config.io.output_dir.empty()
        ? std::filesystem::path("output")
        : std::filesystem::path(config.io.output_dir);
    std::filesystem::create_directories(out_dir);
    const std::filesystem::path out_file = out_dir / "dcr_results.h5";

    // Truncate output each run so the file is always self-consistent.
    H5::H5File file(out_file.string(), H5F_ACC_TRUNC);
    file.createGroup("/grid");
    file.createGroup("/states");
    file.createGroup("/population");
    file.createGroup("/rates");
    file.createGroup("/power_loss");
    file.createGroup("/timing");
    file.createGroup("/metadata");
    write_string_vector(file, "/metadata/spatial_method", {config.solver.spatial_method});

    if (config.solver.spatial_method == "experimental_global_bvp") {
        file.createGroup("/global_bvp");
        write_string_vector(file, "/global_bvp/velocity_profile_type",
            {"fixed_scale_linear_bohm"});
        write_scalar_double(file, "/global_bvp/ion_velocity_transition_length_cm",
            config.global_bvp.ion_velocity_transition_length_cm);
        write_scalar_double(file, "/global_bvp/ion_upstream_speed_fraction",
            config.global_bvp.ion_upstream_speed_fraction);
        write_scalar_double(file, "/global_bvp/final_domain_length_cm",
            config.global_bvp.final_domain_length_cm);
        write_scalar_double(file, "/global_bvp/target_electron_temperature_eV",
            config.plasma.Te_eV);
        write_scalar_double(file, "/global_bvp/target_ion_temperature_eV",
            config.plasma.Ti_eV);
        write_scalar_double(file, "/global_bvp/target_nuclei_density_cm3",
            config.plasma.total_density);
        write_scalar_double(file, "/global_bvp/boundary_exhaust_width_cm",
            config.grid.boundary_poloidal_width_cm);
        write_scalar_double(file, "/global_bvp/spatial_exhaust_width_cm",
            config.grid.spatial_exhaust_width_cm);
    }

    // Grid
    write_vector_double(file, "/grid/x_cm", history.x_cm);

    // State metadata
    const auto& levels = atomic_data.get_levels();
    std::vector<std::string> labels;
    std::vector<int> type_id;
    std::vector<int> charge;
    std::vector<int> atomicity;
    std::vector<int> internal_id;
    labels.reserve(levels.size());
    type_id.reserve(levels.size());
    charge.reserve(levels.size());
    atomicity.reserve(levels.size());
    internal_id.reserve(levels.size());
    for (const auto& lvl : levels) {
        labels.push_back(lvl.label);
        int t = 0;
        if (lvl.type == dcr::atomic::SpeciesType::Atom) t = 0;
        else if (lvl.type == dcr::atomic::SpeciesType::Molecule) t = 1;
        else if (lvl.type == dcr::atomic::SpeciesType::Ion) t = 2;
        type_id.push_back(t);
        charge.push_back(lvl.charge);
        atomicity.push_back(lvl.atomicity);
        internal_id.push_back(lvl.internal_id);
    }
    write_string_vector(file, "/states/labels", labels);
    write_vector_int(file, "/states/type_id", type_id);
    write_vector_int(file, "/states/charge", charge);
    write_vector_int(file, "/states/atomicity", atomicity);
    write_vector_int(file, "/states/internal_id", internal_id);
    write_vector_int(file, "/states/P_indices", boundary.P_indices);
    write_vector_int(file, "/states/A_indices", boundary.A_indices);
    write_vector_int(file, "/states/M_indices", boundary.M_indices);
    if (!history.rate_diagnostics.empty()) {
        const auto& h2plus_indices =
            history.rate_diagnostics.front().molecular_ion_transport.state_indices;
        if (!h2plus_indices.empty()) {
            write_vector_int(file, "/states/H2_plus_indices", h2plus_indices);
            write_string_vector(file, "/states/H2_plus_labels",
                                build_level_labels(h2plus_indices, levels));
        }
    }
    if (!history.qss_local_transient_atomic_indices.empty()) {
        write_vector_int(file, "/states/qss_local_transient_atomic_indices",
                         history.qss_local_transient_atomic_indices);
        write_string_vector(file, "/states/qss_local_transient_atomic_labels",
                            build_level_labels(history.qss_local_transient_atomic_indices, levels));
    }
    if (!history.qss_flow_transient_atomic_indices.empty()) {
        write_vector_int(file, "/states/qss_flow_transient_atomic_indices",
                         history.qss_flow_transient_atomic_indices);
        write_string_vector(file, "/states/qss_flow_transient_atomic_labels",
                            build_level_labels(history.qss_flow_transient_atomic_indices, levels));
    }

    // Populations along x nodes.
    write_matrix(file, "/population/background_full", history.background_full, atomic_data.get_total_states());
    write_matrix(file, "/population/flowA", history.flowA, static_cast<int>(boundary.A_indices.size()));
    write_matrix(file, "/population/flowM", history.flowM, static_cast<int>(boundary.M_indices.size()));
    write_matrix(file, "/population/qss_local_transient_atomic",
                 history.qss_local_transient_atomic,
                 static_cast<int>(history.qss_local_transient_atomic_indices.size()));
    write_matrix(file, "/population/qss_flow_transient_atomic",
                 history.qss_flow_transient_atomic,
                 static_cast<int>(history.qss_flow_transient_atomic_indices.size()));

    const auto total_pop = build_total_population(atomic_data, boundary, history);
    write_matrix(file, "/population/total_full", total_pop, atomic_data.get_total_states());
    const auto n_nuclei_cm3 = build_total_nuclei_profile(total_pop, levels);
    write_vector_double(file, "/grid/n_nuclei_cm3", n_nuclei_cm3);

    if (history.variable_nuclei_balance_enabled) {
        file.createGroup("/variable_nuclei");
        write_scalar_int(file, "/variable_nuclei/enabled", 1);
        write_vector_double(file, "/variable_nuclei/background_nuclei_cm3",
                            build_background_nuclei_profile(history.background_full, levels));
        write_vector_double(file, "/variable_nuclei/flowA_nuclei_cm3",
                            build_compact_nuclei_profile(history.flowA, boundary.A_indices, levels));
        write_vector_double(file, "/variable_nuclei/flowM_nuclei_cm3",
                            build_compact_nuclei_profile(history.flowM, boundary.M_indices, levels));
        write_vector_double(file, "/variable_nuclei/total_nuclei_cm3", n_nuclei_cm3);
        write_vector_double(file, "/variable_nuclei/ion_divergence_nuclei_cm3_s",
                            history.variable_ion_divergence_nuclei_cm3_s);
        write_vector_double(file, "/variable_nuclei/flowA_divergence_nuclei_cm3_s",
                            history.variable_flowA_divergence_nuclei_cm3_s);
        write_vector_double(file, "/variable_nuclei/flowM_divergence_nuclei_cm3_s",
                            history.variable_flowM_divergence_nuclei_cm3_s);
        write_vector_double(file, "/variable_nuclei/neutral_exhaust_nuclei_cm3_s",
                            history.variable_neutral_exhaust_nuclei_cm3_s);
        write_vector_double(file, "/variable_nuclei/balance_rhs_cm3_s",
                            history.variable_balance_rhs_cm3_s);
        write_vector_double(file, "/variable_nuclei/balance_residual_cm3_s",
                             history.variable_balance_residual_cm3_s);
        write_vector_double(file, "/variable_nuclei/prescribed_nuclei_density_cm3",
                            history.prescribed_nuclei_density_cm3);
        write_vector_double(file, "/variable_nuclei/recycling_source_nuclei_cm3_s",
                            history.recycling_source_nuclei_cm3_s);
        write_vector_double(file, "/variable_nuclei/local_atom_exhaust_nuclei_cm3_s",
                            history.local_atom_exhaust_nuclei_cm3_s);
        write_vector_double(file, "/variable_nuclei/local_molecule_exhaust_nuclei_cm3_s",
                            history.local_molecule_exhaust_nuclei_cm3_s);
        write_vector_double(file, "/variable_nuclei/L_I_nuclei_cm3_s",
                            history.ion_divergence_closure_nuclei_cm3_s);
        write_vector_double(file, "/variable_nuclei/ion_balance_coefficient_s",
                            history.ion_balance_coefficient_s);
    }
    if (history.individual_ion_flux_divergence_enabled) {
        file.createGroup("/individual_ion_flux");
        write_scalar_int(file, "/individual_ion_flux/enabled", 1);
        write_scalar_double(file, "/individual_ion_flux/modeling_tolerance",
                            kIndividualIonNucleiModelingTolerance);
        write_vector_double(file, "/individual_ion_flux/background_nuclei_cm3",
                            build_background_nuclei_profile(history.background_full, levels));
        write_vector_double(file, "/individual_ion_flux/flowA_nuclei_cm3",
                            build_compact_nuclei_profile(history.flowA, boundary.A_indices, levels));
        write_vector_double(file, "/individual_ion_flux/flowM_nuclei_cm3",
                            build_compact_nuclei_profile(history.flowM, boundary.M_indices, levels));
        write_vector_double(file, "/individual_ion_flux/total_nuclei_cm3", n_nuclei_cm3);
        write_vector_double(file, "/individual_ion_flux/ion_divergence_nuclei_cm3_s",
                            history.variable_ion_divergence_nuclei_cm3_s);
        write_vector_double(file, "/individual_ion_flux/flowA_divergence_nuclei_cm3_s",
                            history.variable_flowA_divergence_nuclei_cm3_s);
        write_vector_double(file, "/individual_ion_flux/flowM_divergence_nuclei_cm3_s",
                            history.variable_flowM_divergence_nuclei_cm3_s);
        write_vector_double(file, "/individual_ion_flux/neutral_exhaust_nuclei_cm3_s",
                            history.variable_neutral_exhaust_nuclei_cm3_s);
        write_vector_double(file, "/individual_ion_flux/conservation_residual_cm3_s",
                            history.variable_balance_residual_cm3_s);
        write_vector_double(file, "/individual_ion_flux/hminus_equation_residual_cm3_s",
                            history.individual_hminus_omitted_residual_cm3_s);
        write_vector_double(file,
                            "/individual_ion_flux/nuclei_weighted_species_residual_cm3_s",
                            history.individual_nuclei_weighted_species_residual_cm3_s);
        write_vector_double(file, "/individual_ion_flux/identity_relative_error",
                            history.individual_nuclei_identity_relative_error);
        write_vector_int(file, "/individual_ion_flux/identity_consistent",
                         history.individual_nuclei_identity_consistent);
    }

    write_scalar_double(file, "/timing/boundary_elapsed_seconds", history.boundary_elapsed_seconds);
    write_scalar_int(file, "/timing/boundary_iterations", history.boundary_iterations);
    write_vector_double(file, "/timing/marching_cell_elapsed_seconds", history.cell_elapsed_seconds);
    write_vector_int(file, "/timing/marching_cell_iterations", history.cell_iterations);
    write_vector_int(file, "/timing/marching_cell_converged", history.cell_converged);
    write_vector_double(file, "/timing/marching_cell_final_relative_change",
                        history.cell_final_relative_change);
    write_vector_double(file, "/timing/marching_cell_final_residual_relative",
                        history.cell_final_residual_relative);
    write_vector_double(file, "/timing/marching_cell_final_map_residual_norm",
                        history.cell_final_map_residual_norm);

    if (history.adaptive_recycling_diagnostics_enabled) {
        file.createGroup("/adaptive");
        write_scalar_int(file, "/adaptive/enabled", 1);
        write_scalar_int(file, "/adaptive/L1_found", history.adaptive_recycling_L1_found ? 1 : 0);
        write_scalar_int(file, "/adaptive/L1_index", history.adaptive_recycling_L1_index);
        write_scalar_double(file, "/adaptive/L1_cm", history.adaptive_recycling_L1_cm);
        write_vector_double(file, "/adaptive/atomic_flow_fraction", history.adaptive_atomic_flow_fraction);
        write_scalar_int(file, "/adaptive/LM_found", history.adaptive_recycling_LM_found ? 1 : 0);
        write_scalar_int(file, "/adaptive/LM_index", history.adaptive_recycling_LM_index);
        write_scalar_double(file, "/adaptive/LM_cm", history.adaptive_recycling_LM_cm);
        write_vector_double(file, "/adaptive/molecular_flow_fraction", history.adaptive_molecular_flow_fraction);
        write_scalar_int(file, "/adaptive/transport_profile_available",
                         history.adaptive_transport_profile_available ? 1 : 0);
        write_scalar_int(file, "/adaptive/outer_loop_enabled",
                         history.adaptive_outer_loop_enabled ? 1 : 0);
        write_scalar_int(file, "/adaptive/outer_converged",
                         history.adaptive_outer_loop_converged ? 1 : 0);
        write_scalar_int(file, "/adaptive/outer_iterations", history.adaptive_outer_iterations);
        write_vector_double(file, "/adaptive/L1_history_cm", history.adaptive_outer_L1_history_cm);
        write_vector_double(file, "/adaptive/LM_history_cm", history.adaptive_outer_LM_history_cm);
        write_vector_double(file, "/adaptive/F_A_end_history", history.adaptive_outer_F_A_end_history);
        write_vector_double(file, "/adaptive/F_M_end_history", history.adaptive_outer_F_M_end_history);
        if (history.adaptive_transport_profile_available) {
            write_vector_double(file, "/adaptive/ion_divergence_nuclei_cm3_s",
                                history.adaptive_ion_divergence_nuclei_cm3_s);
            write_vector_double(file, "/adaptive/pump_exhaust_nuclei_cm3_s",
                                history.adaptive_pump_exhaust_nuclei_cm3_s);
            write_vector_double(file, "/adaptive/neutral_remainder_nuclei_cm3_s",
                                history.adaptive_neutral_remainder_nuclei_cm3_s);
            write_vector_double(file, "/adaptive/atomic_comp_nuclei_cm3_s",
                                history.adaptive_atomic_comp_nuclei_cm3_s);
            write_vector_double(file, "/adaptive/molecular_comp_nuclei_cm3_s",
                                history.adaptive_molecular_comp_nuclei_cm3_s);
        }
    }

    // Local rate diagnostics.
    write_vector_double(file, "/rates/Te_eV", collect_rate_scalar(history.rate_diagnostics, &RateDiagnosticSnapshot::electron_temperature_eV));
    write_vector_double(file, "/rates/Ti_eV", collect_rate_scalar(history.rate_diagnostics, &RateDiagnosticSnapshot::ion_temperature_eV));
    write_vector_double(file, "/rates/ne_cm3", collect_rate_scalar(history.rate_diagnostics, &RateDiagnosticSnapshot::electron_density_cm3));
    write_vector_double(file, "/rates/atomic_scd_cm3_s",
                        collect_atomic_effective_scalar(history.rate_diagnostics, &AtomicEffectiveRates::scd_cm3_s));
    write_vector_double(file, "/rates/atomic_acd_cm3_s",
                        collect_atomic_effective_scalar(history.rate_diagnostics, &AtomicEffectiveRates::acd_cm3_s));
    write_vector_double(file, "/rates/atomic_effective_eir_rate_cm3_s",
                        collect_atomic_source_scalar(history.rate_diagnostics, &AtomicSourceDiagnostics::effective_eir_rate_cm3_s));
    write_vector_double(file, "/rates/atomic_mar_h_source_rate_cm3_s",
                        collect_atomic_source_scalar(history.rate_diagnostics, &AtomicSourceDiagnostics::mar_h_source_rate_cm3_s));
    write_vector_double(file, "/rates/atomic_flow_h_source_rate_cm3_s",
                        collect_atomic_source_scalar(history.rate_diagnostics, &AtomicSourceDiagnostics::flow_h_source_rate_cm3_s));
    write_vector_double(file, "/rates/molecular_flow_ionization_rate_cm3_s",
                        collect_atomic_source_scalar(history.rate_diagnostics, &AtomicSourceDiagnostics::molecular_flow_ionization_rate_cm3_s));
    write_vector_double(file, "/rates/molecular_flow_charge_exchange_rate_cm3_s",
                        collect_atomic_source_scalar(history.rate_diagnostics, &AtomicSourceDiagnostics::molecular_flow_charge_exchange_rate_cm3_s));
    write_vector_double(file, "/rates/molecular_dissociation_rate_cm3_s",
                        collect_atomic_source_scalar(history.rate_diagnostics, &AtomicSourceDiagnostics::molecular_dissociation_rate_cm3_s));
    write_vector_double(file, "/rates/full_ion_nuclei_source_cm3_s",
                        collect_atomic_source_scalar(history.rate_diagnostics, &AtomicSourceDiagnostics::full_ion_nuclei_source_cm3_s));
    write_vector_double(file, "/rates/full_ion_nuclei_sink_cm3_s",
                        collect_atomic_source_scalar(history.rate_diagnostics, &AtomicSourceDiagnostics::full_ion_nuclei_sink_cm3_s));
    write_vector_double(file, "/rates/molecular_flow_lhs_transport_rate_cm3_s",
                        build_molecular_flow_lhs_transport_rate(boundary, history));
    if (!history.rate_diagnostics.empty()) {
        const int n_h2plus = static_cast<int>(
            history.rate_diagnostics.front().molecular_ion_transport.state_indices.size());
        write_matrix(file, "/rates/H2_plus_mcx_production_cm3_s",
                     collect_molecular_ion_vector(
                         history.rate_diagnostics,
                         &MolecularIonTransportDiagnostics::mcx_production_cm3_s),
                     n_h2plus);
        write_matrix(file, "/rates/H2_plus_mi_production_cm3_s",
                     collect_molecular_ion_vector(
                         history.rate_diagnostics,
                         &MolecularIonTransportDiagnostics::mi_production_cm3_s),
                     n_h2plus);
        write_matrix(file, "/rates/H2_plus_dr_h_source_frequency_s",
                     collect_molecular_ion_vector(
                         history.rate_diagnostics,
                         &MolecularIonTransportDiagnostics::dr_h_source_frequency_s),
                     n_h2plus);
        write_matrix(file, "/rates/H2_plus_target_velocity_cm_s",
                     collect_molecular_ion_vector(
                         history.rate_diagnostics,
                         &MolecularIonTransportDiagnostics::target_velocity_cm_s),
                     n_h2plus);
        write_matrix_stack(file, "/rates/H2_plus_generator_s",
                           collect_molecular_ion_matrix(
                               history.rate_diagnostics,
                               &MolecularIonTransportDiagnostics::generator_s),
                           n_h2plus,
                           n_h2plus);
        write_vector_double(file, "/rates/mar_simple_branching_dr_h_source_cm3_s",
                            collect_molecular_ion_scalar(
                                history.rate_diagnostics,
                                &MolecularIonTransportDiagnostics::simple_branching_dr_mar_h_source_cm3_s));
    }

    write_vector_double(file, "/rates/grouped_source_H_plus_cm3_s",
                        collect_grouped_source_scalar(history.rate_diagnostics, &GroupedSourceDiagnostics::H_plus_cm3_s));
    write_vector_double(file, "/rates/grouped_source_H_cm3_s",
                        collect_grouped_source_scalar(history.rate_diagnostics, &GroupedSourceDiagnostics::H_cm3_s));
    write_vector_double(file, "/rates/grouped_source_H2_cm3_s",
                        collect_grouped_source_scalar(history.rate_diagnostics, &GroupedSourceDiagnostics::H2_cm3_s));
    write_vector_double(file, "/rates/grouped_source_H_minus_cm3_s",
                        collect_grouped_source_scalar(history.rate_diagnostics, &GroupedSourceDiagnostics::H_minus_cm3_s));
    write_vector_double(file, "/rates/grouped_source_H2_plus_cm3_s",
                        collect_grouped_source_scalar(history.rate_diagnostics, &GroupedSourceDiagnostics::H2_plus_cm3_s));

    write_vector_double(file, "/rates/flow_to_local_source_H_plus_cm3_s",
                        collect_flow_to_local_source_scalar(history.rate_diagnostics, &FlowToLocalSourceDiagnostics::H_plus_cm3_s));
    write_vector_double(file, "/rates/flow_to_local_source_H_cm3_s",
                        collect_flow_to_local_source_scalar(history.rate_diagnostics, &FlowToLocalSourceDiagnostics::H_cm3_s));
    write_vector_double(file, "/rates/flow_to_local_source_H2_cm3_s",
                        collect_flow_to_local_source_scalar(history.rate_diagnostics, &FlowToLocalSourceDiagnostics::H2_cm3_s));
    write_vector_double(file, "/rates/flow_to_local_source_H_minus_cm3_s",
                        collect_flow_to_local_source_scalar(history.rate_diagnostics, &FlowToLocalSourceDiagnostics::H_minus_cm3_s));
    write_vector_double(file, "/rates/flow_to_local_source_H2_plus_cm3_s",
                        collect_flow_to_local_source_scalar(history.rate_diagnostics, &FlowToLocalSourceDiagnostics::H2_plus_cm3_s));

    write_vector_double(file, "/power_loss/atomic_ionization_W_cm3",
                        collect_power_loss_scalar(history.rate_diagnostics, &PowerLossDiagnostics::atomic_ionization_W_cm3));
    write_vector_double(file, "/power_loss/molecular_ionization_W_cm3",
                        collect_power_loss_scalar(history.rate_diagnostics, &PowerLossDiagnostics::molecular_ionization_W_cm3));
    write_vector_double(file, "/power_loss/molecular_dissociation_W_cm3",
                        collect_power_loss_scalar(history.rate_diagnostics, &PowerLossDiagnostics::molecular_dissociation_W_cm3));
    write_vector_double(file, "/power_loss/total_electron_inelastic_W_cm3",
                        collect_power_loss_scalar(history.rate_diagnostics, &PowerLossDiagnostics::total_electron_inelastic_W_cm3));

    write_vector_double(file, "/rates/atomic_qss_transport_frequency_s",
                        collect_atomic_qss_scalar(history.rate_diagnostics, &AtomicQSSDiagnostics::transport_frequency_s));
    write_vector_double(file, "/rates/atomic_qss_first_excited_loss_frequency_s",
                        collect_atomic_qss_scalar(history.rate_diagnostics, &AtomicQSSDiagnostics::first_excited_local_loss_frequency_s));
    write_vector_double(file, "/rates/atomic_qss_relaxation_length_cm",
                        collect_atomic_qss_scalar(history.rate_diagnostics, &AtomicQSSDiagnostics::relaxation_length_cm));
    write_vector_double(file, "/rates/atomic_qss_max_transport_to_local_ratio",
                        collect_atomic_qss_scalar(history.rate_diagnostics, &AtomicQSSDiagnostics::max_transport_to_local_ratio));
    write_vector_double(file, "/rates/atomic_qss_max_transport_to_loss_frequency_ratio",
                        collect_atomic_qss_scalar(history.rate_diagnostics, &AtomicQSSDiagnostics::max_transport_to_loss_frequency_ratio));
    write_vector_int(file, "/rates/atomic_qss_first_excited_index",
                     collect_atomic_qss_int(history.rate_diagnostics, &AtomicQSSDiagnostics::first_excited_index));

    if (!history.rate_diagnostics.empty()) {
        std::vector<int> excited_indices;
        for (const auto& snapshot : history.rate_diagnostics) {
            if (!snapshot.atomic_qss.excited_indices.empty()) {
                excited_indices = snapshot.atomic_qss.excited_indices;
                break;
            }
        }
        if (!excited_indices.empty()) {
            write_vector_int(file, "/rates/atomic_excited_indices", excited_indices);
            write_string_vector(file, "/rates/atomic_excited_labels",
                                build_level_labels(excited_indices, levels));
            write_matrix(file, "/rates/atomic_excited_transport_rate_cm3_s",
                         collect_atomic_qss_vector(history.rate_diagnostics, &AtomicQSSDiagnostics::transport_rate_cm3_s),
                         static_cast<int>(excited_indices.size()));
            write_matrix(file, "/rates/atomic_excited_local_source_rate_cm3_s",
                         collect_atomic_qss_vector(history.rate_diagnostics, &AtomicQSSDiagnostics::local_source_rate_cm3_s),
                         static_cast<int>(excited_indices.size()));
            write_matrix(file, "/rates/atomic_excited_local_loss_rate_cm3_s",
                         collect_atomic_qss_vector(history.rate_diagnostics, &AtomicQSSDiagnostics::local_loss_rate_cm3_s),
                         static_cast<int>(excited_indices.size()));
            write_matrix(file, "/rates/atomic_excited_local_loss_frequency_s",
                         collect_atomic_qss_vector(history.rate_diagnostics, &AtomicQSSDiagnostics::local_loss_frequency_s),
                         static_cast<int>(excited_indices.size()));
            write_matrix(file, "/rates/atomic_excited_transport_to_local_ratio",
                         collect_atomic_qss_vector(history.rate_diagnostics, &AtomicQSSDiagnostics::transport_to_local_ratio),
                         static_cast<int>(excited_indices.size()));
            write_matrix(file, "/rates/atomic_excited_transport_to_loss_frequency_ratio",
                         collect_atomic_qss_vector(history.rate_diagnostics, &AtomicQSSDiagnostics::transport_to_loss_frequency_ratio),
                         static_cast<int>(excited_indices.size()));
        }
    }

    if (config.io.verbose_logging) {
        std::cout << "[DCR_Solver] Wrote HDF5 output: " << out_file.string() << "\n";
    }
}

} // namespace dcr::solver
