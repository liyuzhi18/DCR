#include "HDF5Output.hpp"

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
    file.createGroup("/timing");

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

    write_scalar_double(file, "/timing/boundary_elapsed_seconds", history.boundary_elapsed_seconds);
    write_scalar_int(file, "/timing/boundary_iterations", history.boundary_iterations);
    write_vector_double(file, "/timing/marching_cell_elapsed_seconds", history.cell_elapsed_seconds);
    write_vector_int(file, "/timing/marching_cell_iterations", history.cell_iterations);
    write_vector_int(file, "/timing/marching_cell_converged", history.cell_converged);

    // Local rate diagnostics.
    write_vector_double(file, "/rates/Te_eV", collect_rate_scalar(history.rate_diagnostics, &RateDiagnosticSnapshot::electron_temperature_eV));
    write_vector_double(file, "/rates/Ti_eV", collect_rate_scalar(history.rate_diagnostics, &RateDiagnosticSnapshot::ion_temperature_eV));
    write_vector_double(file, "/rates/ne_cm3", collect_rate_scalar(history.rate_diagnostics, &RateDiagnosticSnapshot::electron_density_cm3));
    write_vector_double(file, "/rates/atomic_scd_cm3_s",
                        collect_atomic_effective_scalar(history.rate_diagnostics, &AtomicEffectiveRates::scd_cm3_s));
    write_vector_double(file, "/rates/atomic_acd_cm3_s",
                        collect_atomic_effective_scalar(history.rate_diagnostics, &AtomicEffectiveRates::acd_cm3_s));
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
