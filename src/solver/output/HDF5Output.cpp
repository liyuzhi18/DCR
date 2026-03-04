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

    // Populations along x nodes.
    write_matrix(file, "/population/background_full", history.background_full, atomic_data.get_total_states());
    write_matrix(file, "/population/flowA", history.flowA, static_cast<int>(boundary.A_indices.size()));
    write_matrix(file, "/population/flowM", history.flowM, static_cast<int>(boundary.M_indices.size()));

    const auto total_pop = build_total_population(atomic_data, boundary, history);
    write_matrix(file, "/population/total_full", total_pop, atomic_data.get_total_states());
    const auto n_nuclei_cm3 = build_total_nuclei_profile(total_pop, levels);
    write_vector_double(file, "/grid/n_nuclei_cm3", n_nuclei_cm3);

    if (config.io.verbose_logging) {
        std::cout << "[DCR_Solver] Wrote HDF5 output: " << out_file.string() << "\n";
    }
}

} // namespace dcr::solver
