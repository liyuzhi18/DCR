#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#include "../src/atomic/AtomicData.hpp"
#include "../src/physics/WallBoundary.hpp"
#include "../src/solver/boundary/BoundaryPhase.hpp"
#include "../src/solver/marching/GlobalBVP.hpp"
#include "TestDCRSetup.hpp"

namespace {

namespace fs = std::filesystem;
using Clock = std::chrono::steady_clock;
using Result = dcr::solver::GlobalBVPResult;
using Diagnostics = dcr::solver::GlobalBVPSolveDiagnostics;

constexpr std::uint64_t checkpoint_magic = 0x444352524546494eULL;
constexpr std::uint32_t checkpoint_version = 1;

void require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

template <typename T>
void write_raw(std::ostream& stream, const T& value) {
    static_assert(std::is_trivially_copyable_v<T>);
    stream.write(reinterpret_cast<const char*>(&value), sizeof(value));
    require(stream.good(), "Failed to write refinement checkpoint.");
}

template <typename T>
T read_raw(std::istream& stream) {
    static_assert(std::is_trivially_copyable_v<T>);
    T value{};
    stream.read(reinterpret_cast<char*>(&value), sizeof(value));
    require(stream.good(), "Failed to read refinement checkpoint.");
    return value;
}

void write_string(std::ostream& stream, const std::string& value) {
    write_raw(stream, static_cast<std::uint64_t>(value.size()));
    stream.write(value.data(), static_cast<std::streamsize>(value.size()));
    require(stream.good(), "Failed to write checkpoint string.");
}

std::string read_string(std::istream& stream) {
    const auto size = read_raw<std::uint64_t>(stream);
    require(size < (1ULL << 30), "Checkpoint string is unreasonably large.");
    std::string value(static_cast<size_t>(size), '\0');
    stream.read(value.data(), static_cast<std::streamsize>(size));
    require(stream.good(), "Failed to read checkpoint string.");
    return value;
}

template <typename T>
void write_vector(std::ostream& stream, const std::vector<T>& values) {
    write_raw(stream, static_cast<std::uint64_t>(values.size()));
    for (const auto& value : values) write_raw(stream, value);
}

template <typename T>
std::vector<T> read_vector(std::istream& stream) {
    const auto size = read_raw<std::uint64_t>(stream);
    require(size < (1ULL << 30), "Checkpoint vector is unreasonably large.");
    std::vector<T> values(static_cast<size_t>(size));
    for (auto& value : values) value = read_raw<T>(stream);
    return values;
}

void write_eigen_vector(std::ostream& stream, const dcr::base::Vector& values) {
    write_raw(stream, static_cast<std::int64_t>(values.size()));
    for (int i = 0; i < values.size(); ++i) write_raw(stream, values(i));
}

dcr::base::Vector read_eigen_vector(std::istream& stream) {
    const auto size = read_raw<std::int64_t>(stream);
    require(size >= 0 && size < (1LL << 30), "Invalid checkpoint state-vector size.");
    dcr::base::Vector values(size);
    for (int i = 0; i < values.size(); ++i) values(i) = read_raw<double>(stream);
    return values;
}

void write_eigen_vectors(
    std::ostream& stream,
    const std::vector<dcr::base::Vector>& values) {
    write_raw(stream, static_cast<std::uint64_t>(values.size()));
    for (const auto& value : values) write_eigen_vector(stream, value);
}

std::vector<dcr::base::Vector> read_eigen_vectors(std::istream& stream) {
    const auto size = read_raw<std::uint64_t>(stream);
    require(size < (1ULL << 30), "Checkpoint matrix is unreasonably large.");
    std::vector<dcr::base::Vector> values;
    values.reserve(static_cast<size_t>(size));
    for (std::uint64_t i = 0; i < size; ++i) values.push_back(read_eigen_vector(stream));
    return values;
}

void write_diagnostics(std::ostream& stream, const Diagnostics& value) {
    write_string(stream, value.initialization_source);
    write_string(stream, value.solve_path);
    write_string(stream, value.jacobian_storage);
    write_string(stream, value.factorization_type);
    write_raw(stream, value.unknown_count);
    write_raw(stream, value.continuation_stages);
    write_raw(stream, value.continuation_stage_attempts);
    write_raw(stream, value.newton_iterations);
    write_raw(stream, value.residual_evaluations);
    write_raw(stream, value.residual_cache_hits);
    write_raw(stream, value.molecular_transport_evaluations);
    write_raw(stream, value.molecular_transport_cache_hits);
    write_raw(stream, value.molecular_interval_factorizations);
    write_raw(stream, value.jacobian_assemblies);
    write_raw(stream, value.jacobian_vector_products);
    write_raw(stream, value.linear_solves);
    write_raw(stream, value.linear_iterations);
    write_raw(stream, value.line_search_trial_evaluations);
    write_raw(stream, value.jacobian_rows);
    write_raw(stream, value.jacobian_columns);
    write_raw(stream, value.jacobian_nonzeros);
    write_raw(stream, value.jacobian_nonzeros_per_row);
    write_raw(stream, value.factorization_nonzeros);
    write_raw(stream, value.factorization_fill_ratio);
    write_raw(stream, value.factorization_memory_bytes);
    write_raw(stream, value.reduced_condition_estimate);
    write_raw(stream, value.direct_full_physics_attempted);
    write_raw(stream, value.direct_full_physics_succeeded);
    write_raw(stream, value.continuation_fallback_used);
    write_raw(stream, value.schur_complement_explicitly_assembled);
    write_raw(stream, value.molecular_jacobian_inverse_explicitly_formed);
    write_raw(stream, value.molecular_transport_seconds);
    write_raw(stream, value.residual_seconds);
    write_raw(stream, value.jacobian_seconds);
    write_raw(stream, value.factorization_seconds);
    write_raw(stream, value.linear_solve_seconds);
    write_raw(stream, value.condition_seconds);
    write_raw(stream, value.logging_seconds);
    write_raw(stream, value.total_seconds);
}

Diagnostics read_diagnostics(std::istream& stream) {
    Diagnostics value;
    value.initialization_source = read_string(stream);
    value.solve_path = read_string(stream);
    value.jacobian_storage = read_string(stream);
    value.factorization_type = read_string(stream);
    value.unknown_count = read_raw<int>(stream);
    value.continuation_stages = read_raw<int>(stream);
    value.continuation_stage_attempts = read_raw<int>(stream);
    value.newton_iterations = read_raw<int>(stream);
    value.residual_evaluations = read_raw<int>(stream);
    value.residual_cache_hits = read_raw<int>(stream);
    value.molecular_transport_evaluations = read_raw<int>(stream);
    value.molecular_transport_cache_hits = read_raw<int>(stream);
    value.molecular_interval_factorizations = read_raw<int>(stream);
    value.jacobian_assemblies = read_raw<int>(stream);
    value.jacobian_vector_products = read_raw<int>(stream);
    value.linear_solves = read_raw<int>(stream);
    value.linear_iterations = read_raw<int>(stream);
    value.line_search_trial_evaluations = read_raw<int>(stream);
    value.jacobian_rows = read_raw<int>(stream);
    value.jacobian_columns = read_raw<int>(stream);
    value.jacobian_nonzeros = read_raw<long long>(stream);
    value.jacobian_nonzeros_per_row = read_raw<double>(stream);
    value.factorization_nonzeros = read_raw<long long>(stream);
    value.factorization_fill_ratio = read_raw<double>(stream);
    value.factorization_memory_bytes = read_raw<double>(stream);
    value.reduced_condition_estimate = read_raw<double>(stream);
    value.direct_full_physics_attempted = read_raw<bool>(stream);
    value.direct_full_physics_succeeded = read_raw<bool>(stream);
    value.continuation_fallback_used = read_raw<bool>(stream);
    value.schur_complement_explicitly_assembled = read_raw<bool>(stream);
    value.molecular_jacobian_inverse_explicitly_formed = read_raw<bool>(stream);
    value.molecular_transport_seconds = read_raw<double>(stream);
    value.residual_seconds = read_raw<double>(stream);
    value.jacobian_seconds = read_raw<double>(stream);
    value.factorization_seconds = read_raw<double>(stream);
    value.linear_solve_seconds = read_raw<double>(stream);
    value.condition_seconds = read_raw<double>(stream);
    value.logging_seconds = read_raw<double>(stream);
    value.total_seconds = read_raw<double>(stream);
    return value;
}

fs::path checkpoint_path(const fs::path& directory, int nodes) {
    return directory / ("refinement_" + std::to_string(nodes));
}

void write_checkpoint(
    const fs::path& path,
    const dcr::atomic::AtomicData& atomic_data,
    const dcr::solver::BoundaryPhaseResult& boundary,
    const Result& result) {
    fs::create_directories(path.parent_path());
    const fs::path temporary = path.string() + ".tmp";
    std::ofstream stream(temporary, std::ios::binary | std::ios::trunc);
    require(stream.is_open(), "Could not create refinement checkpoint: " + temporary.string());
    write_raw(stream, checkpoint_magic);
    write_raw(stream, checkpoint_version);
    write_raw(stream, static_cast<int>(result.history.x_cm.size()));
    write_raw(stream, result.converged);

    const auto& levels = atomic_data.get_levels();
    write_raw(stream, static_cast<std::uint64_t>(levels.size()));
    for (const auto& level : levels) {
        write_string(stream, level.label);
        write_raw(stream, static_cast<int>(level.type));
        write_raw(stream, level.charge);
        write_raw(stream, level.atomicity);
        write_raw(stream, level.internal_id);
        write_raw(stream, level.mass_amu);
    }
    write_vector(stream, boundary.P_indices);
    write_vector(stream, boundary.A_indices);
    write_vector(stream, boundary.M_indices);

    write_raw(stream, result.iterations);
    write_raw(stream, result.residual_norm);
    write_raw(stream, result.upstream_proton_flux_cm2_s);
    write_raw(stream, result.max_chemistry_nuclei_relative_error);
    write_raw(stream, result.integrated_nuclei_balance_relative_error);
    write_raw(stream, result.ion_flux_chemistry_relative_error);
    write_raw(stream, result.minimum_line_search_factor);
    write_raw(stream, result.final_ion_residual);
    write_raw(stream, result.final_flow_a_residual);
    write_raw(stream, result.final_flow_m_residual);
    write_raw(stream, result.final_local_residual);
    write_raw(stream, result.final_boundary_residual);
    write_string(stream, result.velocity_profile_type);
    write_raw(stream, result.ion_velocity_transition_length_cm);
    write_raw(stream, result.ion_upstream_speed_fraction);
    write_raw(stream, result.target_electron_temperature_eV);
    write_raw(stream, result.target_ion_temperature_eV);
    write_raw(stream, result.target_nuclei_density_cm3);
    write_raw(stream, result.boundary_exhaust_width_cm);
    write_raw(stream, result.spatial_exhaust_width_cm);
    write_diagnostics(stream, result.diagnostics);

    write_vector(stream, result.history.x_cm);
    write_eigen_vectors(stream, result.history.background_full);
    write_eigen_vectors(stream, result.history.flowA);
    write_eigen_vectors(stream, result.history.flowM);
    write_raw(stream, static_cast<std::uint64_t>(result.history.rate_diagnostics.size()));
    for (const auto& rate : result.history.rate_diagnostics) {
        write_raw(stream, rate.atomic_sources.full_ion_nuclei_source_cm3_s);
        write_raw(stream, rate.atomic_sources.full_ion_nuclei_sink_cm3_s);
    }
    stream.close();
    require(stream.good(), "Failed to close refinement checkpoint: " + temporary.string());
    std::error_code error;
    fs::rename(temporary, path, error);
    if (error) {
        fs::remove(temporary);
        throw std::runtime_error("Could not publish refinement checkpoint: " + error.message());
    }
}

Result read_checkpoint(
    const fs::path& path,
    const dcr::atomic::AtomicData& atomic_data,
    const dcr::solver::BoundaryPhaseResult& boundary) {
    std::ifstream stream(path, std::ios::binary);
    require(stream.is_open(), "Could not open refinement checkpoint: " + path.string());
    require(read_raw<std::uint64_t>(stream) == checkpoint_magic,
        "Invalid refinement checkpoint signature.");
    require(read_raw<std::uint32_t>(stream) == checkpoint_version,
        "Unsupported refinement checkpoint version.");
    const int nodes = read_raw<int>(stream);
    Result result;
    result.converged = read_raw<bool>(stream);
    require(result.converged, "Refinement checkpoint is not marked accepted.");

    const auto& levels = atomic_data.get_levels();
    require(read_raw<std::uint64_t>(stream) == levels.size(),
        "Checkpoint atomic-state count is incompatible.");
    for (const auto& level : levels) {
        require(read_string(stream) == level.label, "Checkpoint state label is incompatible.");
        require(read_raw<int>(stream) == static_cast<int>(level.type),
            "Checkpoint state type is incompatible.");
        require(read_raw<int>(stream) == level.charge, "Checkpoint state charge is incompatible.");
        require(read_raw<int>(stream) == level.atomicity,
            "Checkpoint state atomicity is incompatible.");
        require(read_raw<int>(stream) == level.internal_id,
            "Checkpoint state ID is incompatible.");
        require(read_raw<double>(stream) == level.mass_amu,
            "Checkpoint state mass is incompatible.");
    }
    require(read_vector<int>(stream) == boundary.P_indices, "Checkpoint P layout is incompatible.");
    require(read_vector<int>(stream) == boundary.A_indices, "Checkpoint A layout is incompatible.");
    require(read_vector<int>(stream) == boundary.M_indices, "Checkpoint M layout is incompatible.");

    result.iterations = read_raw<int>(stream);
    result.residual_norm = read_raw<double>(stream);
    result.upstream_proton_flux_cm2_s = read_raw<double>(stream);
    result.max_chemistry_nuclei_relative_error = read_raw<double>(stream);
    result.integrated_nuclei_balance_relative_error = read_raw<double>(stream);
    result.ion_flux_chemistry_relative_error = read_raw<double>(stream);
    result.minimum_line_search_factor = read_raw<double>(stream);
    result.final_ion_residual = read_raw<double>(stream);
    result.final_flow_a_residual = read_raw<double>(stream);
    result.final_flow_m_residual = read_raw<double>(stream);
    result.final_local_residual = read_raw<double>(stream);
    result.final_boundary_residual = read_raw<double>(stream);
    result.velocity_profile_type = read_string(stream);
    result.ion_velocity_transition_length_cm = read_raw<double>(stream);
    result.ion_upstream_speed_fraction = read_raw<double>(stream);
    result.target_electron_temperature_eV = read_raw<double>(stream);
    result.target_ion_temperature_eV = read_raw<double>(stream);
    result.target_nuclei_density_cm3 = read_raw<double>(stream);
    result.boundary_exhaust_width_cm = read_raw<double>(stream);
    result.spatial_exhaust_width_cm = read_raw<double>(stream);
    result.diagnostics = read_diagnostics(stream);

    result.history.x_cm = read_vector<double>(stream);
    result.history.background_full = read_eigen_vectors(stream);
    result.history.flowA = read_eigen_vectors(stream);
    result.history.flowM = read_eigen_vectors(stream);
    const auto rate_count = read_raw<std::uint64_t>(stream);
    result.history.rate_diagnostics.resize(static_cast<size_t>(rate_count));
    for (auto& rate : result.history.rate_diagnostics) {
        rate.atomic_sources.full_ion_nuclei_source_cm3_s = read_raw<double>(stream);
        rate.atomic_sources.full_ion_nuclei_sink_cm3_s = read_raw<double>(stream);
    }
    require(stream.peek() == std::char_traits<char>::eof(),
        "Refinement checkpoint contains trailing incompatible data.");
    require(static_cast<int>(result.history.x_cm.size()) == nodes,
        "Checkpoint node count is inconsistent.");
    require(result.history.background_full.size() == result.history.x_cm.size() &&
        result.history.flowA.size() == result.history.x_cm.size() &&
        result.history.flowM.size() == result.history.x_cm.size() &&
        result.history.rate_diagnostics.size() == result.history.x_cm.size(),
        "Checkpoint profile row counts are inconsistent.");
    for (size_t k = 0; k < result.history.x_cm.size(); ++k) {
        require(std::isfinite(result.history.x_cm[k]) &&
            (k == 0 || result.history.x_cm[k] > result.history.x_cm[k - 1]),
            "Checkpoint coordinates are invalid.");
        require(result.history.background_full[k].size() == static_cast<int>(levels.size()) &&
            result.history.flowA[k].size() == static_cast<int>(boundary.A_indices.size()) &&
            result.history.flowM[k].size() == static_cast<int>(boundary.M_indices.size()),
            "Checkpoint profile dimensions are incompatible.");
        require(result.history.background_full[k].allFinite() &&
            result.history.flowA[k].allFinite() && result.history.flowM[k].allFinite(),
            "Checkpoint populations are non-finite.");
    }
    require(std::isfinite(result.upstream_proton_flux_cm2_s) &&
        result.upstream_proton_flux_cm2_s > 0.0,
        "Checkpoint upstream proton flux is invalid.");
    result.boundary = boundary;
    return result;
}

template <typename VectorAt>
dcr::base::Vector interpolate(
    const std::vector<double>& coarse_x,
    double x,
    VectorAt vector_at) {
    auto upper_iterator = std::upper_bound(coarse_x.begin(), coarse_x.end(), x);
    size_t upper = static_cast<size_t>(std::distance(coarse_x.begin(), upper_iterator));
    if (upper == 0) upper = 1;
    if (upper >= coarse_x.size()) upper = coarse_x.size() - 1;
    const size_t lower = upper - 1;
    const double span = coarse_x[upper] - coarse_x[lower];
    const double fraction = span > 0.0
        ? std::clamp((x - coarse_x[lower]) / span, 0.0, 1.0) : 0.0;
    return (1.0 - fraction) * vector_at(lower) + fraction * vector_at(upper);
}

double profile_difference(const Result& coarse, const Result& fine) {
    double difference_squared = 0.0;
    double fine_squared = 0.0;
    for (size_t k = 0; k < fine.history.x_cm.size(); ++k) {
        const double x = fine.history.x_cm[k];
        const auto p = interpolate(coarse.history.x_cm, x,
            [&](size_t i) -> const dcr::base::Vector& {
                return coarse.history.background_full[i];
            });
        const auto a = interpolate(coarse.history.x_cm, x,
            [&](size_t i) -> const dcr::base::Vector& { return coarse.history.flowA[i]; });
        const auto m = interpolate(coarse.history.x_cm, x,
            [&](size_t i) -> const dcr::base::Vector& { return coarse.history.flowM[i]; });
        difference_squared += (p - fine.history.background_full[k]).squaredNorm();
        difference_squared += (a - fine.history.flowA[k]).squaredNorm();
        difference_squared += (m - fine.history.flowM[k]).squaredNorm();
        fine_squared += fine.history.background_full[k].squaredNorm();
        fine_squared += fine.history.flowA[k].squaredNorm();
        fine_squared += fine.history.flowM[k].squaredNorm();
    }
    return std::sqrt(difference_squared / std::max(1.0, fine_squared));
}

struct IntegratedRates {
    double source = 0.0;
    double sink = 0.0;
};

IntegratedRates integrated_rates(const Result& result) {
    IntegratedRates rates;
    const auto& x = result.history.x_cm;
    require(result.history.rate_diagnostics.size() == x.size(),
        "Rate diagnostics are unavailable for refinement comparison.");
    for (size_t k = 0; k + 1 < x.size(); ++k) {
        const double dx = x[k + 1] - x[k];
        const auto& left = result.history.rate_diagnostics[k].atomic_sources;
        const auto& right = result.history.rate_diagnostics[k + 1].atomic_sources;
        rates.source += 0.5 * dx *
            (left.full_ion_nuclei_source_cm3_s + right.full_ion_nuclei_source_cm3_s);
        rates.sink += 0.5 * dx *
            (left.full_ion_nuclei_sink_cm3_s + right.full_ion_nuclei_sink_cm3_s);
    }
    return rates;
}

double relative_difference(double left, double right) {
    return std::abs(left - right) /
        std::max({1.0, std::abs(left), std::abs(right)});
}

void validate_result(const Result& result, int nodes) {
    require(result.converged && result.residual_norm < 1.0e-8,
        "Grid " + std::to_string(nodes) + " did not meet residual acceptance.");
    require(result.max_chemistry_nuclei_relative_error < 1.0e-10,
        "Grid " + std::to_string(nodes) + " failed chemistry conservation.");
    require(result.integrated_nuclei_balance_relative_error <
        std::max(1.0e-7, 1.0e-6 / std::pow(std::max(1, nodes - 1), 2.0)),
        "Grid " + std::to_string(nodes) + " failed integrated nuclei balance.");
    require(result.ion_flux_chemistry_relative_error < 1.0e-7,
        "Grid " + std::to_string(nodes) + " failed ion flux-chemistry balance.");
    require(static_cast<int>(result.history.x_cm.size()) == nodes,
        "Grid " + std::to_string(nodes) + " has the wrong node count.");
}

void require_physics_compatibility(
    const Result& result,
    const dcr::io::Config& config) {
    const auto target = test_dcr::plasma_temperatures_at(config, 0.0);
    require(result.velocity_profile_type == "fixed_scale_linear_bohm" &&
        result.ion_velocity_transition_length_cm ==
            config.global_bvp.ion_velocity_transition_length_cm &&
        result.ion_upstream_speed_fraction ==
            config.global_bvp.ion_upstream_speed_fraction &&
        result.target_electron_temperature_eV == target.electron_eV &&
        result.target_ion_temperature_eV == target.ion_eV &&
        result.target_nuclei_density_cm3 == config.plasma.total_density &&
        result.boundary_exhaust_width_cm == config.grid.boundary_poloidal_width_cm &&
        result.spatial_exhaust_width_cm == config.grid.spatial_exhaust_width_cm,
        "Refinement checkpoint physics metadata is incompatible.");
}

void print_timing(int nodes, const Result& result, bool loaded, double checkpoint_seconds) {
    const auto& d = result.diagnostics;
    std::cout << std::setprecision(8)
              << "[global-bvp-refinement-timing]"
              << " nodes=" << nodes
              << " unknowns=" << d.unknown_count
              << " initialization_source=" << (loaded ? "checkpoint" : d.initialization_source)
              << " solve_path=" << (loaded ? "checkpoint_resume" : d.solve_path)
              << " continuation_stages=" << d.continuation_stages
              << " continuation_stage_attempts=" << d.continuation_stage_attempts
              << " newton_iterations=" << d.newton_iterations
              << " residual_evaluations=" << d.residual_evaluations
              << " residual_cache_hits=" << d.residual_cache_hits
              << " jacobian_assemblies=" << d.jacobian_assemblies
              << " linear_solves=" << d.linear_solves
              << " linear_iterations=" << d.linear_iterations
              << " line_search_trial_evaluations=" << d.line_search_trial_evaluations
              << " M_transport_evaluations=" << d.molecular_transport_evaluations
              << " M_transport_cache_hits=" << d.molecular_transport_cache_hits
              << " M_interval_factorizations=" << d.molecular_interval_factorizations
              << " M_transport_seconds=" << d.molecular_transport_seconds
              << " residual_seconds=" << d.residual_seconds
              << " jacobian_seconds=" << d.jacobian_seconds
              << " factorization_seconds=" << d.factorization_seconds
              << " linear_solve_seconds=" << d.linear_solve_seconds
              << " condition_seconds=" << d.condition_seconds
              << " logging_seconds=" << d.logging_seconds
              << " checkpoint_seconds=" << checkpoint_seconds
              << " total_seconds=" << d.total_seconds
              << std::endl;
    std::cout << "[global-bvp-reduced-linear-algebra]"
              << " nodes=" << nodes
              << " rows=" << d.jacobian_rows
              << " columns=" << d.jacobian_columns
              << " nonzeros=" << d.jacobian_nonzeros
              << " nonzeros_per_row=" << d.jacobian_nonzeros_per_row
              << " storage=" << d.jacobian_storage
              << " factorization=" << d.factorization_type
              << " factorization_nonzeros=" << d.factorization_nonzeros
              << " fill_ratio=" << d.factorization_fill_ratio
              << " factorization_memory_bytes=" << d.factorization_memory_bytes
              << " condition_estimate=" << d.reduced_condition_estimate
              << " schur_complement_explicit="
              << d.schur_complement_explicitly_assembled
              << " J_MM_inverse_explicit="
              << d.molecular_jacobian_inverse_explicitly_formed
              << " reduced_action=matrix_free_FD_through_causal_M"
              << std::endl;
}

void print_refinement(
    int nodes,
    const Result& result,
    const Result* previous,
    double profile_change) {
    const auto rates = integrated_rates(result);
    const auto previous_rates = previous ? integrated_rates(*previous) : IntegratedRates{};
    const double q_change = previous
        ? relative_difference(result.upstream_proton_flux_cm2_s,
              previous->upstream_proton_flux_cm2_s)
        : 0.0;
    std::cout << "[grid-refinement] nodes=" << nodes
              << " residual=" << result.residual_norm
              << " chemistry=" << result.max_chemistry_nuclei_relative_error
              << " nuclei=" << result.integrated_nuclei_balance_relative_error
              << " ion=" << result.ion_flux_chemistry_relative_error
              << " q_up=" << result.upstream_proton_flux_cm2_s
              << " q_change=" << q_change
              << " profile_change=" << profile_change
              << " integrated_ion_source=" << rates.source
              << " integrated_ion_sink=" << rates.sink
              << " integrated_net_ion_rate=" << (rates.source - rates.sink)
              << " recombination_degree=" << rates.sink / std::max(1.0, rates.source)
              << " source_change="
              << (previous ? relative_difference(rates.source, previous_rates.source) : 0.0)
              << " sink_change="
              << (previous ? relative_difference(rates.sink, previous_rates.sink) : 0.0)
              << std::endl;
}

void print_runtime_ratios(const std::map<int, Result>& accepted) {
    for (const auto [fine, coarse] : std::vector<std::pair<int, int>>{
             {50, 25}, {100, 50}, {200, 100}}) {
        const auto fine_it = accepted.find(fine);
        const auto coarse_it = accepted.find(coarse);
        if (fine_it == accepted.end() || coarse_it == accepted.end()) continue;
        std::cout << "[global-bvp-refinement-runtime-ratio] T_" << fine
                  << "/T_" << coarse << "="
                  << fine_it->second.diagnostics.total_seconds /
                     std::max(1.0e-30, coarse_it->second.diagnostics.total_seconds)
                  << std::endl;
    }
}

double max_population_relative_difference(const Result& left, const Result& right) {
    require(left.history.x_cm == right.history.x_cm,
        "Branch comparison grids differ.");
    double maximum = 0.0;
    auto compare = [&](const std::vector<dcr::base::Vector>& a,
                       const std::vector<dcr::base::Vector>& b) {
        require(a.size() == b.size(), "Branch comparison row counts differ.");
        for (size_t k = 0; k < a.size(); ++k) {
            require(a[k].size() == b[k].size(), "Branch comparison widths differ.");
            for (int i = 0; i < a[k].size(); ++i) {
                maximum = std::max(maximum, std::abs(a[k](i) - b[k](i)) /
                    std::max({1.0e-30, std::abs(a[k](i)), std::abs(b[k](i))}));
            }
        }
    };
    compare(left.history.background_full, right.history.background_full);
    compare(left.history.flowA, right.history.flowA);
    compare(left.history.flowM, right.history.flowM);
    return maximum;
}

} // namespace

int main(int argc, char** argv) {
    const std::string mode = argc > 1 ? argv[1] : "--fast";
    require(mode == "--fast" || mode == "--slow",
        "Usage: check_global_bvp_reduced_refinement [--fast|--slow] [checkpoint_dir]");
    const fs::path checkpoint_directory = argc > 2
        ? fs::path(argv[2])
        : fs::path("global_bvp_refinement_checkpoints");

    auto base_config = test_dcr::load_reference_config(2, false);
    base_config.grid.length_cm = 0.01;
    base_config.numerics.marching_solver = "global_sparse_newton";
    base_config.numerics.marching_tolerance = 1.0e-8;
    base_config.numerics.marching_max_iterations = 120;
    base_config.global_bvp.ion_velocity_transition_length_cm = 1.0;
    dcr::atomic::AtomicData atomic_data(base_config);
    auto boundary_plasma = test_dcr::make_plasma_state(
        base_config, atomic_data.get_total_states());
    const auto temperatures = test_dcr::plasma_temperatures_at(base_config, 0.0);
    test_dcr::EEDFContext eedf(temperatures.electron_eV);
    const double ion_mass_amu = test_dcr::estimate_ion_mass_amu(base_config);
    const auto wall = dcr::physics::compute_wall_recycling(
        base_config.wall.material, temperatures.electron_eV, temperatures.ion_eV,
        ion_mass_amu, base_config.wall.sheath_potential_drop);
    const auto boundary = dcr::solver::run_boundary_phase(
        base_config, atomic_data, boundary_plasma, eedf.grid, ion_mass_amu, wall);
    require(boundary.converged, "Boundary phase did not converge.");

    std::map<int, Result> accepted;
    Result previous;
    bool have_previous = false;
    const std::vector<int> grids = mode == "--fast"
        ? std::vector<int>{2, 5, 10, 25}
        : std::vector<int>{50, 100, 200};

    if (mode == "--slow") {
        const fs::path seed_path = checkpoint_path(checkpoint_directory, 25);
        require(fs::exists(seed_path),
            "Slow refinement requires the accepted refinement_25 checkpoint. Run the fast test first.");
        previous = read_checkpoint(seed_path, atomic_data, boundary);
        require_physics_compatibility(previous, base_config);
        validate_result(previous, 25);
        accepted.emplace(25, previous);
        have_previous = true;
        print_timing(25, previous, true, 0.0);
    }

    double final_profile_change = 0.0;
    for (int nodes : grids) {
        const fs::path path = checkpoint_path(checkpoint_directory, nodes);
        bool loaded = false;
        Result result;
        double checkpoint_seconds = 0.0;
        if (mode == "--slow" && fs::exists(path)) {
            result = read_checkpoint(path, atomic_data, boundary);
            require_physics_compatibility(result, base_config);
            loaded = true;
        } else {
            auto config = base_config;
            config.grid.num_cells = nodes;
            auto plasma = test_dcr::make_plasma_state(
                config, atomic_data.get_total_states());
            result = dcr::solver::solve_global_target_conditioned_bvp(
                config, atomic_data, plasma, eedf.grid, boundary, wall,
                have_previous ? &previous : nullptr);
            const auto checkpoint_start = Clock::now();
            write_checkpoint(path, atomic_data, boundary, result);
            checkpoint_seconds = std::chrono::duration<double>(
                Clock::now() - checkpoint_start).count();
        }
        validate_result(result, nodes);
        const double profile_change = have_previous
            ? profile_difference(previous, result) : 0.0;
        if (have_previous) {
            require(profile_change < 0.1,
                "Grid refinement changed the profile by more than 10%.");
        }
        print_refinement(nodes, result, have_previous ? &previous : nullptr, profile_change);
        print_timing(nodes, result, loaded, checkpoint_seconds);
        accepted[nodes] = result;
        previous = result;
        have_previous = true;
        final_profile_change = profile_change;
    }

    if (mode == "--fast") {
        const Result round_trip = read_checkpoint(
            checkpoint_path(checkpoint_directory, 25), atomic_data, boundary);
        validate_result(round_trip, 25);
        require(profile_difference(previous, round_trip) == 0.0 &&
            previous.upstream_proton_flux_cm2_s == round_trip.upstream_proton_flux_cm2_s,
            "The accepted 25-node checkpoint did not round-trip exactly.");
        std::cout << "[PASS] Fast reduced refinement reached 25 nodes and wrote checkpoints."
                  << std::endl;
        return 0;
    }

    require(final_profile_change < 1.0e-3,
        "The 100-to-200 node profile change exceeds 1e-3.");
    print_runtime_ratios(accepted);

    const Result& first_200 = accepted.at(200);
    auto final_config = base_config;
    final_config.grid.num_cells = 200;
    auto final_plasma = test_dcr::make_plasma_state(
        final_config, atomic_data.get_total_states());
    const auto shifted = dcr::solver::solve_global_target_conditioned_bvp(
        final_config, atomic_data, final_plasma, eedf.grid,
        boundary, wall, &first_200, std::log(0.8));
    validate_result(shifted, 200);
    const double q_difference = relative_difference(
        shifted.upstream_proton_flux_cm2_s, first_200.upstream_proton_flux_cm2_s);
    const double population_difference =
        max_population_relative_difference(first_200, shifted);
    const auto first_rates = integrated_rates(first_200);
    const auto shifted_rates = integrated_rates(shifted);
    const double integrated_rate_difference = std::max(
        relative_difference(first_rates.source, shifted_rates.source),
        relative_difference(first_rates.sink, shifted_rates.sink));
    std::cout << "[global-bvp-branch-check]"
              << " max_population_relative_difference=" << population_difference
              << " q_up_relative_difference=" << q_difference
              << " integrated_rates_relative_difference=" << integrated_rate_difference
              << " first_acceptance_gates=true shifted_acceptance_gates=true"
              << " shifted_solve_path=" << shifted.diagnostics.solve_path
              << std::endl;
    require(q_difference < 1.0e-6,
        "The two 200-node initial guesses converged to different q_up values.");
    require(profile_difference(first_200, shifted) < 1.0e-6,
        "The two 200-node initial guesses converged to different profiles.");

    std::cout << "[PASS] Slow reduced refinement reached 200 nodes and reproduced the branch."
              << std::endl;
    return 0;
}
