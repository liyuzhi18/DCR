#pragma once

#include "MarchingDriver.hpp"
#include "../../physics/WallBoundary.hpp"
#include "../../atomic/EnergyLevel.hpp"
#include <string>
#include <vector>

namespace dcr::solver {

struct GlobalBVPSolveDiagnostics {
    std::string initialization_source;
    std::string solve_path;
    std::string jacobian_storage;
    std::string factorization_type;
    int unknown_count = 0;
    int continuation_stages = 0;
    int continuation_stage_attempts = 0;
    int newton_iterations = 0;
    int residual_evaluations = 0;
    int residual_cache_hits = 0;
    int molecular_transport_evaluations = 0;
    int molecular_transport_cache_hits = 0;
    int molecular_interval_factorizations = 0;
    int jacobian_assemblies = 0;
    int jacobian_vector_products = 0;
    int linear_solves = 0;
    int linear_iterations = 0;
    int line_search_trial_evaluations = 0;
    int jacobian_rows = 0;
    int jacobian_columns = 0;
    long long jacobian_nonzeros = 0;
    double jacobian_nonzeros_per_row = 0.0;
    long long factorization_nonzeros = 0;
    double factorization_fill_ratio = 0.0;
    double factorization_memory_bytes = 0.0;
    double reduced_condition_estimate = 0.0;
    bool direct_full_physics_attempted = false;
    bool direct_full_physics_succeeded = false;
    bool continuation_fallback_used = false;
    bool schur_complement_explicitly_assembled = false;
    bool molecular_jacobian_inverse_explicitly_formed = false;
    double molecular_transport_seconds = 0.0;
    double residual_seconds = 0.0;
    double jacobian_seconds = 0.0;
    double factorization_seconds = 0.0;
    double linear_solve_seconds = 0.0;
    double condition_seconds = 0.0;
    double logging_seconds = 0.0;
    double total_seconds = 0.0;
};

struct GlobalBVPResult {
    BoundaryPhaseResult boundary;
    MarchingHistory history;
    bool converged = false;
    int iterations = 0;
    double residual_norm = 0.0;
    double upstream_proton_flux_cm2_s = 0.0;
    double max_chemistry_nuclei_relative_error = 0.0;
    double integrated_nuclei_balance_relative_error = 0.0;
    double ion_flux_chemistry_relative_error = 0.0;
    double minimum_line_search_factor = 1.0;
    double final_ion_residual = 0.0;
    double final_flow_a_residual = 0.0;
    double final_flow_m_residual = 0.0;
    double final_local_residual = 0.0;
    double final_boundary_residual = 0.0;
    std::string velocity_profile_type;
    double ion_velocity_transition_length_cm = 0.0;
    double ion_upstream_speed_fraction = 0.0;
    double target_electron_temperature_eV = 0.0;
    double target_ion_temperature_eV = 0.0;
    double target_nuclei_density_cm3 = 0.0;
    double boundary_exhaust_width_cm = 0.0;
    double spatial_exhaust_width_cm = 0.0;
    GlobalBVPSolveDiagnostics diagnostics;
};

struct GlobalBVPCountDiagnostics {
    int nodes = 0;
    int intervals = 0;
    int background_states = 0;
    int recycling_atom_states = 0;
    int recycling_molecule_states = 0;
    int positive_ions = 0;
    int nontransported_background_states = 0;
    int unknown_count = 0;
    int residual_count = 0;
    int molecular_unknown_count = 0;
    int molecular_residual_rows = 0;
    int ion_interval_rows = 0;
    int ion_upstream_boundary_rows = 0;
    int local_background_rows = 0;
    int recycling_interval_rows = 0;
    int recycling_target_boundary_rows = 0;
    int target_density_rows = 0;
    int spatial_density_rows = 0;
    int legacy_ion_closure_rows = 0;
    int q_up_coupled_rows = 0;
    int proton_global_index = -1;
};

struct GlobalBVPForwardMTransportDiagnostics {
    std::vector<dcr::base::Vector> density;
    std::vector<dcr::base::Vector> flux;
    double expected_target_ground_flux = 0.0;
    double target_ground_flux = 0.0;
    double maximum_excited_target_flux = 0.0;
    double minimum_density = 0.0;
    double maximum_scaled_midpoint_balance = 0.0;
    double target_total_flux = 0.0;
    double upstream_total_flux = 0.0;
    bool finite_nonnegative = false;
    bool no_floor_activation = true;
};

struct GlobalBVPFDCouplingDiagnostics {
    double relative_error = 0.0;
    double molecular_profile_relative_change = 0.0;
};

struct GlobalBVPJacobianDiagnostics {
    std::vector<std::string> labels;
    std::vector<double> relative_errors;
    std::vector<double> observed_orders;
};

// Signed target-directed velocity prescribed for positive-ion transport.
double global_ion_velocity_cm_s(
    const dcr::io::Config& config,
    const dcr::atomic::EnergyLevel& level,
    double x_cm);

double conservative_flux_divergence_cm3_s(
    double velocity_left_cm_s,
    double density_left_cm3,
    double velocity_right_cm_s,
    double density_right_cm3,
    double dx_cm);

void diagnose_global_bvp_log_state(
    const dcr::io::Config& config,
    const dcr::atomic::AtomicData& atomic_data,
    const dcr::state::PlasmaState& plasma,
    const EEDFGridView& grid,
    const BoundaryPhaseResult& initial_boundary,
    const dcr::physics::WallRecycling& wall,
    const dcr::base::Vector& log_state,
    double chemistry_fraction);

GlobalBVPJacobianDiagnostics check_global_bvp_initial_jacobian(
    const dcr::io::Config& config,
    const dcr::atomic::AtomicData& atomic_data,
    const dcr::state::PlasmaState& plasma,
    const EEDFGridView& grid,
    const BoundaryPhaseResult& initial_boundary,
    const dcr::physics::WallRecycling& wall);

GlobalBVPCountDiagnostics global_bvp_count_diagnostics(
    const dcr::io::Config& config,
    const dcr::atomic::AtomicData& atomic_data,
    const BoundaryPhaseResult& boundary);

GlobalBVPForwardMTransportDiagnostics check_global_bvp_forward_m_transport(
    const dcr::io::Config& config,
    const dcr::atomic::AtomicData& atomic_data,
    const dcr::state::PlasmaState& plasma,
    const EEDFGridView& grid,
    const BoundaryPhaseResult& initial_boundary,
    const dcr::physics::WallRecycling& wall);

GlobalBVPFDCouplingDiagnostics check_global_bvp_fd_m_coupling(
    const dcr::io::Config& config,
    const dcr::atomic::AtomicData& atomic_data,
    const dcr::state::PlasmaState& plasma,
    const EEDFGridView& grid,
    const BoundaryPhaseResult& initial_boundary,
    const dcr::physics::WallRecycling& wall);

GlobalBVPResult solve_global_target_conditioned_bvp(
    const dcr::io::Config& config,
    const dcr::atomic::AtomicData& atomic_data,
    const dcr::state::PlasmaState& plasma,
    const EEDFGridView& grid,
    const BoundaryPhaseResult& initial_boundary,
    const dcr::physics::WallRecycling& wall,
    const GlobalBVPResult* coarse_solution = nullptr,
    double initial_log_shift = 0.0);

} // namespace dcr::solver
