#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cmath>
#include <fstream>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <limits>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <H5Cpp.h>

#include "atomic/AtomicData.hpp"
#include "io/ConfigLoader.hpp"
#include "physics/EnergyGrid.hpp"
#include "physics/Sheath.hpp"
#include "physics/WallBoundary.hpp"
#include "processes/AtomicProcess.hpp"
#include "processes/MolecularProcess.hpp"
#include "solver/boundary/BoundaryPhase.hpp"
#include "solver/core/AlternatingController.hpp"
#include "solver/core/ConfigPaths.hpp"
#include "solver/core/RateAnalysis.hpp"
#include "solver/core/TemperatureProfile.hpp"
#include "solver/marching/CellAdvance.hpp"
#include "solver/marching/CellSolve.hpp"
#include "solver/marching/LocalSystemAssembler.hpp"
#include "solver/marching/MarchingDriver.hpp"
#include "solver/marching/UpstreamIonBoundary.hpp"
#include "solver/output/HDF5Output.hpp"
#include "solver/qss/internal/QSSGrid.hpp"

namespace {

using dcr::base::Matrix;
using dcr::base::Vector;
using dcr::solver::UpstreamIonBoundary;

constexpr double kOuterTolerance = 1.0e-5;
constexpr double kLocalTolerance = 3.0e-7;
constexpr double kLocalResidualTolerance = 3.0e-7;
constexpr double kDensityFloor = 1.0e-60;
constexpr double kPopulationChangeScaleFloor = 1.0;
constexpr double kNewtonResidualTolerance = 3.0e-6;
constexpr double kPhysicalRowScaleFloor = 1.0e-30;
constexpr double kLogMinimum = -138.0;
constexpr double kLogMaximum = 70.0;
constexpr int kMaxOuterIterations = 100;
constexpr int kMaxLocalIterations = 40;
constexpr int kMaxNewtonIterations = 60;
constexpr int kMaxNewtonBacktracks = 20;
constexpr int kMaxNeutralBacktracks = 12;
constexpr int kMaxShootingExpansionPowers = 4;

double estimate_ion_mass_amu(const dcr::io::Config& config) {
    for (const auto& species : config.species) {
        if (!species.is_molecule && species.charge > 0 && species.mass_amu > 0.0) {
            return species.mass_amu;
        }
    }
    return 1.0;
}

Vector compact_background_from_boundary(
    const dcr::solver::BoundaryPhaseResult& boundary) {
    Vector result = Vector::Zero(static_cast<int>(boundary.P_indices.size()));
    for (int pi = 0; pi < result.size(); ++pi) {
        const int gi = boundary.P_indices[static_cast<size_t>(pi)];
        if (gi >= 0 && gi < boundary.population.size()) {
            result(pi) = std::max(boundary.population(gi), 0.0);
        }
    }
    return result;
}

Vector compact_flow_from_boundary(
    const dcr::solver::BoundaryPhaseResult& boundary,
    const std::vector<int>& indices,
    const Vector& retained_flow) {
    Vector result = Vector::Zero(static_cast<int>(indices.size()));
    if (boundary.explicit_recycling) {
        for (int i = 0; i < result.size(); ++i) {
            const int gi = indices[static_cast<size_t>(i)];
            if (gi >= 0 && gi < boundary.population.size()) {
                result(i) = std::max(boundary.population(gi), 0.0);
            }
        }
    } else if (boundary.have_flow_last && retained_flow.size() == result.size()) {
        result = retained_flow.cwiseMax(0.0);
    }
    return result;
}

struct Profiles {
    std::vector<Vector> background;
    std::vector<Vector> flowA;
    std::vector<Vector> flowM;
};

struct CellDiagnostics {
    bool converged = false;
    int replacement_pi = -1;
    int replacement_gi = -1;
    double replacement_population_fraction = 0.0;
    double exact_residual = std::numeric_limits<double>::quiet_NaN();
    double hminus_residual = std::numeric_limits<double>::quiet_NaN();
    double omitted_residual = std::numeric_limits<double>::quiet_NaN();
    double conservation_normalization = std::numeric_limits<double>::quiet_NaN();
    double conservation_epsilon = std::numeric_limits<double>::quiet_NaN();
    double hminus_normalization = 0.0;
    double hminus_epsilon = std::numeric_limits<double>::quiet_NaN();
    bool hminus_active = false;
    double omitted_normalization = 0.0;
    double omitted_epsilon = std::numeric_limits<double>::quiet_NaN();
    bool omitted_active = false;
    double raw_minimum_population = std::numeric_limits<double>::infinity();
    int raw_minimum_pi = -1;
    double final_minimum_population = std::numeric_limits<double>::infinity();
    double steady_backward_error = std::numeric_limits<double>::infinity();
    double pseudo_time_backward_error = std::numeric_limits<double>::infinity();
    double transient_to_steady_row_scale_ratio = 0.0;
    int transient_to_steady_row_scale_pi = -1;
    std::vector<double> transient_to_steady_row_scale_ratios;
    double transient_update_to_steady_row_scale_ratio = 0.0;
    double pseudo_time_physical_residual_steady_scaled = 0.0;
    double pseudo_time_state_change_norm = 0.0;
    double pseudo_time_transient_term_norm = 0.0;
    double steady_residual_norm = 0.0;
    double pseudo_time_residual_norm = 0.0;
    double be_identity_defect_norm = 0.0;
    double be_identity_defect_relative = 0.0;
    int be_identity_defect_pi = -1;
    double conservation_be_identity_defect = 0.0;
    double conservation_be_identity_defect_relative = 0.0;
};

struct SweepDiagnostics {
    bool cells_converged = true;
    double max_exact_residual = 0.0;
    double max_hminus_residual = 0.0;
    int max_exact_cell = -1;
    int max_hminus_cell = -1;
    double max_exact_normalization = 0.0;
    double epsilon_at_max_exact = 0.0;
    double max_conservation_epsilon = 0.0;
    double residual_at_max_epsilon = 0.0;
    double normalization_at_max_epsilon = 0.0;
    int max_epsilon_cell = -1;
    double max_hminus_epsilon = 0.0;
    int max_hminus_epsilon_cell = -1;
    double max_omitted_epsilon = 0.0;
    int max_omitted_epsilon_cell = -1;
    double minimum_population = std::numeric_limits<double>::infinity();
    double max_steady_backward_error = 0.0;
    int max_steady_backward_error_cell = -1;
    double max_pseudo_time_backward_error = 0.0;
    double pseudo_time_profile_change = 0.0;
    double max_transient_to_steady_row_scale_ratio = 0.0;
    int max_transient_ratio_cell = -1;
    int max_transient_ratio_pi = -1;
    int max_transient_ratio_gi = -1;
    double max_transient_update_to_steady_row_scale_ratio = 0.0;
    double max_pseudo_time_physical_residual_steady_scaled = 0.0;
    double max_pseudo_time_state_change_norm = 0.0;
    int max_pseudo_time_profile_change_cell = -1;
    int max_pseudo_time_profile_change_pi = -1;
    int max_pseudo_time_profile_change_gi = -1;
    double max_pseudo_time_profile_change_previous = 0.0;
    double max_pseudo_time_profile_change_next = 0.0;
    double max_pseudo_time_transient_term_norm = 0.0;
    double max_steady_residual_norm = 0.0;
    double max_pseudo_time_residual_norm = 0.0;
    double max_be_identity_defect_norm = 0.0;
    double max_be_identity_defect_relative = 0.0;
    int max_be_identity_defect_cell = -1;
    int max_be_identity_defect_pi = -1;
    int max_be_identity_defect_gi = -1;
    double max_conservation_be_identity_defect = 0.0;
    double max_conservation_be_identity_defect_relative = 0.0;
    std::vector<int> replacement_species_counts;
};

struct IonTrial {
    double amplitude = 0.0;
    double mismatch = 0.0;
    int evaluations = 0;
    std::vector<Vector> background;
    UpstreamIonBoundary upstream_boundary;
    SweepDiagnostics diagnostics;
    std::vector<CellDiagnostics> cell_diagnostics;
};

struct PhysicalCellSystem {
    Matrix lhs;
    Vector rhs;
    Vector row_scales;
};

struct PseudoTimeEstimatorContributor {
    int gi = -1;
    double nuclei_weight_fraction = 0.0;
    double weighted_mean_rate = 0.0;
    double maximum_rate = 0.0;
};

struct PseudoTimeStepEstimate {
    double step = std::numeric_limits<double>::infinity();
    double quantile = 0.95;
    double quantile_rate = 0.0;
    double maximum_rate = 0.0;
    int limiting_cell = -1;
    int limiting_pi = -1;
    int limiting_gi = -1;
    std::vector<PseudoTimeEstimatorContributor> dominant_contributors;
};

struct NeutralBlockStepDiagnostics {
    double modified_backward_error = 0.0;
    double steady_backward_error = 0.0;
    double transient_update_ratio = 0.0;
    double profile_change = 0.0;
    double minimum_population = std::numeric_limits<double>::infinity();
    double state_change_norm = 0.0;
    double transient_term_norm = 0.0;
    double steady_residual_norm = 0.0;
    double be_residual_norm = 0.0;
    double be_identity_defect_norm = 0.0;
    double be_identity_defect_relative = 0.0;
    int be_identity_defect_cell = -1;
    int be_identity_defect_gi = -1;
};

struct NeutralPseudoTimeSweepResult {
    Profiles profiles;
    NeutralBlockStepDiagnostics atom;
    NeutralBlockStepDiagnostics molecule;
};

struct NeutralPseudoTimeStepEstimate {
    double atom_step = std::numeric_limits<double>::infinity();
    double molecule_step = std::numeric_limits<double>::infinity();
    double atom_quantile_rate = 0.0;
    double molecule_quantile_rate = 0.0;
};

struct ContinuationPointResult {
    bool converged = false;
    bool infeasible = false;
    int outer_iterations = 0;
    double amplitude = 0.0;
    double mismatch = std::numeric_limits<double>::infinity();
    double profile_change = std::numeric_limits<double>::infinity();
    double boundary_change = std::numeric_limits<double>::infinity();
    std::string failure;
    Profiles profiles;
    IonTrial ion_trial;
    int anderson_accepted = 0;
    int anderson_rejected = 0;
    int anderson_backtracks = 0;
    int relaxed_fallbacks = 0;
    int rejected_macro_trials = 0;
    int shooting_iterations = 0;
    int first_failing_cell = -1;
    int first_failing_species = -1;
    double attempted_amplitude = std::numeric_limits<double>::quiet_NaN();
    double attempted_mismatch = std::numeric_limits<double>::infinity();
    double fixed_point_residual = std::numeric_limits<double>::infinity();
    double minimum_population = std::numeric_limits<double>::infinity();
    std::string update_type = "none";
    double elapsed_seconds = 0.0;
    double pseudo_time_initial_step = std::numeric_limits<double>::quiet_NaN();
    double pseudo_time_final_step = std::numeric_limits<double>::quiet_NaN();
    int pseudo_time_accepted_steps = 0;
    int pseudo_time_rejected_steps = 0;
    double steady_backward_error = std::numeric_limits<double>::infinity();
    double pseudo_time_backward_error = std::numeric_limits<double>::infinity();
    double pseudo_time_profile_change = std::numeric_limits<double>::infinity();
    double neutral_atom_initial_step = std::numeric_limits<double>::quiet_NaN();
    double neutral_atom_final_step = std::numeric_limits<double>::quiet_NaN();
    double neutral_molecule_initial_step = std::numeric_limits<double>::quiet_NaN();
    double neutral_molecule_final_step = std::numeric_limits<double>::quiet_NaN();
    int neutral_atom_accepted_steps = 0;
    int neutral_atom_rejected_steps = 0;
    int neutral_molecule_accepted_steps = 0;
    int neutral_molecule_rejected_steps = 0;
#ifdef DCR_NESTED_DIAGNOSTIC
    double local_plasma_step_minimum = std::numeric_limits<double>::quiet_NaN();
    double local_plasma_step_q10 = std::numeric_limits<double>::quiet_NaN();
    double local_plasma_step_median = std::numeric_limits<double>::quiet_NaN();
    double local_plasma_step_q90 = std::numeric_limits<double>::quiet_NaN();
    double local_plasma_step_maximum = std::numeric_limits<double>::quiet_NaN();
    int local_plasma_reductions = 0;
#endif
};

struct EndpointAuditResult {
    IonTrial ion_trial;
    Profiles mapped_profiles;
    double neutral_joint_residual = std::numeric_limits<double>::infinity();
    double joint_residual = std::numeric_limits<double>::infinity();
    double profile_change = std::numeric_limits<double>::infinity();
    double boundary_change = std::numeric_limits<double>::infinity();
    double minimum_population = std::numeric_limits<double>::infinity();
};

struct ContinuationCheckpoint {
    double lambda = 0.0;
    double amplitude = 0.0;
    double previous_lambda = 0.0;
    double previous_amplitude = 0.0;
    double next_step = 0.02;
    bool last_step_rejected = false;
    Profiles profiles;
    UpstreamIonBoundary upstream_boundary;
    double current_temperature = std::numeric_limits<double>::quiet_NaN();
    double previous_temperature = std::numeric_limits<double>::quiet_NaN();
    double next_temperature_step = 0.05;
    Vector previous_joint_state;
    double last_attempt_temperature = std::numeric_limits<double>::quiet_NaN();
    bool last_attempt_accepted = true;
    int last_attempt_iterations = 0;
    double last_attempt_amplitude = std::numeric_limits<double>::quiet_NaN();
    double last_attempt_mismatch = std::numeric_limits<double>::quiet_NaN();
    double last_attempt_joint_residual = std::numeric_limits<double>::quiet_NaN();
    double last_attempt_profile_change = std::numeric_limits<double>::quiet_NaN();
    double last_attempt_boundary_change = std::numeric_limits<double>::quiet_NaN();
    double last_attempt_epsilon_sigma = std::numeric_limits<double>::quiet_NaN();
    double last_attempt_epsilon_hminus = std::numeric_limits<double>::quiet_NaN();
    double last_attempt_epsilon_omitted = std::numeric_limits<double>::quiet_NaN();
    double last_attempt_minimum_population = std::numeric_limits<double>::quiet_NaN();
    int last_attempt_failing_cell = -1;
    int last_attempt_failing_species = -1;
    int last_attempt_rejected_anderson_trials = 0;
    std::string last_attempt_update_type = "none";
    std::string last_attempt_status = "not_attempted";
    bool provisional_valid = false;
    double provisional_temperature = std::numeric_limits<double>::quiet_NaN();
    double provisional_amplitude = std::numeric_limits<double>::quiet_NaN();
    Profiles provisional_profiles;
    UpstreamIonBoundary provisional_upstream_boundary;
};

std::uint64_t checkpoint_state_hash(const ContinuationCheckpoint& checkpoint) {
    std::uint64_t hash = 1469598103934665603ULL;
    auto bytes = [&](const void* data, size_t size) {
        const auto* values = static_cast<const unsigned char*>(data);
        for (size_t i = 0; i < size; ++i) {
            hash ^= values[i];
            hash *= 1099511628211ULL;
        }
    };
    auto scalar = [&](const auto& value) { bytes(&value, sizeof(value)); };
    auto vector = [&](const Vector& value) {
        const std::uint64_t size = static_cast<std::uint64_t>(value.size());
        scalar(size);
        if (size > 0) bytes(value.data(), size * sizeof(double));
    };
    auto profile = [&](const std::vector<Vector>& value) {
        const std::uint64_t size = static_cast<std::uint64_t>(value.size());
        scalar(size);
        for (const Vector& state : value) vector(state);
    };
    auto upstream = [&](const UpstreamIonBoundary& value) {
        scalar(value.position_cm);
        vector(value.positive_ion_nuclei_flux_cm2_s);
        vector(value.positive_ion_velocity_cm_s);
        vector(value.positive_ion_composition);
    };
    auto string = [&](const std::string& value) {
        const std::uint64_t size = static_cast<std::uint64_t>(value.size());
        scalar(size);
        if (size > 0) bytes(value.data(), size);
    };

    scalar(checkpoint.lambda);
    scalar(checkpoint.amplitude);
    scalar(checkpoint.previous_lambda);
    scalar(checkpoint.previous_amplitude);
    scalar(checkpoint.next_step);
    scalar(checkpoint.last_step_rejected);
    profile(checkpoint.profiles.background);
    profile(checkpoint.profiles.flowA);
    profile(checkpoint.profiles.flowM);
    upstream(checkpoint.upstream_boundary);
    scalar(checkpoint.current_temperature);
    scalar(checkpoint.previous_temperature);
    scalar(checkpoint.next_temperature_step);
    vector(checkpoint.previous_joint_state);
    scalar(checkpoint.last_attempt_temperature);
    scalar(checkpoint.last_attempt_accepted);
    scalar(checkpoint.last_attempt_iterations);
    scalar(checkpoint.last_attempt_amplitude);
    scalar(checkpoint.last_attempt_mismatch);
    scalar(checkpoint.last_attempt_joint_residual);
    scalar(checkpoint.last_attempt_profile_change);
    scalar(checkpoint.last_attempt_boundary_change);
    scalar(checkpoint.last_attempt_epsilon_sigma);
    scalar(checkpoint.last_attempt_epsilon_hminus);
    scalar(checkpoint.last_attempt_epsilon_omitted);
    scalar(checkpoint.last_attempt_minimum_population);
    scalar(checkpoint.last_attempt_failing_cell);
    scalar(checkpoint.last_attempt_failing_species);
    scalar(checkpoint.last_attempt_rejected_anderson_trials);
    string(checkpoint.last_attempt_update_type);
    string(checkpoint.last_attempt_status);
    scalar(checkpoint.provisional_valid);
    scalar(checkpoint.provisional_temperature);
    scalar(checkpoint.provisional_amplitude);
    profile(checkpoint.provisional_profiles.background);
    profile(checkpoint.provisional_profiles.flowA);
    profile(checkpoint.provisional_profiles.flowM);
    upstream(checkpoint.provisional_upstream_boundary);
    return hash;
}

struct AndersonHistoryEntry {
    Vector state;
    Vector residual;
};

struct ProfilePhysicalSummary {
    double minimum_population = std::numeric_limits<double>::infinity();
    double background_min = std::numeric_limits<double>::infinity();
    double background_max = 0.0;
    double flowA_min = std::numeric_limits<double>::infinity();
    double flowA_max = 0.0;
    double flowM_min = std::numeric_limits<double>::infinity();
    double flowM_max = 0.0;
    double total_nuclei_min = std::numeric_limits<double>::infinity();
    double total_nuclei_max = 0.0;
    int total_nuclei_min_node = -1;
    int total_nuclei_max_node = -1;
    double upstream_ion_nuclei_density = 0.0;
    double upstream_ion_nuclei_flux = 0.0;
    double target_atomic_flux = 0.0;
    double target_molecular_flux = 0.0;
};

struct MatrixDiagnostics {
    int rank = 0;
    double sigma_min = 0.0;
    double condition_number = std::numeric_limits<double>::infinity();
};

struct SolveDiagnostics {
    bool finite = false;
    int rank = 0;
    double system_relative_residual = std::numeric_limits<double>::infinity();
    double unscaled_relative_residual = std::numeric_limits<double>::infinity();
};

struct LocalIterationDiagnostics {
    int iteration = 0;
    double physical_residual_norm = 0.0;
    double physical_residual_relative = 0.0;
    double physical_residual_global_relative = 0.0;
    double linearized_post_residual_relative = 0.0;
    double nonlinear_post_residual_relative =
        std::numeric_limits<double>::infinity();
    double population_change = 0.0;
    double minimum_population = 0.0;
    double raw_solution_minimum = 0.0;
    int raw_nonpositive_count = 0;
    int floor_count = 0;
    int jacobian_rank = 0;
    int line_search_backtracks = 0;
    double accepted_step = 0.0;
    double newton_step_max = 0.0;
    double linear_solve_backward_error_before =
        std::numeric_limits<double>::infinity();
    double linear_solve_backward_error_after =
        std::numeric_limits<double>::infinity();
    int iterative_refinement_steps = 0;
    bool used_gradient_fallback = false;
    int maximum_change_index = -1;
    double maximum_change_previous = 0.0;
    double maximum_change_next = 0.0;
};

struct IntervalSolveResult {
    Vector state;
    bool converged = false;
    bool matrix_singular = false;
    bool line_search_failed = false;
    double interior_seed = kDensityFloor;
    double final_physical_residual_norm = std::numeric_limits<double>::infinity();
    double final_physical_residual_relative = std::numeric_limits<double>::infinity();
    double exact_residual = std::numeric_limits<double>::quiet_NaN();
    double hminus_residual = std::numeric_limits<double>::quiet_NaN();
    std::vector<LocalIterationDiagnostics> history;
};

struct IntervalSystem {
    Matrix lhs;
    Vector rhs;
    double hminus_residual = std::numeric_limits<double>::quiet_NaN();
};

struct SourceChannel {
    double contribution = 0.0;
    std::string label;
};

struct SpeciesSourceBreakdown {
    double production = 0.0;
    double loss = 0.0;
    std::vector<SourceChannel> channels;
};

class UnconvergedReverseCellError : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

double max_relative_change(const Profiles& next, const Profiles& previous) {
    double maximum = 0.0;
    auto visit = [&](const std::vector<Vector>& a, const std::vector<Vector>& b) {
        if (a.size() != b.size()) {
            throw std::runtime_error("Profile block sizes differ");
        }
        for (size_t k = 0; k < a.size(); ++k) {
            if (a[k].size() != b[k].size()) {
                throw std::runtime_error("Profile vector sizes differ");
            }
            if (a[k].size() == 0 || b[k].size() == 0) continue;
            const double denominator = std::max(
                kPopulationChangeScaleFloor, a[k].cwiseAbs().maxCoeff());
            maximum = std::max(
                maximum, (a[k] - b[k]).cwiseAbs().maxCoeff() / denominator);
        }
    };
    visit(next.background, previous.background);
    visit(next.flowA, previous.flowA);
    visit(next.flowM, previous.flowM);
    return maximum;
}

Matrix extract_block(const Matrix& full,
                     const std::vector<int>& rows,
                     const std::vector<int>& columns) {
    Matrix result = Matrix::Zero(static_cast<int>(rows.size()),
                                 static_cast<int>(columns.size()));
    for (int i = 0; i < result.rows(); ++i) {
        const int gi = rows[static_cast<size_t>(i)];
        if (gi < 0 || gi >= full.rows()) continue;
        for (int j = 0; j < result.cols(); ++j) {
            const int gj = columns[static_cast<size_t>(j)];
            if (gj >= 0 && gj < full.cols()) result(i, j) = full(gi, gj);
        }
    }
    return result;
}

void write_csv_field(std::ostream& output, const std::string& value) {
    output << '"';
    for (char character : value) {
        if (character == '"') output << '"';
        output << character;
    }
    output << '"';
}

double weighted_sum(const Vector& values,
                    const std::vector<int>& indices,
                    const std::vector<dcr::atomic::EnergyLevel>& levels) {
    double result = 0.0;
    const int count = std::min<int>(values.size(), static_cast<int>(indices.size()));
    for (int i = 0; i < count; ++i) {
        const int gi = indices[static_cast<size_t>(i)];
        if (gi < 0 || gi >= static_cast<int>(levels.size())) continue;
        result += std::max(1, levels[static_cast<size_t>(gi)].atomicity) * values(i);
    }
    return result;
}

std::string process_kind(const dcr::ProcessBase& process) {
    using namespace crm_detail;
    if (dynamic_cast<const AtomicExcitationProcess*>(&process)) return "atomic_excitation";
    if (dynamic_cast<const AtomicPhotoProcess*>(&process)) return "atomic_photo";
    if (dynamic_cast<const AtomicIonizationProcess*>(&process)) return "atomic_ionization";
    if (dynamic_cast<const AtomicAutoionizationProcess*>(&process)) return "atomic_autoionization";
    if (dynamic_cast<const MolecularVEProcess*>(&process)) return "molecular_VE";
    if (dynamic_cast<const MolecularExcitationProcess*>(&process)) return "molecular_excitation";
    if (dynamic_cast<const MolecularMIProcess*>(&process)) return "molecular_MI";
    if (dynamic_cast<const MolecularDRProcess*>(&process)) return "molecular_DR";
    if (dynamic_cast<const MolecularRAProcess*>(&process)) return "molecular_RA";
    if (dynamic_cast<const MolecularDAProcess*>(&process)) return "molecular_DA";
    if (dynamic_cast<const MolecularDEProcess*>(&process)) return "molecular_DE";
    if (dynamic_cast<const MolecularEDProcess*>(&process)) return "molecular_ED";
    if (dynamic_cast<const MolecularMCXProcess*>(&process)) return "molecular_MCX";
    if (dynamic_cast<const MolecularMIDEProcess*>(&process)) return "molecular_MIDE";
    return "unknown_process";
}

class AlternatingSweepPrototype {
public:
    AlternatingSweepPrototype(dcr::io::Config config,
                              dcr::atomic::AtomicData& atomic_data,
                              dcr::state::PlasmaState& plasma,
                              const EEDFGridView& grid,
                              dcr::solver::BoundaryPhaseResult boundary,
                              dcr::physics::WallRecycling wall,
                              std::string checkpoint_input_path = {},
                              std::string artifact_stem_override = {})
        : config_(std::move(config)),
          atomic_data_(atomic_data),
          plasma_(plasma),
          grid_(grid),
          boundary_(std::move(boundary)),
          wall_(wall),
          levels_(atomic_data_.get_levels()),
          total_states_(atomic_data_.get_total_states()),
          relaxation_(std::clamp(config_.numerics.alternating_sweep_relaxation,
                                 1.0e-6, 1.0)) {
        initialize_layout();
        initialize_grid();
        rate_cache_ = dcr::solver::build_local_rate_cache(
            config_, atomic_data_, plasma_, grid_, 0.0, config_.plasma.total_density);
        std::string case_suffix = std::abs(config_.plasma.Te_eV - 1.0) < 1.0e-12
            ? "" : "_T" + std::to_string(static_cast<int>(std::lround(config_.plasma.Te_eV)));
        const double velocity_length =
            config_.numerics.adaptive_recycling_domain.ion_velocity_length_cm > 0.0
            ? config_.numerics.adaptive_recycling_domain.ion_velocity_length_cm
            : config_.grid.length_cm;
        const double upstream_velocity_fraction = std::max(
            config_.numerics.adaptive_recycling_domain.ion_velocity_floor_fraction,
            1.0 - config_.grid.length_cm / velocity_length);
        if (std::abs(upstream_velocity_fraction - 0.05) > 1.0e-8) {
            case_suffix += "_u0p" + std::to_string(
                static_cast<int>(std::lround(10.0 * upstream_velocity_fraction)));
        }
        const std::filesystem::path default_output_directory =
            config_.io.output_dir.empty()
            ? std::filesystem::path("output")
            : std::filesystem::path(config_.io.output_dir);
        const std::string stem = artifact_stem_override.empty()
            ? (default_output_directory /
               ("alternating_continuation" + case_suffix)).string()
            : std::move(artifact_stem_override);
        const std::filesystem::path stem_path(stem);
        if (!stem_path.parent_path().empty()) {
            std::filesystem::create_directories(stem_path.parent_path());
        }
        continuation_checkpoint_path_ = stem + "_checkpoint.bin";
        continuation_checkpoint_temporary_path_ = stem + "_checkpoint.tmp";
        continuation_summary_path_ = stem + "_stages.csv";
        reference_artifact_stem_ = stem;
        checkpoint_input_path_ = checkpoint_input_path.empty()
            ? continuation_checkpoint_path_ : std::move(checkpoint_input_path);
    }

    int run(double continuation_step_override =
                std::numeric_limits<double>::quiet_NaN(),
            bool reconverge_current_lambda = false,
            bool audit_lambda_one_endpoint = false,
            bool solve_physical_reference = false,
            double checkpoint_velocity_fraction =
                std::numeric_limits<double>::quiet_NaN()) {
        validate_case();
        ContinuationCheckpoint checkpoint;
        const bool checkpoint_loaded = load_continuation_checkpoint(checkpoint);
        if (audit_lambda_one_endpoint || solve_physical_reference) {
            if (std::isfinite(continuation_step_override) ||
                reconverge_current_lambda ||
                std::isfinite(checkpoint_velocity_fraction)) {
                throw std::runtime_error(
                    "Endpoint modes cannot be combined with continuation options");
            }
            if (!checkpoint_loaded) {
                throw std::runtime_error("Endpoint checkpoint is unavailable");
            }
            if (audit_lambda_one_endpoint && solve_physical_reference) {
                throw std::runtime_error(
                    "Endpoint audit and physical-reference solve are mutually exclusive");
            }
            if (solve_physical_reference) {
                return run_physical_reference_case(std::move(checkpoint));
            }
            return run_lambda_one_endpoint_audit(checkpoint);
        }
        if (checkpoint_loaded && checkpoint.lambda >= 0.04) {
            if (std::isfinite(continuation_step_override)) {
                if (!(continuation_step_override > 0.0) ||
                    !std::isfinite(continuation_step_override)) {
                    throw std::runtime_error("Continuation step override must be positive");
                }
                checkpoint.next_step = continuation_step_override;
                checkpoint.last_step_rejected = false;
                if (!save_continuation_checkpoint(checkpoint)) {
                    throw std::runtime_error("Could not save continuation step override");
                }
                std::cout << std::scientific << std::setprecision(12)
                          << "CONTINUATION_STEP_RESET next_step="
                          << checkpoint.next_step << "\n";
            }
            std::cout << std::scientific << std::setprecision(12)
                      << "CONTINUATION_RESUME lambda_rec=" << checkpoint.lambda
                      << " amplitude=" << checkpoint.amplitude
                      << " previous_lambda=" << checkpoint.previous_lambda
                      << " previous_amplitude=" << checkpoint.previous_amplitude
                      << " next_step=" << checkpoint.next_step
                       << " checkpoint=" << std::quoted(continuation_checkpoint_path_)
                       << "\n";
            if (reconverge_current_lambda) {
                const Vector composition =
                    checkpoint.upstream_boundary.positive_ion_composition;
                Profiles profiles = checkpoint.profiles;
                double amplitude = checkpoint.amplitude;
                const double target_velocity_length =
                    config_.numerics.adaptive_recycling_domain.ion_velocity_length_cm;
                const double target_velocity_fraction = std::max(
                    config_.numerics.adaptive_recycling_domain.ion_velocity_floor_fraction,
                    1.0 - config_.grid.length_cm / target_velocity_length);
                double velocity_fraction = std::isfinite(checkpoint_velocity_fraction)
                    ? checkpoint_velocity_fraction : 0.05;
                if (!(velocity_fraction > 0.0) || velocity_fraction >= 1.0) {
                    throw std::runtime_error(
                        "Checkpoint velocity fraction must lie strictly between zero and one");
                }
                double velocity_step = 0.05;
                ContinuationPointResult point;
                while (std::abs(velocity_fraction - target_velocity_fraction) > 1.0e-12) {
                    const double direction = target_velocity_fraction > velocity_fraction
                        ? 1.0 : -1.0;
                    const double attempted_fraction = velocity_fraction + direction *
                        std::min(velocity_step,
                                 std::abs(target_velocity_fraction - velocity_fraction));
                    config_.numerics.adaptive_recycling_domain.ion_velocity_length_cm =
                        std::abs(attempted_fraction - target_velocity_fraction) <= 1.0e-12
                        ? target_velocity_length
                        : config_.grid.length_cm / (1.0 - attempted_fraction);
                    frozen_replacement_rows_ = dominant_replacement_pattern(profiles);
                    point = solve_recycling_continuation_point_anderson(
                        profiles, composition, checkpoint.lambda, amplitude, amplitude,
                        std::abs(attempted_fraction - target_velocity_fraction) <= 1.0e-12,
                        500);
                    std::cout << "VELOCITY_HOMOTOPY_STAGE"
                              << " accepted=" << (point.converged ? 1 : 0)
                              << " attempted_uL_over_uB=" << attempted_fraction
                              << " step=" << velocity_step
                              << " coupled_iterations=" << point.outer_iterations
                              << " failure=" << std::quoted(point.failure)
                              << "\n";
                    if (!point.converged) {
                        velocity_step *= 0.5;
                        if (velocity_step < 1.0e-3) {
                            config_.numerics.adaptive_recycling_domain.ion_velocity_length_cm =
                                target_velocity_length;
                            std::cout << "FIXED_LAMBDA_RECONVERGENCE converged=0"
                                      << " lambda_rec=" << checkpoint.lambda
                                      << " failure=\"velocity_step_below_minimum\""
                                      << " uL_over_uB=" << velocity_fraction
                                      << "\n";
                            return 2;
                        }
                        continue;
                    }
                    profiles = point.profiles;
                    amplitude = point.amplitude;
                    velocity_fraction = attempted_fraction;
                    checkpoint.amplitude = amplitude;
                    checkpoint.previous_lambda = checkpoint.lambda;
                    checkpoint.previous_amplitude = amplitude;
                    checkpoint.last_step_rejected = false;
                    checkpoint.profiles = profiles;
                    checkpoint.upstream_boundary = point.ion_trial.upstream_boundary;
                    if (!save_continuation_checkpoint(checkpoint)) {
                        throw std::runtime_error(
                            "Could not save accepted velocity-homotopy checkpoint");
                    }
                    if (point.outer_iterations <= 15) velocity_step *= 1.5;
                }
                config_.numerics.adaptive_recycling_domain.ion_velocity_length_cm =
                    target_velocity_length;
                checkpoint.amplitude = point.amplitude;
                checkpoint.previous_lambda = checkpoint.lambda;
                checkpoint.previous_amplitude = point.amplitude;
                checkpoint.next_step = 0.02;
                checkpoint.last_step_rejected = false;
                checkpoint.profiles = point.profiles;
                checkpoint.upstream_boundary = point.ion_trial.upstream_boundary;
                if (!save_continuation_checkpoint(checkpoint)) {
                    throw std::runtime_error(
                        "Could not save fixed-lambda reconverged checkpoint");
                }
                initialize_stage_summary();
                const ProfilePhysicalSummary physical = summarize_profiles(
                    point.profiles, &point.ion_trial.upstream_boundary);
                append_stage_summary(
                    "accepted", checkpoint.lambda, checkpoint.lambda, 0.0,
                    "velocity_warm_start", point, &physical);
                std::cout << "FIXED_LAMBDA_RECONVERGENCE converged=1"
                          << " lambda_rec=" << checkpoint.lambda
                          << " amplitude=" << point.amplitude
                          << " mismatch=" << point.mismatch
                          << " profile_change=" << point.profile_change
                          << " boundary_change=" << point.boundary_change
                          << " epsilon_Sigma_max="
                          << point.ion_trial.diagnostics.max_conservation_epsilon
                          << " coupled_iterations=" << point.outer_iterations
                          << " checkpoint=" << std::quoted(continuation_checkpoint_path_)
                          << "\n";
                if (std::abs(checkpoint.lambda - 1.0) <= 1.0e-14) {
                    return run_lambda_one_endpoint_audit(std::move(checkpoint));
                }
                return 0;
            }
            return run_automatic_continuation(checkpoint);
        }
        return run_recycling_base_case();
    }

    int run_outer_iteration() {
        validate_case();
        Profiles profiles = initial_profiles();
        double amplitude = initial_shooting_amplitude(profiles.background.front());
        const Vector ion_composition = upstream_ion_composition(profiles.background.front());

        bool converged = false;
        double final_mismatch = std::numeric_limits<double>::infinity();
        double final_profile_change = std::numeric_limits<double>::infinity();
        double final_amplitude = amplitude;
        SweepDiagnostics final_sweep;
        int completed_iterations = 0;
        std::string failure_reason;

        std::cout << std::scientific << std::setprecision(12);
        std::cout << "prototype=alternating_sweep"
                  << " relaxation=" << relaxation_
                  << " nodes=" << x_.size()
                  << " target_density=" << config_.plasma.total_density
                  << " initial_amplitude=" << amplitude << "\n";

        Profiles initial_profiles_before_shooting = profiles;
        IonTrial accepted_ion;
        try {
            accepted_ion = shoot_ion_amplitude(profiles, amplitude, ion_composition);
        } catch (const std::exception& error) {
            failure_reason = std::string("Initial ion shooting failed: ") + error.what();
            const SweepDiagnostics residuals = evaluate_residuals(
                profiles, make_upstream_ion_boundary(amplitude, ion_composition));
            report(profiles, false, 0, amplitude, final_mismatch,
                   final_profile_change, residuals, failure_reason);
            return 2;
        }
        profiles.background = accepted_ion.background;
        amplitude = accepted_ion.amplitude;
        final_amplitude = amplitude;
        final_mismatch = accepted_ion.mismatch;
        final_profile_change = max_relative_change(profiles, initial_profiles_before_shooting);
        final_sweep = accepted_ion.diagnostics;
        std::cout << "initial_shot amplitude=" << amplitude
                  << " target_mismatch=" << final_mismatch
                  << " shooting_evaluations=" << accepted_ion.evaluations
                  << " profile_change=" << final_profile_change
                  << " cells_converged="
                  << (accepted_ion.diagnostics.cells_converged ? 1 : 0) << "\n";

        for (int outer = 1; outer <= kMaxOuterIterations; ++outer) {
            completed_iterations = outer;
            current_outer_iteration_ = outer;
            const Profiles previous = profiles;
            const Profiles neutral_candidate = sweep_recycled_neutrals(profiles);
            bool update_accepted = false;
            bool local_cell_failure = false;
            int accepted_backtracks = 0;
            double trial_relaxation = relaxation_;
            IonTrial ion_trial;
            Profiles trial_profiles;
            std::string last_rejection;
            for (int backtrack = 0; backtrack <= kMaxNeutralBacktracks; ++backtrack) {
                trial_profiles = profiles;
                relax_neutral_profiles(
                    trial_profiles, neutral_candidate, trial_relaxation);
                try {
                    ion_trial = shoot_ion_amplitude(
                        trial_profiles, amplitude, ion_composition);
                    trial_profiles.background = ion_trial.background;
                    update_accepted = true;
                    accepted_backtracks = backtrack;
                    break;
                } catch (const UnconvergedReverseCellError& error) {
                    failure_reason = error.what();
                    local_cell_failure = true;
                    break;
                } catch (const std::exception& error) {
                    last_rejection = error.what();
                    std::cout << "neutral_backtrack outer=" << outer
                              << " attempt=" << backtrack
                              << " rejected_relaxation=" << trial_relaxation
                              << " reason=" << last_rejection << "\n";
                    trial_relaxation *= 0.5;
                }
            }
            if (local_cell_failure) break;
            if (!update_accepted) {
                failure_reason = "Neutral backtracking exhausted: " + last_rejection;
                break;
            }

            profiles = std::move(trial_profiles);
            relaxation_ = trial_relaxation;
            amplitude = ion_trial.amplitude;
            final_amplitude = amplitude;
            final_profile_change = max_relative_change(profiles, previous);
            final_mismatch = ion_trial.mismatch;
            final_sweep = ion_trial.diagnostics;
            const SweepDiagnostics& sweep = ion_trial.diagnostics;

            const double target_flux = ion_nuclei_flux(profiles.background.front(), 0.0);
            const double upstream_flux = ion_trial.upstream_boundary.
                positive_ion_nuclei_flux_cm2_s.sum();
            std::cout << "outer=" << outer
                      << " amplitude=" << amplitude
                      << " target_flux=" << target_flux
                      << " upstream_flux=" << upstream_flux
                      << " target_mismatch=" << final_mismatch
                      << " shooting_mismatch=" << ion_trial.mismatch
                      << " shooting_evaluations=" << ion_trial.evaluations
                      << " neutral_relaxation=" << relaxation_
                      << " neutral_backtracks=" << accepted_backtracks
                      << " profile_change=" << final_profile_change
                      << " exact_R_sigma=" << sweep.max_exact_residual
                       << " physical_Hminus_residual=" << sweep.max_hminus_residual
                      << " cells_converged=" << (sweep.cells_converged ? 1 : 0)
                      << "\n";

            if (std::abs(final_mismatch) < kOuterTolerance &&
                final_profile_change < kOuterTolerance && sweep.cells_converged) {
                converged = true;
                break;
            }

        }

        const SweepDiagnostics residuals = evaluate_residuals(
            profiles, make_upstream_ion_boundary(final_amplitude, ion_composition));
        report(profiles, converged, completed_iterations, final_amplitude,
               final_mismatch, final_profile_change, residuals, failure_reason);
        return converged ? 0 : 2;
    }

    bool load_temperature_continuation_checkpoint(
        ContinuationCheckpoint& checkpoint) const {
        return load_continuation_checkpoint(checkpoint);
    }

    Vector temperature_joint_state(const ContinuationCheckpoint& checkpoint) const {
        Vector state(pack_recycled_profiles(checkpoint.profiles).size() + 1);
        state(0) = std::log(checkpoint.amplitude);
        state.tail(state.size() - 1) = pack_recycled_profiles(checkpoint.profiles);
        return state;
    }

    Vector temperature_predictor(const ContinuationCheckpoint& checkpoint,
                                 double attempted_temperature) const {
        const Vector current = temperature_joint_state(checkpoint);
        Vector predictor = current;
        const double span = checkpoint.current_temperature -
            checkpoint.previous_temperature;
        if (span > 0.0 && checkpoint.previous_joint_state.size() == current.size()) {
            predictor += (attempted_temperature - checkpoint.current_temperature) /
                span * (current - checkpoint.previous_joint_state);
        }
        auto valid = [&](const Vector& candidate) {
            return candidate.size() == current.size() && candidate.allFinite() &&
                std::isfinite(std::exp(candidate(0))) &&
                candidate.tail(candidate.size() - 1).minCoeff() >= 0.0;
        };
        if (valid(predictor)) return predictor;
        for (double damping = 0.5; damping >= 1.0 / 1024.0; damping *= 0.5) {
            Vector damped = current + damping * (predictor - current);
            if (valid(damped)) return damped;
        }
        return current;
    }

    ContinuationPointResult solve_temperature_stage(
        const ContinuationCheckpoint& checkpoint,
        const Vector& predictor,
        EndpointAuditResult& accepted_audit,
        bool use_pseudo_time = false,
        bool plain_backward_euler_once = false,
        bool use_neutral_pseudo_time = false,
        const ContinuationCheckpoint* accepted_checkpoint_context = nullptr,
        bool fixed_amplitude_diagnostic = false,
        int diagnostic_max_inner_macro_iterations = 2000,
        int diagnostic_stagnation_window = 100,
        double diagnostic_neutral_molecule_step_scale = 1.0,
        bool fixed_gamma_activity_controller = false,
        double activity_trace_protection_fraction = 1.0e-3) {
        frozen_replacement_rows_ = dominant_replacement_pattern(checkpoint.profiles);
        return solve_recycling_continuation_point_anderson(
            checkpoint.profiles,
            checkpoint.upstream_boundary.positive_ion_composition,
            1.0, checkpoint.amplitude, checkpoint.amplitude, true,
             500, &predictor, &accepted_audit, use_pseudo_time,
             plain_backward_euler_once, use_neutral_pseudo_time,
             accepted_checkpoint_context != nullptr
                 ? accepted_checkpoint_context : &checkpoint,
              fixed_amplitude_diagnostic,
               diagnostic_max_inner_macro_iterations,
               diagnostic_stagnation_window,
               diagnostic_neutral_molecule_step_scale,
               fixed_gamma_activity_controller,
               activity_trace_protection_fraction);
    }

    EndpointAuditResult audit_temperature_stage(
        const ContinuationPointResult& point) const {
        ContinuationCheckpoint candidate;
        candidate.lambda = 1.0;
        candidate.amplitude = point.amplitude;
        candidate.profiles = point.profiles;
        candidate.upstream_boundary = point.ion_trial.upstream_boundary;
        return perform_lambda_one_endpoint_audit(candidate);
    }

    EndpointAuditResult audit_temperature_checkpoint(
        const ContinuationCheckpoint& checkpoint) const {
        return perform_lambda_one_endpoint_audit(checkpoint);
    }

    bool temperature_audit_accepted(const EndpointAuditResult& audit) const {
        return endpoint_audit_accepted(audit);
    }

    bool temperature_inner_audit_accepted(
        const EndpointAuditResult& audit) const {
        return endpoint_inner_audit_accepted(audit);
    }

    void print_temperature_cell34_molecular_diagnostic(
        const Profiles& profiles,
        const char* phase) const {
        constexpr int center = 34;
        if (profiles.flowM.size() <= static_cast<size_t>(center + 1) ||
            profiles.background.size() <= static_cast<size_t>(center + 1) ||
            x_.size() <= static_cast<size_t>(center + 1) ||
            dx_.size() <= static_cast<size_t>(center)) {
            std::cout << "CELL34_MOLECULAR_DIAGNOSTIC"
                      << " phase=" << phase
                      << " available=0 reason=profile_layout\n";
            return;
        }
        dcr::solver::AtomicRateCalculator rate_calculator(atomic_data_);
        const double injected_molecules = std::max(
            profiles.flowM.front().sum(), 1.0e-300);
        int layer_index = -1;
        for (size_t node = 0;
             node < profiles.flowM.size() && node < x_.size(); ++node) {
            if (profiles.flowM[node].sum() / injected_molecules < 0.01) {
                layer_index = static_cast<int>(node);
                break;
            }
        }
        std::cout << "CELL34_MOLECULAR_LAYER"
                  << " phase=" << phase
                  << " cell=34 x_cm=" << x_[center]
                  << " x_over_L=" << x_[center] / config_.grid.length_cm
                  << " layer_threshold=1e-2"
                  << " layer_index=" << layer_index
                  << " layer_x_cm=" << (layer_index >= 0
                      ? x_[static_cast<size_t>(layer_index)]
                      : std::numeric_limits<double>::quiet_NaN())
                  << "\n";

        for (int node = center - 1; node <= center + 1; ++node) {
            const Vector background_full = dcr::solver::make_background_full(
                profiles.background[static_cast<size_t>(node)],
                boundary_, total_states_);
            const auto local = dcr::solver::assemble_local_system(
                config_, atomic_data_, plasma_, grid_, boundary_,
                background_full, profiles.flowA[static_cast<size_t>(node)],
                profiles.flowM[static_cast<size_t>(node)], x_[node], false,
                &rate_cache_);
            const auto chemistry =
                dcr::solver::assemble_local_chemistry_sources(
                    local, boundary_, atomic_data_, background_full,
                    profiles.flowA[static_cast<size_t>(node)],
                    profiles.flowM[static_cast<size_t>(node)]);
            const Vector zero_molecule = Vector::Zero(
                profiles.flowM[static_cast<size_t>(node)].size());
            const auto no_molecule_chemistry =
                dcr::solver::assemble_local_chemistry_sources(
                    local, boundary_, atomic_data_, background_full,
                    profiles.flowA[static_cast<size_t>(node)], zero_molecule);
            const Vector molecular_plasma_source =
                chemistry.background - no_molecule_chemistry.background;

            double gross_molecular_production = 0.0;
            double gross_molecular_loss = 0.0;
            for (size_t local_row = 0;
                 local_row < boundary_.M_indices.size(); ++local_row) {
                const int gi = boundary_.M_indices[local_row];
                for (size_t local_column = 0;
                     local_column < boundary_.M_indices.size(); ++local_column) {
                    const int gj = boundary_.M_indices[local_column];
                    const double contribution = local.R_full(gi, gj) *
                        profiles.flowM[static_cast<size_t>(node)](
                            static_cast<int>(local_column));
                    if (contribution >= 0.0) {
                        gross_molecular_production += contribution;
                    } else {
                        gross_molecular_loss -= contribution;
                    }
                }
            }
            const auto rates = rate_calculator.evaluate(
                config_, boundary_, local, background_full, x_[node],
                plasma_, grid_);
            const double molecule_density =
                profiles.flowM[static_cast<size_t>(node)].sum();
            const double flow_fraction = molecule_density / injected_molecules;
            const double dx_before = dx_[static_cast<size_t>(node - 1)];
            const double transport_time =
                dx_before / std::abs(boundary_.u_M);
            const double previous_density =
                profiles.flowM[static_cast<size_t>(node - 1)].sum();
            const double deposition_rate = std::max(
                0.0, -boundary_.u_M *
                    (molecule_density - previous_density) / dx_before);
            const double effective_rate = deposition_rate /
                std::max(molecule_density, 1.0e-300);
            const double diagnostic_damkohler =
                effective_rate * transport_time;
            std::cout << "CELL34_MOLECULAR_DIAGNOSTIC"
                      << " phase=" << phase
                      << " available=1 cell=" << node
                      << " x_cm=" << x_[node]
                      << " x_over_L=" << x_[node] / config_.grid.length_cm
                      << " molecule_density_cm-3=" << molecule_density
                      << " flow_fraction=" << flow_fraction
                      << " chemistry_net_cm-3_s="
                      << chemistry.flowM.sum()
                      << " chemistry_gross_production_cm-3_s="
                      << gross_molecular_production
                      << " chemistry_gross_loss_cm-3_s="
                      << gross_molecular_loss
                      << " exhaust_loss_cm-3_s="
                      << molecule_exhaust_rate_ * molecule_density
                      << " molecular_plasma_source_nuclei_cm-3_s="
                      << muP_.dot(molecular_plasma_source)
                      << " molecular_plasma_source_max_cm-3_s="
                      << molecular_plasma_source.cwiseAbs().maxCoeff()
                      << " molecular_flow_ionization_cm-3_s="
                      << rates.atomic_sources
                          .molecular_flow_ionization_rate_cm3_s
                      << " molecular_flow_charge_exchange_cm-3_s="
                      << rates.atomic_sources
                          .molecular_flow_charge_exchange_rate_cm3_s
                      << " molecular_dissociation_cm-3_s="
                      << rates.atomic_sources
                          .molecular_dissociation_rate_cm3_s
                      << " dx_before_cm=" << dx_before
                      << " u_M_cm_s=" << boundary_.u_M
                      << " dx_over_uM_s=" << transport_time
                      << " diagnostic_nu_M_eff_s-1=" << effective_rate
                      << " diagnostic_Da_M=" << diagnostic_damkohler
                      << " diagnostic_Da_definition="
                      << std::quoted("transport_attenuation_estimate")
                      << "\n";
        }
    }

#ifdef DCR_NESTED_DIAGNOSTIC
    void print_saved_molecular_residual_diagnostic(
        const ContinuationCheckpoint& checkpoint) const {
        if (!checkpoint.provisional_valid ||
            checkpoint.provisional_profiles.background.empty()) {
            throw std::runtime_error(
                "Molecular residual postprocessor requires a provisional state");
        }
        const Profiles& state = checkpoint.provisional_profiles;
        const Profiles image = sweep_recycled_neutrals(state, 1.0);
        const double nu_ref = estimate_neutral_pseudo_time_steps(state, 1.0)
            .molecule_quantile_rate;
        const double n_nuc_ref = std::max(1.0, config_.plasma.total_density);

        struct RowDiagnostic {
            int node = -1;
            int gi = -1;
            int mu = 1;
            double x_cm = 0.0;
            double population = 0.0;
            double nuclei_density = 0.0;
            double image_population = 0.0;
            double map_defect = 0.0;
            double map_scale = 1.0;
            double map_residual = 0.0;
            double raw_steady_residual = 0.0;
            double physical_activity = 0.0;
            double physical_scaled_residual = 0.0;
            double dominant_production = 0.0;
            double dominant_loss = 0.0;
            std::string dominant_production_term;
            std::string dominant_loss_term;
        };

        std::vector<RowDiagnostic> rows;
        rows.reserve(state.background.size() * boundary_.M_indices.size());
        auto add_row = [&](int node,
                           int local_row,
                           double raw_steady_residual,
                           double physical_activity,
                           double dominant_production,
                           const std::string& dominant_production_term,
                           double dominant_loss,
                           const std::string& dominant_loss_term) {
            RowDiagnostic row;
            row.node = node;
            row.gi = boundary_.M_indices[static_cast<size_t>(local_row)];
            row.mu = std::max(
                1, levels_[static_cast<size_t>(row.gi)].atomicity);
            row.x_cm = x_[static_cast<size_t>(node)];
            row.population = state.flowM[static_cast<size_t>(node)](local_row);
            row.nuclei_density = total_nuclei(
                state, static_cast<size_t>(node));
            row.image_population =
                image.flowM[static_cast<size_t>(node)](local_row);
            row.map_defect = row.image_population - row.population;
            row.map_scale = std::max({
                1.0, std::abs(row.population),
                std::abs(row.image_population)});
            row.map_residual = std::abs(row.map_defect) / row.map_scale;
            row.raw_steady_residual = raw_steady_residual;
            row.physical_activity = physical_activity;
            row.physical_scaled_residual = std::abs(raw_steady_residual) /
                std::max(physical_activity, 1.0e-300);
            row.dominant_production = dominant_production;
            row.dominant_loss = dominant_loss;
            row.dominant_production_term = dominant_production_term;
            row.dominant_loss_term = dominant_loss_term;
            rows.push_back(std::move(row));
        };

        const double boundary_rate = boundary_.u_M / dx_.front();
        for (int local_row = 0;
             local_row < state.flowM.front().size(); ++local_row) {
            const double current = state.flowM.front()(local_row);
            const double target = image.flowM.front()(local_row);
            add_row(
                0, local_row, boundary_rate * (current - target),
                boundary_rate * (std::abs(current) + std::abs(target)),
                boundary_rate * std::abs(target), "boundary_inflow",
                boundary_rate * std::abs(current), "boundary_outflow");
        }

        for (size_t interval = 0;
             interval + 1 < state.background.size(); ++interval) {
            const size_t node = interval + 1;
            const Vector background_full = dcr::solver::make_background_full(
                state.background[node], boundary_, total_states_);
            const auto local = dcr::solver::assemble_local_system(
                config_, atomic_data_, plasma_, grid_, boundary_,
                background_full, image.flowA[interval], image.flowM[interval],
                x_[node], false, &rate_cache_);
            const auto advanced = dcr::solver::advance_recycling_flow_one_step(
                config_, atomic_data_, boundary_, local,
                image.flowA[interval], image.flowM[interval], dx_[interval]);
            const Matrix R_MM = extract_block(
                local.R_full, boundary_.M_indices, boundary_.M_indices);
            const Vector& current = state.flowM[node];
            const Vector physical_residual =
                advanced.diagnosticsM.steady_lhs * current -
                advanced.diagnosticsM.steady_rhs;
            const double spatial_rate = boundary_.u_M / dx_[interval];
            const double exhaust_rate = advanced.c_s_M /
                std::max(config_.grid.spatial_exhaust_width_cm, 1.0e-12);

            for (int row_index = 0; row_index < current.size(); ++row_index) {
                double activity = std::abs(
                    advanced.diagnosticsM.steady_rhs(row_index));
                for (int column = 0;
                     column < current.size(); ++column) {
                    activity += std::abs(
                        advanced.diagnosticsM.steady_lhs(row_index, column) *
                        current(column));
                }

                double dominant_production = spatial_rate *
                    std::abs(image.flowM[interval](row_index));
                std::string production_term = "spatial_inflow";
                double dominant_loss = spatial_rate *
                    std::abs(current(row_index));
                std::string loss_term = "spatial_outflow";
                const double exhaust_loss = exhaust_rate *
                    std::abs(current(row_index));
                if (exhaust_loss > dominant_loss) {
                    dominant_loss = exhaust_loss;
                    loss_term = "spatial_exhaust";
                }
                for (int column = 0;
                     column < current.size(); ++column) {
                    const double contribution =
                        R_MM(row_index, column) * current(column);
                    const int source_gi = boundary_.M_indices[
                        static_cast<size_t>(column)];
                    if (contribution > dominant_production) {
                        dominant_production = contribution;
                        production_term = "chemistry_from_" +
                            species_label(source_gi);
                    }
                    if (-contribution > dominant_loss) {
                        dominant_loss = -contribution;
                        loss_term = "chemistry_from_" +
                            species_label(source_gi);
                    }
                }
                add_row(
                    static_cast<int>(node), row_index,
                    physical_residual(row_index), activity,
                    dominant_production, production_term,
                    dominant_loss, loss_term);
            }
        }

        std::sort(rows.begin(), rows.end(), [](const auto& left, const auto& right) {
            return left.map_residual > right.map_residual;
        });
        const double reported_residual = rows.empty()
            ? 0.0 : rows.front().map_residual;
        std::cout << std::scientific << std::setprecision(12)
                  << "MOLECULAR_RESIDUAL_POSTPROCESS"
                  << " checkpoint_temperature="
                  << checkpoint.provisional_temperature
                  << " rows=" << rows.size()
                  << " current_norm_definition=" << std::quoted(
                      "max_abs(image-state)/max(1,abs(image),abs(state))")
                  << " M_residual=" << reported_residual
                  << " n_nuc_ref=" << n_nuc_ref
                  << " nu_ref=" << nu_ref
                  << " nu_ref_definition=" << std::quoted(
                      "molecular_nuclei_weighted_q95_row_activity_rate")
                  << "\n";

        const size_t count = std::min<size_t>(10, rows.size());
        for (size_t rank = 0; rank < count; ++rank) {
            const RowDiagnostic& row = rows[rank];
            std::cout << "MOLECULAR_RESIDUAL_ROW"
                      << " rank=" << rank + 1
                      << " cell=" << row.node
                      << " x_cm=" << row.x_cm
                      << " gi=" << row.gi
                      << " state=" << std::quoted(species_label(row.gi))
                      << " mu_s=" << row.mu
                      << " n_M=" << row.population
                      << " n_M_over_n_nuc=" << row.population /
                          std::max(row.nuclei_density, 1.0e-300)
                      << " image_n_M=" << row.image_population
                      << " map_raw_defect=" << row.map_defect
                      << " current_map_scale=" << row.map_scale
                      << " current_map_scaled_residual=" << row.map_residual
                      << " raw_steady_residual=" << row.raw_steady_residual
                      << " physical_row_activity_scale="
                      << row.physical_activity
                      << " physical_scaled_residual="
                      << row.physical_scaled_residual
                      << " dominant_production_term="
                      << std::quoted(row.dominant_production_term)
                      << " dominant_production=" << row.dominant_production
                      << " dominant_loss_term="
                      << std::quoted(row.dominant_loss_term)
                      << " dominant_loss=" << row.dominant_loss
                      << "\n";
        }

        for (double epsilon_abs : {1.0e-8, 1.0e-10, 1.0e-12, 1.0e-14}) {
            double hybrid_residual = 0.0;
            int limiting_cell = -1;
            int limiting_gi = -1;
            for (const RowDiagnostic& row : rows) {
                const double weighted_residual =
                    row.mu * std::abs(row.raw_steady_residual);
                const double weighted_activity =
                    row.mu * row.physical_activity;
                const double absolute_scale = epsilon_abs * row.mu *
                    n_nuc_ref * nu_ref;
                const double scaled = weighted_residual /
                    std::max({weighted_activity, absolute_scale, 1.0e-300});
                if (scaled > hybrid_residual) {
                    hybrid_residual = scaled;
                    limiting_cell = row.node;
                    limiting_gi = row.gi;
                }
            }
            std::cout << "MOLECULAR_HYBRID_RESIDUAL"
                      << " epsilon_abs=" << epsilon_abs
                      << " M_residual=" << hybrid_residual
                      << " limiting_cell=" << limiting_cell
                      << " limiting_gi=" << limiting_gi
                      << " limiting_state=" << std::quoted(
                          limiting_gi >= 0 ? species_label(limiting_gi) : "none")
                      << "\n";
        }
    }

    void print_saved_activity_spectrum_diagnostic(
        const ContinuationCheckpoint& checkpoint,
        bool run_trace_deweighting_scan = false,
        double protection_fraction = 1.0e-3) const {
        const Profiles& state = checkpoint.provisional_profiles;
        const double nu_ref = estimate_neutral_pseudo_time_steps(state, 1.0)
            .molecule_quantile_rate;
        const double n_nuc_ref = std::max(1.0, config_.plasma.total_density);
        constexpr double important_fraction_of_species_peak = 1.0e-6;
        const std::vector<std::string> protected_pathway_names{
            "target_nuclei_density",
            "sheath_entrance_ion_flux",
            "recycled_atomic_boundary_flux",
            "recycled_molecular_boundary_flux",
            "total_DR_source",
            "MCX_source",
            "EIR_source",
            "MAR_source"};

        struct ActivityRow {
            std::string block;
            int cell = -1;
            int gi = -1;
            int mu = 1;
            double x_cm = 0.0;
            double population = 0.0;
            double population_fraction = 0.0;
            double raw_residual = 0.0;
            double activity = 0.0;
            double pathway_contribution = 0.0;
            double dominant_production = 0.0;
            double dominant_loss = 0.0;
            std::string production_term;
            std::string loss_term;
            std::vector<double> protected_pathway_contributions;
            bool important = false;
        };
        std::vector<ActivityRow> rows;
        size_t unavailable_A = 0;
        size_t unavailable_M = 0;

        struct H2PlusLayerRow {
            int cell = -1;
            double x_cm = 0.0;
            double recycled_h2_nuclei = 0.0;
            double recycled_h2_deposition = 0.0;
            double h2plus_nuclei = 0.0;
            double molecular_ionization = 0.0;
            double molecular_charge_exchange = 0.0;
            double dr_mar_h_source = 0.0;
            double positive_ion_nuclei = 0.0;
            double positive_ion_profile_change = 0.0;
        };
        std::vector<H2PlusLayerRow> h2plus_layer;
        h2plus_layer.reserve(state.background.size());

        auto append = [&](const char* block,
                          int cell,
                          int gi,
                          double population,
                          double raw_residual,
                          double activity,
                          double pathway_contribution,
                          double dominant_production,
                          const std::string& production_term,
                          double dominant_loss,
                          const std::string& loss_term) {
            ActivityRow row;
            row.block = block;
            row.cell = cell;
            row.gi = gi;
            row.mu = std::max(
                1, levels_[static_cast<size_t>(gi)].atomicity);
            row.x_cm = x_[static_cast<size_t>(cell)];
            row.population = population;
            row.population_fraction = population /
                std::max(total_nuclei(state, static_cast<size_t>(cell)), 1.0e-300);
            row.raw_residual = row.mu * raw_residual;
            row.activity = row.mu * activity;
            row.pathway_contribution = row.mu * pathway_contribution;
            row.dominant_production = row.mu * dominant_production;
            row.dominant_loss = row.mu * dominant_loss;
            row.production_term = production_term;
            row.loss_term = loss_term;
            row.protected_pathway_contributions.assign(
                protected_pathway_names.size(), 0.0);
            rows.push_back(std::move(row));
        };

        dcr::solver::AtomicRateCalculator rate_calculator(atomic_data_);
        for (size_t cell = 0; cell < state.background.size(); ++cell) {
            const Vector* upstream = cell + 1 < state.background.size()
                ? &state.background[cell + 1] : nullptr;
            const UpstreamIonBoundary* upstream_face = upstream == nullptr
                ? &checkpoint.provisional_upstream_boundary : nullptr;
            const PhysicalCellSystem physical = assemble_physical_cell_system(
                static_cast<int>(cell), state.background[cell], upstream,
                upstream_face, state.flowA, state.flowM);
            const Vector raw = physical.lhs * state.background[cell] - physical.rhs;
            const Vector background_full = dcr::solver::make_background_full(
                state.background[cell], boundary_, total_states_);
            const auto local = dcr::solver::assemble_local_system(
                config_, atomic_data_, plasma_, grid_, boundary_, background_full,
                state.flowA[cell], state.flowM[cell], x_[cell], false, &rate_cache_);
            const auto rate_snapshot = rate_calculator.evaluate(
                config_, boundary_, local, background_full, x_[cell], plasma_, grid_);
            const Matrix& rates = local.R_source_full;
            const double dx = dx_[cell];

            H2PlusLayerRow layer_row;
            layer_row.cell = static_cast<int>(cell);
            layer_row.x_cm = x_[cell];
            for (int mi = 0; mi < state.flowM[cell].size(); ++mi) {
                const int gi = boundary_.M_indices[static_cast<size_t>(mi)];
                layer_row.recycled_h2_nuclei +=
                    levels_[static_cast<size_t>(gi)].atomicity * state.flowM[cell](mi);
            }
            if (cell > 0) {
                double preceding_h2_nuclei = 0.0;
                for (int mi = 0; mi < state.flowM[cell - 1].size(); ++mi) {
                    const int gi = boundary_.M_indices[static_cast<size_t>(mi)];
                    preceding_h2_nuclei += levels_[static_cast<size_t>(gi)].atomicity *
                        state.flowM[cell - 1](mi);
                }
                layer_row.recycled_h2_deposition = boundary_.u_M /
                    dx_[cell - 1] * std::abs(
                        preceding_h2_nuclei - layer_row.recycled_h2_nuclei);
            }
            for (int pi = 0; pi < state.background[cell].size(); ++pi) {
                const int gi = boundary_.P_indices[static_cast<size_t>(pi)];
                const auto& level = levels_[static_cast<size_t>(gi)];
                if (level.charge > 0) {
                    layer_row.positive_ion_nuclei +=
                        level.atomicity * state.background[cell](pi);
                }
                if (level.charge > 0 && level.atomicity == 2) {
                    layer_row.h2plus_nuclei +=
                        level.atomicity * state.background[cell](pi);
                }
            }
            const auto& ion_transport = rate_snapshot.molecular_ion_transport;
            const double mcx_total =
                ion_transport.mcx_production_cm3_s.sum();
            const double mi_total =
                ion_transport.mi_production_cm3_s.sum();
            const double mar_branch_fraction = mcx_total /
                std::max(mcx_total + mi_total, 1.0e-300);
            layer_row.molecular_ionization =
                ion_transport.mi_production_cm3_s.sum();
            layer_row.molecular_charge_exchange =
                ion_transport.mcx_production_cm3_s.sum();
            layer_row.dr_mar_h_source =
                ion_transport.simple_branching_dr_mar_h_source_cm3_s;
            h2plus_layer.push_back(layer_row);

            for (int pi = 0; pi < state.background[cell].size(); ++pi) {
                const int gi = boundary_.P_indices[static_cast<size_t>(pi)];
                const auto& level = levels_[static_cast<size_t>(gi)];
                double production = 0.0;
                double loss = 0.0;
                std::string production_term = "none";
                std::string loss_term = "none";
                for (int gj = 0;
                     gj < rates.cols() && gj < local.population_for_source.size(); ++gj) {
                    const double contribution =
                        rates(gi, gj) * local.population_for_source(gj);
                    if (contribution > production) {
                        production = contribution;
                        production_term = "chemistry_from_" + species_label(gj);
                    }
                    if (-contribution > loss) {
                        loss = -contribution;
                        loss_term = "chemistry_from_" + species_label(gj);
                    }
                }
                if (is_positive_ion(pi)) {
                    const double outgoing = ion_speed(gi, x_[cell]) *
                        state.background[cell](pi) / dx;
                    double incoming = 0.0;
                    if (upstream != nullptr) {
                        incoming = ion_speed(gi, x_[cell + 1]) *
                            (*upstream)(pi) / dx;
                    } else {
                        incoming = dcr::solver::
                            upstream_face_positive_ion_incoming_term(
                                upstream_face->positive_ion_nuclei_flux_cm2_s(pi),
                                muP_(pi), dx);
                    }
                    if (incoming > production) {
                        production = incoming;
                        production_term = "ion_transport_inflow";
                    }
                    if (outgoing > loss) {
                        loss = outgoing;
                        loss_term = "ion_transport_outflow";
                    }
                } else {
                    const double exhaust = background_exhaust_rate(pi) *
                        state.background[cell](pi);
                    if (exhaust > loss) {
                        loss = exhaust;
                        loss_term = "background_exhaust";
                    }
                }

                double pathway = std::max(production, loss);
                if (level.atomicity == 2 && level.charge > 0) {
                    const auto& transport = rate_snapshot.molecular_ion_transport;
                    for (int i = 0;
                         i < static_cast<int>(transport.state_indices.size()); ++i) {
                        if (transport.state_indices[static_cast<size_t>(i)] != gi) continue;
                        const double mcx = transport.mcx_production_cm3_s(i);
                        const double mi = transport.mi_production_cm3_s(i);
                        const double dr = state.background[cell](pi) *
                            transport.dr_h_source_frequency_s(i);
                        pathway = std::max({mcx, mi, dr});
                        if (mcx >= mi && mcx >= dr) {
                            production_term = "MCX_H2plus_production";
                            production = std::max(production, mcx);
                        } else if (mi >= dr) {
                            production_term = "MI_H2plus_production";
                            production = std::max(production, mi);
                        } else {
                            loss_term = "DR_MAR_H_source";
                            loss = std::max(loss, dr);
                        }
                        break;
                    }
                } else if ((level.type == dcr::atomic::SpeciesType::Ion &&
                            level.charge < 0 && level.atomicity == 1) ||
                           (level.type == dcr::atomic::SpeciesType::Atom &&
                            level.charge == 0 && level.internal_id >= 8)) {
                    const SpeciesSourceBreakdown breakdown = source_breakdown(
                        pi, local, state.background[cell], state.flowA[cell],
                        state.flowM[cell], x_[cell]);
                    pathway = 0.0;
                    for (const SourceChannel& channel : breakdown.channels) {
                        const bool relevant = level.charge < 0
                            ? (channel.label.find("molecular_DA") != std::string::npos ||
                               channel.label.find("molecular_RA") != std::string::npos ||
                               channel.label.find("molecular_ED") != std::string::npos)
                            : channel.label.find("molecular_DR") != std::string::npos;
                        if (relevant && std::abs(channel.contribution) > pathway) {
                            pathway = std::abs(channel.contribution);
                            if (channel.contribution > 0.0) {
                                production_term = channel.label;
                                production = std::max(production, channel.contribution);
                            } else {
                                loss_term = channel.label;
                                loss = std::max(loss, -channel.contribution);
                            }
                        }
                    }
                }
                append(
                    "P", static_cast<int>(cell), gi,
                    state.background[cell](pi), raw(pi),
                    std::max({physical.row_scales(pi), production, loss}),
                    pathway, production, production_term, loss, loss_term);
                ActivityRow& appended_row = rows.back();
                if (cell == 0) {
                    appended_row.protected_pathway_contributions[0] =
                        appended_row.mu * state.background[cell](pi);
                    if (level.charge > 0) {
                        appended_row.protected_pathway_contributions[1] =
                            appended_row.mu * ion_speed(gi, x_[cell]) *
                            state.background[cell](pi);
                    }
                }
                if (rate_snapshot.atomic_effective.valid &&
                    gi == rate_snapshot.atomic_effective.ion_ground_index) {
                    appended_row.protected_pathway_contributions[6] =
                        dx * rate_snapshot.atomic_sources.effective_eir_rate_cm3_s;
                }
                if (level.atomicity == 2 && level.charge > 0) {
                    for (int i = 0;
                         i < static_cast<int>(ion_transport.state_indices.size()); ++i) {
                        if (ion_transport.state_indices[static_cast<size_t>(i)] != gi) {
                            continue;
                        }
                        const double dr_source = state.background[cell](pi) *
                            ion_transport.dr_h_source_frequency_s(i);
                        appended_row.protected_pathway_contributions[4] =
                            dx * dr_source;
                        appended_row.protected_pathway_contributions[5] =
                            dx * ion_transport.mcx_production_cm3_s(i);
                        appended_row.protected_pathway_contributions[7] =
                            dx * mar_branch_fraction * dr_source;
                        break;
                    }
                }

                if (cell < 138 || cell > 147 ||
                    level.atomicity != 2 || level.charge <= 0 ||
                    level.internal_id < 1 || level.internal_id > 6) {
                    continue;
                }

                struct ExactTerm {
                    double contribution = 0.0;
                    std::string source_block;
                    int donor_gi = -1;
                };
                std::vector<ExactTerm> exact_terms;
                auto add_exact_terms = [&](const char* source_block,
                                           const std::vector<int>& indices,
                                           const Vector& populations) {
                    for (int j = 0; j < populations.size(); ++j) {
                        const int gj = indices[static_cast<size_t>(j)];
                        const double contribution = rates(gi, gj) * populations(j);
                        if (contribution != 0.0) {
                            exact_terms.push_back({contribution, source_block, gj});
                        }
                    }
                };
                add_exact_terms("P", boundary_.P_indices, state.background[cell]);
                add_exact_terms("A", boundary_.A_indices, state.flowA[cell]);
                add_exact_terms("M", boundary_.M_indices, state.flowM[cell]);

                double chemistry_production = 0.0;
                double chemistry_destruction = 0.0;
                double recycled_A_net = 0.0;
                double recycled_M_net = 0.0;
                double P_h2plus_net = 0.0;
                double P_background_h2_net = 0.0;
                double P_atomic_plasma_net = 0.0;
                double exact_activity = 0.0;
                for (const ExactTerm& term : exact_terms) {
                    chemistry_production += std::max(term.contribution, 0.0);
                    chemistry_destruction += std::max(-term.contribution, 0.0);
                    exact_activity = std::max(exact_activity, std::abs(term.contribution));
                    if (term.source_block == "A") {
                        recycled_A_net += term.contribution;
                    } else if (term.source_block == "M") {
                        recycled_M_net += term.contribution;
                    } else {
                        const auto& donor = levels_[static_cast<size_t>(term.donor_gi)];
                        if (donor.atomicity == 2 && donor.charge > 0) {
                            P_h2plus_net += term.contribution;
                        } else if (donor.type == dcr::atomic::SpeciesType::Molecule &&
                                   donor.charge == 0) {
                            P_background_h2_net += term.contribution;
                        } else {
                            P_atomic_plasma_net += term.contribution;
                        }
                    }
                }

                const double transport_out = ion_speed(gi, x_[cell]) *
                    state.background[cell](pi) / dx;
                const double transport_in = upstream != nullptr
                    ? ion_speed(gi, x_[cell + 1]) * (*upstream)(pi) / dx
                    : dcr::solver::upstream_face_positive_ion_incoming_term(
                        upstream_face->positive_ion_nuclei_flux_cm2_s(pi),
                        muP_(pi), dx);
                exact_activity = std::max(
                    {exact_activity, transport_in, transport_out, 1.0e-300});
                const double exact_residual = transport_out + chemistry_destruction -
                    transport_in - chemistry_production;
                const double weighted_activity = level.atomicity * exact_activity;
                const double hybrid_denominator = std::max(
                    weighted_activity,
                    1.0e-14 * level.atomicity * n_nuc_ref * nu_ref);
                const double hybrid_residual =
                    level.atomicity * std::abs(exact_residual) / hybrid_denominator;

                double mcx = 0.0;
                double mi = 0.0;
                double dr_mar_h_source_equivalent = 0.0;
                for (int i = 0;
                     i < static_cast<int>(ion_transport.state_indices.size()); ++i) {
                    if (ion_transport.state_indices[static_cast<size_t>(i)] != gi) continue;
                    mcx = ion_transport.mcx_production_cm3_s(i);
                    mi = ion_transport.mi_production_cm3_s(i);
                    dr_mar_h_source_equivalent = state.background[cell](pi) *
                        ion_transport.dr_h_source_frequency_s(i);
                    break;
                }

                std::cout << "H2PLUS_ROW_BALANCE"
                          << " cell=" << cell
                          << " x_cm=" << x_[cell]
                          << " gi=" << gi
                          << " state=" << std::quoted(species_label(gi))
                          << " population=" << state.background[cell](pi)
                          << " n_over_n_nuc=" << state.background[cell](pi) /
                              std::max(total_nuclei(state, cell), 1.0e-300)
                          << " raw_steady_residual=" << exact_residual
                          << " nuclei_weighted_raw_residual=" <<
                              level.atomicity * exact_residual
                          << " physical_activity=" << exact_activity
                          << " nuclei_weighted_physical_activity=" << weighted_activity
                          << " hybrid_scaled_residual=" << hybrid_residual
                          << " transport_in=" << transport_in
                          << " transport_out=" << transport_out
                          << " chemistry_production=" << chemistry_production
                          << " chemistry_destruction=" << chemistry_destruction
                          << " positive_side_out_plus_destruction=" <<
                              transport_out + chemistry_destruction
                          << " negative_side_in_plus_production=" <<
                              transport_in + chemistry_production
                          << " recycled_A_net=" << recycled_A_net
                          << " recycled_M_net=" << recycled_M_net
                          << " P_h2plus_net=" << P_h2plus_net
                          << " P_background_h2_net=" << P_background_h2_net
                          << " P_atomic_plasma_net=" << P_atomic_plasma_net
                          << " MCX_production=" << mcx
                          << " MI_production=" << mi
                          << " DR_MAR_H_source_equivalent=" <<
                              dr_mar_h_source_equivalent
                          << " decomposition_defect=" << exact_residual - raw(pi)
                          << "\n";

                std::sort(exact_terms.begin(), exact_terms.end(),
                    [](const ExactTerm& lhs, const ExactTerm& rhs) {
                        return std::abs(lhs.contribution) > std::abs(rhs.contribution);
                    });
                int printed_terms = 0;
                for (const ExactTerm& term : exact_terms) {
                    if (printed_terms >= 12 ||
                        std::abs(term.contribution) < 1.0e-3 * exact_activity) {
                        break;
                    }
                    std::cout << "H2PLUS_DOMINANT_TERM"
                              << " cell=" << cell
                              << " gi=" << gi
                              << " state=" << std::quoted(species_label(gi))
                              << " rank=" << (printed_terms + 1)
                              << " role=" << (term.contribution > 0.0
                                  ? "production" : "destruction")
                              << " source_block=" << term.source_block
                              << " donor_gi=" << term.donor_gi
                              << " donor=" << std::quoted(
                                  species_label(term.donor_gi))
                              << " signed_contribution=" << term.contribution
                              << " fraction_of_activity=" <<
                                  std::abs(term.contribution) / exact_activity
                              << "\n";
                    ++printed_terms;
                }
            }
        }

        for (size_t cell = 1; cell < h2plus_layer.size(); ++cell) {
            h2plus_layer[cell].positive_ion_profile_change = std::abs(
                h2plus_layer[cell].positive_ion_nuclei -
                h2plus_layer[cell - 1].positive_ion_nuclei) /
                std::max({h2plus_layer[cell].positive_ion_nuclei,
                          h2plus_layer[cell - 1].positive_ion_nuclei, 1.0});
        }
        auto print_layer_peak = [&](const char* metric, auto value) {
            const auto peak = std::max_element(
                h2plus_layer.begin(), h2plus_layer.end(),
                [&](const H2PlusLayerRow& lhs, const H2PlusLayerRow& rhs) {
                    return value(lhs) < value(rhs);
                });
            std::cout << "H2PLUS_LAYER_PEAK"
                      << " metric=" << metric
                      << " cell=" << peak->cell
                      << " x_cm=" << peak->x_cm
                      << " value=" << value(*peak)
                      << "\n";
        };
        print_layer_peak("recycled_H2_nuclei",
            [](const H2PlusLayerRow& row) { return row.recycled_h2_nuclei; });
        print_layer_peak("recycled_H2_deposition",
            [](const H2PlusLayerRow& row) { return row.recycled_h2_deposition; });
        print_layer_peak("H2plus_nuclei",
            [](const H2PlusLayerRow& row) { return row.h2plus_nuclei; });
        print_layer_peak("molecular_ionization",
            [](const H2PlusLayerRow& row) { return row.molecular_ionization; });
        print_layer_peak("molecular_charge_exchange",
            [](const H2PlusLayerRow& row) { return row.molecular_charge_exchange; });
        print_layer_peak("DR_MAR_H_source",
            [](const H2PlusLayerRow& row) { return row.dr_mar_h_source; });
        print_layer_peak("positive_ion_profile_change",
            [](const H2PlusLayerRow& row) { return row.positive_ion_profile_change; });
        for (const H2PlusLayerRow& row : h2plus_layer) {
            if (row.cell < 138 || row.cell > 147) continue;
            std::cout << "H2PLUS_LAYER_PROFILE"
                      << " cell=" << row.cell
                      << " x_cm=" << row.x_cm
                      << " recycled_H2_nuclei=" << row.recycled_h2_nuclei
                      << " recycled_H2_deposition=" << row.recycled_h2_deposition
                      << " H2plus_nuclei=" << row.h2plus_nuclei
                      << " molecular_ionization=" << row.molecular_ionization
                      << " molecular_charge_exchange=" <<
                          row.molecular_charge_exchange
                      << " DR_MAR_H_source=" << row.dr_mar_h_source
                      << " positive_ion_nuclei=" << row.positive_ion_nuclei
                      << " positive_ion_profile_change=" <<
                          row.positive_ion_profile_change
                      << "\n";
        }

        auto [target_A, target_M] = target_recycling(state.background.front());
        auto append_boundary_block = [&](const char* block,
                                         const Vector& current,
                                         const Vector& target,
                                         const std::vector<int>& indices,
                                         double velocity) {
            const double rate = velocity / dx_.front();
            for (int row = 0; row < current.size(); ++row) {
                append(
                    block, 0, indices[static_cast<size_t>(row)], current(row),
                    rate * (current(row) - target(row)),
                    std::max(rate * std::abs(current(row)),
                             rate * std::abs(target(row))),
                    std::max(rate * std::abs(current(row)),
                             rate * std::abs(target(row))),
                    rate * std::abs(target(row)), "boundary_inflow",
                    rate * std::abs(current(row)), "boundary_outflow");
                ActivityRow& appended_row = rows.back();
                appended_row.protected_pathway_contributions[0] =
                    appended_row.mu * current(row);
                const size_t pathway = std::string(block) == "A" ? 2 : 3;
                appended_row.protected_pathway_contributions[pathway] =
                    appended_row.mu * velocity * current(row);
            }
        };
        append_boundary_block(
            "A", state.flowA.front(), target_A, boundary_.A_indices, boundary_.u_A);
        append_boundary_block(
            "M", state.flowM.front(), target_M, boundary_.M_indices, boundary_.u_M);

        for (size_t node = 1; node < state.background.size(); ++node) {
            const Vector background_full = dcr::solver::make_background_full(
                state.background[node], boundary_, total_states_);
            const auto local = dcr::solver::assemble_local_system(
                config_, atomic_data_, plasma_, grid_, boundary_, background_full,
                state.flowA[node - 1], state.flowM[node - 1], x_[node],
                false, &rate_cache_);
            const auto advanced = dcr::solver::advance_recycling_flow_one_step(
                config_, atomic_data_, boundary_, local,
                state.flowA[node - 1], state.flowM[node - 1], dx_[node - 1]);

            auto append_flow_block = [&](const char* block,
                                         const Vector& inflow,
                                         const Vector& current,
                                         const std::vector<int>& indices,
                                         const Matrix& rates,
                                         const dcr::solver::RecyclingBlockStepDiagnostics& diagnostics,
                                         double velocity,
                                         double exhaust_rate) {
                const Vector raw = diagnostics.steady_lhs * current -
                    diagnostics.steady_rhs;
                const double spatial_rate = velocity / dx_[node - 1];
                for (int row = 0; row < current.size(); ++row) {
                    double production = spatial_rate * std::abs(inflow(row));
                    double loss = spatial_rate * std::abs(current(row));
                    std::string production_term = "spatial_inflow";
                    std::string loss_term = "spatial_outflow";
                    const double exhaust = exhaust_rate * std::abs(current(row));
                    if (exhaust > loss) {
                        loss = exhaust;
                        loss_term = "spatial_exhaust";
                    }
                    for (int column = 0; column < current.size(); ++column) {
                        const double contribution = rates(row, column) * current(column);
                        const int donor = indices[static_cast<size_t>(column)];
                        if (contribution > production) {
                            production = contribution;
                            production_term = "chemistry_from_" + species_label(donor);
                        }
                        if (-contribution > loss) {
                            loss = -contribution;
                            loss_term = "chemistry_from_" + species_label(donor);
                        }
                    }
                    append(
                        block, static_cast<int>(node),
                        indices[static_cast<size_t>(row)], current(row), raw(row),
                        std::max(production, loss), std::max(production, loss),
                        production, production_term, loss, loss_term);
                }
            };
            append_flow_block(
                "A", state.flowA[node - 1], state.flowA[node],
                boundary_.A_indices,
                extract_block(local.R_full, boundary_.A_indices, boundary_.A_indices),
                advanced.diagnosticsA, boundary_.u_A,
                advanced.c_s_A /
                    std::max(config_.grid.spatial_exhaust_width_cm, 1.0e-12));
            append_flow_block(
                "M", state.flowM[node - 1], state.flowM[node],
                boundary_.M_indices,
                extract_block(local.R_full, boundary_.M_indices, boundary_.M_indices),
                advanced.diagnosticsM, boundary_.u_M,
                advanced.c_s_M /
                    std::max(config_.grid.spatial_exhaust_width_cm, 1.0e-12));
        }
        unavailable_A = state.flowA.back().size();
        unavailable_M = state.flowM.back().size();

        std::vector<double> activities;
        activities.reserve(rows.size());
        for (const ActivityRow& row : rows) {
            if (row.activity > 0.0 && std::isfinite(row.activity)) {
                activities.push_back(row.activity);
            }
        }
        std::sort(activities.begin(), activities.end());
        auto quantile = [&](double probability) {
            if (activities.empty()) return 0.0;
            const size_t index = static_cast<size_t>(std::floor(
                probability * static_cast<double>(activities.size() - 1)));
            return activities[index];
        };
        size_t gap_index = 0;
        double gap_ratio = 1.0;
        const size_t gap_margin = std::max<size_t>(1, activities.size() / 100);
        for (size_t i = gap_margin;
             i + gap_margin < activities.size(); ++i) {
            const double ratio = activities[i] /
                std::max(activities[i - 1], 1.0e-300);
            if (ratio > gap_ratio) {
                gap_ratio = ratio;
                gap_index = i;
            }
        }
        std::cout << std::scientific << std::setprecision(12)
                  << "ACTIVITY_SPECTRUM"
                  << " rows=" << rows.size()
                  << " unavailable_A=" << unavailable_A
                  << " unavailable_M=" << unavailable_M
                  << " minimum_nonzero=" << quantile(0.0)
                  << " q01=" << quantile(0.01)
                  << " q05=" << quantile(0.05)
                  << " q10=" << quantile(0.10)
                  << " median=" << quantile(0.50)
                  << " q90=" << quantile(0.90)
                  << " q95=" << quantile(0.95)
                  << " q99=" << quantile(0.99)
                  << " maximum=" << quantile(1.0)
                  << " largest_internal_gap_ratio=" << gap_ratio
                  << " gap_lower=" << (gap_index > 0
                      ? activities[gap_index - 1] : 0.0)
                  << " gap_upper=" << (gap_index < activities.size()
                      ? activities[gap_index] : 0.0)
                  << " clear_two_decade_gap=" << (gap_ratio >= 100.0 ? 1 : 0)
                  << " n_nuc_ref=" << n_nuc_ref
                  << " nu_ref=" << nu_ref
                  << " activity_definition=" << std::quoted(
                      "mu_s*max(transport_in,transport_out,exhaust,production,loss)")
                  << "\n";

        struct TraceGroup {
            std::string block;
            int gi = -1;
        };
        std::vector<TraceGroup> trace_groups;
        for (const char* block : {"P", "A", "M"}) {
            for (int gi = 0; gi < total_states_; ++gi) {
                const auto& level = levels_[static_cast<size_t>(gi)];
                const bool named_background_trace = std::string(block) == "P" && (
                    (level.type == dcr::atomic::SpeciesType::Ion && level.charge < 0) ||
                    (level.type == dcr::atomic::SpeciesType::Ion &&
                     level.charge > 0 && level.atomicity == 2) ||
                    (level.type == dcr::atomic::SpeciesType::Atom &&
                     level.charge == 0 && level.internal_id >= 8) ||
                    (level.type == dcr::atomic::SpeciesType::Molecule &&
                     level.charge == 0));
                double maximum_fraction = 0.0;
                bool represented = false;
                for (const ActivityRow& row : rows) {
                    if (row.block == block && row.gi == gi) {
                        represented = true;
                        maximum_fraction = std::max(
                            maximum_fraction, row.population_fraction);
                    }
                }
                const bool recycled_trace =
                    represented && (std::string(block) == "A" ||
                                    std::string(block) == "M");
                if (represented && (named_background_trace || recycled_trace ||
                    (maximum_fraction > 0.0 && maximum_fraction < 1.0e-8))) {
                    trace_groups.push_back({block, gi});
                }
            }
        }

        for (const TraceGroup& group : trace_groups) {
            double peak_pathway = 0.0;
            for (const ActivityRow& row : rows) {
                if (row.block == group.block && row.gi == group.gi) {
                    peak_pathway = std::max(
                        peak_pathway, row.pathway_contribution);
                }
            }
            const double important_threshold =
                important_fraction_of_species_peak * peak_pathway;
            for (ActivityRow& row : rows) {
                if (row.block == group.block && row.gi == group.gi &&
                    row.pathway_contribution >= important_threshold &&
                    row.pathway_contribution > 0.0) {
                    row.important = true;
                }
            }
        }

        for (const TraceGroup& group : trace_groups) {
            const ActivityRow* minimum_important = nullptr;
            double minimum_population = std::numeric_limits<double>::infinity();
            double maximum_population = 0.0;
            double minimum_fraction = std::numeric_limits<double>::infinity();
            double maximum_fraction = 0.0;
            double minimum_activity = std::numeric_limits<double>::infinity();
            double maximum_activity = 0.0;
            int important_count = 0;
            for (const ActivityRow& row : rows) {
                if (row.block != group.block || row.gi != group.gi) continue;
                minimum_population = std::min(minimum_population, row.population);
                maximum_population = std::max(maximum_population, row.population);
                minimum_fraction = std::min(minimum_fraction, row.population_fraction);
                maximum_fraction = std::max(maximum_fraction, row.population_fraction);
                minimum_activity = std::min(minimum_activity, row.activity);
                maximum_activity = std::max(maximum_activity, row.activity);
                if (row.important &&
                    (minimum_important == nullptr ||
                     row.activity < minimum_important->activity)) {
                    minimum_important = &row;
                }
                if (row.important) ++important_count;
            }
            if (minimum_important == nullptr) continue;
            const int gi = group.gi;
            const auto& level = levels_[static_cast<size_t>(gi)];
            const char* family =
                group.block == "M" ? "local_recycled_H2" :
                group.block == "A" && level.internal_id >= 8
                ? "high_n_recycled_atomic_H" :
                group.block == "A" ? "recycled_atomic_H" :
                (level.type == dcr::atomic::SpeciesType::Ion && level.charge < 0)
                ? "Hminus_DA_MN" :
                (level.type == dcr::atomic::SpeciesType::Ion &&
                 level.charge > 0 && level.atomicity == 2)
                ? "H2plus_MCX_MI_DR_MAR" :
                (level.type == dcr::atomic::SpeciesType::Atom &&
                 level.internal_id >= 8)
                ? "high_n_atomic_H" :
                (level.type == dcr::atomic::SpeciesType::Molecule &&
                  level.charge == 0)
                ? "background_H2" : "other_low_population_state";
            std::cout << "TRACE_ACTIVITY"
                      << " family=" << family
                      << " block=" << group.block
                      << " gi=" << gi
                      << " state=" << std::quoted(species_label(gi))
                      << " important_definition=" << std::quoted(
                          "pathway_contribution>=1e-6*species_peak")
                      << " important_rows=" << important_count
                      << " population_min=" << minimum_population
                      << " population_max=" << maximum_population
                      << " fraction_min=" << minimum_fraction
                      << " fraction_max=" << maximum_fraction
                      << " activity_min=" << minimum_activity
                      << " activity_max=" << maximum_activity
                      << " minimum_important_activity="
                      << minimum_important->activity
                      << " minimum_important_cell=" << minimum_important->cell
                      << " minimum_important_x_cm=" << minimum_important->x_cm
                      << " minimum_important_fraction="
                      << minimum_important->population_fraction
                      << " minimum_important_pathway_contribution="
                      << minimum_important->pathway_contribution
                      << " dominant_production="
                      << std::quoted(minimum_important->production_term)
                      << " dominant_loss="
                      << std::quoted(minimum_important->loss_term)
                      << "\n";
        }

        for (double epsilon_abs : {1.0e-8, 1.0e-10, 1.0e-12, 1.0e-14}) {
            int important_below_floor = 0;
            double largest_pathway_below_floor = 0.0;
            for (const ActivityRow& row : rows) {
                if (!row.important) continue;
                const double floor = epsilon_abs * row.mu * n_nuc_ref * nu_ref;
                if (row.activity < floor) {
                    ++important_below_floor;
                    largest_pathway_below_floor = std::max(
                        largest_pathway_below_floor, row.pathway_contribution);
                }
            }
            const bool admissible = important_below_floor == 0;
            std::cout << "ACTIVITY_FLOOR_AUDIT"
                      << " epsilon_abs=" << epsilon_abs
                      << " H_floor=" << epsilon_abs * n_nuc_ref * nu_ref
                      << " H2_floor=" << 2.0 * epsilon_abs * n_nuc_ref * nu_ref
                      << " important_rows_below_floor=" << important_below_floor
                      << " largest_pathway_below_floor="
                      << largest_pathway_below_floor
                      << " physically_admissible=" << (admissible ? 1 : 0)
                      << "\n";
            if (!admissible) continue;

            double global = 0.0;
            double block_maximum[3] = {0.0, 0.0, 0.0};
            auto block_index = [](const std::string& block) {
                return block == "P" ? 0 : (block == "A" ? 1 : 2);
            };
            for (const ActivityRow& row : rows) {
                const double denominator = std::max(
                    row.activity, epsilon_abs * row.mu * n_nuc_ref * nu_ref);
                const double scaled = std::abs(row.raw_residual) /
                    std::max(denominator, 1.0e-300);
                const int index = block_index(row.block);
                block_maximum[index] = std::max(block_maximum[index], scaled);
                global = std::max(global, scaled);
            }
            for (int index = 0; index < 3; ++index) {
                const std::string block = index == 0 ? "P" : (index == 1 ? "A" : "M");
                std::cout << "BLOCK_HYBRID_RESIDUAL"
                          << " epsilon_abs=" << epsilon_abs
                          << " block=" << block
                          << " residual=" << block_maximum[index]
                          << " global_residual=" << global
                          << " contribution_to_global="
                          << block_maximum[index] / std::max(global, 1.0e-300)
                          << "\n";
                std::vector<const ActivityRow*> ranked;
                for (const ActivityRow& row : rows) {
                    if (row.block == block) ranked.push_back(&row);
                }
                std::sort(ranked.begin(), ranked.end(), [&](const auto* left, const auto* right) {
                    const double left_denominator = std::max(
                        left->activity, epsilon_abs * left->mu * n_nuc_ref * nu_ref);
                    const double right_denominator = std::max(
                        right->activity, epsilon_abs * right->mu * n_nuc_ref * nu_ref);
                    return std::abs(left->raw_residual) / left_denominator >
                        std::abs(right->raw_residual) / right_denominator;
                });
                const size_t count = std::min<size_t>(10, ranked.size());
                for (size_t rank = 0; rank < count; ++rank) {
                    const ActivityRow& row = *ranked[rank];
                    const double denominator = std::max(
                        row.activity, epsilon_abs * row.mu * n_nuc_ref * nu_ref);
                    std::cout << "BLOCK_HYBRID_ROW"
                              << " epsilon_abs=" << epsilon_abs
                              << " block=" << block
                              << " rank=" << rank + 1
                              << " cell=" << row.cell
                              << " x_cm=" << row.x_cm
                              << " gi=" << row.gi
                              << " state=" << std::quoted(species_label(row.gi))
                              << " n=" << row.population
                              << " n_over_n_nuc=" << row.population_fraction
                              << " raw_steady_residual=" << row.raw_residual
                              << " physical_activity=" << row.activity
                              << " hybrid_denominator=" << denominator
                              << " hybrid_scaled_residual="
                              << std::abs(row.raw_residual) /
                                  std::max(denominator, 1.0e-300)
                              << "\n";
                }
            }
        }

        if (!run_trace_deweighting_scan) return;
        if (!(protection_fraction >= 0.0) ||
            !std::isfinite(protection_fraction)) {
            throw std::runtime_error(
                "Trace protection fraction must be finite and nonnegative");
        }

        std::vector<double> pathway_signed(
            protected_pathway_names.size(), 0.0);
        std::vector<double> pathway_gross(
            protected_pathway_names.size(), 0.0);
        double total_nuclei_inventory = 0.0;
        for (const ActivityRow& row : rows) {
            total_nuclei_inventory += row.mu * row.population;
            for (size_t pathway = 0;
                 pathway < protected_pathway_names.size(); ++pathway) {
                const double contribution =
                    row.protected_pathway_contributions[pathway];
                pathway_signed[pathway] += contribution;
                pathway_gross[pathway] += std::abs(contribution);
            }
        }
        std::cout << "ACTIVITY_TRACE_SCAN_START"
                  << " enabled=1"
                  << " default_enabled=0"
                  << " f_protect=" << protection_fraction
                  << " rows=" << rows.size()
                  << " classification_uses_population_fraction=0"
                  << " species_name_protection=0"
                  << " physical_equations_changed=0"
                  << "\n";
        for (size_t pathway = 0;
             pathway < protected_pathway_names.size(); ++pathway) {
            std::cout << "PROTECTED_PATHWAY_TOTAL"
                      << " pathway=" << protected_pathway_names[pathway]
                      << " signed_total=" << pathway_signed[pathway]
                      << " absolute_total=" << pathway_gross[pathway]
                      << "\n";
        }

        const std::vector<double> candidate_epsilons{
            1.0e-8, 1.0e-10, 1.0e-12, 1.0e-14};
        int surviving_candidates = 0;
        auto block_index = [](const std::string& block) {
            return block == "P" ? 0 : (block == "A" ? 1 : 2);
        };
        for (double epsilon_abs : candidate_epsilons) {
            std::vector<dcr::solver::ActivityTraceClassification>
                classifications;
            classifications.reserve(rows.size());
            int trace_count = 0;
            int protected_count = 0;
            int low_activity_protected_count = 0;
            double trace_inventory = 0.0;
            std::vector<double> trace_pathway_signed(
                protected_pathway_names.size(), 0.0);
            std::vector<double> trace_pathway_gross(
                protected_pathway_names.size(), 0.0);
            double block_residual[3] = {0.0, 0.0, 0.0};
            double trace_residual[3] = {0.0, 0.0, 0.0};
            double nontrace_residual[3] = {0.0, 0.0, 0.0};
            for (const ActivityRow& row : rows) {
                const double floor = epsilon_abs * row.mu * n_nuc_ref * nu_ref;
                const auto classification =
                    dcr::solver::classify_activity_trace_row(
                        row.activity, floor,
                        row.protected_pathway_contributions,
                        pathway_gross, protection_fraction);
                classifications.push_back(classification);
                if (classification.protected_pathway) ++protected_count;
                if (classification.protected_pathway && row.activity < floor) {
                    ++low_activity_protected_count;
                }
                if (classification.trace) {
                    ++trace_count;
                    trace_inventory += row.mu * row.population;
                    for (size_t pathway = 0;
                         pathway < protected_pathway_names.size(); ++pathway) {
                        const double contribution =
                            row.protected_pathway_contributions[pathway];
                        trace_pathway_signed[pathway] += contribution;
                        trace_pathway_gross[pathway] += std::abs(contribution);
                    }
                }
                const double denominator = classification.trace
                    ? std::max(row.activity, floor)
                    : row.activity;
                const double scaled = std::abs(row.raw_residual) /
                    std::max(denominator, 1.0e-300);
                const int index = block_index(row.block);
                block_residual[index] = std::max(block_residual[index], scaled);
                if (classification.trace) {
                    trace_residual[index] = std::max(
                        trace_residual[index], scaled);
                } else {
                    nontrace_residual[index] = std::max(
                        nontrace_residual[index], scaled);
                }
            }

            const int deweighted_protected_rows = 0;
            const bool survives_pathway_protection =
                deweighted_protected_rows == 0;
            if (survives_pathway_protection) ++surviving_candidates;
            std::cout << "ACTIVITY_TRACE_THRESHOLD"
                      << " epsilon_abs=" << epsilon_abs
                      << " total_rows=" << rows.size()
                      << " trace_rows=" << trace_count
                      << " protected_rows=" << protected_count
                      << " low_activity_protected_rows="
                      << low_activity_protected_count
                       << " fraction_trace=" << static_cast<double>(trace_count) /
                           std::max<size_t>(1, rows.size())
                       << " total_nuclei_inventory=" << total_nuclei_inventory
                       << " trace_nuclei_inventory=" << trace_inventory
                       << " trace_nuclei_fraction=" << trace_inventory /
                           std::max(total_nuclei_inventory, 1.0e-300)
                       << " deweighted_protected_rows=" <<
                          deweighted_protected_rows
                      << " survives_pathway_protection=" <<
                          (survives_pathway_protection ? 1 : 0)
                      << " production_valid=0"
                      << " P_hybrid_residual=" << block_residual[0]
                      << " A_hybrid_residual=" << block_residual[1]
                      << " M_hybrid_residual=" << block_residual[2]
                      << " P_trace_residual=" << trace_residual[0]
                      << " A_trace_residual=" << trace_residual[1]
                      << " M_trace_residual=" << trace_residual[2]
                      << " P_nontrace_residual=" << nontrace_residual[0]
                      << " A_nontrace_residual=" << nontrace_residual[1]
                      << " M_nontrace_residual=" << nontrace_residual[2]
                      << " audit_rows_evaluated=" << rows.size()
                      << " audit_complete=1"
                      << "\n";
            for (size_t pathway = 0;
                 pathway < protected_pathway_names.size(); ++pathway) {
                std::cout << "ACTIVITY_TRACE_PATHWAY"
                          << " epsilon_abs=" << epsilon_abs
                          << " pathway=" << protected_pathway_names[pathway]
                          << " all_signed=" << pathway_signed[pathway]
                          << " all_absolute=" << pathway_gross[pathway]
                          << " trace_signed=" << trace_pathway_signed[pathway]
                          << " trace_absolute=" << trace_pathway_gross[pathway]
                          << "\n";
            }
            for (int index = 0; index < 3; ++index) {
                const std::string block = index == 0
                    ? "P" : (index == 1 ? "A" : "M");
                std::vector<size_t> ranked;
                for (size_t row_index = 0;
                     row_index < rows.size(); ++row_index) {
                    if (rows[row_index].block == block) {
                        ranked.push_back(row_index);
                    }
                }
                std::sort(ranked.begin(), ranked.end(),
                    [&](size_t left, size_t right) {
                        auto scaled = [&](size_t row_index) {
                            const ActivityRow& row = rows[row_index];
                            const double floor = epsilon_abs * row.mu *
                                n_nuc_ref * nu_ref;
                            const double denominator =
                                classifications[row_index].trace
                                ? std::max(row.activity, floor)
                                : row.activity;
                            return std::abs(row.raw_residual) /
                                std::max(denominator, 1.0e-300);
                        };
                        return scaled(left) > scaled(right);
                    });
                for (size_t rank = 0;
                     rank < std::min<size_t>(10, ranked.size()); ++rank) {
                    const size_t row_index = ranked[rank];
                    const ActivityRow& row = rows[row_index];
                    const double floor = epsilon_abs * row.mu *
                        n_nuc_ref * nu_ref;
                    const double denominator = classifications[row_index].trace
                        ? std::max(row.activity, floor) : row.activity;
                    std::cout << "ACTIVITY_TRACE_DOMINANT_ROW"
                              << " epsilon_abs=" << epsilon_abs
                              << " block=" << block
                              << " rank=" << rank + 1
                              << " cell=" << row.cell
                              << " gi=" << row.gi
                              << " state=" << std::quoted(species_label(row.gi))
                              << " activity=" << row.activity
                              << " raw_residual=" << row.raw_residual
                              << " hybrid_residual=" <<
                                  std::abs(row.raw_residual) /
                                      std::max(denominator, 1.0e-300)
                              << " trace=" <<
                                  (classifications[row_index].trace ? 1 : 0)
                              << " protected=" <<
                                  (classifications[row_index].protected_pathway
                                      ? 1 : 0)
                              << "\n";
                }
            }
        }

        std::cout << "ACTIVITY_TRACE_GLOBAL_VERDICT"
                  << " surviving_candidates=" << surviving_candidates
                  << " no_safe_global_activity_floor="
                  << (surviving_candidates == 0 ? 1 : 0)
                  << " production_validated=0"
                  << " requires_output_insensitivity_rerun=1"
                  << "\n";

        if (surviving_candidates == 0) {
            for (const char* block : {"P", "A", "M"}) {
                for (int gi = 0; gi < total_states_; ++gi) {
                    std::vector<const ActivityRow*> family_rows;
                    for (const ActivityRow& row : rows) {
                        if (row.block == block && row.gi == gi) {
                            family_rows.push_back(&row);
                        }
                    }
                    if (family_rows.empty()) continue;
                    double minimum_protected_activity =
                        std::numeric_limits<double>::infinity();
                    std::vector<bool> retained_pathways(
                        protected_pathway_names.size(), false);
                    for (const ActivityRow* row : family_rows) {
                        const auto classification =
                            dcr::solver::classify_activity_trace_row(
                                row->activity,
                                std::numeric_limits<double>::max(),
                                row->protected_pathway_contributions,
                                pathway_gross, protection_fraction);
                        if (!classification.protected_pathway) continue;
                        minimum_protected_activity = std::min(
                            minimum_protected_activity, row->activity);
                        for (int pathway : classification.protected_pathways) {
                            retained_pathways[static_cast<size_t>(pathway)] = true;
                        }
                    }
                    double selected_epsilon = 0.0;
                    for (double epsilon_abs : candidate_epsilons) {
                        const double floor = epsilon_abs *
                            levels_[static_cast<size_t>(gi)].atomicity *
                            n_nuc_ref * nu_ref;
                        if (floor < minimum_protected_activity) {
                            selected_epsilon = epsilon_abs;
                            break;
                        }
                    }
                    int family_trace_rows = 0;
                    for (const ActivityRow* row : family_rows) {
                        if (selected_epsilon > 0.0 &&
                            row->activity < selected_epsilon * row->mu *
                                n_nuc_ref * nu_ref) {
                            ++family_trace_rows;
                        }
                    }
                    std::cout << "ACTIVITY_TRACE_FAMILY_FALLBACK"
                              << " block=" << block
                              << " gi=" << gi
                              << " family=" << std::quoted(species_label(gi))
                              << " minimum_protected_activity="
                              << minimum_protected_activity
                              << " proposed_epsilon_abs=" << selected_epsilon
                              << " proposed_maximum_safe_floor=" <<
                                  (selected_epsilon > 0.0
                                      ? selected_epsilon *
                                          levels_[static_cast<size_t>(gi)].atomicity *
                                          n_nuc_ref * nu_ref
                                      : 0.0)
                              << " trace_rows=" << family_trace_rows
                              << " protected_pathways=";
                    bool first = true;
                    for (size_t pathway = 0;
                         pathway < retained_pathways.size(); ++pathway) {
                        if (!retained_pathways[pathway]) continue;
                        if (!first) std::cout << ',';
                        std::cout << protected_pathway_names[pathway];
                        first = false;
                    }
                    if (first) std::cout << "none";
                    std::cout << "\n";
                }
            }
        }
    }
#endif

    void save_temperature_continuation_checkpoint(
        const ContinuationCheckpoint& checkpoint) const {
        if (!save_continuation_checkpoint(checkpoint)) {
            throw std::runtime_error("Could not atomically save temperature checkpoint");
        }
    }

    void export_temperature_endpoint(const ContinuationCheckpoint& checkpoint,
                                     const EndpointAuditResult& audit) const {
        write_alternating_hdf5(checkpoint, audit);
    }

    std::string temperature_species_label(int global_index) const {
        return global_index >= 0 ? species_label(global_index) : "none";
    }

private:
    bool save_continuation_checkpoint(const ContinuationCheckpoint& checkpoint) const {
        if (!valid_profile_layout(checkpoint.profiles) ||
            !valid_upstream_ion_boundary(checkpoint.upstream_boundary)) {
            return false;
        }
        std::ofstream output(
            continuation_checkpoint_temporary_path_,
            std::ios::binary | std::ios::trunc);
        if (!output) return false;
        const std::uint64_t magic = 0x414c545245433031ULL;
        const std::uint32_t version = 7;
        output.write(reinterpret_cast<const char*>(&magic), sizeof(magic));
        output.write(reinterpret_cast<const char*>(&version), sizeof(version));
        output.write(reinterpret_cast<const char*>(&checkpoint.lambda), sizeof(double));
        output.write(reinterpret_cast<const char*>(&checkpoint.amplitude), sizeof(double));
        output.write(reinterpret_cast<const char*>(&checkpoint.previous_lambda), sizeof(double));
        output.write(reinterpret_cast<const char*>(&checkpoint.previous_amplitude), sizeof(double));
        output.write(reinterpret_cast<const char*>(&checkpoint.next_step), sizeof(double));
        const std::uint8_t last_step_rejected = checkpoint.last_step_rejected ? 1 : 0;
        output.write(
            reinterpret_cast<const char*>(&last_step_rejected),
            sizeof(last_step_rejected));
        auto write_profile = [&](const std::vector<Vector>& profile) {
            const std::uint64_t nodes = profile.size();
            output.write(reinterpret_cast<const char*>(&nodes), sizeof(nodes));
            for (const Vector& vector : profile) {
                const std::uint64_t size = static_cast<std::uint64_t>(vector.size());
                output.write(reinterpret_cast<const char*>(&size), sizeof(size));
                output.write(
                    reinterpret_cast<const char*>(vector.data()),
                    static_cast<std::streamsize>(size * sizeof(double)));
            }
        };
        write_profile(checkpoint.profiles.background);
        write_profile(checkpoint.profiles.flowA);
        write_profile(checkpoint.profiles.flowM);
        auto write_vector = [&](const Vector& vector) {
            const std::uint64_t size = static_cast<std::uint64_t>(vector.size());
            output.write(reinterpret_cast<const char*>(&size), sizeof(size));
            output.write(
                reinterpret_cast<const char*>(vector.data()),
                static_cast<std::streamsize>(size * sizeof(double)));
        };
        output.write(
            reinterpret_cast<const char*>(&checkpoint.upstream_boundary.position_cm),
            sizeof(double));
        write_vector(checkpoint.upstream_boundary.positive_ion_nuclei_flux_cm2_s);
        write_vector(checkpoint.upstream_boundary.positive_ion_velocity_cm_s);
        write_vector(checkpoint.upstream_boundary.positive_ion_composition);
        const double current_temperature = std::isfinite(checkpoint.current_temperature)
            ? checkpoint.current_temperature : config_.plasma.Te_eV;
        const double previous_temperature = std::isfinite(checkpoint.previous_temperature)
            ? checkpoint.previous_temperature : current_temperature;
        const double next_temperature_step =
            checkpoint.next_temperature_step > 0.0 &&
                std::isfinite(checkpoint.next_temperature_step)
            ? checkpoint.next_temperature_step : 0.05;
        const Vector current_joint_state = temperature_joint_state(checkpoint);
        const bool previous_joint_state_valid =
            checkpoint.previous_joint_state.size() == current_joint_state.size() &&
            checkpoint.previous_joint_state.allFinite() &&
            std::isfinite(std::exp(checkpoint.previous_joint_state(0))) &&
            checkpoint.previous_joint_state.tail(
                checkpoint.previous_joint_state.size() - 1).minCoeff() >= 0.0;
        const Vector previous_joint_state = previous_joint_state_valid
            ? checkpoint.previous_joint_state : current_joint_state;
        output.write(reinterpret_cast<const char*>(&current_temperature), sizeof(double));
        output.write(reinterpret_cast<const char*>(&previous_temperature), sizeof(double));
        output.write(reinterpret_cast<const char*>(&next_temperature_step), sizeof(double));
        write_vector(previous_joint_state);
        output.write(
            reinterpret_cast<const char*>(&checkpoint.last_attempt_temperature),
            sizeof(double));
        const std::uint8_t last_attempt_accepted =
            checkpoint.last_attempt_accepted ? 1 : 0;
        output.write(
            reinterpret_cast<const char*>(&last_attempt_accepted),
            sizeof(last_attempt_accepted));
        output.write(
            reinterpret_cast<const char*>(&checkpoint.last_attempt_iterations),
            sizeof(checkpoint.last_attempt_iterations));
        const double diagnostic_values[] = {
            checkpoint.last_attempt_amplitude,
            checkpoint.last_attempt_mismatch,
            checkpoint.last_attempt_joint_residual,
            checkpoint.last_attempt_profile_change,
            checkpoint.last_attempt_boundary_change,
            checkpoint.last_attempt_epsilon_sigma,
            checkpoint.last_attempt_epsilon_hminus,
            checkpoint.last_attempt_epsilon_omitted,
            checkpoint.last_attempt_minimum_population};
        output.write(
            reinterpret_cast<const char*>(diagnostic_values),
            sizeof(diagnostic_values));
        output.write(
            reinterpret_cast<const char*>(&checkpoint.last_attempt_failing_cell),
            sizeof(checkpoint.last_attempt_failing_cell));
        output.write(
            reinterpret_cast<const char*>(&checkpoint.last_attempt_failing_species),
            sizeof(checkpoint.last_attempt_failing_species));
        output.write(
            reinterpret_cast<const char*>(
                &checkpoint.last_attempt_rejected_anderson_trials),
            sizeof(checkpoint.last_attempt_rejected_anderson_trials));
        auto write_string = [&](const std::string& value) {
            const std::uint64_t size = static_cast<std::uint64_t>(value.size());
            output.write(reinterpret_cast<const char*>(&size), sizeof(size));
            output.write(value.data(), static_cast<std::streamsize>(size));
        };
        write_string(checkpoint.last_attempt_update_type);
        write_string(checkpoint.last_attempt_status);
        const std::uint8_t provisional_valid = checkpoint.provisional_valid ? 1 : 0;
        output.write(
            reinterpret_cast<const char*>(&provisional_valid),
            sizeof(provisional_valid));
        if (checkpoint.provisional_valid) {
            output.write(
                reinterpret_cast<const char*>(&checkpoint.provisional_temperature),
                sizeof(double));
            output.write(
                reinterpret_cast<const char*>(&checkpoint.provisional_amplitude),
                sizeof(double));
            write_profile(checkpoint.provisional_profiles.background);
            write_profile(checkpoint.provisional_profiles.flowA);
            write_profile(checkpoint.provisional_profiles.flowM);
            output.write(
                reinterpret_cast<const char*>(
                    &checkpoint.provisional_upstream_boundary.position_cm),
                sizeof(double));
            write_vector(checkpoint.provisional_upstream_boundary
                .positive_ion_nuclei_flux_cm2_s);
            write_vector(checkpoint.provisional_upstream_boundary
                .positive_ion_velocity_cm_s);
            write_vector(checkpoint.provisional_upstream_boundary
                .positive_ion_composition);
        }
        output.close();
        if (!output) return false;
        return std::rename(
            continuation_checkpoint_temporary_path_.c_str(),
            continuation_checkpoint_path_.c_str()) == 0;
    }

    bool load_continuation_checkpoint(ContinuationCheckpoint& checkpoint) const {
        std::ifstream input(checkpoint_input_path_, std::ios::binary);
        if (!input) return false;
        std::uint64_t magic = 0;
        std::uint32_t version = 0;
        input.read(reinterpret_cast<char*>(&magic), sizeof(magic));
        input.read(reinterpret_cast<char*>(&version), sizeof(version));
        if (magic != 0x414c545245433031ULL ||
            (version != 3 && version != 4 && version != 5 &&
             version != 6 && version != 7)) {
            return false;
        }
        input.read(reinterpret_cast<char*>(&checkpoint.lambda), sizeof(double));
        input.read(reinterpret_cast<char*>(&checkpoint.amplitude), sizeof(double));
        input.read(reinterpret_cast<char*>(&checkpoint.previous_lambda), sizeof(double));
        input.read(reinterpret_cast<char*>(&checkpoint.previous_amplitude), sizeof(double));
        input.read(reinterpret_cast<char*>(&checkpoint.next_step), sizeof(double));
        if (version >= 4) {
            std::uint8_t last_step_rejected = 0;
            input.read(
                reinterpret_cast<char*>(&last_step_rejected),
                sizeof(last_step_rejected));
            checkpoint.last_step_rejected = last_step_rejected != 0;
        } else {
            checkpoint.last_step_rejected = false;
        }
        auto read_profile = [&](std::vector<Vector>& profile,
                                size_t expected_nodes,
                                size_t expected_vector_size) {
            std::uint64_t nodes = 0;
            input.read(reinterpret_cast<char*>(&nodes), sizeof(nodes));
            if (!input || nodes != expected_nodes) return false;
            profile.resize(static_cast<size_t>(nodes));
            for (Vector& vector : profile) {
                std::uint64_t size = 0;
                input.read(reinterpret_cast<char*>(&size), sizeof(size));
                if (!input || size != expected_vector_size) return false;
                vector.resize(static_cast<int>(size));
                input.read(
                    reinterpret_cast<char*>(vector.data()),
                    static_cast<std::streamsize>(size * sizeof(double)));
                if (!input || !vector.allFinite() || vector.minCoeff() < 0.0) return false;
            }
            return true;
        };
        const size_t background_nodes = version >= 5 ? dx_.size() : x_.size();
        if (!read_profile(
                checkpoint.profiles.background, background_nodes,
                boundary_.P_indices.size()) ||
            !read_profile(
                checkpoint.profiles.flowA, x_.size(), boundary_.A_indices.size()) ||
            !read_profile(
                checkpoint.profiles.flowM, x_.size(), boundary_.M_indices.size())) {
            return false;
        }
        if (version >= 5) {
            auto read_vector = [&](Vector& vector, size_t expected_size) {
                std::uint64_t size = 0;
                input.read(reinterpret_cast<char*>(&size), sizeof(size));
                if (!input || size != expected_size) return false;
                vector.resize(static_cast<int>(size));
                input.read(
                    reinterpret_cast<char*>(vector.data()),
                    static_cast<std::streamsize>(size * sizeof(double)));
                return input && vector.allFinite() && vector.minCoeff() >= 0.0;
            };
            input.read(
                reinterpret_cast<char*>(&checkpoint.upstream_boundary.position_cm),
                sizeof(double));
            if (!input || !read_vector(
                    checkpoint.upstream_boundary.positive_ion_nuclei_flux_cm2_s,
                    boundary_.P_indices.size()) ||
                !read_vector(
                    checkpoint.upstream_boundary.positive_ion_velocity_cm_s,
                    boundary_.P_indices.size()) ||
                !read_vector(
                    checkpoint.upstream_boundary.positive_ion_composition,
                    boundary_.P_indices.size())) {
                return false;
            }
        } else {
            const Vector legacy_upstream = checkpoint.profiles.background.back();
            Vector composition = Vector::Zero(legacy_upstream.size());
            double flux_sum = 0.0;
            for (int pi : ion_p_indices_) {
                const int gi = boundary_.P_indices[static_cast<size_t>(pi)];
                const double nuclei_flux = muP_(pi) * ion_speed(gi, x_.back()) *
                    legacy_upstream(pi);
                composition(pi) = nuclei_flux;
                flux_sum += nuclei_flux;
            }
            if (!(flux_sum > 0.0)) return false;
            composition /= flux_sum;
            checkpoint.upstream_boundary = make_upstream_ion_boundary(
                checkpoint.amplitude, composition);
            checkpoint.profiles.background.pop_back();
            std::cout << "CHECKPOINT_CONVERSION from_version=" << version
                       << " to_version=6"
                      << " discarded_upstream_nonion_state=1"
                      << " legacy_face_flux=" << flux_sum
                       << " stored_amplitude=" << checkpoint.amplitude
                       << "\n";
        }
        if (version >= 6) {
            auto read_vector = [&](Vector& vector, size_t expected_size) {
                std::uint64_t size = 0;
                input.read(reinterpret_cast<char*>(&size), sizeof(size));
                if (!input || size != expected_size) return false;
                vector.resize(static_cast<int>(size));
                input.read(
                    reinterpret_cast<char*>(vector.data()),
                    static_cast<std::streamsize>(size * sizeof(double)));
                return input && vector.allFinite();
            };
            input.read(
                reinterpret_cast<char*>(&checkpoint.current_temperature),
                sizeof(double));
            input.read(
                reinterpret_cast<char*>(&checkpoint.previous_temperature),
                sizeof(double));
            input.read(
                reinterpret_cast<char*>(&checkpoint.next_temperature_step),
                sizeof(double));
            const size_t joint_size = static_cast<size_t>(
                1 + pack_recycled_profiles(checkpoint.profiles).size());
            if (!input ||
                !read_vector(checkpoint.previous_joint_state, joint_size)) {
                return false;
            }
            input.read(
                reinterpret_cast<char*>(&checkpoint.last_attempt_temperature),
                sizeof(double));
            std::uint8_t last_attempt_accepted = 0;
            input.read(
                reinterpret_cast<char*>(&last_attempt_accepted),
                sizeof(last_attempt_accepted));
            checkpoint.last_attempt_accepted = last_attempt_accepted != 0;
            input.read(
                reinterpret_cast<char*>(&checkpoint.last_attempt_iterations),
                sizeof(checkpoint.last_attempt_iterations));
            double diagnostic_values[9];
            input.read(
                reinterpret_cast<char*>(diagnostic_values),
                sizeof(diagnostic_values));
            checkpoint.last_attempt_amplitude = diagnostic_values[0];
            checkpoint.last_attempt_mismatch = diagnostic_values[1];
            checkpoint.last_attempt_joint_residual = diagnostic_values[2];
            checkpoint.last_attempt_profile_change = diagnostic_values[3];
            checkpoint.last_attempt_boundary_change = diagnostic_values[4];
            checkpoint.last_attempt_epsilon_sigma = diagnostic_values[5];
            checkpoint.last_attempt_epsilon_hminus = diagnostic_values[6];
            checkpoint.last_attempt_epsilon_omitted = diagnostic_values[7];
            checkpoint.last_attempt_minimum_population = diagnostic_values[8];
            input.read(
                reinterpret_cast<char*>(&checkpoint.last_attempt_failing_cell),
                sizeof(checkpoint.last_attempt_failing_cell));
            input.read(
                reinterpret_cast<char*>(&checkpoint.last_attempt_failing_species),
                sizeof(checkpoint.last_attempt_failing_species));
            input.read(
                reinterpret_cast<char*>(
                    &checkpoint.last_attempt_rejected_anderson_trials),
                sizeof(checkpoint.last_attempt_rejected_anderson_trials));
            auto read_string = [&](std::string& value) {
                std::uint64_t size = 0;
                input.read(reinterpret_cast<char*>(&size), sizeof(size));
                if (!input || size > 4096) return false;
                value.resize(static_cast<size_t>(size));
                input.read(value.data(), static_cast<std::streamsize>(size));
                return static_cast<bool>(input);
            };
            if (!input ||
                !read_string(checkpoint.last_attempt_update_type) ||
                !read_string(checkpoint.last_attempt_status) ||
                !std::isfinite(checkpoint.current_temperature) ||
                !std::isfinite(checkpoint.previous_temperature) ||
                checkpoint.current_temperature < 1.0 ||
                checkpoint.current_temperature > 5.0 ||
                checkpoint.previous_temperature > checkpoint.current_temperature ||
                !(checkpoint.next_temperature_step > 0.0) ||
                !std::isfinite(checkpoint.next_temperature_step) ||
                !std::isfinite(std::exp(checkpoint.previous_joint_state(0))) ||
                checkpoint.previous_joint_state.tail(
                    checkpoint.previous_joint_state.size() - 1).minCoeff() < 0.0) {
                return false;
            }
            if (version >= 7) {
                std::uint8_t provisional_valid = 0;
                input.read(
                    reinterpret_cast<char*>(&provisional_valid),
                    sizeof(provisional_valid));
                checkpoint.provisional_valid = provisional_valid != 0;
                if (checkpoint.provisional_valid) {
                    input.read(
                        reinterpret_cast<char*>(
                            &checkpoint.provisional_temperature),
                        sizeof(double));
                    input.read(
                        reinterpret_cast<char*>(
                            &checkpoint.provisional_amplitude),
                        sizeof(double));
                    if (!input || !read_profile(
                            checkpoint.provisional_profiles.background,
                            dx_.size(), boundary_.P_indices.size()) ||
                        !read_profile(
                            checkpoint.provisional_profiles.flowA,
                            x_.size(), boundary_.A_indices.size()) ||
                        !read_profile(
                            checkpoint.provisional_profiles.flowM,
                            x_.size(), boundary_.M_indices.size())) {
                        return false;
                    }
                    input.read(
                        reinterpret_cast<char*>(
                            &checkpoint.provisional_upstream_boundary.position_cm),
                        sizeof(double));
                    if (!input || !read_vector(
                            checkpoint.provisional_upstream_boundary
                                .positive_ion_nuclei_flux_cm2_s,
                            boundary_.P_indices.size()) ||
                        !read_vector(
                            checkpoint.provisional_upstream_boundary
                                .positive_ion_velocity_cm_s,
                            boundary_.P_indices.size()) ||
                        !read_vector(
                            checkpoint.provisional_upstream_boundary
                                .positive_ion_composition,
                            boundary_.P_indices.size()) ||
                        !std::isfinite(checkpoint.provisional_temperature) ||
                        !std::isfinite(checkpoint.provisional_amplitude) ||
                        checkpoint.provisional_amplitude <= 0.0) {
                        return false;
                    }
                }
            }
        } else {
            checkpoint.current_temperature = config_.plasma.Te_eV;
            checkpoint.previous_temperature = checkpoint.current_temperature;
            checkpoint.next_temperature_step = 0.05;
            checkpoint.previous_joint_state = temperature_joint_state(checkpoint);
        }
        return std::isfinite(checkpoint.lambda) &&
            std::isfinite(checkpoint.amplitude) && checkpoint.amplitude > 0.0 &&
            valid_profile_layout(checkpoint.profiles) &&
            valid_upstream_ion_boundary(checkpoint.upstream_boundary);
    }

    ProfilePhysicalSummary summarize_profiles(
        const Profiles& profiles,
        const UpstreamIonBoundary* upstream_boundary = nullptr) const {
        ProfilePhysicalSummary summary;
        for (size_t k = 0; k < profiles.background.size(); ++k) {
            const Vector& background = profiles.background[k];
            summary.background_min = std::min(
                summary.background_min, background.minCoeff());
            summary.background_max = std::max(
                summary.background_max, background.maxCoeff());
            summary.minimum_population = std::min(
                summary.minimum_population, background.minCoeff());
            const double nuclei = total_nuclei(profiles, k);
            if (nuclei < summary.total_nuclei_min) {
                summary.total_nuclei_min = nuclei;
                summary.total_nuclei_min_node = static_cast<int>(k);
            }
            if (nuclei > summary.total_nuclei_max) {
                summary.total_nuclei_max = nuclei;
                summary.total_nuclei_max_node = static_cast<int>(k);
            }
        }
        for (const Vector& flow : profiles.flowA) {
            if (flow.size() == 0) continue;
            summary.flowA_min = std::min(summary.flowA_min, flow.minCoeff());
            summary.flowA_max = std::max(summary.flowA_max, flow.maxCoeff());
            summary.minimum_population = std::min(
                summary.minimum_population, flow.minCoeff());
        }
        for (const Vector& flow : profiles.flowM) {
            if (flow.size() == 0) continue;
            summary.flowM_min = std::min(summary.flowM_min, flow.minCoeff());
            summary.flowM_max = std::max(summary.flowM_max, flow.maxCoeff());
            summary.minimum_population = std::min(
                summary.minimum_population, flow.minCoeff());
        }
        if (upstream_boundary != nullptr) {
            for (int pi : ion_p_indices_) {
                const double nuclei_flux =
                    upstream_boundary->positive_ion_nuclei_flux_cm2_s(pi);
                const double velocity =
                    upstream_boundary->positive_ion_velocity_cm_s(pi);
                summary.upstream_ion_nuclei_flux += nuclei_flux;
                summary.upstream_ion_nuclei_density +=
                    nuclei_flux / std::max(velocity, 1.0e-300);
            }
        }
        summary.target_atomic_flux = boundary_.u_A * weighted_sum(
            profiles.flowA.front(), boundary_.A_indices, levels_);
        summary.target_molecular_flux = boundary_.u_M * weighted_sum(
            profiles.flowM.front(), boundary_.M_indices, levels_);
        return summary;
    }

    EndpointAuditResult perform_lambda_one_endpoint_audit(
        const ContinuationCheckpoint& checkpoint) const {
        EndpointAuditResult audit;
        const Vector composition =
            checkpoint.upstream_boundary.positive_ion_composition;
        audit.ion_trial = evaluate_ion_trial(
            checkpoint.profiles, checkpoint.amplitude, composition);
        Profiles reverse_profiles = checkpoint.profiles;
        reverse_profiles.background = audit.ion_trial.background;
        audit.mapped_profiles = sweep_recycled_neutrals(reverse_profiles, 1.0);

        const Vector current_recycled = pack_recycled_profiles(checkpoint.profiles);
        const Vector mapped_recycled = pack_recycled_profiles(audit.mapped_profiles);
        audit.neutral_joint_residual = 0.0;
        for (int i = 0; i < current_recycled.size(); ++i) {
            const double scale = std::max({
                1.0, std::abs(current_recycled(i)), std::abs(mapped_recycled(i))});
            audit.neutral_joint_residual = std::max(
                audit.neutral_joint_residual,
                std::abs(mapped_recycled(i) - current_recycled(i)) / scale);
        }
        audit.joint_residual = std::max(
            std::abs(audit.ion_trial.mismatch), audit.neutral_joint_residual);
        audit.profile_change = max_relative_change(
            audit.mapped_profiles, checkpoint.profiles);
        auto [physical_target_A, physical_target_M] =
            target_recycling(audit.ion_trial.background.front());
        audit.boundary_change = std::max(
            vector_relative_change(
                checkpoint.profiles.flowA.front(), physical_target_A),
            vector_relative_change(
                checkpoint.profiles.flowM.front(), physical_target_M));
        audit.minimum_population =
            summarize_profiles(audit.mapped_profiles).minimum_population;
        return audit;
    }

    void print_lambda_one_endpoint_audit(
        const EndpointAuditResult& audit,
        const char* phase) const {
        for (size_t k = 0; k < audit.ion_trial.cell_diagnostics.size(); ++k) {
            const CellDiagnostics& cell = audit.ion_trial.cell_diagnostics[k];
            std::cout << "ENDPOINT_CELL_AUDIT"
                      << " phase=" << phase
                      << " cell=" << k
                      << " x=" << x_[k]
                      << " selected_pi=" << cell.replacement_pi
                      << " selected_species="
                      << std::quoted(species_label(cell.replacement_gi))
                      << " selected_lagged_P_nuclei_fraction="
                      << cell.replacement_population_fraction
                      << " R_Sigma=" << cell.exact_residual
                      << " abs_R_Sigma=" << std::abs(cell.exact_residual)
                      << " R_Hminus=" << cell.hminus_residual
                      << " R_omitted=" << cell.omitted_residual
                      << " epsilon_Sigma=" << cell.conservation_epsilon
                      << " epsilon_Hminus=" << cell.hminus_epsilon
                      << " Hminus_active=" << (cell.hminus_active ? 1 : 0)
                      << " epsilon_omitted=" << cell.omitted_epsilon
                      << " omitted_active=" << (cell.omitted_active ? 1 : 0)
                      << " raw_minimum_population="
                      << cell.raw_minimum_population
                      << " final_minimum_population="
                      << cell.final_minimum_population
                      << "\n";
        }
        const SweepDiagnostics& sweep = audit.ion_trial.diagnostics;
        for (size_t gi = 0; gi < sweep.replacement_species_counts.size(); ++gi) {
            const int count = sweep.replacement_species_counts[gi];
            if (count == 0) continue;
            std::cout << "ENDPOINT_ROW_SELECTION_SUMMARY"
                      << " phase=" << phase
                      << " species=" << std::quoted(species_label(static_cast<int>(gi)))
                      << " count=" << count << "\n";
        }
        std::cout << "ENDPOINT_AUDIT_SUMMARY"
                  << " phase=" << phase
                  << " target_mismatch=" << audit.ion_trial.mismatch
                  << " neutral_joint_residual=" << audit.neutral_joint_residual
                   << " joint_residual=" << audit.joint_residual
                   << " profile_change=" << audit.profile_change
                   << " boundary_change=" << audit.boundary_change
                  << " minimum_population=" << audit.minimum_population
                  << " max_epsilon_Sigma=" << sweep.max_conservation_epsilon
                  << " epsilon_Sigma_cell=" << sweep.max_epsilon_cell
                  << " max_epsilon_Hminus=" << sweep.max_hminus_epsilon
                  << " epsilon_Hminus_cell=" << sweep.max_hminus_epsilon_cell
                  << " max_epsilon_omitted=" << sweep.max_omitted_epsilon
                  << " epsilon_omitted_cell=" << sweep.max_omitted_epsilon_cell
                  << "\n";
    }

    bool endpoint_audit_accepted(const EndpointAuditResult& audit) const {
        const SweepDiagnostics& sweep = audit.ion_trial.diagnostics;
        return std::isfinite(audit.minimum_population) &&
            std::isfinite(audit.ion_trial.mismatch) &&
            std::isfinite(audit.joint_residual) &&
            std::isfinite(audit.profile_change) &&
            std::isfinite(audit.boundary_change) &&
            std::isfinite(sweep.max_conservation_epsilon) &&
            std::isfinite(sweep.max_hminus_epsilon) &&
            std::isfinite(sweep.max_omitted_epsilon) &&
            std::isfinite(sweep.max_steady_backward_error) &&
            valid_profile_layout(audit.mapped_profiles) &&
            valid_upstream_ion_boundary(audit.ion_trial.upstream_boundary) &&
            audit.minimum_population >= 0.0 &&
            audit.ion_trial.diagnostics.cells_converged &&
            std::abs(audit.ion_trial.mismatch) < kOuterTolerance &&
            audit.joint_residual < kOuterTolerance &&
            audit.profile_change < kOuterTolerance &&
            audit.boundary_change < kOuterTolerance &&
            sweep.max_steady_backward_error < kLocalResidualTolerance &&
            sweep.max_conservation_epsilon <= 1.0e-6 &&
            sweep.max_hminus_epsilon <= kNewtonResidualTolerance &&
            sweep.max_omitted_epsilon <= kNewtonResidualTolerance;
    }

    bool endpoint_inner_audit_accepted(const EndpointAuditResult& audit) const {
        const SweepDiagnostics& sweep = audit.ion_trial.diagnostics;
        return std::isfinite(audit.minimum_population) &&
            std::isfinite(audit.neutral_joint_residual) &&
            std::isfinite(audit.profile_change) &&
            std::isfinite(audit.boundary_change) &&
            std::isfinite(sweep.max_conservation_epsilon) &&
            std::isfinite(sweep.max_hminus_epsilon) &&
            std::isfinite(sweep.max_omitted_epsilon) &&
            std::isfinite(sweep.max_steady_backward_error) &&
            valid_profile_layout(audit.mapped_profiles) &&
            valid_upstream_ion_boundary(audit.ion_trial.upstream_boundary) &&
            audit.minimum_population >= 0.0 &&
            sweep.cells_converged &&
            audit.neutral_joint_residual < kOuterTolerance &&
            audit.profile_change < kOuterTolerance &&
            audit.boundary_change < kOuterTolerance &&
            sweep.max_steady_backward_error < kLocalResidualTolerance &&
            sweep.max_conservation_epsilon <= 1.0e-6 &&
            sweep.max_hminus_epsilon <= kNewtonResidualTolerance &&
            sweep.max_omitted_epsilon <= kNewtonResidualTolerance;
    }

    int run_lambda_one_endpoint_audit(
        const ContinuationCheckpoint& checkpoint) {
        if (std::abs(checkpoint.lambda - 1.0) > 1.0e-14) {
            throw std::runtime_error("Endpoint audit requires a lambda_rec=1 checkpoint");
        }
        if (std::abs(config_.plasma.total_density - 1.0e16) > 1.0) {
            throw std::runtime_error(
                "Endpoint audit requires n_nuc,target=1e16 cm^-3");
        }
        std::cout << std::scientific << std::setprecision(12)
                  << "ENDPOINT_AUDIT_BANNER lambda_rec=" << checkpoint.lambda
                  << " n_nuc_target_cm-3=" << config_.plasma.total_density
                  << " upstream_u_over_uB=" << upstream_velocity_fraction()
                  << " checkpoint=" << std::quoted(continuation_checkpoint_path_)
                  << "\n";

        const std::uint64_t state_hash_before = checkpoint_state_hash(checkpoint);
        EndpointAuditResult audit = perform_lambda_one_endpoint_audit(checkpoint);
        EndpointAuditResult reproduced =
            perform_lambda_one_endpoint_audit(checkpoint);
        const std::uint64_t state_hash_after = checkpoint_state_hash(checkpoint);
        print_lambda_one_endpoint_audit(audit, "loaded_checkpoint");
        const double reproduction_difference = std::max({
            std::abs(audit.ion_trial.mismatch - reproduced.ion_trial.mismatch),
            std::abs(audit.joint_residual - reproduced.joint_residual),
            std::abs(audit.profile_change - reproduced.profile_change),
            std::abs(audit.boundary_change - reproduced.boundary_change)});
        const bool state_unchanged = state_hash_before == state_hash_after;
        const bool accepted = endpoint_audit_accepted(audit) &&
            endpoint_audit_accepted(reproduced) && state_unchanged &&
            reproduction_difference <= 1.0e-12;
        std::cout << "ENDPOINT_AUDIT_RESULT accepted=" << (accepted ? 1 : 0)
                  << " lambda_rec=1.000000000000e+00"
                  << " target_mismatch=" << audit.ion_trial.mismatch
                  << " joint_residual=" << audit.joint_residual
                  << " profile_change=" << audit.profile_change
                  << " minimum_population=" << audit.minimum_population
                  << " epsilon_Sigma_max="
                  << audit.ion_trial.diagnostics.max_conservation_epsilon
                  << " epsilon_Hminus_max="
                  << audit.ion_trial.diagnostics.max_hminus_epsilon
                   << " epsilon_omitted_max="
                   << audit.ion_trial.diagnostics.max_omitted_epsilon
                   << " true_steady_residual="
                   << audit.ion_trial.diagnostics.max_steady_backward_error
                   << " state_hash_before=" << state_hash_before
                   << " state_hash_after=" << state_hash_after
                   << " state_unchanged=" << (state_unchanged ? 1 : 0)
                   << " reproduction_difference=" << reproduction_difference
                   << "\n";
        return accepted ? 0 : 2;
    }

    void write_alternating_hdf5(
        const ContinuationCheckpoint& checkpoint,
        const EndpointAuditResult& audit) const {
        const Profiles& profiles = checkpoint.profiles;
        const size_t nodes = profiles.background.size();
        if (!valid_profile_layout(profiles) ||
            !valid_upstream_ion_boundary(audit.ion_trial.upstream_boundary)) {
            throw std::runtime_error(
                "Cannot write alternating HDF5 output: invalid profile layout");
        }

        dcr::solver::MarchingHistory history;
        history.x_cm.assign(x_.begin(), x_.begin() + static_cast<std::ptrdiff_t>(nodes));
        history.boundary_elapsed_seconds = boundary_.elapsed_seconds;
        history.boundary_iterations = boundary_.iterations;
        history.background_full.reserve(nodes);
        history.flowA.reserve(nodes);
        history.flowM.reserve(nodes);
        history.rate_diagnostics.reserve(nodes);

        dcr::solver::AtomicRateCalculator rate_calculator(atomic_data_);
        for (size_t node = 0; node < nodes; ++node) {
            const Vector& background = profiles.background[node];
            const Vector& flow_a = profiles.flowA[node];
            const Vector& flow_m = profiles.flowM[node];
            if (background.size() != static_cast<int>(boundary_.P_indices.size()) ||
                flow_a.size() != static_cast<int>(boundary_.A_indices.size()) ||
                flow_m.size() != static_cast<int>(boundary_.M_indices.size()) ||
                !background.allFinite() || !flow_a.allFinite() || !flow_m.allFinite() ||
                background.minCoeff() < 0.0 || flow_a.minCoeff() < 0.0 ||
                flow_m.minCoeff() < 0.0) {
                throw std::runtime_error(
                    "Cannot write alternating HDF5 output: invalid profile block");
            }

            Vector background_full = dcr::solver::make_background_full(
                background, boundary_, total_states_);
            history.background_full.push_back(background_full);
            history.flowA.push_back(flow_a);
            history.flowM.push_back(flow_m);
            const auto local = dcr::solver::assemble_local_system(
                config_, atomic_data_, plasma_, grid_, boundary_, background_full,
                flow_a, flow_m, x_[node], false, &rate_cache_);
            history.rate_diagnostics.push_back(rate_calculator.evaluate(
                config_, boundary_, local, background_full, x_[node], plasma_, grid_));
        }

        const size_t cells = dx_.size();
        const double unavailable = std::numeric_limits<double>::quiet_NaN();
        history.cell_elapsed_seconds.assign(cells, unavailable);
        history.cell_iterations.assign(cells, -1);
        history.cell_converged.assign(cells, 1);
        history.cell_final_relative_change.assign(cells, unavailable);
        history.cell_final_residual_relative.assign(cells, unavailable);
        history.cell_final_map_residual_norm.assign(cells, unavailable);
        for (size_t cell = 0;
             cell < cells && cell < audit.ion_trial.cell_diagnostics.size(); ++cell) {
            history.cell_converged[cell] =
                audit.ion_trial.cell_diagnostics[cell].converged ? 1 : 0;
        }

        dcr::solver::BoundaryPhaseResult final_boundary = boundary_;
        for (int pi = 0; pi < profiles.background.front().size(); ++pi) {
            const int gi = boundary_.P_indices[static_cast<size_t>(pi)];
            if (gi >= 0 && gi < final_boundary.population.size()) {
                final_boundary.population(gi) = profiles.background.front()(pi);
            }
        }
        final_boundary.flowA_last = profiles.flowA.front();
        final_boundary.flowM_last = profiles.flowM.front();
        final_boundary.have_flow_last = true;

        dcr::io::Config output_config = config_;
        output_config.solver.spatial_method = "alternating_sweep";
        const std::filesystem::path artifact_path(reference_artifact_stem_);
        if (!artifact_path.parent_path().empty()) {
            output_config.io.output_dir = artifact_path.parent_path().string();
        }
        dcr::solver::write_hdf5_output(
            output_config, atomic_data_, final_boundary, history);
        const std::filesystem::path hdf5_path =
            std::filesystem::path(output_config.io.output_dir) / "dcr_results.h5";
        {
            H5::H5File file(hdf5_path.string(), H5F_ACC_RDWR);
            H5::Group group = file.createGroup("/upstream_boundary");
            auto write_scalar = [&](const char* name, double value) {
                const hsize_t dimensions[1] = {1};
                H5::DataSpace space(1, dimensions);
                H5::DataSet dataset = group.createDataSet(
                    name, H5::PredType::NATIVE_DOUBLE, space);
                dataset.write(&value, H5::PredType::NATIVE_DOUBLE);
            };
            auto write_vector = [&](const char* name, const Vector& values) {
                const hsize_t dimensions[1] = {
                    static_cast<hsize_t>(values.size())};
                H5::DataSpace space(1, dimensions);
                H5::DataSet dataset = group.createDataSet(
                    name, H5::PredType::NATIVE_DOUBLE, space);
                dataset.write(values.data(), H5::PredType::NATIVE_DOUBLE);
            };
            const UpstreamIonBoundary& upstream = audit.ion_trial.upstream_boundary;
            write_scalar("position_cm", upstream.position_cm);
            write_scalar(
                "positive_ion_nuclei_flux_total_cm2_s",
                upstream.positive_ion_nuclei_flux_cm2_s.sum());
            write_vector(
                "positive_ion_nuclei_flux_cm2_s",
                upstream.positive_ion_nuclei_flux_cm2_s);
            write_vector(
                "positive_ion_velocity_cm_s",
                upstream.positive_ion_velocity_cm_s);
            write_vector(
                "positive_ion_composition",
                upstream.positive_ion_composition);
            write_vector("flowA_outflow_cm3", profiles.flowA.back());
            write_vector("flowM_outflow_cm3", profiles.flowM.back());
        }
        std::cout << "ALTERNATING_HDF5_EXPORT"
                  << " path=" << std::quoted(hdf5_path.string())
                  << " background_nodes=" << nodes
                  << " flow_nodes=" << profiles.flowA.size()
                  << " upstream_face_x=" << x_.back()
                  << " target_mismatch=" << audit.ion_trial.mismatch
                  << " joint_residual=" << audit.joint_residual
                  << " minimum_population=" << audit.minimum_population
                  << "\n";
    }

    void export_physical_reference_profiles(
        const ContinuationCheckpoint& checkpoint,
        const EndpointAuditResult& audit,
        const ContinuationPointResult& point) const {
        const std::string densities_path = reference_artifact_stem_ + "_densities.csv";
        const std::string flux_velocity_path =
            reference_artifact_stem_ + "_flux_velocity.csv";
        const std::string recycling_path = reference_artifact_stem_ + "_recycling.csv";
        const std::string residuals_path = reference_artifact_stem_ + "_residuals.csv";
        const std::string summary_path = reference_artifact_stem_ + "_summary.csv";
        const std::string upstream_boundary_path =
            reference_artifact_stem_ + "_upstream_boundary.csv";

        std::ofstream densities(densities_path, std::ios::trunc);
        std::ofstream flux_velocity(flux_velocity_path, std::ios::trunc);
        std::ofstream recycling(recycling_path, std::ios::trunc);
        std::ofstream residuals(residuals_path, std::ios::trunc);
        std::ofstream summary(summary_path, std::ios::trunc);
        std::ofstream upstream_boundary(upstream_boundary_path, std::ios::trunc);
        if (!densities || !flux_velocity || !recycling || !residuals || !summary ||
            !upstream_boundary) {
            throw std::runtime_error("Could not create physical-reference profile exports");
        }
        densities << std::scientific << std::setprecision(12);
        flux_velocity << std::scientific << std::setprecision(12);
        recycling << std::scientific << std::setprecision(12);
        residuals << std::scientific << std::setprecision(12);
        summary << std::scientific << std::setprecision(12);
        upstream_boundary << std::scientific << std::setprecision(12);

        densities << "node,x_cm,block,local_index,global_index,species,"
                     "density_cm-3,mapped_density_cm-3\n";
        flux_velocity << "node,x_cm,block,local_index,global_index,species,"
                         "density_cm-3,velocity_magnitude_cm_s-1,"
                         "signed_velocity_cm_s-1,signed_flux_cm-2_s-1,"
                         "mapped_density_cm-3,mapped_signed_flux_cm-2_s-1\n";

        auto write_state_profiles = [&] (
            const char* block,
            const std::vector<int>& indices,
            const std::vector<Vector>& current,
            const std::vector<Vector>& mapped) {
            if (current.size() != mapped.size()) {
                throw std::runtime_error("CSV profile block sizes differ");
            }
            for (size_t node = 0; node < current.size(); ++node) {
                for (int local = 0; local < current[node].size(); ++local) {
                    const int global = indices[static_cast<size_t>(local)];
                    double signed_velocity = 0.0;
                    if (block[0] == 'P' && is_positive_ion(local)) {
                        signed_velocity = -ion_speed(global, x_[node]);
                    } else if (block[0] == 'A') {
                        signed_velocity = boundary_.u_A;
                    } else if (block[0] == 'M') {
                        signed_velocity = boundary_.u_M;
                    }
                    const double density = current[node](local);
                    const double mapped_density = mapped[node](local);

                    densities << node << ',' << x_[node] << ',' << block << ','
                              << local << ',' << global << ',';
                    write_csv_field(densities, species_label(global));
                    densities << ',' << density << ',' << mapped_density << '\n';

                    flux_velocity << node << ',' << x_[node] << ',' << block << ','
                                  << local << ',' << global << ',';
                    write_csv_field(flux_velocity, species_label(global));
                    flux_velocity << ',' << density << ',' << std::abs(signed_velocity)
                                  << ',' << signed_velocity << ','
                                  << density * signed_velocity << ',' << mapped_density
                                  << ',' << mapped_density * signed_velocity << '\n';
                }
            }
        };
        write_state_profiles(
            "P", boundary_.P_indices, checkpoint.profiles.background,
            audit.mapped_profiles.background);
        write_state_profiles(
            "A", boundary_.A_indices, checkpoint.profiles.flowA,
            audit.mapped_profiles.flowA);
        write_state_profiles(
            "M", boundary_.M_indices, checkpoint.profiles.flowM,
            audit.mapped_profiles.flowM);

        upstream_boundary << "x_cm,local_index,global_index,species,"
                             "nuclei_flux_cm-2_s-1,velocity_cm_s-1,composition\n";
        const UpstreamIonBoundary& upstream = audit.ion_trial.upstream_boundary;
        for (int pi : ion_p_indices_) {
            const int global = boundary_.P_indices[static_cast<size_t>(pi)];
            upstream_boundary << upstream.position_cm << ',' << pi << ',' << global << ',';
            write_csv_field(upstream_boundary, species_label(global));
            upstream_boundary << ','
                              << upstream.positive_ion_nuclei_flux_cm2_s(pi) << ','
                              << upstream.positive_ion_velocity_cm_s(pi) << ','
                              << upstream.positive_ion_composition(pi) << '\n';
        }

        recycling << "node,x_cm,block,local_index,global_index,species,"
                     "current_density_cm-3,mapped_density_cm-3,"
                     "fixed_point_residual_cm-3,fixed_point_relative_residual,"
                     "velocity_cm_s-1,current_flux_cm-2_s-1,mapped_flux_cm-2_s-1\n";
        auto write_recycling_block = [&] (
            const char* block,
            const std::vector<int>& indices,
            const std::vector<Vector>& current,
            const std::vector<Vector>& mapped,
            double velocity) {
            for (size_t node = 0; node < x_.size(); ++node) {
                for (int local = 0; local < current[node].size(); ++local) {
                    const int global = indices[static_cast<size_t>(local)];
                    const double value = current[node](local);
                    const double mapped_value = mapped[node](local);
                    const double difference = mapped_value - value;
                    const double scale = std::max({1.0, std::abs(value),
                                                   std::abs(mapped_value)});
                    recycling << node << ',' << x_[node] << ',' << block << ','
                              << local << ',' << global << ',';
                    write_csv_field(recycling, species_label(global));
                    recycling << ',' << value << ',' << mapped_value << ','
                              << difference << ',' << difference / scale << ','
                              << velocity << ',' << velocity * value << ','
                              << velocity * mapped_value << '\n';
                }
            }
        };
        write_recycling_block(
            "A", boundary_.A_indices, checkpoint.profiles.flowA,
            audit.mapped_profiles.flowA, boundary_.u_A);
        write_recycling_block(
            "M", boundary_.M_indices, checkpoint.profiles.flowM,
            audit.mapped_profiles.flowM, boundary_.u_M);

        residuals << "cell,x_cm,dx_cm,local_row,global_index,species,selected_row,"
                     "Hminus_row,physical_residual_cm-3_s-1,row_normalization_cm-3_s-1,"
                     "physical_epsilon,R_Sigma_cm-3_s-1,Sigma_normalization_cm-3_s-1,"
                     "epsilon_Sigma,R_Hminus_cm-3_s-1,Hminus_normalization_cm-3_s-1,"
                     "epsilon_Hminus,R_omitted_cm-3_s-1,omitted_normalization_cm-3_s-1,"
                     "epsilon_omitted,selected_lagged_P_nuclei_fraction,"
                     "raw_minimum_population_cm-3,final_minimum_population_cm-3\n";
        Profiles audited_profiles = checkpoint.profiles;
        audited_profiles.background = audit.ion_trial.background;
        for (int cell_index = 0; cell_index + 1 < static_cast<int>(x_.size());
             ++cell_index) {
            const Vector* upstream = cell_index + 1 <
                static_cast<int>(audited_profiles.background.size())
                ? &audited_profiles.background[static_cast<size_t>(cell_index + 1)]
                : nullptr;
            const PhysicalCellSystem original = assemble_physical_cell_system(
                cell_index,
                audited_profiles.background[static_cast<size_t>(cell_index)],
                upstream,
                upstream == nullptr ? &audit.ion_trial.upstream_boundary : nullptr,
                audited_profiles.flowA,
                audited_profiles.flowM);
            const Vector physical_residual = original.lhs *
                audited_profiles.background[static_cast<size_t>(cell_index)] - original.rhs;
            const CellDiagnostics& cell =
                audit.ion_trial.cell_diagnostics[static_cast<size_t>(cell_index)];
            for (int pi = 0; pi < physical_residual.size(); ++pi) {
                const int global = boundary_.P_indices[static_cast<size_t>(pi)];
                const double scale = original.row_scales(pi);
                const double epsilon = scale > kPhysicalRowScaleFloor
                    ? std::abs(physical_residual(pi)) / scale
                    : std::numeric_limits<double>::quiet_NaN();
                residuals << cell_index << ',' << x_[static_cast<size_t>(cell_index)]
                          << ',' << dx_[static_cast<size_t>(cell_index)] << ','
                          << pi << ',' << global << ',';
                write_csv_field(residuals, species_label(global));
                residuals << ',' << (pi == cell.replacement_pi ? 1 : 0)
                          << ',' << (pi == hminus_pi_ ? 1 : 0)
                          << ',' << physical_residual(pi) << ',' << scale << ','
                          << epsilon << ',' << cell.exact_residual << ','
                          << cell.conservation_normalization << ','
                          << cell.conservation_epsilon << ',' << cell.hminus_residual
                          << ',' << cell.hminus_normalization << ','
                          << cell.hminus_epsilon << ',' << cell.omitted_residual
                          << ',' << cell.omitted_normalization << ','
                          << cell.omitted_epsilon << ','
                          << cell.replacement_population_fraction << ','
                          << cell.raw_minimum_population << ','
                          << cell.final_minimum_population << '\n';
            }
        }

        const SweepDiagnostics& sweep = audit.ion_trial.diagnostics;
        auto metric = [&](const char* name, double value) {
            summary << name << ',' << value << '\n';
        };
        summary << "metric,value\n";
        metric("Te_eV", config_.plasma.Te_eV);
        metric("Ti_eV", config_.plasma.Ti_eV);
        metric("length_cm", config_.grid.length_cm);
        metric("nodes", static_cast<double>(config_.grid.num_cells));
        metric("background_locations", static_cast<double>(
            checkpoint.profiles.background.size()));
        metric("flow_locations", static_cast<double>(checkpoint.profiles.flowA.size()));
        metric("upstream_boundary_position_cm", upstream.position_cm);
        metric("upstream_positive_ion_nuclei_flux_cm-2_s-1",
               upstream.positive_ion_nuclei_flux_cm2_s.sum());
        metric("n_nuc_target_cm-3", config_.plasma.total_density);
        metric("upstream_u_over_uB", upstream_velocity_fraction());
        metric("R_rec", wall_.alpha_atom + wall_.alpha_molecule);
        metric("lambda_rec", checkpoint.lambda);
        metric("f_pump", 1.0 / config_.grid.spatial_exhaust_width_cm);
        metric("shooting_amplitude", checkpoint.amplitude);
        metric("joint_solver_iterations", static_cast<double>(point.outer_iterations));
        metric("joint_solver_sweep_evaluations", static_cast<double>(point.shooting_iterations));
        metric("joint_solver_target_mismatch", point.mismatch);
        metric("joint_solver_profile_change", point.profile_change);
        metric("joint_solver_boundary_change", point.boundary_change);
        metric("audit_target_mismatch", audit.ion_trial.mismatch);
        metric("audit_neutral_joint_residual", audit.neutral_joint_residual);
        metric("audit_joint_residual", audit.joint_residual);
        metric("audit_profile_change", audit.profile_change);
        metric("minimum_population_cm-3", audit.minimum_population);
        metric("epsilon_Sigma_max", sweep.max_conservation_epsilon);
        metric("epsilon_Sigma_cell", static_cast<double>(sweep.max_epsilon_cell));
        metric("epsilon_Hminus_max", sweep.max_hminus_epsilon);
        metric("epsilon_Hminus_cell", static_cast<double>(sweep.max_hminus_epsilon_cell));
        metric("epsilon_omitted_max", sweep.max_omitted_epsilon);
        metric("epsilon_omitted_cell", static_cast<double>(sweep.max_omitted_epsilon_cell));

        if (!densities || !flux_velocity || !recycling || !residuals || !summary ||
            !upstream_boundary) {
            throw std::runtime_error("Physical-reference profile export failed");
        }
        std::cout << "PHYSICAL_REFERENCE_EXPORT"
                  << " densities=" << std::quoted(densities_path)
                  << " flux_velocity=" << std::quoted(flux_velocity_path)
                  << " recycling=" << std::quoted(recycling_path)
                  << " residuals=" << std::quoted(residuals_path)
                  << " summary=" << std::quoted(summary_path)
                  << " upstream_boundary=" << std::quoted(upstream_boundary_path)
                  << "\n";
    }

    int run_physical_reference_case(ContinuationCheckpoint checkpoint) {
        auto close = [](double a, double b) {
            return std::abs(a - b) <=
                1.0e-12 * std::max({1.0, std::abs(a), std::abs(b)});
        };
        if (!close(checkpoint.lambda, 1.0) ||
            !close(config_.plasma.Te_eV, 2.0) ||
            !close(config_.plasma.Ti_eV, 2.0) ||
            !close(config_.grid.length_cm, 50.0) ||
            config_.grid.num_cells != 200 ||
            !close(config_.plasma.total_density, 1.0e16) ||
            !close(1.0 / config_.grid.spatial_exhaust_width_cm, 0.01) ||
            !close(wall_.alpha_atom + wall_.alpha_molecule, 1.0)) {
            throw std::runtime_error("Physical-reference parameters do not match the requested case");
        }
        if (checkpoint_input_path_ == continuation_checkpoint_path_) {
            throw std::runtime_error(
                "Physical-reference solve requires a distinct output artifact stem");
        }

        std::cout << std::scientific << std::setprecision(12)
                  << "PHYSICAL_REFERENCE_BANNER"
                  << " Te_eV=" << config_.plasma.Te_eV
                  << " Ti_eV=" << config_.plasma.Ti_eV
                  << " length_cm=" << config_.grid.length_cm
                  << " nodes=" << config_.grid.num_cells
                  << " n_nuc_target_cm-3=" << config_.plasma.total_density
                  << " upstream_u_over_uB=" << upstream_velocity_fraction()
                  << " R_rec=" << wall_.alpha_atom + wall_.alpha_molecule
                  << " lambda_rec=" << checkpoint.lambda
                  << " f_pump=" << 1.0 / config_.grid.spatial_exhaust_width_cm
                  << " checkpoint_input=" << std::quoted(checkpoint_input_path_)
                  << " checkpoint_output=" << std::quoted(continuation_checkpoint_path_)
                  << "\n";

        const auto solve_start = std::chrono::steady_clock::now();
        frozen_replacement_rows_ = dominant_replacement_pattern(checkpoint.profiles);
        auto strictly_accepted = [&](const EndpointAuditResult& candidate) {
            const SweepDiagnostics& candidate_sweep = candidate.ion_trial.diagnostics;
            return endpoint_audit_accepted(candidate) &&
                candidate_sweep.max_conservation_epsilon < 1.0e-6 &&
                candidate_sweep.max_hminus_epsilon < 1.0e-6 &&
                candidate_sweep.max_omitted_epsilon < 1.0e-6;
        };

        EndpointAuditResult audit = perform_lambda_one_endpoint_audit(checkpoint);
        int joint_evaluations = 1;
        int polishing_iterations = 0;
        std::cout << "PHYSICAL_REFERENCE_PREUPDATE_CHECK"
                  << " accepted=" << (strictly_accepted(audit) ? 1 : 0)
                  << " target_mismatch=" << audit.ion_trial.mismatch
                  << " joint_residual=" << audit.joint_residual
                  << " profile_change=" << audit.profile_change
                  << " minimum_population=" << audit.minimum_population
                  << "\n";

        // A fallback polish is safeguarded globally: only a lower-residual state can
        // replace the current iterate, and the first fully accepted iterate is retained.
        for (int iteration = 0;
             !strictly_accepted(audit) && iteration < 500;
             ++iteration) {
            const double current_score = std::max(
                audit.joint_residual, audit.profile_change);
            bool improved = false;
            double trial_relaxation = 0.3;
            for (int backtrack = 0; backtrack < 12; ++backtrack) {
                ContinuationCheckpoint candidate = checkpoint;
                candidate.amplitude = checkpoint.amplitude * std::exp(
                    -trial_relaxation * audit.ion_trial.mismatch);
                candidate.profiles.background = audit.ion_trial.background;
                for (size_t node = 0; node < x_.size(); ++node) {
                    candidate.profiles.flowA[node] += trial_relaxation *
                        (audit.mapped_profiles.flowA[node] -
                         checkpoint.profiles.flowA[node]);
                    candidate.profiles.flowM[node] += trial_relaxation *
                        (audit.mapped_profiles.flowM[node] -
                         checkpoint.profiles.flowM[node]);
                }
                EndpointAuditResult candidate_audit =
                    perform_lambda_one_endpoint_audit(candidate);
                ++joint_evaluations;
                candidate.profiles.background = candidate_audit.ion_trial.background;
                candidate_audit = perform_lambda_one_endpoint_audit(candidate);
                ++joint_evaluations;
                const double candidate_score = std::max(
                    candidate_audit.joint_residual, candidate_audit.profile_change);
                const SweepDiagnostics& candidate_sweep =
                    candidate_audit.ion_trial.diagnostics;
                const bool physically_valid = candidate_audit.minimum_population >= 0.0 &&
                    candidate_sweep.cells_converged &&
                    std::isfinite(candidate_score);
                std::cout << "PHYSICAL_REFERENCE_POLISH_TRIAL"
                          << " iteration=" << iteration + 1
                          << " backtrack=" << backtrack
                          << " relaxation=" << trial_relaxation
                          << " score=" << candidate_score
                          << " current_score=" << current_score
                          << " epsilon_Sigma_max="
                          << candidate_sweep.max_conservation_epsilon
                          << " epsilon_Hminus_max="
                          << candidate_sweep.max_hminus_epsilon
                          << " epsilon_omitted_max="
                          << candidate_sweep.max_omitted_epsilon
                          << " retained="
                          << ((physically_valid && candidate_score < current_score) ? 1 : 0)
                          << "\n";
                if (physically_valid && candidate_score < current_score) {
                    checkpoint = std::move(candidate);
                    audit = std::move(candidate_audit);
                    polishing_iterations = iteration + 1;
                    improved = true;
                    break;
                }
                trial_relaxation *= 0.5;
            }
            if (!improved) break;
        }
        if (!strictly_accepted(audit)) {
            std::cout << "PHYSICAL_REFERENCE_SOLVE_RESULT accepted=0"
                      << " solver_converged=0"
                      << " failure=safeguarded_polish_failed"
                      << " iterations=" << polishing_iterations << "\n";
            return 2;
        }

        Profiles converged_pattern_profiles = checkpoint.profiles;
        converged_pattern_profiles.background = audit.ion_trial.background;
        const std::vector<int> recomputed_pattern =
            dominant_replacement_pattern(converged_pattern_profiles);
        int changed_cells = 0;
        for (size_t cell = 0; cell < recomputed_pattern.size(); ++cell) {
            if (recomputed_pattern[cell] != frozen_replacement_rows_[cell]) {
                ++changed_cells;
            }
        }
        frozen_replacement_rows_ = recomputed_pattern;
        EndpointAuditResult final_audit = perform_lambda_one_endpoint_audit(checkpoint);
        ++joint_evaluations;
        std::cout << "PHYSICAL_REFERENCE_PATTERN_RECHECK"
                  << " changed_cells=" << changed_cells
                  << " final_sweep=1"
                  << " accepted=" << (strictly_accepted(final_audit) ? 1 : 0)
                  << "\n";
        if (!strictly_accepted(final_audit)) {
            std::cout << "PHYSICAL_REFERENCE_SOLVE_RESULT accepted=0"
                      << " solver_converged=0"
                      << " failure=recomputed_pattern_audit_failed\n";
            return 2;
        }
        audit = std::move(final_audit);

        ContinuationPointResult point;
        checkpoint.lambda = 1.0;
        checkpoint.previous_lambda = 1.0;
        checkpoint.previous_amplitude = checkpoint.amplitude;
        checkpoint.next_step = 0.02;
        checkpoint.last_step_rejected = false;
        checkpoint.upstream_boundary = audit.ion_trial.upstream_boundary;
        point.converged = true;
        point.outer_iterations = polishing_iterations;
        point.shooting_iterations = joint_evaluations;
        point.amplitude = checkpoint.amplitude;
        point.mismatch = audit.ion_trial.mismatch;
        point.profile_change = audit.profile_change;
        point.boundary_change = audit.neutral_joint_residual;
        point.profiles = checkpoint.profiles;
        point.ion_trial = audit.ion_trial;
        point.elapsed_seconds = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - solve_start).count();

        print_lambda_one_endpoint_audit(audit, "physical_reference_final");
        const SweepDiagnostics& sweep = audit.ion_trial.diagnostics;
        const bool accepted = strictly_accepted(audit);
        if (!accepted) {
            std::cout << "PHYSICAL_REFERENCE_SOLVE_RESULT accepted=0"
                      << " solver_converged=1"
                      << " target_mismatch=" << audit.ion_trial.mismatch
                      << " joint_residual=" << audit.joint_residual
                      << " minimum_population=" << audit.minimum_population
                      << " epsilon_Sigma_max=" << sweep.max_conservation_epsilon
                      << " epsilon_Hminus_max=" << sweep.max_hminus_epsilon
                      << " epsilon_omitted_max=" << sweep.max_omitted_epsilon
                      << "\n";
            return 2;
        }
        if (!save_continuation_checkpoint(checkpoint)) {
            throw std::runtime_error("Could not save physical-reference checkpoint");
        }
        export_physical_reference_profiles(checkpoint, audit, point);
        write_alternating_hdf5(checkpoint, audit);
        initialize_stage_summary();
        const ProfilePhysicalSummary physical = summarize_profiles(
            point.profiles, &audit.ion_trial.upstream_boundary);
        append_stage_summary(
            "accepted", 1.0, 1.0, 0.0, "physical_reference", point, &physical);

        std::cout << "PHYSICAL_REFERENCE_SOLVE_RESULT accepted=1"
                  << " solver_converged=1"
                  << " iterations=" << point.outer_iterations
                  << " joint_evaluations=" << joint_evaluations
                  << " preupdate_accepted=" << (polishing_iterations == 0 ? 1 : 0)
                  << " pattern_changed_cells=" << changed_cells
                  << " amplitude=" << point.amplitude
                  << " target_mismatch=" << audit.ion_trial.mismatch
                  << " joint_residual=" << audit.joint_residual
                  << " profile_change=" << audit.profile_change
                  << " minimum_population=" << audit.minimum_population
                  << " epsilon_Sigma_max=" << sweep.max_conservation_epsilon
                  << " epsilon_Hminus_max=" << sweep.max_hminus_epsilon
                  << " epsilon_omitted_max=" << sweep.max_omitted_epsilon
                  << " checkpoint=" << std::quoted(continuation_checkpoint_path_)
                  << "\n";
        return 0;
    }

    void report_minimum_population_locations(const Profiles& profiles) const {
        const ProfilePhysicalSummary summary = summarize_profiles(profiles);
        auto report_block = [&](const char* block,
                                const std::vector<Vector>& profile,
                                const std::vector<int>& global_indices) {
            if (profile.empty()) return;
            for (int local = 0; local < profile.front().size(); ++local) {
                int first_node = -1;
                int last_node = -1;
                int count = 0;
                for (size_t node = 0; node < profile.size(); ++node) {
                    if (profile[node](local) != summary.minimum_population) continue;
                    if (first_node < 0) first_node = static_cast<int>(node);
                    last_node = static_cast<int>(node);
                    ++count;
                }
                if (count == 0) continue;
                const int global = global_indices[static_cast<size_t>(local)];
                std::cout << "MINIMUM_POPULATION_LOCATION value="
                          << summary.minimum_population
                          << " block=" << block
                          << " species=" << std::quoted(species_label(global))
                          << " first_node=" << first_node
                          << " first_x=" << x_[static_cast<size_t>(first_node)]
                          << " last_node=" << last_node
                          << " last_x=" << x_[static_cast<size_t>(last_node)]
                          << " tied_node_count=" << count << "\n";
            }
        };
        report_block("P", profiles.background, boundary_.P_indices);
        report_block("A", profiles.flowA, boundary_.A_indices);
        report_block("M", profiles.flowM, boundary_.M_indices);
    }

    void initialize_stage_summary() const {
        std::ofstream output(continuation_summary_path_, std::ios::trunc);
        output << "status,from_lambda,lambda,step,difficulty,coupled_iterations,"
               << "shooting_iterations,amplitude,mismatch,profile_change,boundary_change,"
               << "upstream_ion_nuclei_density,target_atomic_flux,target_molecular_flux,"
               << "minimum_population,total_nuclei_min,total_nuclei_min_node,"
               << "total_nuclei_max,total_nuclei_max_node,epsilon_sigma_max,"
               << "epsilon_sigma_cell,anderson_accepted,anderson_rejected,"
               << "anderson_backtracks,relaxed_fallbacks,elapsed_seconds,"
               << "first_failing_cell,reason\n";
    }

    void append_stage_summary(
        const char* status,
        double from_lambda,
        double lambda,
        double step,
        const char* difficulty,
        const ContinuationPointResult& point,
        const ProfilePhysicalSummary* physical) const {
        const double nan = std::numeric_limits<double>::quiet_NaN();
        std::ofstream output(continuation_summary_path_, std::ios::app);
        output << std::scientific << std::setprecision(12)
               << status << ',' << from_lambda << ',' << lambda << ',' << step << ','
               << difficulty << ',' << point.outer_iterations << ','
               << point.shooting_iterations << ',' << point.amplitude << ','
               << point.mismatch << ',' << point.profile_change << ','
               << point.boundary_change << ','
               << (physical ? physical->upstream_ion_nuclei_density : nan) << ','
               << (physical ? physical->target_atomic_flux : nan) << ','
               << (physical ? physical->target_molecular_flux : nan) << ','
               << (physical ? physical->minimum_population : nan) << ','
               << (physical ? physical->total_nuclei_min : nan) << ','
               << (physical ? physical->total_nuclei_min_node : -1) << ','
               << (physical ? physical->total_nuclei_max : nan) << ','
               << (physical ? physical->total_nuclei_max_node : -1) << ','
               << point.ion_trial.diagnostics.max_conservation_epsilon << ','
               << point.ion_trial.diagnostics.max_epsilon_cell << ','
               << point.anderson_accepted << ',' << point.anderson_rejected << ','
               << point.anderson_backtracks << ',' << point.relaxed_fallbacks << ','
               << point.elapsed_seconds << ',' << point.first_failing_cell << ','
               << std::quoted(point.failure) << '\n';
    }

    int run_recycling_base_case() {
        constexpr double lambda_recycling = 0.0;
        std::cout << std::scientific << std::setprecision(12);
        Profiles profiles = initial_profiles();
        const auto [physical_target_A, physical_target_M] =
            target_recycling(profiles.background.front());
        profiles = sweep_recycled_neutrals(profiles, lambda_recycling);

        const double physical_atomic_flux = boundary_.u_A * weighted_sum(
            physical_target_A, boundary_.A_indices, levels_);
        const double physical_molecular_flux = boundary_.u_M * weighted_sum(
            physical_target_M, boundary_.M_indices, levels_);
        const double applied_atomic_flux = boundary_.u_A * weighted_sum(
            profiles.flowA.front(), boundary_.A_indices, levels_);
        const double applied_molecular_flux = boundary_.u_M * weighted_sum(
            profiles.flowM.front(), boundary_.M_indices, levels_);
        double maximum_flow_population = 0.0;
        for (size_t k = 0; k < profiles.flowA.size(); ++k) {
            if (profiles.flowA[k].size() > 0) {
                maximum_flow_population = std::max(
                    maximum_flow_population, profiles.flowA[k].maxCoeff());
            }
            if (profiles.flowM[k].size() > 0) {
                maximum_flow_population = std::max(
                    maximum_flow_population, profiles.flowM[k].maxCoeff());
            }
        }
        std::cout << "RECYCLING_BASE_SETUP lambda_rec=" << lambda_recycling
                  << " physical_atomic_flux=" << physical_atomic_flux
                  << " physical_molecular_flux=" << physical_molecular_flux
                  << " applied_atomic_flux=" << applied_atomic_flux
                  << " applied_molecular_flux=" << applied_molecular_flux
                  << " maximum_recycled_population=" << maximum_flow_population
                  << " chemistry_unmodified=1 deposition_unmodified=1"
                  << "\n";

        const double initial_amplitude = initial_shooting_amplitude(
            profiles.background.front());
        const Vector composition = upstream_ion_composition(profiles.background.front());
        current_outer_iteration_ = 0;
        bool base_cells_converged = false;
        bool base_nonnegative = false;
        double base_minimum_population = std::numeric_limits<double>::quiet_NaN();
        double base_mismatch = std::numeric_limits<double>::quiet_NaN();
        try {
            const IonTrial base_trial = evaluate_ion_trial(
                profiles, initial_amplitude, composition);
            base_minimum_population = std::numeric_limits<double>::infinity();
            for (const Vector& state : base_trial.background) {
                base_minimum_population = std::min(
                    base_minimum_population, state.minCoeff());
            }
            base_cells_converged = base_trial.diagnostics.cells_converged;
            base_nonnegative = base_minimum_population >= 0.0;
            base_mismatch = base_trial.mismatch;
            std::cout << "RECYCLING_BASE_SWEEP lambda_rec=" << lambda_recycling
                      << " amplitude=" << initial_amplitude
                      << " all_reverse_cells_converged="
                      << (base_cells_converged ? 1 : 0)
                      << " nonnegative=" << (base_nonnegative ? 1 : 0)
                      << " minimum_population=" << base_minimum_population
                      << " target_mismatch=" << base_mismatch
                      << "\n";

            const IonTrial trial = shoot_recycling_base_controlled(
                profiles, initial_amplitude, composition);
            double minimum_population = std::numeric_limits<double>::infinity();
            for (const Vector& state : trial.background) {
                minimum_population = std::min(minimum_population, state.minCoeff());
            }
            const bool nonnegative = minimum_population >= 0.0;
            const bool feasible = trial.diagnostics.cells_converged && nonnegative;
            std::cout << "RECYCLING_BASE_RESULT lambda_rec=" << lambda_recycling
                      << " feasible=" << (feasible ? 1 : 0)
                      << " base_sweep_feasible="
                      << ((base_cells_converged && base_nonnegative) ? 1 : 0)
                      << " shooting_converged=1"
                      << " all_reverse_cells_converged="
                      << (trial.diagnostics.cells_converged ? 1 : 0)
                      << " nonnegative=" << (nonnegative ? 1 : 0)
                      << " minimum_population=" << minimum_population
                      << " shooting_amplitude=" << trial.amplitude
                      << " shooting_mismatch=" << trial.mismatch
                      << " shooting_evaluations=" << trial.evaluations
                      << " maximum_exact_residual="
                      << trial.diagnostics.max_exact_residual
                      << " maximum_exact_cell=" << trial.diagnostics.max_exact_cell
                      << " maximum_Hminus_residual="
                      << trial.diagnostics.max_hminus_residual
                      << " maximum_Hminus_cell=" << trial.diagnostics.max_hminus_cell
                      << "\n";
            if (!feasible) return 2;
            ContinuationPointResult point_001 = solve_recycling_continuation_point(
                trial, profiles, composition, 0.01, 4.3156783e22,
                std::numeric_limits<double>::quiet_NaN());
            if (!point_001.converged) return 2;

            ContinuationPointResult point_002 = solve_recycling_continuation_point(
                point_001.ion_trial, point_001.profiles, composition,
                0.02, 2.10e22, point_001.amplitude);
            if (point_002.converged) {
                ContinuationPointResult point_003 = solve_recycling_continuation_point(
                    point_002.ion_trial, point_002.profiles, composition,
                    0.03, 1.78519e22, point_002.amplitude);
                if (!point_003.converged) return 2;
                ContinuationPointResult point_004 = solve_recycling_continuation_point(
                    point_003.ion_trial, point_003.profiles, composition,
                    0.04, 1.5308063e22, point_003.amplitude);
                if (!point_004.converged) return 2;
                ContinuationCheckpoint checkpoint;
                checkpoint.lambda = 0.04;
                checkpoint.amplitude = point_004.amplitude;
                checkpoint.previous_lambda = 0.03;
                checkpoint.previous_amplitude = point_003.amplitude;
                checkpoint.next_step = 0.02;
                checkpoint.profiles = point_004.profiles;
                checkpoint.upstream_boundary = point_004.ion_trial.upstream_boundary;
                if (!save_continuation_checkpoint(checkpoint)) {
                    std::cout << "CONTINUATION_CHECKPOINT_FAILURE lambda_rec=4.000000000000e-02\n";
                    return 2;
                }
                initialize_stage_summary();
                const ProfilePhysicalSummary physical_004 = summarize_profiles(
                    point_004.profiles, &point_004.ion_trial.upstream_boundary);
                append_stage_summary(
                    "accepted", 0.03, 0.04, 0.01, "checkpoint",
                    point_004, &physical_004);
                return run_automatic_continuation(checkpoint);
            }
            if (!point_002.infeasible) return 2;

            std::cout << "CONTINUATION_FALLBACK failed_lambda_rec=2.000000000000e-02"
                      << " retry_lambda_rec=1.500000000000e-02\n";
            const double fallback_predictor =
                0.5 * (point_001.amplitude + 2.10e22);
            ContinuationPointResult point_0015 = solve_recycling_continuation_point(
                point_001.ion_trial, point_001.profiles, composition,
                0.015, fallback_predictor, point_001.amplitude);
            return point_0015.converged ? 0 : 2;
        } catch (const std::exception& error) {
            std::cout << "RECYCLING_BASE_RESULT lambda_rec=" << lambda_recycling
                      << " feasible=0 base_sweep_feasible="
                      << ((base_cells_converged && base_nonnegative) ? 1 : 0)
                      << " all_reverse_cells_converged="
                      << (base_cells_converged ? 1 : 0)
                      << " nonnegative=" << (base_nonnegative ? 1 : 0)
                      << " base_minimum_population=" << base_minimum_population
                      << " base_target_mismatch=" << base_mismatch
                      << " shooting_converged=0"
                      << " failure=" << std::quoted(error.what())
                      << "\n";
            return 2;
        }
    }

    ContinuationPointResult solve_recycling_continuation_point(
        const IonTrial& base_trial,
        const Profiles& base_profiles,
        const Vector& composition,
        double lambda_recycling,
        double requested_initial_amplitude,
        double safe_endpoint_amplitude) {
        ContinuationPointResult result;
        Profiles profiles = base_profiles;
        profiles.background = base_trial.background;
        double amplitude = requested_initial_amplitude;

        std::cout << "CONTINUATION_START lambda_rec=" << lambda_recycling
                  << " initial_amplitude=" << amplitude
                  << " base_amplitude=" << base_trial.amplitude
                  << " safe_endpoint_amplitude=" << safe_endpoint_amplitude
                  << "\n";
        for (int outer = 1; outer <= kMaxOuterIterations; ++outer) {
            current_outer_iteration_ = outer;
            const Profiles previous = profiles;
            const Profiles neutral_candidate = sweep_recycled_neutrals(
                profiles, lambda_recycling);
            Profiles candidate = neutral_candidate;
            const double neutral_relaxation = lambda_recycling >= 0.03
                ? relaxation_ : 1.0;
            if (neutral_relaxation < 1.0) {
                candidate = profiles;
                relax_neutral_profiles(
                    candidate, neutral_candidate, neutral_relaxation);
            }
            IonTrial ion_trial;
            try {
                ion_trial = shoot_nearby_controlled(
                    candidate, amplitude, composition, lambda_recycling,
                    safe_endpoint_amplitude);
            } catch (const std::exception& error) {
                result.infeasible = true;
                result.outer_iterations = outer;
                result.failure = error.what();
                std::cout << "CONTINUATION_RESULT lambda_rec=" << lambda_recycling
                          << " converged=0 outer_iterations=" << outer
                          << " failure=" << std::quoted(error.what())
                          << "\n";
                return result;
            }
            candidate.background = ion_trial.background;
            profiles = std::move(candidate);
            amplitude = ion_trial.amplitude;

            const double profile_change = max_relative_change(profiles, previous);
            auto [physical_target_A, physical_target_M] =
                target_recycling(profiles.background.front());
            physical_target_A *= lambda_recycling;
            physical_target_M *= lambda_recycling;
            const double target_A_change = vector_relative_change(
                profiles.flowA.front(), physical_target_A);
            const double target_M_change = vector_relative_change(
                profiles.flowM.front(), physical_target_M);
            const double boundary_change = std::max(target_A_change, target_M_change);
            std::cout << "CONTINUATION_OUTER lambda_rec=" << lambda_recycling
                      << " outer=" << outer
                      << " amplitude=" << amplitude
                      << " shooting_mismatch=" << ion_trial.mismatch
                      << " profile_change=" << profile_change
                      << " recycling_boundary_change=" << boundary_change
                      << " neutral_relaxation=" << neutral_relaxation
                      << " all_reverse_cells_converged="
                      << (ion_trial.diagnostics.cells_converged ? 1 : 0)
                      << "\n";

            if (std::abs(ion_trial.mismatch) < kOuterTolerance &&
                profile_change < kOuterTolerance &&
                boundary_change < kOuterTolerance &&
                ion_trial.diagnostics.cells_converged) {
                double minimum_population = std::numeric_limits<double>::infinity();
                for (const Vector& state : profiles.background) {
                    minimum_population = std::min(minimum_population, state.minCoeff());
                }
                double upstream_ion_nuclei_density = 0.0;
                for (int pi : ion_p_indices_) {
                    upstream_ion_nuclei_density +=
                        ion_trial.upstream_boundary.positive_ion_nuclei_flux_cm2_s(pi) /
                        ion_trial.upstream_boundary.positive_ion_velocity_cm_s(pi);
                }
                const double target_atomic_flux = boundary_.u_A * weighted_sum(
                    profiles.flowA.front(), boundary_.A_indices, levels_);
                const double target_molecular_flux = boundary_.u_M * weighted_sum(
                    profiles.flowM.front(), boundary_.M_indices, levels_);
                std::cout << "CONTINUATION_RESULT lambda_rec=" << lambda_recycling
                          << " converged=1 outer_iterations=" << outer
                          << " shooting_amplitude=" << amplitude
                          << " shooting_mismatch=" << ion_trial.mismatch
                          << " profile_change=" << profile_change
                          << " recycling_boundary_change=" << boundary_change
                          << " minimum_population=" << minimum_population
                          << " upstream_ion_nuclei_density="
                          << upstream_ion_nuclei_density
                          << " target_atomic_flux=" << target_atomic_flux
                          << " target_molecular_flux=" << target_molecular_flux
                          << " maximum_exact_residual="
                          << ion_trial.diagnostics.max_exact_residual
                          << " maximum_exact_cell="
                          << ion_trial.diagnostics.max_exact_cell
                          << " conservation_normalization_at_max_exact="
                          << ion_trial.diagnostics.max_exact_normalization
                          << " epsilon_Sigma_at_max_exact="
                          << ion_trial.diagnostics.epsilon_at_max_exact
                          << " maximum_epsilon_Sigma="
                          << ion_trial.diagnostics.max_conservation_epsilon
                          << " maximum_epsilon_cell="
                          << ion_trial.diagnostics.max_epsilon_cell
                          << " residual_at_max_epsilon="
                          << ion_trial.diagnostics.residual_at_max_epsilon
                          << " normalization_at_max_epsilon="
                          << ion_trial.diagnostics.normalization_at_max_epsilon
                          << "\n";
                result.converged = minimum_population >= 0.0;
                result.outer_iterations = outer;
                result.amplitude = amplitude;
                result.mismatch = ion_trial.mismatch;
                result.profile_change = profile_change;
                result.boundary_change = boundary_change;
                result.profiles = profiles;
                result.ion_trial = ion_trial;
                return result;
            }
        }
        result.outer_iterations = kMaxOuterIterations;
        result.failure = "outer_iteration_limit";
        std::cout << "CONTINUATION_RESULT lambda_rec=" << lambda_recycling
                  << " converged=0 outer_iterations=" << kMaxOuterIterations
                  << " failure=outer_iteration_limit"
                  << "\n";
        return result;
    }

#ifdef DCR_NESTED_DIAGNOSTIC
    struct ControllerActivityRow {
        const char* block = "none";
        int cell = -1;
        int local = -1;
        int gi = -1;
        int mu = 1;
        double activity = 0.0;
        bool trace = false;
        bool protected_pathway = false;
        std::vector<double> pathway_contributions;
    };

    struct ControllerActivitySnapshot {
        std::vector<ControllerActivityRow> P;
        std::vector<ControllerActivityRow> A;
        std::vector<ControllerActivityRow> M;
    };

    struct ControllerChangeSummary {
        double raw_maximum = 0.0;
        double filtered_maximum = 0.0;
        int raw_index = -1;
        int filtered_index = -1;
        int trace_rows = 0;
        int nontrace_rows = 0;
    };

    ControllerActivitySnapshot controller_activity_snapshot(
        const Profiles& state,
        const UpstreamIonBoundary& upstream_boundary,
        double lambda_recycling,
        double protection_fraction,
        bool include_P = true,
        bool include_A = true,
        bool include_M = true) const {
        constexpr double epsilon_abs = 1.0e-10;
        constexpr size_t pathway_count = 8;
        const double n_nuc_ref = std::max(1.0, config_.plasma.total_density);
        const double nu_ref = estimate_neutral_pseudo_time_steps(
            state, lambda_recycling).molecule_quantile_rate;
        ControllerActivitySnapshot snapshot;
        std::vector<ControllerActivityRow*> all_rows;
        std::vector<double> pathway_gross(pathway_count, 0.0);

        auto append = [&](std::vector<ControllerActivityRow>& rows,
                          const char* block,
                          int cell,
                          int local,
                          int gi,
                          double activity) -> ControllerActivityRow& {
            ControllerActivityRow row;
            row.block = block;
            row.cell = cell;
            row.local = local;
            row.gi = gi;
            row.mu = std::max(
                1, levels_[static_cast<size_t>(gi)].atomicity);
            row.activity = row.mu * activity;
            row.pathway_contributions.assign(pathway_count, 0.0);
            rows.push_back(std::move(row));
            all_rows.push_back(&rows.back());
            return rows.back();
        };

        snapshot.P.reserve(state.background.size() * boundary_.P_indices.size());
        snapshot.A.reserve(state.flowA.size() * boundary_.A_indices.size());
        snapshot.M.reserve(state.flowM.size() * boundary_.M_indices.size());
        all_rows.reserve(
            snapshot.P.capacity() + snapshot.A.capacity() + snapshot.M.capacity());

        dcr::solver::AtomicRateCalculator rate_calculator(atomic_data_);
        if (include_P) {
        for (size_t cell = 0; cell < state.background.size(); ++cell) {
            const Vector* upstream = cell + 1 < state.background.size()
                ? &state.background[cell + 1] : nullptr;
            const UpstreamIonBoundary* upstream_face = upstream == nullptr
                ? &upstream_boundary : nullptr;
            const PhysicalCellSystem physical = assemble_physical_cell_system(
                static_cast<int>(cell), state.background[cell], upstream,
                upstream_face, state.flowA, state.flowM);
            const Vector background_full = dcr::solver::make_background_full(
                state.background[cell], boundary_, total_states_);
            const auto local = dcr::solver::assemble_local_system(
                config_, atomic_data_, plasma_, grid_, boundary_, background_full,
                state.flowA[cell], state.flowM[cell], x_[cell], false, &rate_cache_);
            const auto rates = rate_calculator.evaluate(
                config_, boundary_, local, background_full, x_[cell], plasma_, grid_);
            const auto& ion_transport = rates.molecular_ion_transport;
            const double mcx_total = ion_transport.mcx_production_cm3_s.sum();
            const double mi_total = ion_transport.mi_production_cm3_s.sum();
            const double mar_branch_fraction = mcx_total /
                std::max(mcx_total + mi_total, 1.0e-300);
            for (int pi = 0; pi < state.background[cell].size(); ++pi) {
                const int gi = boundary_.P_indices[static_cast<size_t>(pi)];
                const auto& level = levels_[static_cast<size_t>(gi)];
                ControllerActivityRow& row = append(
                    snapshot.P, "P", static_cast<int>(cell), pi, gi,
                    physical.row_scales(pi));
                if (cell == 0) {
                    row.pathway_contributions[0] =
                        row.mu * state.background[cell](pi);
                    if (level.charge > 0) {
                        row.pathway_contributions[1] = row.mu *
                            ion_speed(gi, x_[cell]) * state.background[cell](pi);
                    }
                }
                if (rates.atomic_effective.valid &&
                    gi == rates.atomic_effective.ion_ground_index) {
                    row.pathway_contributions[6] = dx_[cell] *
                        rates.atomic_sources.effective_eir_rate_cm3_s;
                }
                if (level.atomicity == 2 && level.charge > 0) {
                    for (int i = 0;
                         i < static_cast<int>(ion_transport.state_indices.size()); ++i) {
                        if (ion_transport.state_indices[static_cast<size_t>(i)] != gi) {
                            continue;
                        }
                        const double dr_source = state.background[cell](pi) *
                            ion_transport.dr_h_source_frequency_s(i);
                        row.pathway_contributions[4] = dx_[cell] * dr_source;
                        row.pathway_contributions[5] = dx_[cell] *
                            ion_transport.mcx_production_cm3_s(i);
                        row.pathway_contributions[7] = dx_[cell] *
                            mar_branch_fraction * dr_source;
                        break;
                    }
                }
            }
        }
        } else {
            for (int pi = 0; pi < state.background.front().size(); ++pi) {
                pathway_gross[0] += std::abs(
                    muP_(pi) * state.background.front()(pi));
            }
        }

        auto [target_A, target_M] = target_recycling(state.background.front());
        target_A *= std::clamp(lambda_recycling, 0.0, 1.0);
        target_M *= std::clamp(lambda_recycling, 0.0, 1.0);
        auto append_boundary = [&](std::vector<ControllerActivityRow>& rows,
                                   const char* block,
                                   const Vector& current,
                                   const Vector& target,
                                   const std::vector<int>& indices,
                                   double velocity,
                                   size_t pathway) {
            const double rate = velocity / dx_.front();
            for (int local = 0; local < current.size(); ++local) {
                ControllerActivityRow& row = append(
                    rows, block, 0, local,
                    indices[static_cast<size_t>(local)],
                    std::max(rate * std::abs(current(local)),
                             rate * std::abs(target(local))));
                row.pathway_contributions[0] = row.mu * current(local);
                row.pathway_contributions[pathway] =
                    row.mu * velocity * current(local);
            }
        };
        auto add_omitted_boundary = [&](const Vector& current,
                                        const std::vector<int>& indices,
                                        double velocity,
                                        size_t pathway) {
            for (int local = 0; local < current.size(); ++local) {
                const int gi = indices[static_cast<size_t>(local)];
                const int mu = std::max(
                    1, levels_[static_cast<size_t>(gi)].atomicity);
                pathway_gross[0] += std::abs(mu * current(local));
                pathway_gross[pathway] +=
                    std::abs(mu * velocity * current(local));
            }
        };
        if (include_A) {
            append_boundary(
                snapshot.A, "A", state.flowA.front(), target_A,
                boundary_.A_indices, boundary_.u_A, 2);
        } else {
            add_omitted_boundary(
                state.flowA.front(), boundary_.A_indices, boundary_.u_A, 2);
        }
        if (include_M) {
            append_boundary(
                snapshot.M, "M", state.flowM.front(), target_M,
                boundary_.M_indices, boundary_.u_M, 3);
        } else {
            add_omitted_boundary(
                state.flowM.front(), boundary_.M_indices, boundary_.u_M, 3);
        }

        auto recycling_activity = [](const Vector& inflow,
                                     const Vector& current,
                                     const Matrix& rates,
                                     int row,
                                     double spatial_rate,
                                     double exhaust_rate) {
            double production = spatial_rate * std::abs(inflow(row));
            double loss = std::max(
                spatial_rate * std::abs(current(row)),
                exhaust_rate * std::abs(current(row)));
            for (int column = 0; column < current.size(); ++column) {
                const double contribution = rates(row, column) * current(column);
                production = std::max(production, contribution);
                loss = std::max(loss, -contribution);
            }
            return std::max(production, loss);
        };
        if (include_A || include_M) {
        for (size_t node = 1; node < state.background.size(); ++node) {
            const Vector background_full = dcr::solver::make_background_full(
                state.background[node], boundary_, total_states_);
            const auto local = dcr::solver::assemble_local_system(
                config_, atomic_data_, plasma_, grid_, boundary_, background_full,
                state.flowA[node - 1], state.flowM[node - 1], x_[node],
                false, &rate_cache_);
            const Matrix rates_A = extract_block(
                local.R_full, boundary_.A_indices, boundary_.A_indices);
            const Matrix rates_M = extract_block(
                local.R_full, boundary_.M_indices, boundary_.M_indices);
            const double spatial_A = boundary_.u_A / dx_[node - 1];
            const double spatial_M = boundary_.u_M / dx_[node - 1];
            const double exhaust_A = atom_exhaust_rate_;
            const double exhaust_M = molecule_exhaust_rate_;
            if (include_A) {
            for (int ai = 0; ai < state.flowA[node].size(); ++ai) {
                append(snapshot.A, "A", static_cast<int>(node), ai,
                    boundary_.A_indices[static_cast<size_t>(ai)],
                    recycling_activity(
                        state.flowA[node - 1], state.flowA[node], rates_A,
                        ai, spatial_A, exhaust_A));
            }
            }
            if (include_M) {
            for (int mi = 0; mi < state.flowM[node].size(); ++mi) {
                append(snapshot.M, "M", static_cast<int>(node), mi,
                    boundary_.M_indices[static_cast<size_t>(mi)],
                    recycling_activity(
                        state.flowM[node - 1], state.flowM[node], rates_M,
                        mi, spatial_M, exhaust_M));
            }
            }
        }
        }

        for (const ControllerActivityRow* row : all_rows) {
            for (size_t pathway = 0; pathway < pathway_count; ++pathway) {
                pathway_gross[pathway] +=
                    std::abs(row->pathway_contributions[pathway]);
            }
        }
        for (ControllerActivityRow* row : all_rows) {
            const double floor = epsilon_abs * row->mu * n_nuc_ref * nu_ref;
            const auto classification =
                dcr::solver::classify_activity_trace_row(
                    row->activity, floor, row->pathway_contributions,
                    pathway_gross, protection_fraction);
            row->trace = classification.trace;
            row->protected_pathway = classification.protected_pathway;
        }
        return snapshot;
    }

    static ControllerChangeSummary controller_change_summary(
        const std::vector<Vector>& previous,
        const std::vector<Vector>& next,
        const std::vector<ControllerActivityRow>& rows) {
        ControllerChangeSummary result;
        if (rows.empty() || previous.size() != next.size()) return result;
        const int first_cell = rows.front().cell;
        for (size_t row_index = 0; row_index < rows.size(); ++row_index) {
            const ControllerActivityRow& row = rows[row_index];
            const int cell = row.cell - first_cell;
            if (cell < 0 || cell >= static_cast<int>(previous.size()) ||
                row.local < 0 || row.local >= previous[cell].size() ||
                row.local >= next[cell].size()) {
                continue;
            }
            const double old_value = previous[cell](row.local);
            const double new_value = next[cell](row.local);
            const double scale = row.block[0] == 'P' && row.block[1] == '\0'
                ? std::max({1.0, std::abs(old_value), std::abs(new_value)})
                : std::max(
                    kPopulationChangeScaleFloor,
                    previous[cell].cwiseAbs().maxCoeff());
            const double change = std::abs(new_value - old_value) / scale;
            if (change > result.raw_maximum) {
                result.raw_maximum = change;
                result.raw_index = static_cast<int>(row_index);
            }
            if (row.trace) {
                ++result.trace_rows;
                continue;
            }
            ++result.nontrace_rows;
            if (change > result.filtered_maximum) {
                result.filtered_maximum = change;
                result.filtered_index = static_cast<int>(row_index);
            }
        }
        return result;
    }

    static ControllerChangeSummary controller_packed_change_summary(
        const Vector& previous,
        const Vector& next,
        const std::vector<ControllerActivityRow>& rows) {
        ControllerChangeSummary result;
        if (previous.size() != next.size() ||
            static_cast<size_t>(previous.size()) != rows.size()) {
            return result;
        }
        std::vector<bool> trace_mask;
        trace_mask.reserve(rows.size());
        for (const ControllerActivityRow& row : rows) {
            trace_mask.push_back(row.trace);
        }
        const std::vector<bool> no_trace(rows.size(), false);
        const auto raw = dcr::solver::maximum_nontrace_relative_change(
            previous, next, no_trace);
        const auto filtered = dcr::solver::maximum_nontrace_relative_change(
            previous, next, trace_mask);
        result.raw_maximum = raw.maximum;
        result.filtered_maximum = filtered.maximum;
        result.raw_index = raw.index;
        result.filtered_index = filtered.index;
        result.trace_rows = static_cast<int>(std::count(
            trace_mask.begin(), trace_mask.end(), true));
        result.nontrace_rows = filtered.evaluated_rows;
        return result;
    }

    static std::vector<bool> finite_ptc_trace_mask(
        const ControllerActivitySnapshot& snapshot) {
        std::vector<bool> mask;
        mask.reserve(snapshot.P.size() + snapshot.A.size() + snapshot.M.size());
        auto append = [&](const std::vector<ControllerActivityRow>& rows) {
            for (const ControllerActivityRow& row : rows) mask.push_back(row.trace);
        };
        append(snapshot.P);
        append(snapshot.A);
        append(snapshot.M);
        return mask;
    }
#endif

    Vector pack_recycled_profiles(const Profiles& profiles) const {
        const int atom_size = profiles.flowA.empty() ? 0 : profiles.flowA.front().size();
        const int molecule_size = profiles.flowM.empty() ? 0 : profiles.flowM.front().size();
        Vector packed(static_cast<int>(profiles.background.size()) *
                      (atom_size + molecule_size));
        int offset = 0;
        for (size_t k = 0; k < profiles.background.size(); ++k) {
            if (atom_size > 0) {
                packed.segment(offset, atom_size) = profiles.flowA[k];
                offset += atom_size;
            }
            if (molecule_size > 0) {
                packed.segment(offset, molecule_size) = profiles.flowM[k];
                offset += molecule_size;
            }
        }
        return packed;
    }

    void unpack_recycled_profiles(const Vector& packed, Profiles& profiles) const {
        const int atom_size = profiles.flowA.empty() ? 0 : profiles.flowA.front().size();
        const int molecule_size = profiles.flowM.empty() ? 0 : profiles.flowM.front().size();
        const int expected = static_cast<int>(profiles.background.size()) *
            (atom_size + molecule_size);
        if (packed.size() != expected) {
            throw std::runtime_error("Anderson recycled-profile size mismatch");
        }
        int offset = 0;
        for (size_t k = 0; k < profiles.background.size(); ++k) {
            if (atom_size > 0) {
                profiles.flowA[k] = packed.segment(offset, atom_size);
                offset += atom_size;
            }
            if (molecule_size > 0) {
                profiles.flowM[k] = packed.segment(offset, molecule_size);
                offset += molecule_size;
            }
        }
    }

    static Vector pack_finite_ptc_state(
        double log_amplitude,
        const Profiles& profiles) {
        int size = 1;
        auto add_size = [&](const std::vector<Vector>& block) {
            for (const Vector& values : block) size += values.size();
        };
        add_size(profiles.background);
        add_size(profiles.flowA);
        add_size(profiles.flowM);
        Vector packed(size);
        packed(0) = log_amplitude;
        int offset = 1;
        auto pack_block = [&](const std::vector<Vector>& block) {
            for (const Vector& values : block) {
                packed.segment(offset, values.size()) = values;
                offset += values.size();
            }
        };
        pack_block(profiles.background);
        pack_block(profiles.flowA);
        pack_block(profiles.flowM);
        return packed;
    }

    static void unpack_finite_ptc_state(
        const Vector& packed,
        double& log_amplitude,
        Profiles& profiles) {
        int expected = 1;
        auto add_size = [&](const std::vector<Vector>& block) {
            for (const Vector& values : block) expected += values.size();
        };
        add_size(profiles.background);
        add_size(profiles.flowA);
        add_size(profiles.flowM);
        if (packed.size() != expected) {
            throw std::runtime_error("Anderson finite-PTC state size mismatch");
        }
        log_amplitude = packed(0);
        int offset = 1;
        auto unpack_block = [&](std::vector<Vector>& block) {
            for (Vector& values : block) {
                values = packed.segment(offset, values.size());
                offset += values.size();
            }
        };
        unpack_block(profiles.background);
        unpack_block(profiles.flowA);
        unpack_block(profiles.flowM);
    }

    static double recycled_fixed_point_residual(
        const Vector& state,
        const Vector& image) {
        if (state.size() == 0) return 0.0;
        double residual = 0.0;
        for (int i = 0; i < state.size(); ++i) {
            const double scale = std::max({
                1.0, std::abs(state(i)), std::abs(image(i))});
            residual = std::max(
                residual, std::abs(image(i) - state(i)) / scale);
        }
        return residual;
    }

    PseudoTimeStepEstimate estimate_pseudo_time_step(
        const Profiles& profiles,
        double amplitude,
        const Vector& composition) const {
        struct WeightedRate {
            double rate = 0.0;
            double weight = 0.0;
            int cell = -1;
            int pi = -1;
            int gi = -1;
        };

        const UpstreamIonBoundary upstream_face =
            make_upstream_ion_boundary(amplitude, composition);
        PseudoTimeStepEstimate estimate;
        std::vector<WeightedRate> samples;
        std::vector<double> species_weights(
            static_cast<size_t>(total_states_), 0.0);
        std::vector<double> species_weighted_rates(
            static_cast<size_t>(total_states_), 0.0);
        std::vector<double> species_maximum_rates(
            static_cast<size_t>(total_states_), 0.0);
        double total_weight = 0.0;
        for (int k = static_cast<int>(profiles.background.size()) - 1; k >= 0; --k) {
            const Vector& state = profiles.background[static_cast<size_t>(k)];
            const Vector* upstream =
                k + 1 < static_cast<int>(profiles.background.size())
                ? &profiles.background[static_cast<size_t>(k + 1)] : nullptr;
            const PhysicalCellSystem system = assemble_physical_cell_system(
                k, state, upstream,
                upstream == nullptr ? &upstream_face : nullptr,
                profiles.flowA, profiles.flowM);
            const double local_nuclei = std::max(
                kDensityFloor,
                total_nuclei(profiles, static_cast<size_t>(k)));
            for (int row = 0; row < system.lhs.rows(); ++row) {
                const int gi = boundary_.P_indices[static_cast<size_t>(row)];
                const double population = std::max(0.0, state(row));
                const double rate = system.row_scales(row) /
                    std::max(population, kDensityFloor);
                const double weight = muP_(row) * population / local_nuclei;
                if (!(weight > 0.0) || !std::isfinite(rate)) continue;
                samples.push_back({rate, weight, k, row, gi});
                total_weight += weight;
                species_weights[static_cast<size_t>(gi)] += weight;
                species_weighted_rates[static_cast<size_t>(gi)] += weight * rate;
                species_maximum_rates[static_cast<size_t>(gi)] = std::max(
                    species_maximum_rates[static_cast<size_t>(gi)], rate);
                estimate.maximum_rate = std::max(estimate.maximum_rate, rate);
            }
        }
        if (samples.empty() || !(total_weight > 0.0)) {
            throw std::runtime_error(
                "Cannot estimate pseudo-time step from nuclei-weighted row rates");
        }
        std::sort(samples.begin(), samples.end(), [](const auto& lhs, const auto& rhs) {
            return lhs.rate < rhs.rate;
        });
        const double target_weight = estimate.quantile * total_weight;
        double cumulative_weight = 0.0;
        const WeightedRate* quantile_sample = &samples.back();
        for (const WeightedRate& sample : samples) {
            cumulative_weight += sample.weight;
            if (cumulative_weight >= target_weight) {
                quantile_sample = &sample;
                break;
            }
        }
        estimate.quantile_rate = quantile_sample->rate;
        estimate.limiting_cell = quantile_sample->cell;
        estimate.limiting_pi = quantile_sample->pi;
        estimate.limiting_gi = quantile_sample->gi;
        estimate.step = 0.1 / std::max(estimate.quantile_rate, 1.0e-30);

        for (int gi = 0; gi < total_states_; ++gi) {
            const double weight = species_weights[static_cast<size_t>(gi)];
            if (!(weight > 0.0)) continue;
            estimate.dominant_contributors.push_back({
                gi,
                weight / total_weight,
                species_weighted_rates[static_cast<size_t>(gi)] / weight,
                species_maximum_rates[static_cast<size_t>(gi)]});
        }
        std::sort(
            estimate.dominant_contributors.begin(),
            estimate.dominant_contributors.end(),
            [](const auto& lhs, const auto& rhs) {
                return lhs.nuclei_weight_fraction > rhs.nuclei_weight_fraction;
            });
        if (estimate.dominant_contributors.size() > 5) {
            estimate.dominant_contributors.resize(5);
        }
        return estimate;
    }

    ContinuationPointResult solve_recycling_continuation_point_anderson(
        const Profiles& base_profiles,
        const Vector& composition,
        double lambda_recycling,
        double requested_initial_amplitude,
        double safe_endpoint_amplitude,
        bool require_conservation = true,
        int maximum_outer_iterations = kMaxOuterIterations,
        const Vector* complete_predictor = nullptr,
        EndpointAuditResult* strict_endpoint_audit = nullptr,
        bool use_pseudo_time = false,
        bool plain_backward_euler_once = false,
        bool use_neutral_pseudo_time = false,
        const ContinuationCheckpoint* checkpoint_context = nullptr,
        bool fixed_amplitude_diagnostic = false,
        int diagnostic_max_inner_macro_iterations = 2000,
        int diagnostic_stagnation_window = 100,
        double diagnostic_neutral_molecule_step_scale = 1.0,
        bool fixed_gamma_activity_controller = false,
        double activity_trace_protection_fraction = 1.0e-3) {
#ifndef DCR_NESTED_DIAGNOSTIC
        (void)fixed_gamma_activity_controller;
        (void)activity_trace_protection_fraction;
#endif
        ContinuationPointResult result;
        const auto stage_start = std::chrono::steady_clock::now();
        Profiles profiles = base_profiles;
        dcr::solver::LogAmplitudeTrustRegion amplitude_trust;
        amplitude_trust.center = std::log(safe_endpoint_amplitude);
        const dcr::solver::LogAmplitudeBounds physical_amplitude_bounds{
            std::log(std::numeric_limits<double>::min()),
            std::log(std::numeric_limits<double>::max()),
            dcr::solver::LogAmplitudeBoundKind::physical_admissibility};
        auto amplitude_trust_bounds = [&]() {
            return amplitude_trust.bounds();
        };
        auto amplitude_in_bounds = [&](double candidate) {
            if (!(candidate > 0.0) || !std::isfinite(candidate)) return false;
            const double candidate_log = std::log(candidate);
            const auto trust = amplitude_trust_bounds();
            return candidate_log > trust.lower && candidate_log < trust.upper &&
                candidate_log > physical_amplitude_bounds.lower &&
                candidate_log < physical_amplitude_bounds.upper;
        };
        double amplitude = safe_endpoint_amplitude;
        if (amplitude_in_bounds(requested_initial_amplitude)) {
            amplitude = requested_initial_amplitude;
        }
        const double stage_tolerance = lambda_recycling >= 1.0 - 1.0e-14
            ? 1.0e-5 : 1.0e-4;
        std::vector<AndersonHistoryEntry> history;
        const Vector base_recycled = pack_recycled_profiles(base_profiles);
        const Vector base_map_recycled = pack_recycled_profiles(
            sweep_recycled_neutrals(base_profiles, lambda_recycling));
        Vector anderson_scale = Vector::Ones(base_recycled.size() + 1);
        for (int i = 0; i < base_recycled.size(); ++i) {
            anderson_scale(i + 1) = std::max({
                1.0, std::abs(base_recycled(i)), std::abs(base_map_recycled(i))});
        }
        Vector current_state(base_recycled.size() + 1);
        current_state(0) = std::log(amplitude);
        current_state.tail(base_recycled.size()) = base_recycled;
        Vector finite_ptc_anderson_scale = pack_finite_ptc_state(
            current_state(0), base_profiles);
        finite_ptc_anderson_scale = finite_ptc_anderson_scale
            .cwiseAbs().cwiseMax(Vector::Ones(
                finite_ptc_anderson_scale.size()));
        finite_ptc_anderson_scale(0) = 1.0;
        if (complete_predictor != nullptr &&
            complete_predictor->size() == current_state.size()) {
            auto predictor_valid = [&](const Vector& state) {
                if (!state.allFinite() ||
                    state.tail(base_recycled.size()).minCoeff() < 0.0) {
                    return false;
                }
                const double predicted_amplitude = std::exp(state(0));
                return amplitude_in_bounds(predicted_amplitude);
            };
            Vector candidate = *complete_predictor;
            double damping = 1.0;
            while (!predictor_valid(candidate) && damping > 1.0 / 1024.0) {
                damping *= 0.5;
                candidate = current_state + damping * (*complete_predictor - current_state);
            }
            if (predictor_valid(candidate)) current_state = std::move(candidate);
        }
        amplitude = std::exp(current_state(0));
        const double diagnostic_fixed_log_amplitude = current_state(0);
        if (fixed_amplitude_diagnostic) {
            std::cout << "FIXED_AMPLITUDE_DIAGNOSTIC_START"
                      << " ell=" << diagnostic_fixed_log_amplitude
                      << " Gamma_up=" << amplitude
                      << " target_mismatch_is_gate=0"
                      << "\n";
        }
#ifdef DCR_NESTED_DIAGNOSTIC
        if (fixed_gamma_activity_controller) {
            std::cout << "FIXED_GAMMA_ACTIVITY_CONTROLLER_START"
                      << " enabled=1"
                      << " default_off=1"
                      << " epsilon_abs=1e-10"
                      << " protection_fraction=" <<
                          activity_trace_protection_fraction
                       << " physical_equations_changed=0"
                       << " merit_changed=0"
                       << " convergence_gates_changed=0"
                       << " local_plasma_pseudo_time=0"
                       << " classification_state=event_state"
                      << "\n";
        }
#endif
        const PseudoTimeStepEstimate pseudo_time_estimate = use_pseudo_time
            ? estimate_pseudo_time_step(base_profiles, amplitude, composition)
            : PseudoTimeStepEstimate{};
        double pseudo_time_step = pseudo_time_estimate.step;
        const double initial_pseudo_time_step = pseudo_time_step;
        const double maximum_pseudo_time_step = initial_pseudo_time_step *
            std::max(1.0, config_.numerics.temperature_ptc_dtau_max_factor);
        const double minimum_pseudo_time_step =
            initial_pseudo_time_step * 1.0e-6;
#ifdef DCR_NESTED_DIAGNOSTIC
        constexpr bool use_local_plasma_pseudo_time = false;
        const double local_plasma_minimum_step =
            initial_pseudo_time_step * 1.0e-3;
        std::vector<Vector> local_plasma_steps;
        if (use_local_plasma_pseudo_time) {
            local_plasma_steps = base_profiles.background;
            const UpstreamIonBoundary initial_upstream_face =
                make_upstream_ion_boundary(amplitude, composition);
            for (size_t cell = 0; cell < base_profiles.background.size(); ++cell) {
                const Vector* upstream = cell + 1 < base_profiles.background.size()
                    ? &base_profiles.background[cell + 1] : nullptr;
                const UpstreamIonBoundary* upstream_face = upstream == nullptr
                    ? &initial_upstream_face : nullptr;
                const PhysicalCellSystem physical = assemble_physical_cell_system(
                    static_cast<int>(cell), base_profiles.background[cell], upstream,
                    upstream_face, base_profiles.flowA, base_profiles.flowM);
                for (int pi = 0; pi < local_plasma_steps[cell].size(); ++pi) {
                    const double rate = physical.row_scales(pi) /
                        std::max(base_profiles.background[cell](pi), kDensityFloor);
                    local_plasma_steps[cell](pi) = std::clamp(
                        0.1 / std::max(rate, 1.0e-30),
                        local_plasma_minimum_step, maximum_pseudo_time_step);
                }
            }
        }
        auto local_plasma_step_values = [&]() {
            std::vector<double> values;
            for (const Vector& cell : local_plasma_steps) {
                values.insert(values.end(), cell.data(), cell.data() + cell.size());
            }
            std::sort(values.begin(), values.end());
            return values;
        };
        auto local_plasma_quantile = [](const std::vector<double>& values,
                                        double probability) {
            if (values.empty()) return std::numeric_limits<double>::quiet_NaN();
            const size_t index = static_cast<size_t>(std::floor(
                probability * static_cast<double>(values.size() - 1)));
            return values[index];
        };
        auto local_plasma_hash = [&]() {
            std::uint64_t hash = 1469598103934665603ULL;
            for (const Vector& cell : local_plasma_steps) {
                const auto* bytes = reinterpret_cast<const unsigned char*>(cell.data());
                for (size_t i = 0;
                     i < static_cast<size_t>(cell.size()) * sizeof(double); ++i) {
                    hash ^= bytes[i];
                    hash *= 1099511628211ULL;
                }
            }
            return hash;
        };
        auto sync_local_plasma_summary_step = [&]() {
            if (!use_local_plasma_pseudo_time) return;
            const std::vector<double> values = local_plasma_step_values();
            pseudo_time_step = values.empty()
                ? initial_pseudo_time_step : values.front();
            if (!values.empty()) {
                result.local_plasma_step_minimum = values.front();
                result.local_plasma_step_q10 =
                    local_plasma_quantile(values, 0.10);
                result.local_plasma_step_median =
                    local_plasma_quantile(values, 0.50);
                result.local_plasma_step_q90 =
                    local_plasma_quantile(values, 0.90);
                result.local_plasma_step_maximum = values.back();
            }
        };
        auto local_inverse_plasma_steps = [&]() {
            std::vector<Vector> inverse = local_plasma_steps;
            for (Vector& cell : inverse) {
                cell = cell.cwiseInverse();
            }
            return inverse;
        };
        if (use_local_plasma_pseudo_time) {
            sync_local_plasma_summary_step();
            const std::vector<double> values = local_plasma_step_values();
            std::cout << "LOCAL_PSEUDO_TIME_INITIAL"
                      << " rows=" << values.size()
                      << " minimum=" << values.front()
                      << " q01=" << local_plasma_quantile(values, 0.01)
                      << " q10=" << local_plasma_quantile(values, 0.10)
                      << " median=" << local_plasma_quantile(values, 0.50)
                      << " q90=" << local_plasma_quantile(values, 0.90)
                      << " q99=" << local_plasma_quantile(values, 0.99)
                      << " maximum=" << values.back()
                      << " configured_minimum=" << local_plasma_minimum_step
                      << " configured_maximum=" << maximum_pseudo_time_step
                      << " hash=" << local_plasma_hash()
                      << " derivation=existing_row_rate_0p1_over_nu"
                      << "\n";
        }
#endif
        const NeutralPseudoTimeStepEstimate neutral_pseudo_time_estimate =
            use_neutral_pseudo_time
            ? estimate_neutral_pseudo_time_steps(base_profiles, lambda_recycling)
            : NeutralPseudoTimeStepEstimate{};
        double neutral_atom_step = neutral_pseudo_time_estimate.atom_step;
        double neutral_molecule_step = neutral_pseudo_time_estimate.molecule_step *
            diagnostic_neutral_molecule_step_scale;
        const double initial_neutral_atom_step = neutral_atom_step;
        const double initial_neutral_molecule_step = neutral_molecule_step;
        const double maximum_neutral_atom_step = initial_neutral_atom_step *
            std::max(1.0, config_.numerics.temperature_ptc_dtau_max_factor);
        const double maximum_neutral_molecule_step = initial_neutral_molecule_step *
            std::max(1.0, config_.numerics.temperature_ptc_dtau_max_factor);
        const double minimum_neutral_atom_step = initial_neutral_atom_step * 1.0e-6;
        const double minimum_neutral_molecule_step =
            initial_neutral_molecule_step * 1.0e-6;
        if (fixed_amplitude_diagnostic) {
            std::cout << "NESTED_DIAGNOSTIC_TIMESTEP_SCALE"
                      << " neutral_molecule="
                      << diagnostic_neutral_molecule_step_scale
                      << " initial_dtau_M=" << initial_neutral_molecule_step
                      << " maximum_dtau_M=" << maximum_neutral_molecule_step
                      << "\n";
        }
        std::vector<Vector> pseudo_time_old_background = profiles.background;
        double previous_steady_backward_error =
            std::numeric_limits<double>::infinity();
        double previous_accepted_combined_merit =
            std::numeric_limits<double>::infinity();
        int accepted_macro_step_streak = 0;
        result.pseudo_time_initial_step = pseudo_time_step;
        result.pseudo_time_final_step = pseudo_time_step;
        result.neutral_atom_initial_step = neutral_atom_step;
        result.neutral_atom_final_step = neutral_atom_step;
        result.neutral_molecule_initial_step = neutral_molecule_step;
        result.neutral_molecule_final_step = neutral_molecule_step;
        auto elapsed_seconds = [&]() {
            return std::chrono::duration<double>(
                std::chrono::steady_clock::now() - stage_start).count();
        };
        auto failure_cell = [](const std::string& message) {
            const std::string marker = "Reverse cell ";
            const size_t begin = message.find(marker);
            if (begin == std::string::npos) return -1;
            const size_t value_begin = begin + marker.size();
            try {
                return std::stoi(message.substr(value_begin));
            } catch (...) {
                return -1;
            }
        };
        auto failure_species = [](const std::string& message) {
            const std::string marker = "species_gi=";
            const size_t begin = message.find(marker);
            if (begin == std::string::npos) return -1;
            try {
                return std::stoi(message.substr(begin + marker.size()));
            } catch (...) {
                return -1;
            }
        };

        struct TrialEvaluation {
            bool valid = false;
            Vector state;
            Vector residual;
            Profiles profiles;
            IonTrial ion_trial;
            double fixed_point_residual = std::numeric_limits<double>::infinity();
            double target_residual = std::numeric_limits<double>::infinity();
            double recycled_profile_residual = std::numeric_limits<double>::infinity();
            double combined_merit = std::numeric_limits<double>::infinity();
            std::string failure;
        };

        std::cout << "AUTO_POINT_START lambda_rec=" << lambda_recycling
                  << " predictor=" << requested_initial_amplitude
                  << " safe_endpoint=" << safe_endpoint_amplitude
                   << " stage_tolerance=" << stage_tolerance
                   << " joint_unknown=log_amplitude_A_M"
                   << " anderson_depth=8 fallback_relaxation=0.2,0.15,0.1"
                   << " pseudo_time=" << (use_pseudo_time ? 1 : 0)
                   << " dtau=" << pseudo_time_step
                   << " neutral_pseudo_time="
                   << (use_neutral_pseudo_time ? 1 : 0)
                   << " neutral_dtau_A=" << neutral_atom_step
                   << " neutral_dtau_M=" << neutral_molecule_step
                   << "\n";

        auto evaluate_state = [&](const Vector& state) {
            TrialEvaluation evaluation;
            evaluation.state = state;
            if (state.size() != current_state.size() || !state.allFinite()) {
                evaluation.failure = "non-finite joint Anderson state";
                return evaluation;
            }
            const double trial_amplitude = std::exp(state(0));
            if (!amplitude_in_bounds(trial_amplitude)) {
                evaluation.failure = "controlled log-amplitude bound";
                return evaluation;
            }
            const Vector recycled_state = state.tail(base_recycled.size());
            if (!recycled_state.allFinite() || recycled_state.minCoeff() < 0.0) {
                evaluation.failure = "nonnegative recycled-profile bound";
                return evaluation;
            }
            Profiles candidate = profiles;
            unpack_recycled_profiles(recycled_state, candidate);
            try {
                evaluation.ion_trial = evaluate_ion_trial(
                    candidate, trial_amplitude, composition,
                    use_pseudo_time ? &pseudo_time_old_background : nullptr,
                    use_pseudo_time ? 1.0 / pseudo_time_step : 0.0);
            } catch (const std::exception& error) {
                evaluation.failure = error.what();
                return evaluation;
            }
            ++result.shooting_iterations;
            candidate.background = evaluation.ion_trial.background;
            double minimum_population = recycled_state.minCoeff();
            bool populations_finite = true;
            for (const Vector& background : candidate.background) {
                populations_finite = populations_finite && background.allFinite();
                minimum_population = std::min(
                    minimum_population, background.minCoeff());
            }
            if (!populations_finite || minimum_population < 0.0 ||
                !evaluation.ion_trial.diagnostics.cells_converged) {
                evaluation.failure = "nonnegative/local-cell feasibility";
                return evaluation;
            }
            if (use_pseudo_time &&
                evaluation.ion_trial.diagnostics.max_pseudo_time_backward_error >=
                    kLocalResidualTolerance) {
                evaluation.failure = "pseudo-time backward-Euler residual tolerance";
                return evaluation;
            }
            const Vector swept_recycled = pack_recycled_profiles(
                sweep_recycled_neutrals(candidate, lambda_recycling));
            evaluation.residual = Vector::Zero(state.size());
            evaluation.residual(0) = -evaluation.ion_trial.mismatch;
            evaluation.residual.tail(base_recycled.size()) =
                (swept_recycled - recycled_state).cwiseQuotient(
                    anderson_scale.tail(base_recycled.size()));
            evaluation.target_residual = std::abs(evaluation.ion_trial.mismatch);
            evaluation.recycled_profile_residual =
                recycled_fixed_point_residual(recycled_state, swept_recycled);
            evaluation.fixed_point_residual = std::max(
                evaluation.target_residual,
                evaluation.recycled_profile_residual);
            evaluation.combined_merit = std::max({
                evaluation.target_residual / stage_tolerance,
                evaluation.ion_trial.diagnostics.max_steady_backward_error /
                    stage_tolerance,
                evaluation.recycled_profile_residual / stage_tolerance});
            evaluation.profiles = std::move(candidate);
            evaluation.valid =
                std::isfinite(evaluation.fixed_point_residual) &&
                std::isfinite(evaluation.combined_merit);
            return evaluation;
        };

        auto physical_state = [&](const Vector& scaled_state) {
            return scaled_state.cwiseProduct(anderson_scale);
        };

        if (use_pseudo_time && !plain_backward_euler_once) {
            constexpr int plasma_steps_per_macro_cycle = 5;
            constexpr double macro_relaxation = 0.1;
            constexpr double maximum_plasma_state_change = 0.05;
            const double scheduler_minimum_pseudo_time_step =
                initial_pseudo_time_step * 1.0e-3;
            int successful_be_step_streak = 0;
            int steady_transition_streak = 0;
            int neutral_atom_success_streak = 0;
            int neutral_molecule_success_streak = 0;
            int neutral_atom_steady_macro_streak = 0;
            int neutral_molecule_steady_macro_streak = 0;
            int stable_macro_cycle_streak = 0;
            bool plasma_step_calibrated = false;
            bool neutral_atom_step_calibrated = false;
            bool neutral_molecule_step_calibrated =
                fixed_amplitude_diagnostic &&
                diagnostic_neutral_molecule_step_scale != 1.0;
            bool have_previous_amplitude_point = false;
            double previous_log_amplitude =
                std::numeric_limits<double>::quiet_NaN();
            double previous_amplitude_mismatch =
                std::numeric_limits<double>::quiet_NaN();
            bool fixed_ptc_anderson_mode = false;
            std::optional<dcr::solver::FixedPtcMapParameters>
                fixed_ptc_map_parameters;
#ifdef DCR_NESTED_DIAGNOSTIC
            std::optional<std::vector<Vector>> frozen_local_plasma_steps;
            auto same_local_plasma_steps = [](
                const std::vector<Vector>& left,
                const std::vector<Vector>& right) {
                if (left.size() != right.size()) return false;
                for (size_t cell = 0; cell < left.size(); ++cell) {
                    if (left[cell].size() != right[cell].size() ||
                        !(left[cell].array() == right[cell].array()).all()) {
                        return false;
                    }
                }
                return true;
            };
#endif
            if (config_.numerics.temperature_anderson_watchdog <
                2 * config_.numerics
                    .temperature_anderson_nonmonotone_window) {
                throw std::runtime_error(
                    "temperature_anderson_watchdog must be at least twice "
                    "temperature_anderson_nonmonotone_window");
            }
            dcr::solver::NonmonotoneMeritWatchdog anderson_merit(
                config_.numerics.temperature_anderson_nonmonotone_window,
                config_.numerics.temperature_anderson_watchdog);
            bool plasma_handoff_ready = false;
            bool neutral_atom_handoff_ready = !use_neutral_pseudo_time;
            bool neutral_molecule_handoff_ready = !use_neutral_pseudo_time;
            int anderson_ptc_restarts = 0;
            int anderson_phase_iterations = 0;
            int amplitude_bound_hits = 0;
            int local_step_cap_hits = 0;
            int trust_region_clip_hits = 0;
            int physical_bound_clip_hits = 0;
            int consecutive_local_step_caps = 0;
            int consecutive_trust_region_clips = 0;
            int consecutive_physical_bound_clips = 0;
            std::vector<double> ptc_true_merit_history;
            std::vector<double> macro_target_residual_history;
            std::vector<double> macro_profile_residual_history;
            Profiles initial_profiles = profiles;
            unpack_recycled_profiles(
                current_state.tail(base_recycled.size()), initial_profiles);
            profiles = std::move(initial_profiles);
            amplitude = std::exp(current_state(0));
            const int h2_ground_mi = molecule_ground_mi_;
            if (use_neutral_pseudo_time) {
                const Profiles base_neutral_image =
                    sweep_recycled_neutrals(profiles, lambda_recycling);
                Profiles perturbed = profiles;
                constexpr double directional_fraction = 1.0e-3;
                long double direction_norm_squared = 0.0L;
                for (size_t k = 0; k < profiles.flowM.size(); ++k) {
                    const Vector direction =
                        base_neutral_image.flowM[k] - profiles.flowM[k];
                    perturbed.flowM[k] += directional_fraction * direction;
                    direction_norm_squared += static_cast<long double>(
                        direction.squaredNorm());
                }
                const Profiles perturbed_image =
                    sweep_recycled_neutrals(perturbed, lambda_recycling);
                long double projected_change = 0.0L;
                for (size_t k = 0; k < profiles.flowM.size(); ++k) {
                    const Vector direction =
                        base_neutral_image.flowM[k] - profiles.flowM[k];
                    projected_change += static_cast<long double>(direction.dot(
                        perturbed_image.flowM[k] -
                        base_neutral_image.flowM[k]));
                }
                const double directional_amplification =
                    direction_norm_squared > 0.0L
                    ? static_cast<double>(projected_change /
                        (directional_fraction * direction_norm_squared))
                    : std::numeric_limits<double>::quiet_NaN();
                std::cout << "NEUTRAL_PSEUDO_TIME_INITIAL"
                          << " dtau_A=" << neutral_atom_step
                          << " dtau_M=" << neutral_molecule_step
                          << " q95_rate_A="
                          << neutral_pseudo_time_estimate.atom_quantile_rate
                          << " q95_rate_M="
                          << neutral_pseudo_time_estimate.molecule_quantile_rate
                          << " directional_amplification_M="
                          << directional_amplification
                          << " directional_fraction="
                          << directional_fraction
                          << "\n";
            }
            double previous_plasma_steady_residual = evaluate_residuals(
                profiles, make_upstream_ion_boundary(amplitude, composition))
                .max_steady_backward_error;

            auto profile_minimum = [](const Profiles& candidate) {
                double minimum = std::numeric_limits<double>::infinity();
                auto visit = [&](const std::vector<Vector>& block) {
                    for (const Vector& state : block) {
                        if (state.size() > 0) {
                            minimum = std::min(minimum, state.minCoeff());
                        }
                    }
                };
                visit(candidate.background);
                visit(candidate.flowA);
                visit(candidate.flowM);
                return minimum;
            };
            auto block_all_finite = [](const std::vector<Vector>& block) {
                return std::all_of(
                    block.begin(), block.end(),
                    [](const Vector& state) { return state.allFinite(); });
            };
            auto target_mismatch = [&](const Profiles& candidate) {
                const double nuclei =
                    weighted_sum(
                        candidate.background.front(),
                        boundary_.P_indices, levels_) +
                    weighted_sum(
                        candidate.flowA.front(),
                        boundary_.A_indices, levels_) +
                    weighted_sum(
                        candidate.flowM.front(),
                        boundary_.M_indices, levels_);
                return (nuclei - config_.plasma.total_density) /
                    config_.plasma.total_density;
            };
            struct JointStateInfeasibility {
                std::string reason;
                const char* block = "none";
                int cell = -1;
                int gi = -1;
                double value = std::numeric_limits<double>::quiet_NaN();
            };
            auto joint_state_infeasibility = [&](const Vector& state) {
                JointStateInfeasibility failure;
                if (state.size() != current_state.size()) {
                    failure.reason = "joint_state_size";
                    failure.block = "joint";
                    failure.value = static_cast<double>(state.size());
                    return failure;
                }
                auto locate_recycled = [&](int packed_index) {
                    const int atom_size = static_cast<int>(
                        boundary_.A_indices.size());
                    const int molecule_size = static_cast<int>(
                        boundary_.M_indices.size());
                    const int cell_width = atom_size + molecule_size;
                    failure.cell = packed_index / cell_width;
                    const int local = packed_index % cell_width;
                    if (local < atom_size) {
                        failure.block = "A";
                        failure.gi = boundary_.A_indices[
                            static_cast<size_t>(local)];
                    } else {
                        failure.block = "M";
                        failure.gi = boundary_.M_indices[
                            static_cast<size_t>(local - atom_size)];
                    }
                };
                for (int index = 0; index < state.size(); ++index) {
                    if (!std::isfinite(state(index))) {
                        failure.reason = "nonfinite_joint_state";
                        failure.value = state(index);
                        if (index == 0) {
                            failure.block = "amplitude";
                            failure.cell = 0;
                        } else {
                            locate_recycled(index - 1);
                        }
                        return failure;
                    }
                }
                for (int index = 0; index < base_recycled.size(); ++index) {
                    if (state(index + 1) < 0.0) {
                        failure.reason = "negative_recycled_population";
                        failure.value = state(index + 1);
                        locate_recycled(index);
                        return failure;
                    }
                }
                const double candidate_amplitude = std::exp(state(0));
                if (!std::isfinite(candidate_amplitude)) {
                    failure.reason = "nonfinite_amplitude";
                    failure.block = "amplitude";
                    failure.cell = 0;
                    failure.value = candidate_amplitude;
                    return failure;
                }
                if (!amplitude_in_bounds(candidate_amplitude)) {
                    failure.reason = "amplitude_trust_bound";
                    failure.block = "amplitude";
                    failure.cell = 0;
                    failure.value = candidate_amplitude;
                    return failure;
                }
                return failure;
            };
            auto joint_state_rejection_reason = [&](const Vector& state) {
                return joint_state_infeasibility(state).reason;
            };
            auto joint_state_is_feasible = [&](const Vector& state) {
                return joint_state_rejection_reason(state).empty();
            };
            auto finite_ptc_state_rejection_reason = [&](const Vector& state) {
                Profiles candidate = profiles;
                double candidate_log_amplitude =
                    std::numeric_limits<double>::quiet_NaN();
                try {
                    unpack_finite_ptc_state(
                        state, candidate_log_amplitude, candidate);
                } catch (const std::exception&) {
                    return std::string("finite_ptc_state_size");
                }
                if (!state.allFinite()) {
                    return std::string("nonfinite_finite_ptc_state");
                }
                const double candidate_amplitude =
                    std::exp(candidate_log_amplitude);
                if (!amplitude_in_bounds(candidate_amplitude)) {
                    return std::string("amplitude_trust_bound");
                }
                if (profile_minimum(candidate) < 0.0) {
                    return std::string("negative_finite_ptc_population");
                }
                return std::string{};
            };
            auto neutral_block_residual = [](const std::vector<Vector>& state,
                                             const std::vector<Vector>& image) {
                double maximum = 0.0;
                for (size_t k = 0; k < state.size(); ++k) {
                    for (int local = 0; local < state[k].size(); ++local) {
                        const double scale = std::max({
                            1.0, std::abs(state[k](local)),
                            std::abs(image[k](local))});
                        maximum = std::max(
                            maximum,
                            std::abs(image[k](local) - state[k](local)) / scale);
                    }
                }
                return maximum;
            };

            struct DominantNeutralResidual {
                const char* block = "none";
                int cell = -1;
                int gi = -1;
                double numerator = 0.0;
                double normalization = 1.0;
                double residual = 0.0;
            };
            auto dominant_neutral_residual = [&profiles, this](
                const Vector& state,
                const Vector& image) {
                DominantNeutralResidual dominant;
                int offset = 0;
                auto visit = [&](const char* block,
                                 int cell,
                                 const Vector& values,
                                 const std::vector<int>& global_indices) {
                    for (int local = 0; local < values.size(); ++local) {
                        const double numerator = image(offset) - state(offset);
                        const double normalization = std::max({
                            1.0, std::abs(state(offset)),
                            std::abs(image(offset))});
                        const double residual =
                            std::abs(numerator) / normalization;
                        if (residual > dominant.residual) {
                            dominant.block = block;
                            dominant.cell = cell;
                            dominant.gi = global_indices[
                                static_cast<size_t>(local)];
                            dominant.numerator = numerator;
                            dominant.normalization = normalization;
                            dominant.residual = residual;
                        }
                        ++offset;
                    }
                };
                for (size_t k = 0; k < profiles.background.size(); ++k) {
                    visit("A", static_cast<int>(k), profiles.flowA[k],
                          boundary_.A_indices);
                    visit("M", static_cast<int>(k), profiles.flowM[k],
                          boundary_.M_indices);
                }
                return dominant;
            };
            auto median = [](std::vector<double> values) {
                std::sort(values.begin(), values.end());
                const size_t middle = values.size() / 2;
                return values.size() % 2 == 0
                    ? 0.5 * (values[middle - 1] + values[middle])
                    : values[middle];
            };
#ifdef DCR_NESTED_DIAGNOSTIC
            int local_plasma_reduction_sequence = 0;
            auto report_activity_controller_event = [&] (
                const char* event,
                const char* block,
                int macro,
                int inner,
                const ControllerChangeSummary& change,
                const std::vector<ControllerActivityRow>& rows,
                const ControllerActivityRow* event_row,
                double legacy_raw_maximum,
                const char* action,
                int restored_trace_components,
                int nontrace_negative_components) {
                if (!fixed_gamma_activity_controller) return;
                const ControllerActivityRow* raw_row = change.raw_index >= 0 &&
                    change.raw_index < static_cast<int>(rows.size())
                    ? &rows[static_cast<size_t>(change.raw_index)] : nullptr;
                const ControllerActivityRow* filtered_row =
                    change.filtered_index >= 0 &&
                    change.filtered_index < static_cast<int>(rows.size())
                    ? &rows[static_cast<size_t>(change.filtered_index)] : nullptr;
                std::cout << "FIXED_GAMMA_ACTIVITY_CONTROLLER_EVENT"
                          << " event=" << event
                          << " action=" << action
                          << " macro=" << macro
                          << " inner=" << inner
                          << " epsilon_abs=1e-10"
                          << " protection_fraction=" <<
                              activity_trace_protection_fraction
                          << " raw_max_change=" << change.raw_maximum
                          << " controller_max_change=" <<
                              change.filtered_maximum
                          << " legacy_raw_max_change=" << legacy_raw_maximum
                          << " limiting_block=" <<
                              (filtered_row != nullptr
                                  ? filtered_row->block : block)
                          << " limiting_cell=" <<
                              (filtered_row != nullptr ? filtered_row->cell : -1)
                          << " limiting_gi=" <<
                              (filtered_row != nullptr ? filtered_row->gi : -1)
                          << " raw_limiting_cell=" <<
                              (raw_row != nullptr ? raw_row->cell : -1)
                          << " raw_limiting_block=" <<
                              (raw_row != nullptr ? raw_row->block : block)
                          << " raw_limiting_gi=" <<
                              (raw_row != nullptr ? raw_row->gi : -1)
                          << " raw_limiter_trace=" <<
                              (raw_row != nullptr && raw_row->trace ? 1 : 0)
                          << " raw_limiter_protected=" <<
                              (raw_row != nullptr && raw_row->protected_pathway ? 1 : 0)
                          << " event_cell=" <<
                              (event_row != nullptr ? event_row->cell : -1)
                          << " event_gi=" <<
                              (event_row != nullptr ? event_row->gi : -1)
                          << " event_trace=" <<
                              (event_row != nullptr && event_row->trace ? 1 : 0)
                          << " event_protected=" <<
                              (event_row != nullptr &&
                               event_row->protected_pathway ? 1 : 0)
                          << " trace_rows_evaluated=" << change.trace_rows
                          << " nontrace_rows_evaluated=" << change.nontrace_rows
                          << " restored_trace_components=" <<
                              restored_trace_components
                          << " nontrace_negative_components=" <<
                              nontrace_negative_components
                          << " classification_state=event_state"
                          << "\n";
            };
            auto report_local_plasma_reduction = [&](
                const char* kind,
                int macro,
                int inner,
                size_t cell,
                int pi,
                double before,
                double after,
                const IonTrial& trial) {
                double activity = std::numeric_limits<double>::quiet_NaN();
                double hybrid_residual = std::numeric_limits<double>::quiet_NaN();
                const char* classification = "ambiguous";
                if (cell < trial.background.size() &&
                    pi >= 0 && pi < trial.background[cell].size() &&
                    trial.background[cell].allFinite()) {
                    const Vector* upstream = cell + 1 < trial.background.size()
                        ? &trial.background[cell + 1] : nullptr;
                    const UpstreamIonBoundary* upstream_face = upstream == nullptr
                        ? &trial.upstream_boundary : nullptr;
                    const PhysicalCellSystem physical = assemble_physical_cell_system(
                        static_cast<int>(cell), trial.background[cell], upstream,
                        upstream_face, profiles.flowA, profiles.flowM);
                    const Vector residual = physical.lhs * trial.background[cell] -
                        physical.rhs;
                    const int gi = boundary_.P_indices[static_cast<size_t>(pi)];
                    const double mu = levels_[static_cast<size_t>(gi)].atomicity;
                    activity = mu * physical.row_scales(pi);
                    const double operational_floor = 1.0e-14 * mu *
                        std::max(1.0, config_.plasma.total_density) *
                        neutral_pseudo_time_estimate.molecule_quantile_rate;
                    hybrid_residual = mu * std::abs(residual(pi)) /
                        std::max({activity, operational_floor, 1.0e-300});
                    if (activity < operational_floor) {
                        classification = "negligible_trace_state";
                    } else if (activity < 100.0 * operational_floor) {
                        classification = "ambiguous";
                    } else if (hybrid_residual <= stage_tolerance) {
                        classification = "physically_active_fast_state";
                    } else {
                        classification =
                            "physically_active_far_from_equilibrium";
                    }
                }
                const int gi = pi >= 0 &&
                    pi < static_cast<int>(boundary_.P_indices.size())
                    ? boundary_.P_indices[static_cast<size_t>(pi)] : -1;
                std::cout << "PLASMA_TIMESTEP_REDUCTION"
                          << " sequence=" << ++local_plasma_reduction_sequence
                          << " kind=" << kind
                          << " macro=" << macro
                          << " inner=" << inner
                          << " cell=" << cell
                          << " pi=" << pi
                          << " gi=" << gi
                          << " state=" << std::quoted(
                              gi >= 0 ? species_label(gi) : "none")
                          << " Delta_tau_before=" << before
                          << " Delta_tau_after=" << after
                          << " reduction_factor=" << before / after
                          << " physical_activity=" << activity
                          << " hybrid_residual=" << hybrid_residual
                          << " throughput_classification=" << classification
                          << " density_not_used_for_classification=1"
                          << "\n";
                result.local_plasma_reductions =
                    local_plasma_reduction_sequence;
            };
#endif

            const int maximum_macro_cycles = fixed_amplitude_diagnostic
                ? std::max(1, diagnostic_max_inner_macro_iterations)
                : std::max(2000, maximum_outer_iterations);
            std::vector<double> diagnostic_inner_merit_history;
            for (int macro = 1; macro <= maximum_macro_cycles; ++macro) {
                current_outer_iteration_ = macro;
                const Profiles previous_macro_profiles = profiles;
                bool macro_had_retry = false;
                bool anderson_map_stationary_at_macro_start =
                    fixed_ptc_anderson_mode;
                if (fixed_ptc_anderson_mode) {
                    const dcr::solver::FixedPtcMapParameters current_parameters{
                        pseudo_time_step, neutral_atom_step,
                        neutral_molecule_step, amplitude_trust.center,
                        amplitude_trust.radius};
                    if (!fixed_ptc_map_parameters.has_value() ||
                        !dcr::solver::same_fixed_ptc_map(
                            *fixed_ptc_map_parameters, current_parameters)
#ifdef DCR_NESTED_DIAGNOSTIC
                        || (use_local_plasma_pseudo_time &&
                            (!frozen_local_plasma_steps.has_value() ||
                             !same_local_plasma_steps(
                                 *frozen_local_plasma_steps,
                                 local_plasma_steps)))
#endif
                        ) {
                        history.clear();
                        anderson_merit.clear();
                        result.failure = fixed_amplitude_diagnostic
                            ? "anderson_map_not_frozen"
                            : "fixed PTC map parameters changed with active Anderson history";
                        result.outer_iterations = macro;
                        result.elapsed_seconds = elapsed_seconds();
                        return result;
                    }
                }
                IonTrial last_plasma_trial;
                double last_be_step_change = std::numeric_limits<double>::infinity();
                const int plasma_steps_this_macro =
                    plasma_steps_per_macro_cycle;
                double last_transient_negligibility =
                    std::numeric_limits<double>::infinity();
                bool restart_ptc_after_steady_failure = false;
                bool force_anderson_handoff = false;

                for (int inner = 1; inner <= plasma_steps_this_macro; ++inner) {
                    bool accepted_be_step = false;
                    while (!accepted_be_step) {
                        const double used_pseudo_time_step = pseudo_time_step;
                        const std::vector<Vector> old_background = profiles.background;
#ifdef DCR_NESTED_DIAGNOSTIC
                        const std::vector<Vector> local_inverse_steps =
                            use_local_plasma_pseudo_time
                            ? local_inverse_plasma_steps()
                            : std::vector<Vector>{};
#endif
                        IonTrial plasma_trial;
                        std::string rejection_reason;
                        try {
                            plasma_trial = evaluate_ion_trial(
                                profiles, amplitude, composition,
                                &old_background,
                                1.0 / used_pseudo_time_step
#ifdef DCR_NESTED_DIAGNOSTIC
                                , use_local_plasma_pseudo_time
                                    ? &local_inverse_steps : nullptr
#endif
                                );
                        } catch (const std::exception& error) {
                            rejection_reason = error.what();
                        }
#ifdef DCR_NESTED_DIAGNOSTIC
                        constexpr bool emit_be_mismatch_forensics = false;
                        if (emit_be_mismatch_forensics &&
                            rejection_reason.empty() && macro >= 75 && macro <= 79 &&
                            plasma_trial.background.size() > 184 &&
                            old_background.size() == plasma_trial.background.size()) {
                            constexpr int diagnostic_cell = 183;
                            constexpr int diagnostic_pi = 31;
                            const Vector& diagnostic_state =
                                plasma_trial.background[diagnostic_cell];
                            const Vector& diagnostic_old = old_background[diagnostic_cell];
                            const Vector* diagnostic_upstream =
                                &plasma_trial.background[diagnostic_cell + 1];
                            const double inverse_step = 1.0 / used_pseudo_time_step;
                            const PhysicalCellSystem steady =
                                assemble_physical_cell_system(
                                    diagnostic_cell, diagnostic_state,
                                    diagnostic_upstream, nullptr,
                                    profiles.flowA, profiles.flowM);
                            PhysicalCellSystem backward_euler = steady;
                            for (int pi = 0; pi < diagnostic_state.size(); ++pi) {
                                backward_euler.lhs(pi, pi) += inverse_step;
                                backward_euler.rhs(pi) +=
                                    inverse_step * diagnostic_old(pi);
                            }
                            const Vector f_be = backward_euler.lhs * diagnostic_state -
                                backward_euler.rhs;
                            const Vector f_steady = steady.lhs * diagnostic_state -
                                steady.rhs;
                            const Vector transient = inverse_step *
                                (diagnostic_state - diagnostic_old);
                            const Vector defect = f_be - (transient + f_steady);
                            const double row_scale = std::max(
                                kPhysicalRowScaleFloor,
                                steady.row_scales(diagnostic_pi));
                            const double identity_scale = std::max({
                                kPhysicalRowScaleFloor, row_scale,
                                std::abs(f_be(diagnostic_pi)),
                                std::abs(transient(diagnostic_pi)),
                                std::abs(f_steady(diagnostic_pi))});

                            int conservation_cell = -1;
                            double conservation_relative = -1.0;
                            for (size_t cell = 0;
                                 cell < plasma_trial.cell_diagnostics.size(); ++cell) {
                                const double relative = plasma_trial
                                    .cell_diagnostics[cell]
                                    .conservation_be_identity_defect_relative;
                                if (relative > conservation_relative) {
                                    conservation_relative = relative;
                                    conservation_cell = static_cast<int>(cell);
                                }
                            }
                            const int diagnostic_gi = boundary_.P_indices[
                                static_cast<size_t>(diagnostic_pi)];
                            const CellDiagnostics& row_cell_diagnostics =
                                plasma_trial.cell_diagnostics[diagnostic_cell];
                            std::cout << "P_BE_IDENTITY_HISTORY"
                                      << " macro=" << macro
                                      << " inner=" << inner
                                      << " cell=" << diagnostic_cell
                                      << " x_cm=" << x_[diagnostic_cell]
                                      << " pi=" << diagnostic_pi
                                      << " gi=" << diagnostic_gi
                                      << " species=" << std::quoted(
                                             species_label(diagnostic_gi))
                                      << " replacement_pi="
                                      << row_cell_diagnostics.replacement_pi
                                      << " physical_row_replaced_in_solve="
                                      << (row_cell_diagnostics.replacement_pi ==
                                          diagnostic_pi ? 1 : 0)
                                      << " state=" << diagnostic_state(diagnostic_pi)
                                      << " old_state=" << diagnostic_old(diagnostic_pi)
                                      << " dtau=" << used_pseudo_time_step
                                      << " F_BE=" << f_be(diagnostic_pi)
                                      << " transient_term="
                                      << transient(diagnostic_pi)
                                      << " F_steady=" << f_steady(diagnostic_pi)
                                      << " D_BE=" << defect(diagnostic_pi)
                                      << " row_scale=" << row_scale
                                      << " identity_scale=" << identity_scale
                                      << " D_BE_scaled="
                                      << defect(diagnostic_pi) / identity_scale
                                      << " conservation_max_cell="
                                      << conservation_cell
                                      << " conservation_D_BE_relative="
                                      << conservation_relative
                                      << "\n";

                            if (macro == 79) {
                                const double gross_transient_lhs =
                                    inverse_step * diagnostic_state(diagnostic_pi);
                                const double gross_transient_rhs =
                                    inverse_step * diagnostic_old(diagnostic_pi);
                                const double represented_inverse_step =
                                    backward_euler.lhs(diagnostic_pi, diagnostic_pi) -
                                    steady.lhs(diagnostic_pi, diagnostic_pi);
                                const double represented_rhs_increment =
                                    backward_euler.rhs(diagnostic_pi) -
                                    steady.rhs(diagnostic_pi);
                                long double f_be_long =
                                    -static_cast<long double>(
                                        backward_euler.rhs(diagnostic_pi));
                                long double f_steady_long =
                                    -static_cast<long double>(steady.rhs(diagnostic_pi));
                                for (int column = 0;
                                     column < diagnostic_state.size(); ++column) {
                                    f_be_long += static_cast<long double>(
                                        backward_euler.lhs(diagnostic_pi, column)) *
                                        static_cast<long double>(
                                            diagnostic_state(column));
                                    f_steady_long += static_cast<long double>(
                                        steady.lhs(diagnostic_pi, column)) *
                                        static_cast<long double>(
                                            diagnostic_state(column));
                                }
                                const long double transient_long =
                                    static_cast<long double>(inverse_step) *
                                    (static_cast<long double>(
                                         diagnostic_state(diagnostic_pi)) -
                                     static_cast<long double>(
                                         diagnostic_old(diagnostic_pi)));
                                const long double defect_long = f_be_long -
                                    (transient_long + f_steady_long);
                                std::cout << "P_BE_IDENTITY_DETAIL"
                                          << " macro=" << macro
                                          << " inner=" << inner
                                          << " block=P"
                                          << " cell=" << diagnostic_cell
                                          << " x_cm=" << x_[diagnostic_cell]
                                          << " pi=" << diagnostic_pi
                                          << " gi=" << diagnostic_gi
                                          << " species=" << std::quoted(
                                                 species_label(diagnostic_gi))
                                          << " state="
                                          << diagnostic_state(diagnostic_pi)
                                          << " old_state="
                                          << diagnostic_old(diagnostic_pi)
                                          << " partner_A_cell=" << diagnostic_cell
                                          << " partner_M_cell=" << diagnostic_cell
                                          << " upstream_background_cell="
                                          << diagnostic_cell + 1
                                          << " upstream_used_by_row="
                                          << (is_positive_ion(diagnostic_pi) ? 1 : 0)
                                          << " boundary_face_used=0"
                                          << " replacement_pi="
                                          << row_cell_diagnostics.replacement_pi
                                          << " replacement_gi="
                                          << row_cell_diagnostics.replacement_gi
                                          << " identity_uses_physical_row=1"
                                          << " physical_row_replaced_in_solve="
                                          << (row_cell_diagnostics.replacement_pi ==
                                              diagnostic_pi ? 1 : 0)
                                          << " row_scale=" << row_scale
                                          << " identity_scale=" << identity_scale
                                          << " F_BE_raw=" << f_be(diagnostic_pi)
                                          << " transient_raw="
                                          << transient(diagnostic_pi)
                                          << " F_steady_raw="
                                          << f_steady(diagnostic_pi)
                                          << " D_BE_raw=" << defect(diagnostic_pi)
                                          << " F_BE_row_scaled="
                                          << f_be(diagnostic_pi) / row_scale
                                          << " transient_row_scaled="
                                          << transient(diagnostic_pi) / row_scale
                                          << " F_steady_row_scaled="
                                          << f_steady(diagnostic_pi) / row_scale
                                          << " D_BE_diagnostic_scaled="
                                          << defect(diagnostic_pi) / identity_scale
                                          << " F_BE_identity_scaled="
                                          << f_be(diagnostic_pi) / identity_scale
                                          << " transient_identity_scaled="
                                          << transient(diagnostic_pi) / identity_scale
                                          << " F_steady_identity_scaled="
                                          << f_steady(diagnostic_pi) / identity_scale
                                          << " inverse_dt=" << inverse_step
                                          << " gross_temporal_lhs="
                                          << gross_transient_lhs
                                          << " gross_temporal_rhs="
                                          << gross_transient_rhs
                                          << " intended_temporal_difference="
                                          << transient(diagnostic_pi)
                                          << " represented_inverse_dt="
                                          << represented_inverse_step
                                          << " represented_temporal_lhs="
                                          << represented_inverse_step *
                                                 diagnostic_state(diagnostic_pi)
                                          << " represented_temporal_rhs="
                                          << represented_rhs_increment
                                          << " represented_temporal_difference="
                                          << represented_inverse_step *
                                                 diagnostic_state(diagnostic_pi) -
                                                 represented_rhs_increment
                                          << " F_BE_long_double="
                                          << static_cast<double>(f_be_long)
                                          << " transient_long_double="
                                          << static_cast<double>(transient_long)
                                          << " F_steady_long_double="
                                          << static_cast<double>(f_steady_long)
                                          << " D_BE_long_double="
                                          << static_cast<double>(defect_long)
                                          << "\n";

                                for (int column = 0;
                                     column < diagnostic_state.size(); ++column) {
                                    const double coefficient =
                                        steady.lhs(diagnostic_pi, column);
                                    if (coefficient == 0.0) continue;
                                    const int column_gi = boundary_.P_indices[
                                        static_cast<size_t>(column)];
                                    std::cout << "P_BE_STEADY_LHS_TERM"
                                              << " macro=" << macro
                                              << " inner=" << inner
                                              << " row_pi=" << diagnostic_pi
                                              << " column_pi=" << column
                                              << " column_gi=" << column_gi
                                              << " column_species=" << std::quoted(
                                                     species_label(column_gi))
                                              << " coefficient=" << coefficient
                                              << " state="
                                              << diagnostic_state(column)
                                              << " contribution="
                                              << coefficient *
                                                     diagnostic_state(column)
                                              << "\n";
                                }
                                std::cout << "P_BE_RHS_TERM"
                                          << " macro=" << macro
                                          << " inner=" << inner
                                          << " row_pi=" << diagnostic_pi
                                          << " category=steady_rhs"
                                          << " contribution="
                                          << steady.rhs(diagnostic_pi)
                                          << " residual_sign=-1\n";
                                std::cout << "P_BE_TEMPORAL_TERM"
                                          << " macro=" << macro
                                          << " inner=" << inner
                                          << " row_pi=" << diagnostic_pi
                                          << " category=BE_lhs_diagonal"
                                          << " coefficient=" << inverse_step
                                          << " state="
                                          << diagnostic_state(diagnostic_pi)
                                          << " contribution="
                                          << gross_transient_lhs
                                          << "\n";
                                std::cout << "P_BE_TEMPORAL_TERM"
                                          << " macro=" << macro
                                          << " inner=" << inner
                                          << " row_pi=" << diagnostic_pi
                                          << " category=BE_rhs_old_state"
                                          << " coefficient=" << inverse_step
                                          << " old_state="
                                          << diagnostic_old(diagnostic_pi)
                                          << " contribution="
                                          << gross_transient_rhs
                                          << " residual_sign=-1\n";

                                const auto diagnostic_full =
                                    dcr::solver::make_background_full(
                                        diagnostic_state, boundary_, total_states_);
                                const auto diagnostic_local =
                                    dcr::solver::assemble_local_system(
                                        config_, atomic_data_, plasma_, grid_,
                                        boundary_, diagnostic_full,
                                        profiles.flowA[diagnostic_cell],
                                        profiles.flowM[diagnostic_cell],
                                        x_[diagnostic_cell], false, &rate_cache_);
                                const auto diagnostic_chemistry =
                                    dcr::solver::assemble_local_chemistry_sources(
                                        diagnostic_local, boundary_, atomic_data_,
                                        diagnostic_state,
                                        profiles.flowA[diagnostic_cell],
                                        profiles.flowM[diagnostic_cell]);
                                const SpeciesSourceBreakdown breakdown =
                                    source_breakdown(
                                        diagnostic_pi, diagnostic_local,
                                        diagnostic_state,
                                        profiles.flowA[diagnostic_cell],
                                        profiles.flowM[diagnostic_cell],
                                        x_[diagnostic_cell]);
                                const double exhaust_rate =
                                    background_exhaust_rate(diagnostic_pi);
                                std::cout << "P_BE_PHYSICAL_TERM_SUMMARY"
                                          << " macro=" << macro
                                          << " inner=" << inner
                                          << " row_pi=" << diagnostic_pi
                                          << " transport_mode="
                                          << (is_positive_ion(diagnostic_pi)
                                                  ? "reverse_upstream" :
                                                    "local_exhaust")
                                          << " exhaust_rate=" << exhaust_rate
                                          << " exhaust="
                                          << exhaust_rate *
                                                 diagnostic_state(diagnostic_pi)
                                          << " chemistry_assembled="
                                          << diagnostic_chemistry.background(
                                                 diagnostic_pi)
                                          << " chemistry_channels_production="
                                          << breakdown.production
                                          << " chemistry_channels_loss="
                                          << breakdown.loss
                                          << " chemistry_channel_net="
                                          << breakdown.production - breakdown.loss
                                          << " chemistry_reconstruction_error="
                                          << breakdown.production - breakdown.loss -
                                                 diagnostic_chemistry.background(
                                                     diagnostic_pi)
                                          << " F_steady_reconstructed="
                                          << exhaust_rate *
                                                 diagnostic_state(diagnostic_pi) -
                                                 diagnostic_chemistry.background(
                                                     diagnostic_pi)
                                          << "\n";
                                print_source_channels(
                                    "be_steady_macro79", diagnostic_pi,
                                    breakdown, "P_BE_SOURCE_TERM");

                                if (conservation_cell >= 0) {
                                    const Vector& conservation_state =
                                        plasma_trial.background[conservation_cell];
                                    const Vector& conservation_old =
                                        old_background[conservation_cell];
                                    const Vector* conservation_upstream =
                                        conservation_cell + 1 < static_cast<int>(
                                            plasma_trial.background.size())
                                        ? &plasma_trial.background[
                                              conservation_cell + 1]
                                        : nullptr;
                                    const UpstreamIonBoundary* conservation_face =
                                        conservation_upstream == nullptr
                                        ? &plasma_trial.upstream_boundary : nullptr;
                                    const PhysicalCellSystem conservation_steady =
                                        assemble_physical_cell_system(
                                            conservation_cell, conservation_state,
                                            conservation_upstream, conservation_face,
                                            profiles.flowA, profiles.flowM);
                                    PhysicalCellSystem conservation_be =
                                        conservation_steady;
                                    for (int pi = 0;
                                         pi < conservation_state.size(); ++pi) {
                                        conservation_be.lhs(pi, pi) += inverse_step;
                                        conservation_be.rhs(pi) +=
                                            inverse_step * conservation_old(pi);
                                    }
                                    const Vector conservation_f_be =
                                        conservation_be.lhs * conservation_state -
                                        conservation_be.rhs;
                                    const Vector conservation_f_steady =
                                        conservation_steady.lhs * conservation_state -
                                        conservation_steady.rhs;
                                    const Vector conservation_transient = inverse_step *
                                        (conservation_state - conservation_old);
                                    const Vector conservation_defect = conservation_f_be -
                                        (conservation_transient +
                                         conservation_f_steady);
                                    const double projected_be =
                                        muP_.dot(conservation_f_be);
                                    const double projected_transient =
                                        muP_.dot(conservation_transient);
                                    const double projected_steady =
                                        muP_.dot(conservation_f_steady);
                                    const double projected_defect = projected_be -
                                        (projected_transient + projected_steady);
                                    const double projected_scale = std::max({
                                        kPhysicalRowScaleFloor,
                                        std::abs(projected_be),
                                        std::abs(projected_transient),
                                        std::abs(projected_steady)});
                                    long double row_defect_projection = 0.0L;
                                    for (int pi = 0;
                                         pi < conservation_defect.size(); ++pi) {
                                        row_defect_projection +=
                                            static_cast<long double>(muP_(pi)) *
                                            static_cast<long double>(
                                                conservation_defect(pi));
                                    }
                                    const CellDiagnostics& conservation_diagnostics =
                                        plasma_trial.cell_diagnostics[
                                            static_cast<size_t>(conservation_cell)];
                                    std::cout << "P_BE_CONSERVATION_DETAIL"
                                              << " macro=" << macro
                                              << " inner=" << inner
                                              << " cell=" << conservation_cell
                                              << " x_cm=" << x_[conservation_cell]
                                              << " replacement_pi="
                                              << conservation_diagnostics.replacement_pi
                                              << " replacement_gi="
                                              << conservation_diagnostics.replacement_gi
                                              << " projection_uses_physical_rows=1"
                                              << " projected_F_BE=" << projected_be
                                              << " projected_transient="
                                              << projected_transient
                                              << " projected_F_steady="
                                              << projected_steady
                                              << " projected_D_BE="
                                              << projected_defect
                                              << " projected_scale="
                                              << projected_scale
                                              << " projected_D_BE_scaled="
                                              << projected_defect / projected_scale
                                              << " projection_of_row_D_BE="
                                              << static_cast<double>(
                                                     row_defect_projection)
                                              << " stored_projected_D_BE="
                                              << conservation_diagnostics
                                                     .conservation_be_identity_defect
                                              << " stored_projected_D_BE_scaled="
                                              << conservation_diagnostics
                                                     .conservation_be_identity_defect_relative
                                              << "\n";
                                    for (int pi = 0;
                                         pi < conservation_defect.size(); ++pi) {
                                        if (conservation_defect(pi) == 0.0) continue;
                                        const int gi = boundary_.P_indices[
                                            static_cast<size_t>(pi)];
                                        std::cout << "P_BE_CONSERVATION_ROW_TERM"
                                                  << " macro=" << macro
                                                  << " inner=" << inner
                                                  << " cell=" << conservation_cell
                                                  << " pi=" << pi
                                                  << " gi=" << gi
                                                  << " species=" << std::quoted(
                                                         species_label(gi))
                                                  << " mu=" << muP_(pi)
                                                  << " row_D_BE="
                                                  << conservation_defect(pi)
                                                  << " projected_contribution="
                                                  << muP_(pi) *
                                                         conservation_defect(pi)
                                                  << "\n";
                                    }
                                }
                            }
                        }
#endif
                        ++result.shooting_iterations;

                        double minimum_population =
                            std::numeric_limits<double>::infinity();
                        bool populations_finite = true;
                        for (const Vector& state : plasma_trial.background) {
                            populations_finite = populations_finite && state.allFinite();
                            if (state.size() > 0) {
                                minimum_population = std::min(
                                    minimum_population, state.minCoeff());
                            }
                        }
                        const double plasma_equation_residual =
                            plasma_trial.diagnostics
                                .max_pseudo_time_backward_error;
                        const double raw_plasma_state_change =
                            plasma_trial.diagnostics.pseudo_time_profile_change;
                        double plasma_state_change = raw_plasma_state_change;
#ifdef DCR_NESTED_DIAGNOSTIC
                        std::optional<ControllerActivitySnapshot>
                            plasma_activity_snapshot;
                        ControllerChangeSummary plasma_change;
                        const bool classify_plasma_controller_event =
                            fixed_gamma_activity_controller &&
                            (!plasma_step_calibrated ||
                             raw_plasma_state_change >
                                 maximum_plasma_state_change ||
                             !rejection_reason.empty() ||
                             !plasma_trial.diagnostics.cells_converged ||
                             plasma_equation_residual >=
                                 kLocalResidualTolerance ||
                             !populations_finite || minimum_population < 0.0);
                        if (classify_plasma_controller_event) {
                            Profiles event_profiles = profiles;
                            if (plasma_trial.background.size() ==
                                profiles.background.size()) {
                                event_profiles.background = plasma_trial.background;
                            }
                            plasma_activity_snapshot = controller_activity_snapshot(
                                event_profiles,
                                make_upstream_ion_boundary(amplitude, composition),
                                lambda_recycling,
                                activity_trace_protection_fraction,
                                true, false, false);
                            plasma_change = controller_change_summary(
                                old_background, event_profiles.background,
                                plasma_activity_snapshot->P);
                            plasma_state_change = plasma_change.filtered_maximum;
                        }
#endif
                        const bool be_converged = rejection_reason.empty() &&
                            plasma_trial.diagnostics.cells_converged &&
                            plasma_equation_residual < kLocalResidualTolerance;
                        const bool populations_valid = rejection_reason.empty() &&
                            populations_finite && minimum_population >= 0.0;
                        const bool state_change_valid = rejection_reason.empty() &&
                            plasma_state_change <= maximum_plasma_state_change;
                        const bool physical_residual_valid =
                            rejection_reason.empty();

                        if (!be_converged || !populations_valid ||
                            !state_change_valid || !physical_residual_valid) {
                            if (rejection_reason.empty()) {
                                if (!be_converged) {
                                    rejection_reason = "backward_euler_residual";
                                } else if (!populations_valid) {
                                    rejection_reason = "population_feasibility";
                                } else if (!state_change_valid) {
                                    rejection_reason = "excessive_state_change";
                                } else {
                                    rejection_reason =
                                        "catastrophic_physical_residual_growth";
                                }
                            }
#ifdef DCR_NESTED_DIAGNOSTIC
                            if (classify_plasma_controller_event) {
                                report_activity_controller_event(
                                    "P_timestep_limiter", "P", macro, inner,
                                    plasma_change, plasma_activity_snapshot->P,
                                    nullptr, raw_plasma_state_change,
                                    "rejected", 0, 0);
                            }
#endif
                            ++result.pseudo_time_rejected_steps;
                            successful_be_step_streak = 0;
                            macro_had_retry = true;
                            if (fixed_ptc_anderson_mode) {
                                ++anderson_ptc_restarts;
                                if (anderson_ptc_restarts >
                                    config_.numerics
                                        .temperature_anderson_max_restarts) {
                                    result.infeasible = true;
                                    result.outer_iterations = macro;
                                    result.failure =
                                        "anderson finite PTC map failed after restarts: " +
                                        rejection_reason;
                                    result.attempted_amplitude = amplitude;
                                    result.elapsed_seconds = elapsed_seconds();
                                    return result;
                                }
                                profiles = previous_macro_profiles;
                                history.clear();
                                anderson_merit.clear();
                                fixed_ptc_map_parameters.reset();
#ifdef DCR_NESTED_DIAGNOSTIC
                                frozen_local_plasma_steps.reset();
#endif
                                fixed_ptc_anderson_mode = false;
                                plasma_handoff_ready = false;
                                neutral_atom_handoff_ready =
                                    !use_neutral_pseudo_time;
                                neutral_molecule_handoff_ready =
                                    !use_neutral_pseudo_time;
                                steady_transition_streak = 0;
                                neutral_atom_steady_macro_streak = 0;
                                neutral_molecule_steady_macro_streak = 0;
                                successful_be_step_streak = 0;
                                neutral_atom_success_streak = 0;
                                neutral_molecule_success_streak = 0;
                                plasma_step_calibrated = false;
                                neutral_atom_step_calibrated = false;
                                neutral_molecule_step_calibrated =
                                    fixed_amplitude_diagnostic &&
                                    diagnostic_neutral_molecule_step_scale != 1.0;
#ifdef DCR_NESTED_DIAGNOSTIC
                                if (use_local_plasma_pseudo_time) {
                                    bool reduced_any = false;
                                    for (size_t cell = 0;
                                         cell < local_plasma_steps.size(); ++cell) {
                                        for (int pi = 0;
                                             pi < local_plasma_steps[cell].size(); ++pi) {
                                            const ControllerActivityRow* activity_row =
                                                fixed_gamma_activity_controller
                                                ? &plasma_activity_snapshot->P[
                                                    cell * boundary_.P_indices.size() +
                                                    static_cast<size_t>(pi)]
                                                : nullptr;
                                            if (activity_row != nullptr &&
                                                activity_row->trace) {
                                                continue;
                                            }
                                            const bool have_trial_value =
                                                cell < plasma_trial.background.size() &&
                                                pi < plasma_trial.background[cell].size();
                                            const double change = have_trial_value
                                                ? std::abs(
                                                    plasma_trial.background[cell](pi) -
                                                    old_background[cell](pi)) /
                                                    std::max({
                                                        1.0,
                                                        std::abs(old_background[cell](pi)),
                                                        std::abs(plasma_trial.background[cell](pi))})
                                                : std::numeric_limits<double>::infinity();
                                            const bool implicated =
                                                rejection_reason !=
                                                    "excessive_state_change" ||
                                                change > maximum_plasma_state_change;
                                            if (!implicated) continue;
                                            const double before =
                                                local_plasma_steps[cell](pi);
                                            const double after = std::max(
                                                local_plasma_minimum_step,
                                                0.25 * before);
                                            if (after < before) {
                                                local_plasma_steps[cell](pi) = after;
                                                 report_local_plasma_reduction(
                                                     "anderson_restart", macro, inner,
                                                     cell, pi, before, after,
                                                     plasma_trial);
                                                if (fixed_gamma_activity_controller) {
                                                    report_activity_controller_event(
                                                        "P_local_timestep_reduction",
                                                        "P", macro, inner,
                                                        plasma_change,
                                                        plasma_activity_snapshot->P,
                                                        activity_row,
                                                        raw_plasma_state_change,
                                                        "anderson_restart", 0, 0);
                                                }
                                                 reduced_any = true;
                                            }
                                        }
                                    }
                                    if (!reduced_any) force_anderson_handoff = true;
                                    sync_local_plasma_summary_step();
                                } else
#endif
                                {
                                    pseudo_time_step = std::max(
                                        scheduler_minimum_pseudo_time_step,
                                        0.25 * pseudo_time_step);
                                }
                                neutral_atom_step = std::max(
                                    minimum_neutral_atom_step,
                                    0.25 * neutral_atom_step);
                                neutral_molecule_step = std::max(
                                    minimum_neutral_molecule_step,
                                    0.25 * neutral_molecule_step);
                                ptc_true_merit_history.clear();
                                result.pseudo_time_final_step = pseudo_time_step;
                                restart_ptc_after_steady_failure = true;
                                accepted_be_step = true;
                                std::cout << "ANDERSON_RETURN_TO_PTC"
                                          << " macro=" << macro
                                          << " restart="
                                          << anderson_ptc_restarts
                                          << " reason="
                                          << std::quoted(rejection_reason)
                                          << " dtau_P=" << pseudo_time_step
                                          << " dtau_A=" << neutral_atom_step
                                          << " dtau_M=" << neutral_molecule_step
                                          << " limiting_block=P"
                                          << " limiting_cell="
                                          << plasma_trial.diagnostics
                                              .max_pseudo_time_profile_change_cell
                                          << " limiting_gi="
                                          << plasma_trial.diagnostics
                                              .max_pseudo_time_profile_change_gi
                                          << " limiting_species=" << std::quoted(
                                              plasma_trial.diagnostics
                                                      .max_pseudo_time_profile_change_gi >= 0
                                                  ? species_label(plasma_trial.diagnostics
                                                        .max_pseudo_time_profile_change_gi)
                                                  : "none")
                                          << " normalized_state_change="
                                          << plasma_state_change
                                          << "\n";
                                break;
                            }
#ifdef DCR_NESTED_DIAGNOSTIC
                            if (use_local_plasma_pseudo_time) {
                                bool reduced_any = false;
                                for (size_t cell = 0;
                                     cell < local_plasma_steps.size(); ++cell) {
                                    for (int pi = 0;
                                         pi < local_plasma_steps[cell].size(); ++pi) {
                                        const ControllerActivityRow* activity_row =
                                            fixed_gamma_activity_controller
                                            ? &plasma_activity_snapshot->P[
                                                cell * boundary_.P_indices.size() +
                                                static_cast<size_t>(pi)]
                                            : nullptr;
                                        if (activity_row != nullptr &&
                                            activity_row->trace) {
                                            continue;
                                        }
                                        const bool have_trial_value =
                                            cell < plasma_trial.background.size() &&
                                            pi < plasma_trial.background[cell].size();
                                        const double change = have_trial_value
                                            ? std::abs(
                                                plasma_trial.background[cell](pi) -
                                                old_background[cell](pi)) /
                                                std::max({
                                                    1.0,
                                                    std::abs(old_background[cell](pi)),
                                                    std::abs(plasma_trial.background[cell](pi))})
                                            : std::numeric_limits<double>::infinity();
                                        const bool implicated =
                                            rejection_reason !=
                                                "excessive_state_change" ||
                                            change > maximum_plasma_state_change;
                                        if (!implicated) continue;
                                        const double before =
                                            local_plasma_steps[cell](pi);
                                        const double after = std::max(
                                            local_plasma_minimum_step,
                                            0.5 * before);
                                        if (after < before) {
                                            local_plasma_steps[cell](pi) = after;
                                             report_local_plasma_reduction(
                                                 "rejected_step", macro, inner,
                                                 cell, pi, before, after,
                                                 plasma_trial);
                                            if (fixed_gamma_activity_controller) {
                                                report_activity_controller_event(
                                                    "P_local_timestep_reduction",
                                                    "P", macro, inner,
                                                    plasma_change,
                                                    plasma_activity_snapshot->P,
                                                    activity_row,
                                                    raw_plasma_state_change,
                                                    "rejected_step", 0, 0);
                                            }
                                             reduced_any = true;
                                        }
                                    }
                                }
                                if (reduced_any) {
                                    history.clear();
                                    anderson_merit.clear();
                                    sync_local_plasma_summary_step();
                                    result.pseudo_time_final_step =
                                        pseudo_time_step;
                                    continue;
                                }
                                force_anderson_handoff = true;
                                accepted_be_step = true;
                                std::cout <<
                                    "PSEUDO_TIME_ANDERSON_HANDOFF_REQUEST"
                                          << " macro=" << macro
                                          << " reason=" << std::quoted(
                                              "local_minimum_dtau_without_accepted_step")
                                          << " rejection_reason="
                                          << std::quoted(rejection_reason)
                                          << " dtau_P_min=" << pseudo_time_step
                                          << "\n";
                                break;
                            }
#endif
                            if (pseudo_time_step <=
                                scheduler_minimum_pseudo_time_step) {
                                force_anderson_handoff = true;
                                accepted_be_step = true;
                                std::cout << "PSEUDO_TIME_ANDERSON_HANDOFF_REQUEST"
                                          << " macro=" << macro
                                          << " reason="
                                          << std::quoted(
                                              "minimum_dtau_without_accepted_step")
                                          << " rejection_reason="
                                          << std::quoted(rejection_reason)
                                          << " dtau_P=" << pseudo_time_step
                                          << "\n";
                                break;
                            }
                            pseudo_time_step = std::max(
                                scheduler_minimum_pseudo_time_step,
                                0.5 * pseudo_time_step);
                            result.pseudo_time_final_step = pseudo_time_step;
                            std::cout << "PSEUDO_TIME_PLASMA_STEP"
                                      << " accepted=0 macro=" << macro
                                      << " inner=" << inner
                                      << " dtau=" << used_pseudo_time_step
                                      << " next_dtau=" << pseudo_time_step
                                      << " be_residual="
                                      << plasma_equation_residual
                                      << " steady_residual="
                                      << plasma_trial.diagnostics
                                          .max_steady_backward_error
                                      << " state_change="
                                      << plasma_state_change
                                      << " limiting_block=P"
                                      << " limiting_cell="
                                      << plasma_trial.diagnostics
                                          .max_pseudo_time_profile_change_cell
                                      << " limiting_gi="
                                      << plasma_trial.diagnostics
                                          .max_pseudo_time_profile_change_gi
                                      << " limiting_species=" << std::quoted(
                                          plasma_trial.diagnostics
                                                  .max_pseudo_time_profile_change_gi >= 0
                                              ? species_label(plasma_trial.diagnostics
                                                    .max_pseudo_time_profile_change_gi)
                                              : "none")
                                      << " reason=" << std::quoted(rejection_reason)
                                      << "\n";
                            continue;
                        }

#ifdef DCR_NESTED_DIAGNOSTIC
#endif
                        const double transient_negligibility =
                            plasma_trial.diagnostics
                                .max_transient_update_to_steady_row_scale_ratio;
                        if (!fixed_ptc_anderson_mode &&
                            !plasma_step_calibrated) {
                            constexpr double calibration_minimum_change = 1.0e-4;
                            constexpr double calibration_maximum_change = 1.0e-2;
#ifdef DCR_NESTED_DIAGNOSTIC
                            if (use_local_plasma_pseudo_time) {
                                bool changed_any = false;
                                for (size_t cell = 0;
                                     cell < local_plasma_steps.size(); ++cell) {
                                    for (int pi = 0;
                                         pi < local_plasma_steps[cell].size(); ++pi) {
                                        const ControllerActivityRow* activity_row =
                                            fixed_gamma_activity_controller
                                            ? &plasma_activity_snapshot->P[
                                                cell * boundary_.P_indices.size() +
                                                static_cast<size_t>(pi)]
                                            : nullptr;
                                        if (activity_row != nullptr &&
                                            activity_row->trace) {
                                            continue;
                                        }
                                        const double change = std::abs(
                                            plasma_trial.background[cell](pi) -
                                            old_background[cell](pi)) /
                                            std::max({
                                                1.0,
                                                std::abs(old_background[cell](pi)),
                                                std::abs(plasma_trial.background[cell](pi))});
                                        const double before =
                                            local_plasma_steps[cell](pi);
                                        double after = before;
                                        if (change > calibration_maximum_change) {
                                            after = std::max(
                                                local_plasma_minimum_step,
                                                0.5 * before);
                                            if (after < before) {
                                                 report_local_plasma_reduction(
                                                     "calibration", macro, inner,
                                                     cell, pi, before, after,
                                                     plasma_trial);
                                                if (fixed_gamma_activity_controller) {
                                                    report_activity_controller_event(
                                                        "P_local_timestep_reduction",
                                                        "P", macro, inner,
                                                        plasma_change,
                                                        plasma_activity_snapshot->P,
                                                        activity_row,
                                                        raw_plasma_state_change,
                                                        "calibration", 0, 0);
                                                }
                                             }
                                        } else if (change < calibration_minimum_change &&
                                                   transient_negligibility >= 1.0e-8) {
                                            after = std::min(
                                                maximum_pseudo_time_step,
                                                10.0 * before);
                                            if (after > before) {
                                                const int gi = boundary_.P_indices[
                                                    static_cast<size_t>(pi)];
                                                std::cout <<
                                                    "LOCAL_PSEUDO_TIME_INCREASE"
                                                          << " macro=" << macro
                                                          << " inner=" << inner
                                                          << " cell=" << cell
                                                          << " gi=" << gi
                                                          << " Delta_tau_before="
                                                          << before
                                                          << " Delta_tau_after="
                                                          << after
                                                          << " increase_factor="
                                                          << after / before
                                                          << "\n";
                                            }
                                        }
                                        if (after != before) {
                                            local_plasma_steps[cell](pi) = after;
                                            changed_any = true;
                                        }
                                    }
                                }
                                if (changed_any) {
                                    history.clear();
                                    anderson_merit.clear();
                                    sync_local_plasma_summary_step();
                                    result.pseudo_time_final_step =
                                        pseudo_time_step;
                                    continue;
                                }
                                plasma_step_calibrated = true;
                                const std::vector<double> values =
                                    local_plasma_step_values();
                                std::cout << "LOCAL_PSEUDO_TIME_CALIBRATED"
                                          << " macro=" << macro
                                          << " inner=" << inner
                                          << " minimum=" << values.front()
                                          << " median=" <<
                                              local_plasma_quantile(values, 0.50)
                                          << " maximum=" << values.back()
                                          << " hash=" << local_plasma_hash()
                                          << "\n";
                            } else
#endif
                            {
                            if (plasma_state_change > calibration_maximum_change) {
                                const double reduced_step = std::max(
                                    scheduler_minimum_pseudo_time_step,
                                    0.5 * pseudo_time_step);
                                if (reduced_step < pseudo_time_step) {
                                    pseudo_time_step = reduced_step;
                                    result.pseudo_time_final_step = pseudo_time_step;
                                    std::cout << "PSEUDO_TIME_PLASMA_CALIBRATION"
                                              << " accepted=0 macro=" << macro
                                              << " inner=" << inner
                                              << " dtau=" << used_pseudo_time_step
                                              << " next_dtau=" << pseudo_time_step
                                              << " state_change="
                                              << plasma_state_change
                                              << " reason=change_above_target\n";
#ifdef DCR_NESTED_DIAGNOSTIC
                                    if (fixed_gamma_activity_controller) {
                                        report_activity_controller_event(
                                            "P_timestep_calibration", "P", macro,
                                            inner, plasma_change,
                                            plasma_activity_snapshot->P, nullptr,
                                            raw_plasma_state_change,
                                            "change_above_target", 0, 0);
                                    }
#endif
                                    continue;
                                }
                            } else if (plasma_state_change <
                                           calibration_minimum_change &&
                                       transient_negligibility >= 1.0e-8) {
                                pseudo_time_step *= 10.0;
                                result.pseudo_time_final_step = pseudo_time_step;
                                std::cout << "PSEUDO_TIME_PLASMA_CALIBRATION"
                                          << " accepted=0 macro=" << macro
                                          << " inner=" << inner
                                          << " dtau=" << used_pseudo_time_step
                                          << " next_dtau=" << pseudo_time_step
                                          << " state_change="
                                          << plasma_state_change
                                          << " reason=change_below_target\n";
#ifdef DCR_NESTED_DIAGNOSTIC
                                if (fixed_gamma_activity_controller) {
                                    report_activity_controller_event(
                                        "P_timestep_calibration", "P", macro,
                                        inner, plasma_change,
                                        plasma_activity_snapshot->P, nullptr,
                                        raw_plasma_state_change,
                                        "change_below_target", 0, 0);
                                }
#endif
                                continue;
                            }
                            plasma_step_calibrated = true;
                            std::cout << "PSEUDO_TIME_PLASMA_CALIBRATION"
                                      << " accepted=1 macro=" << macro
                                      << " inner=" << inner
                                      << " dtau=" << used_pseudo_time_step
                                      << " state_change=" << plasma_state_change
                                      << "\n";
#ifdef DCR_NESTED_DIAGNOSTIC
                            if (fixed_gamma_activity_controller) {
                                report_activity_controller_event(
                                    "P_timestep_calibration", "P", macro, inner,
                                    plasma_change, plasma_activity_snapshot->P,
                                    nullptr, raw_plasma_state_change, "accepted",
                                    0, 0);
                            }
#endif
                            }
                        }

                        if (fixed_amplitude_diagnostic &&
                            (plasma_trial.diagnostics
                                 .max_be_identity_defect_relative >
                                 kLocalResidualTolerance ||
                             plasma_trial.diagnostics
                                 .max_conservation_be_identity_defect_relative >
                                 kLocalResidualTolerance)) {
                            result.failure = "be_steady_operator_mismatch";
                            result.outer_iterations = macro;
                            result.attempted_amplitude = amplitude;
                            result.amplitude = amplitude;
                            result.profiles = profiles;
                            result.first_failing_cell = plasma_trial.diagnostics
                                .max_be_identity_defect_cell;
                            result.first_failing_species = plasma_trial.diagnostics
                                .max_be_identity_defect_gi;
                            result.elapsed_seconds = elapsed_seconds();
                            std::cout << "NESTED_INNER_STATUS"
                                      << " status=be_steady_operator_mismatch"
                                      << " block=P"
                                      << " macro=" << macro
                                      << " cell=" << result.first_failing_cell
                                      << " gi=" << result.first_failing_species
                                      << " row=" << plasma_trial.diagnostics
                                          .max_be_identity_defect_pi
                                      << " D_BE_relative="
                                      << plasma_trial.diagnostics
                                          .max_be_identity_defect_relative
                                      << " D_BE="
                                      << plasma_trial.diagnostics
                                          .max_be_identity_defect_norm
                                      << " conservation_D_BE_relative="
                                      << plasma_trial.diagnostics
                                          .max_conservation_be_identity_defect_relative
                                      << " conservation_D_BE="
                                      << plasma_trial.diagnostics
                                          .max_conservation_be_identity_defect
                                      << " tolerance="
                                      << kLocalResidualTolerance
                                      << "\n";
                            return result;
                        }

                        profiles.background = plasma_trial.background;
                        previous_plasma_steady_residual =
                            plasma_trial.diagnostics.max_steady_backward_error;
                        last_be_step_change =
                            plasma_state_change;
                        last_transient_negligibility = transient_negligibility;
                        last_plasma_trial = std::move(plasma_trial);
                        ++result.pseudo_time_accepted_steps;
                        if (!fixed_ptc_anderson_mode &&
                            transient_negligibility >= 1.0e-8) {
                            steady_transition_streak = 0;
                            plasma_handoff_ready = false;
                            ++successful_be_step_streak;
                            if (successful_be_step_streak >= 3) {
#ifdef DCR_NESTED_DIAGNOSTIC
                                if (use_local_plasma_pseudo_time) {
                                    bool changed_any = false;
                                    for (size_t cell_index = 0;
                                         cell_index < local_plasma_steps.size();
                                         ++cell_index) {
                                        Vector& cell =
                                            local_plasma_steps[cell_index];
                                        for (int pi = 0; pi < cell.size(); ++pi) {
                                            const double component_change = std::abs(
                                                last_plasma_trial.background[
                                                    cell_index](pi) -
                                                old_background[cell_index](pi)) /
                                                std::max({
                                                    1.0,
                                                    std::abs(last_plasma_trial.background[
                                                        cell_index](pi)),
                                                    std::abs(old_background[
                                                        cell_index](pi))});
                                            if (component_change >= 1.0e-4) continue;
                                            const double increased = std::min(
                                                maximum_pseudo_time_step,
                                                1.5 * cell(pi));
                                            changed_any = changed_any ||
                                                increased != cell(pi);
                                            cell(pi) = increased;
                                        }
                                    }
                                    if (changed_any) {
                                        history.clear();
                                        anderson_merit.clear();
                                        sync_local_plasma_summary_step();
                                    }
                                } else
#endif
                                {
                                pseudo_time_step = std::min(
                                    maximum_pseudo_time_step,
                                    1.5 * pseudo_time_step);
                                }
                                successful_be_step_streak = 0;
                            }
                            result.pseudo_time_final_step = pseudo_time_step;
                        } else if (!fixed_ptc_anderson_mode) {
                            ++steady_transition_streak;
                        }
                        std::cout << "PSEUDO_TIME_PLASMA_STEP"
                                  << " accepted=1 macro=" << macro
                                  << " inner=" << inner
                                  << " mode="
                                  << (fixed_ptc_anderson_mode
                                      ? "fixed_ptc" : "adaptive_ptc")
                                  << " dtau=" << used_pseudo_time_step
                                  << " next_dtau=" << pseudo_time_step
                                  << " eta_tau=" << transient_negligibility;
                        std::cout << " be_residual=" << plasma_equation_residual
                                  << " steady_residual="
                                  << last_plasma_trial.diagnostics
                                      .max_steady_backward_error
                                  << " state_change=" << last_be_step_change
                                  << " raw_state_change="
                                  << last_plasma_trial.diagnostics
                                      .max_pseudo_time_state_change_norm
                                  << " raw_transient_norm="
                                  << last_plasma_trial.diagnostics
                                      .max_pseudo_time_transient_term_norm
                                  << " raw_steady_norm="
                                  << last_plasma_trial.diagnostics
                                      .max_steady_residual_norm
                                  << " raw_be_norm="
                                  << last_plasma_trial.diagnostics
                                      .max_pseudo_time_residual_norm
                                  << " D_BE="
                                  << last_plasma_trial.diagnostics
                                      .max_be_identity_defect_norm
                                  << " D_BE_relative="
                                  << last_plasma_trial.diagnostics
                                      .max_be_identity_defect_relative
                                  << " D_BE_cell="
                                  << last_plasma_trial.diagnostics
                                      .max_be_identity_defect_cell
                                  << " D_BE_gi="
                                  << last_plasma_trial.diagnostics
                                      .max_be_identity_defect_gi
                                  << " conservation_D_BE="
                                  << last_plasma_trial.diagnostics
                                      .max_conservation_be_identity_defect
                                  << " conservation_D_BE_relative="
                                  << last_plasma_trial.diagnostics
                                      .max_conservation_be_identity_defect_relative
                                  << "\n";
                        accepted_be_step = true;
                    }
                    if (restart_ptc_after_steady_failure) break;
                    if (force_anderson_handoff) break;
                }
                if (restart_ptc_after_steady_failure) continue;
                if (force_anderson_handoff) {
                    fixed_ptc_anderson_mode = true;
                    fixed_ptc_map_parameters.reset();
                    anderson_phase_iterations = 0;
                    history.clear();
                    anderson_merit.clear();
                    consecutive_local_step_caps = 0;
                    consecutive_trust_region_clips = 0;
                    consecutive_physical_bound_clips = 0;
                }
                if (!fixed_ptc_anderson_mode &&
                    steady_transition_streak >= 3) {
                    plasma_handoff_ready = true;
                    successful_be_step_streak = 0;
                    std::cout << "PSEUDO_TIME_HANDOFF_READY"
                              << " block=P"
                              << " macro=" << macro
                              << " consecutive_steps="
                              << steady_transition_streak
                              << " eta_tau=" << last_transient_negligibility
                              << " steady_residual="
                              << last_plasma_trial.diagnostics
                                  .max_steady_backward_error
                              << "\n";
                }

                NeutralPseudoTimeSweepResult last_neutral_sweep;
                if (use_neutral_pseudo_time) {
                    constexpr int neutral_steps_per_macro_cycle = 3;
                    constexpr double maximum_neutral_state_change = 0.05;
                    for (int neutral_inner = 1;
                         neutral_inner <= neutral_steps_per_macro_cycle;
                         ++neutral_inner) {
                        bool atom_step_accepted = false;
                        bool molecule_step_accepted = false;
                        bool accepted_neutral_step = false;
                        while (!accepted_neutral_step) {
                            const double attempted_atom_step = neutral_atom_step;
                            const double attempted_molecule_step =
                                neutral_molecule_step;
                            const double inverse_dt_A =
                                1.0 / attempted_atom_step;
                            const double inverse_dt_M =
                                1.0 / attempted_molecule_step;
                            NeutralPseudoTimeSweepResult trial;
                            std::string solve_failure;
                            try {
                                trial = sweep_recycled_neutrals_pseudo_time(
                                    profiles, inverse_dt_A, inverse_dt_M,
                                    lambda_recycling, !atom_step_accepted,
                                    !molecule_step_accepted);
                            } catch (const std::exception& error) {
                                solve_failure = error.what();
                            }

#ifdef DCR_NESTED_DIAGNOSTIC
                            std::optional<ControllerActivitySnapshot>
                                neutral_activity_snapshot;
                            std::vector<ControllerActivityRow>
                                atom_timestep_activity_rows;
                            std::vector<ControllerActivityRow>
                                molecule_timestep_activity_rows;
                            ControllerChangeSummary atom_change;
                            ControllerChangeSummary molecule_change;
                            const bool classify_neutral_controller_event =
                                fixed_gamma_activity_controller &&
                                (!solve_failure.empty() ||
                                 (!atom_step_accepted &&
                                  (!neutral_atom_step_calibrated ||
                                   trial.atom.profile_change >
                                       maximum_neutral_state_change ||
                                   trial.atom.modified_backward_error >= 1.0e-6 ||
                                   !std::isfinite(trial.atom.minimum_population) ||
                                   trial.atom.minimum_population < 0.0)) ||
                                 (!molecule_step_accepted &&
                                  (!neutral_molecule_step_calibrated ||
                                   trial.molecule.profile_change >
                                       maximum_neutral_state_change ||
                                   trial.molecule.modified_backward_error >=
                                       1.0e-6 ||
                                   !std::isfinite(
                                       trial.molecule.minimum_population) ||
                                   trial.molecule.minimum_population < 0.0)));
                            if (classify_neutral_controller_event) {
                                const Profiles& event_profiles = solve_failure.empty()
                                    ? trial.profiles : profiles;
                                neutral_activity_snapshot =
                                    controller_activity_snapshot(
                                        event_profiles,
                                        make_upstream_ion_boundary(
                                            amplitude, composition),
                                        lambda_recycling,
                                        activity_trace_protection_fraction,
                                        false, true, true);
                                const size_t atom_boundary_rows =
                                    boundary_.A_indices.size();
                                const size_t molecule_boundary_rows =
                                    boundary_.M_indices.size();
                                atom_timestep_activity_rows.assign(
                                    neutral_activity_snapshot->A.begin() +
                                        static_cast<std::ptrdiff_t>(
                                            atom_boundary_rows),
                                    neutral_activity_snapshot->A.end());
                                molecule_timestep_activity_rows.assign(
                                    neutral_activity_snapshot->M.begin() +
                                        static_cast<std::ptrdiff_t>(
                                            molecule_boundary_rows),
                                    neutral_activity_snapshot->M.end());
                                const std::vector<Vector> previous_A(
                                    profiles.flowA.begin() + 1,
                                    profiles.flowA.end());
                                const std::vector<Vector> event_A(
                                    event_profiles.flowA.begin() + 1,
                                    event_profiles.flowA.end());
                                const std::vector<Vector> previous_M(
                                    profiles.flowM.begin() + 1,
                                    profiles.flowM.end());
                                const std::vector<Vector> event_M(
                                    event_profiles.flowM.begin() + 1,
                                    event_profiles.flowM.end());
                                atom_change = controller_change_summary(
                                    previous_A, event_A,
                                    atom_timestep_activity_rows);
                                molecule_change = controller_change_summary(
                                    previous_M, event_M,
                                    molecule_timestep_activity_rows);
                            }
#endif
                            const double atom_controller_change =
#ifdef DCR_NESTED_DIAGNOSTIC
                                classify_neutral_controller_event
                                ? atom_change.filtered_maximum :
#endif
                                trial.atom.profile_change;
                            const double molecule_controller_change =
#ifdef DCR_NESTED_DIAGNOSTIC
                                classify_neutral_controller_event
                                ? molecule_change.filtered_maximum :
#endif
                                trial.molecule.profile_change;

                            bool atom_failed = false;
                            bool molecule_failed = false;
                            std::string atom_reason;
                            std::string molecule_reason;
                            if (!solve_failure.empty()) {
                                atom_failed = !atom_step_accepted &&
                                    solve_failure.find("Atomic neutral block:") !=
                                        std::string::npos;
                                molecule_failed = !molecule_step_accepted &&
                                    solve_failure.find("Molecular neutral block:") !=
                                        std::string::npos;
                                if (!atom_failed && !molecule_failed) {
                                    atom_failed = !atom_step_accepted;
                                    molecule_failed = !molecule_step_accepted;
                                }
                                atom_reason = atom_failed
                                    ? "implicit_solve_failure" : "";
                                molecule_reason = molecule_failed
                                    ? "implicit_solve_failure" : "";
                            } else {
                                const bool atom_finite =
                                    block_all_finite(trial.profiles.flowA) &&
                                    std::isfinite(trial.atom.minimum_population);
                                const bool molecule_finite =
                                    block_all_finite(trial.profiles.flowM) &&
                                    std::isfinite(trial.molecule.minimum_population);
                                if (!atom_step_accepted && !atom_finite) {
                                    atom_failed = true;
                                    atom_reason = "nonfinite_state";
                                } else if (!atom_step_accepted &&
                                           trial.atom.minimum_population < 0.0) {
                                    atom_failed = true;
                                    atom_reason = "negative_population";
                                } else if (!atom_step_accepted &&
                                           trial.atom.modified_backward_error >= 1.0e-6) {
                                    atom_failed = true;
                                    atom_reason = "implicit_solve_failure";
                                } else if (!atom_step_accepted &&
                                           atom_controller_change >
                                           maximum_neutral_state_change) {
                                    atom_failed = true;
                                    atom_reason = "catastrophic_state_change";
                                }
                                if (!molecule_step_accepted && !molecule_finite) {
                                    molecule_failed = true;
                                    molecule_reason = "nonfinite_state";
                                } else if (!molecule_step_accepted &&
                                           trial.molecule.minimum_population < 0.0) {
                                    molecule_failed = true;
                                    molecule_reason = "negative_population";
                                } else if (!molecule_step_accepted &&
                                           trial.molecule.modified_backward_error >= 1.0e-6) {
                                    molecule_failed = true;
                                    molecule_reason = "implicit_solve_failure";
                                } else if (!molecule_step_accepted &&
                                           molecule_controller_change >
                                           maximum_neutral_state_change) {
                                    molecule_failed = true;
                                    molecule_reason = "catastrophic_state_change";
                                }
                            }

                            if (solve_failure.empty() && !atom_step_accepted) {
                                last_neutral_sweep.atom = trial.atom;
                            }
                            if (solve_failure.empty() &&
                                !molecule_step_accepted) {
                                last_neutral_sweep.molecule = trial.molecule;
                            }
                            if (fixed_amplitude_diagnostic &&
                                solve_failure.empty() &&
                                (trial.atom.be_identity_defect_relative >
                                     kLocalResidualTolerance ||
                                 trial.molecule.be_identity_defect_relative >
                                     kLocalResidualTolerance)) {
                                const bool atom_mismatch =
                                    trial.atom.be_identity_defect_relative >=
                                    trial.molecule.be_identity_defect_relative;
                                const auto& block = atom_mismatch
                                    ? trial.atom : trial.molecule;
                                result.failure =
                                    "be_steady_operator_mismatch";
                                result.outer_iterations = macro;
                                result.attempted_amplitude = amplitude;
                                result.amplitude = amplitude;
                                result.profiles = profiles;
                                result.first_failing_cell =
                                    block.be_identity_defect_cell;
                                result.first_failing_species =
                                    block.be_identity_defect_gi;
                                result.elapsed_seconds = elapsed_seconds();
                                std::cout << "NESTED_INNER_STATUS"
                                          << " status=be_steady_operator_mismatch"
                                          << " block="
                                          << (atom_mismatch ? "A" : "M")
                                          << " macro=" << macro
                                          << " cell="
                                          << block.be_identity_defect_cell
                                          << " gi="
                                          << block.be_identity_defect_gi
                                          << " D_BE_relative="
                                          << block.be_identity_defect_relative
                                          << " D_BE="
                                          << block.be_identity_defect_norm
                                          << " tolerance="
                                          << kLocalResidualTolerance
                                          << "\n";
                                return result;
                            }
                            bool atom_calibration_retry = false;
                            bool molecule_calibration_retry = false;
                            constexpr double calibration_minimum_change = 1.0e-4;
                            constexpr double calibration_maximum_change = 1.0e-2;
                            if (solve_failure.empty() && !atom_step_accepted &&
                                !atom_failed &&
                                !fixed_ptc_anderson_mode &&
                                !neutral_atom_step_calibrated) {
                                if (atom_controller_change >
                                    calibration_maximum_change) {
                                    const double reduced_step = std::max(
                                        minimum_neutral_atom_step,
                                        0.5 * neutral_atom_step);
                                    if (reduced_step < neutral_atom_step) {
                                        neutral_atom_step = reduced_step;
                                        atom_reason =
                                            "calibration_change_above_target";
                                        atom_calibration_retry = true;
                                    }
                                } else if (atom_controller_change <
                                               calibration_minimum_change &&
                                           trial.atom.transient_update_ratio >=
                                               1.0e-8) {
                                    const auto increased_step = dcr::solver::
                                        increased_pseudo_time_step(
                                            neutral_atom_step,
                                            maximum_neutral_atom_step);
                                    if (increased_step.has_value()) {
                                        neutral_atom_step = *increased_step;
                                        atom_reason =
                                            "calibration_change_below_target";
                                        atom_calibration_retry = true;
                                    }
                                }
                                if (!atom_calibration_retry) {
                                    neutral_atom_step_calibrated = true;
                                }
                                result.neutral_atom_final_step =
                                    neutral_atom_step;
                            }
                            if (solve_failure.empty() &&
                                !molecule_step_accepted && !molecule_failed &&
                                !fixed_ptc_anderson_mode &&
                                !neutral_molecule_step_calibrated) {
                                if (molecule_controller_change >
                                    calibration_maximum_change) {
                                    const double reduced_step = std::max(
                                        minimum_neutral_molecule_step,
                                        0.5 * neutral_molecule_step);
                                    if (reduced_step < neutral_molecule_step) {
                                        neutral_molecule_step = reduced_step;
                                        molecule_reason =
                                            "calibration_change_above_target";
                                        molecule_calibration_retry = true;
                                    }
                                } else if (molecule_controller_change <
                                               calibration_minimum_change &&
                                           trial.molecule.transient_update_ratio >=
                                               1.0e-8) {
                                    const auto increased_step = dcr::solver::
                                        increased_pseudo_time_step(
                                            neutral_molecule_step,
                                            maximum_neutral_molecule_step);
                                    if (increased_step.has_value()) {
                                        neutral_molecule_step = *increased_step;
                                        molecule_reason =
                                            "calibration_change_below_target";
                                        molecule_calibration_retry = true;
                                    }
                                }
                                if (!molecule_calibration_retry) {
                                    neutral_molecule_step_calibrated = true;
                                }
                                result.neutral_molecule_final_step =
                                    neutral_molecule_step;
                            }
#ifdef DCR_NESTED_DIAGNOSTIC
                            if (classify_neutral_controller_event &&
                                !atom_step_accepted &&
                                (atom_failed || atom_calibration_retry)) {
                                report_activity_controller_event(
                                    "A_timestep_limiter", "A", macro,
                                    neutral_inner, atom_change,
                                    atom_timestep_activity_rows, nullptr,
                                    trial.atom.profile_change,
                                    atom_failed ? atom_reason.c_str() :
                                    (atom_calibration_retry
                                        ? atom_reason.c_str() : "passed"),
                                    0, 0);
                            }
                            if (classify_neutral_controller_event &&
                                !molecule_step_accepted &&
                                (molecule_failed || molecule_calibration_retry)) {
                                report_activity_controller_event(
                                    "M_timestep_limiter", "M", macro,
                                    neutral_inner, molecule_change,
                                    molecule_timestep_activity_rows, nullptr,
                                    trial.molecule.profile_change,
                                    molecule_failed ? molecule_reason.c_str() :
                                    (molecule_calibration_retry
                                        ? molecule_reason.c_str() : "passed"),
                                    0, 0);
                            }
#endif
                            if (solve_failure.empty() &&
                                !atom_step_accepted && !atom_failed &&
                                !atom_calibration_retry) {
                                profiles.flowA = trial.profiles.flowA;
                                ++result.neutral_atom_accepted_steps;
                                if (!fixed_ptc_anderson_mode &&
                                    ++neutral_atom_success_streak >= 3) {
                                    neutral_atom_step = std::min(
                                        maximum_neutral_atom_step,
                                        1.5 * neutral_atom_step);
                                    neutral_atom_success_streak = 0;
                                    result.neutral_atom_final_step =
                                        neutral_atom_step;
                                }
                                atom_step_accepted = true;
                            }
                            if (solve_failure.empty() &&
                                !molecule_step_accepted && !molecule_failed &&
                                !molecule_calibration_retry) {
                                profiles.flowM = trial.profiles.flowM;
                                ++result.neutral_molecule_accepted_steps;
                                if (!fixed_ptc_anderson_mode &&
                                    ++neutral_molecule_success_streak >= 3) {
                                    neutral_molecule_step = std::min(
                                        maximum_neutral_molecule_step,
                                        1.5 * neutral_molecule_step);
                                    neutral_molecule_success_streak = 0;
                                    result.neutral_molecule_final_step =
                                        neutral_molecule_step;
                                }
                                molecule_step_accepted = true;
                            }
                            last_neutral_sweep.profiles = profiles;

                            if (atom_failed || molecule_failed) {
                                macro_had_retry = true;
                                if (fixed_ptc_anderson_mode) {
                                    history.clear();
                                    anderson_merit.clear();
                                    fixed_ptc_map_parameters.reset();
                                    fixed_ptc_anderson_mode = false;
                                    ++anderson_ptc_restarts;
                                    std::cout << "ANDERSON_RETURN_TO_PTC"
                                              << " macro=" << macro
                                              << " restart="
                                              << anderson_ptc_restarts
                                              << " reason=" << std::quoted(
                                                  "neutral_finite_ptc_map_failure")
                                              << "\n";
                                }
                                if (atom_failed) {
                                    ++result.neutral_atom_rejected_steps;
                                    neutral_atom_success_streak = 0;
                                    if (neutral_atom_step <=
                                               minimum_neutral_atom_step) {
                                        result.failure =
                                            "atomic neutral pseudo-time reached minimum step: " +
                                            atom_reason;
                                        result.elapsed_seconds = elapsed_seconds();
                                        return result;
                                    } else {
                                        neutral_atom_step = std::max(
                                            minimum_neutral_atom_step,
                                            0.5 * neutral_atom_step);
                                        result.neutral_atom_final_step =
                                            neutral_atom_step;
                                    }
                                }
                                if (molecule_failed) {
                                    ++result.neutral_molecule_rejected_steps;
                                    neutral_molecule_success_streak = 0;
                                    if (neutral_molecule_step <=
                                            minimum_neutral_molecule_step) {
                                        result.failure =
                                            "molecular neutral pseudo-time reached minimum step: " +
                                            molecule_reason;
                                        result.elapsed_seconds = elapsed_seconds();
                                        return result;
                                    } else {
                                        neutral_molecule_step = std::max(
                                            minimum_neutral_molecule_step,
                                            0.5 * neutral_molecule_step);
                                        result.neutral_molecule_final_step =
                                            neutral_molecule_step;
                                    }
                                }
                            }

                            std::cout << "NEUTRAL_PSEUDO_TIME_STEP"
                                      << " accepted="
                                      << ((atom_step_accepted &&
                                           molecule_step_accepted) ? 1 : 0)
                                      << " accepted_A="
                                      << (atom_step_accepted ? 1 : 0)
                                      << " accepted_M="
                                      << (molecule_step_accepted ? 1 : 0)
                                      << " macro=" << macro
                                      << " inner=" << neutral_inner
                                      << " dtau_A=" << attempted_atom_step
                                      << " next_dtau_A=" << neutral_atom_step
                                      << " dtau_M=" << attempted_molecule_step
                                      << " next_dtau_M=" << neutral_molecule_step
                                      << " be_A="
                                      << last_neutral_sweep.atom
                                          .modified_backward_error
                                      << " be_M="
                                      << last_neutral_sweep.molecule
                                          .modified_backward_error
                                      << " eta_A="
                                      << last_neutral_sweep.atom
                                          .transient_update_ratio
                                      << " eta_M="
                                      << last_neutral_sweep.molecule
                                          .transient_update_ratio
                                      << " change_A="
                                      << last_neutral_sweep.atom.profile_change
                                      << " change_M="
                                      << last_neutral_sweep.molecule.profile_change
                                      << " raw_change_A="
                                      << last_neutral_sweep.atom.state_change_norm
                                      << " raw_change_M="
                                      << last_neutral_sweep.molecule.state_change_norm
                                      << " raw_transient_A="
                                      << last_neutral_sweep.atom.transient_term_norm
                                      << " raw_transient_M="
                                      << last_neutral_sweep.molecule.transient_term_norm
                                      << " raw_steady_A="
                                      << last_neutral_sweep.atom.steady_residual_norm
                                      << " raw_steady_M="
                                      << last_neutral_sweep.molecule.steady_residual_norm
                                      << " raw_be_A="
                                      << last_neutral_sweep.atom.be_residual_norm
                                      << " raw_be_M="
                                      << last_neutral_sweep.molecule.be_residual_norm
                                      << " D_BE_A="
                                      << last_neutral_sweep.atom.be_identity_defect_norm
                                      << " D_BE_M="
                                      << last_neutral_sweep.molecule.be_identity_defect_norm
                                      << " D_BE_relative_A="
                                      << last_neutral_sweep.atom
                                          .be_identity_defect_relative
                                      << " D_BE_relative_M="
                                      << last_neutral_sweep.molecule
                                          .be_identity_defect_relative
                                      << " D_BE_cell_A="
                                      << last_neutral_sweep.atom
                                          .be_identity_defect_cell
                                      << " D_BE_gi_A="
                                      << last_neutral_sweep.atom
                                          .be_identity_defect_gi
                                      << " D_BE_cell_M="
                                      << last_neutral_sweep.molecule
                                          .be_identity_defect_cell
                                      << " D_BE_gi_M="
                                      << last_neutral_sweep.molecule
                                          .be_identity_defect_gi
                                      << " atom_reason="
                                      << std::quoted(atom_reason)
                                      << " molecule_reason="
                                      << std::quoted(molecule_reason)
                                      << "\n";
                            accepted_neutral_step =
                                atom_step_accepted && molecule_step_accepted;
                        }
                    }
                }

                if (macro_had_retry ||
                    (use_neutral_pseudo_time &&
                     !anderson_map_stationary_at_macro_start)) {
                    stable_macro_cycle_streak = 0;
                } else {
                    ++stable_macro_cycle_streak;
                }

                const Vector recycled_state =
                    current_state.tail(base_recycled.size());
                const Vector evaluated_recycled =
                    pack_recycled_profiles(profiles);
                const Vector swept_recycled = pack_recycled_profiles(
                    sweep_recycled_neutrals(profiles, lambda_recycling));
                const Vector mapped_recycled = use_neutral_pseudo_time
                    ? evaluated_recycled : swept_recycled;
                Vector joint_residual = Vector::Zero(current_state.size());
                const double uncapped_log_amplitude_step =
                    fixed_amplitude_diagnostic
                    ? 0.0 : -macro_relaxation * target_mismatch(profiles);
                const double capped_log_amplitude_step = std::clamp(
                    uncapped_log_amplitude_step, -0.05, 0.05);
                const auto fixed_amplitude_step =
                    dcr::solver::feasible_log_amplitude_step(
                        current_state(0), capped_log_amplitude_step,
                        amplitude_trust_bounds(), physical_amplitude_bounds);
                const double fixed_map_log_amplitude_step =
                    fixed_amplitude_step.valid
                    ? fixed_amplitude_step.delta_log : 0.0;
                joint_residual(0) = fixed_map_log_amplitude_step;
                if (anderson_map_stationary_at_macro_start &&
                    !fixed_amplitude_diagnostic) {
                    if (consecutive_physical_bound_clips >= 12) {
                        history.clear();
                        anderson_merit.clear();
                        fixed_ptc_map_parameters.reset();
                        result.infeasible = true;
                        result.failure =
                            "persistent physical amplitude-bound clipping";
                        result.outer_iterations = macro;
                        result.elapsed_seconds = elapsed_seconds();
                        std::cout << "PSEUDO_TIME_AMPLITUDE_CLIP_RECOVERY"
                                  << " macro=" << macro
                                  << " source=" << std::quoted(
                                      "physical_bound")
                                  << " action=reject_physical_bound"
                                  << "\n";
                        return result;
                    }
                    if (consecutive_trust_region_clips >= 12) {
                        history.clear();
                        anderson_merit.clear();
                        fixed_ptc_map_parameters.reset();
                        fixed_ptc_anderson_mode = false;
                        anderson_map_stationary_at_macro_start = false;
                        amplitude_trust.center = current_state(0);
                        amplitude_trust.radius = std::min(
                            amplitude_trust.maximum_radius,
                            1.5 * amplitude_trust.radius);
                        consecutive_trust_region_clips = 0;
                        std::cout << "PSEUDO_TIME_AMPLITUDE_CLIP_RECOVERY"
                                  << " macro=" << macro
                                  << " source=" << std::quoted(
                                      "algorithmic_trust_region")
                                  << " action=clear_history_recenter_trust"
                                  << "\n";
                    }
                    if (capped_log_amplitude_step !=
                        uncapped_log_amplitude_step) {
                        ++local_step_cap_hits;
                        ++consecutive_local_step_caps;
                    } else {
                        consecutive_local_step_caps = 0;
                    }
                    if (anderson_map_stationary_at_macro_start &&
                        consecutive_local_step_caps >= 12) {
                        history.clear();
                        anderson_merit.clear();
                        fixed_ptc_map_parameters.reset();
                        fixed_ptc_anderson_mode = false;
                        anderson_map_stationary_at_macro_start = false;
                        consecutive_local_step_caps = 0;
                        std::cout << "PSEUDO_TIME_AMPLITUDE_CLIP_RECOVERY"
                                  << " macro=" << macro
                                  << " source=" << std::quoted(
                                      "local_step_cap")
                                  << " action=clear_history_return_to_ptc"
                                  << "\n";
                    }
                }
                joint_residual.tail(base_recycled.size()) =
                    (mapped_recycled - recycled_state).cwiseQuotient(
                        anderson_scale.tail(base_recycled.size()));
                const Vector finite_ptc_current_state = pack_finite_ptc_state(
                    current_state(0), previous_macro_profiles);
                const Vector finite_ptc_mapped_state = pack_finite_ptc_state(
                    current_state(0) + fixed_map_log_amplitude_step,
                    profiles);
                const Vector finite_ptc_scaled_current =
                    finite_ptc_current_state.cwiseQuotient(
                        finite_ptc_anderson_scale);
                const Vector finite_ptc_joint_residual =
                    (finite_ptc_mapped_state - finite_ptc_current_state)
                        .cwiseQuotient(finite_ptc_anderson_scale);
                const Profiles evaluated_profiles = profiles;
                const double evaluated_amplitude = amplitude;
                const UpstreamIonBoundary evaluated_boundary =
                    make_upstream_ion_boundary(evaluated_amplitude, composition);
                const SweepDiagnostics steady_diagnostics =
                    evaluate_residuals(evaluated_profiles, evaluated_boundary);
                previous_plasma_steady_residual =
                    steady_diagnostics.max_steady_backward_error;
                const double mismatch = target_mismatch(evaluated_profiles);
                const double neutral_profile_residual =
                    recycled_fixed_point_residual(
                        evaluated_recycled, swept_recycled);
                const Profiles steady_neutral_profiles = [&]() {
                    Profiles candidate = evaluated_profiles;
                    unpack_recycled_profiles(swept_recycled, candidate);
                    return candidate;
                }();
                const double neutral_atom_residual = neutral_block_residual(
                    evaluated_profiles.flowA, steady_neutral_profiles.flowA);
                const double neutral_molecule_residual = neutral_block_residual(
                    evaluated_profiles.flowM, steady_neutral_profiles.flowM);

                if (use_neutral_pseudo_time) {
                    if (!fixed_ptc_anderson_mode &&
                        last_neutral_sweep.atom.transient_update_ratio < 1.0e-8) {
                        ++neutral_atom_steady_macro_streak;
                    } else if (!fixed_ptc_anderson_mode) {
                        neutral_atom_steady_macro_streak = 0;
                        neutral_atom_handoff_ready = false;
                    }
                    if (!fixed_ptc_anderson_mode &&
                        last_neutral_sweep.molecule.transient_update_ratio < 1.0e-8) {
                        ++neutral_molecule_steady_macro_streak;
                    } else if (!fixed_ptc_anderson_mode) {
                        neutral_molecule_steady_macro_streak = 0;
                        neutral_molecule_handoff_ready = false;
                    }
                    if (!fixed_ptc_anderson_mode &&
                        neutral_atom_steady_macro_streak >= 3) {
                        neutral_atom_handoff_ready = true;
                        neutral_atom_success_streak = 0;
                        std::cout << "PSEUDO_TIME_HANDOFF_READY"
                                  << " block=A macro=" << macro
                                  << " consecutive_macros="
                                  << neutral_atom_steady_macro_streak
                                  << " eta="
                                  << last_neutral_sweep.atom.transient_update_ratio
                                  << " residual=" << neutral_atom_residual
                                  << "\n";
                    }
                    if (!fixed_ptc_anderson_mode &&
                        neutral_molecule_steady_macro_streak >= 3) {
                        neutral_molecule_handoff_ready = true;
                        neutral_molecule_success_streak = 0;
                        std::cout << "PSEUDO_TIME_HANDOFF_READY"
                                  << " block=M macro=" << macro
                                  << " consecutive_macros="
                                  << neutral_molecule_steady_macro_streak
                                  << " eta="
                                  << last_neutral_sweep.molecule
                                      .transient_update_ratio
                                  << " residual=" << neutral_molecule_residual
                                  << "\n";
                    }
                    constexpr int diagnostic_cell = 184;
                    if (h2_ground_mi >= 0 &&
                        diagnostic_cell < static_cast<int>(
                            evaluated_profiles.flowM.size())) {
                        const double state_value = evaluated_profiles.flowM[
                            static_cast<size_t>(diagnostic_cell)](h2_ground_mi);
                        const double image_value = steady_neutral_profiles.flowM[
                            static_cast<size_t>(diagnostic_cell)](h2_ground_mi);
                        const double numerator = image_value - state_value;
                        const double normalization = std::max({
                            1.0, std::abs(state_value), std::abs(image_value)});
                        std::cout << "NEUTRAL_H2_GROUND_HISTORY"
                                  << " macro=" << macro
                                  << " cell=" << diagnostic_cell
                                  << " gi=" << boundary_.M_indices[
                                      static_cast<size_t>(h2_ground_mi)]
                                  << " state=" << state_value
                                  << " image=" << image_value
                                  << " signed_numerator=" << numerator
                                  << " normalization=" << normalization
                                  << " residual="
                                  << std::abs(numerator) / normalization
                                  << "\n";
                    }
                }
                if (anderson_map_stationary_at_macro_start &&
                    !macro_had_retry) {
                    macro_target_residual_history.push_back(std::abs(mismatch));
                    macro_profile_residual_history.push_back(
                        neutral_profile_residual);
                } else {
                    macro_target_residual_history.clear();
                    macro_profile_residual_history.clear();
                }
                const double combined_merit = fixed_amplitude_diagnostic
                    ? std::max(
                        steady_diagnostics.max_steady_backward_error /
                            stage_tolerance,
                        neutral_profile_residual / stage_tolerance)
                    : std::max({
                        std::abs(mismatch) / stage_tolerance,
                        steady_diagnostics.max_steady_backward_error /
                            stage_tolerance,
                        neutral_profile_residual / stage_tolerance});
                previous_accepted_combined_merit = combined_merit;
                const double macro_profile_change =
                    max_relative_change(evaluated_profiles, previous_macro_profiles);
                if (!fixed_ptc_anderson_mode) {
                    ptc_true_merit_history.push_back(combined_merit);
                    const size_t stall_window = static_cast<size_t>(std::max(
                        3, config_.numerics.temperature_ptc_stall_window));
                    bool residual_stalled = false;
                    bool split_oscillating = false;
                    if (ptc_true_merit_history.size() >= stall_window) {
                        const size_t first =
                            ptc_true_merit_history.size() - stall_window;
                        residual_stalled = combined_merit >=
                            0.95 * ptc_true_merit_history[first];
                        int alternating_differences = 0;
                        double previous_difference = 0.0;
                        for (size_t i = first + 1;
                             i < ptc_true_merit_history.size(); ++i) {
                            const double difference = ptc_true_merit_history[i] -
                                ptc_true_merit_history[i - 1];
                            if (difference * previous_difference < 0.0) {
                                ++alternating_differences;
                            }
                            if (difference != 0.0) previous_difference = difference;
                        }
                        split_oscillating = alternating_differences >= 3;
                    }
                    const bool dtau_ceiling_reached =
                        pseudo_time_step >= maximum_pseudo_time_step ||
                        (use_neutral_pseudo_time &&
                         (neutral_atom_step >= maximum_neutral_atom_step ||
                          neutral_molecule_step >= maximum_neutral_molecule_step));
                    const bool all_blocks_ready = plasma_handoff_ready &&
                        neutral_atom_handoff_ready &&
                        neutral_molecule_handoff_ready;
                    const bool settled_for_fixed_ptc_handoff = !macro_had_retry &&
                        macro_profile_change < 1.0e-2;
                    if (all_blocks_ready || dtau_ceiling_reached ||
                        (settled_for_fixed_ptc_handoff &&
                         (residual_stalled || split_oscillating))) {
                        fixed_ptc_anderson_mode = true;
                        fixed_ptc_map_parameters.reset();
                        anderson_phase_iterations = 0;
                        history.clear();
                        anderson_merit.clear();
                        consecutive_local_step_caps = 0;
                        consecutive_trust_region_clips = 0;
                        consecutive_physical_bound_clips = 0;
                        std::cout << "PSEUDO_TIME_ANDERSON_HANDOFF"
                                  << " macro=" << macro
                                  << " reason=" << std::quoted(
                                      all_blocks_ready ? "zero_transient_update" :
                                      (dtau_ceiling_reached ? "dtau_ceiling" :
                                      (split_oscillating ? "split_oscillation" :
                                       "steady_residual_stall")))
                                  << " dtau_P=" << pseudo_time_step
                                  << " dtau_A=" << neutral_atom_step
                                  << " dtau_M=" << neutral_molecule_step
                                  << " true_steady_merit=" << combined_merit
                                  << "\n";
                    }
                } else {
                    ++anderson_phase_iterations;
                }
                auto [physical_target_A, physical_target_M] =
                    target_recycling(evaluated_profiles.background.front());
                physical_target_A *= lambda_recycling;
                physical_target_M *= lambda_recycling;
                const double boundary_change = std::max(
                    vector_relative_change(
                        evaluated_profiles.flowA.front(), physical_target_A),
                    vector_relative_change(
                        evaluated_profiles.flowM.front(), physical_target_M));
                const double minimum_population =
                    profile_minimum(evaluated_profiles);

                result.outer_iterations = macro;
                result.amplitude = evaluated_amplitude;
                result.attempted_amplitude = evaluated_amplitude;
                result.mismatch = mismatch;
                result.attempted_mismatch = mismatch;
                result.fixed_point_residual = fixed_amplitude_diagnostic
                    ? neutral_profile_residual
                    : std::max(std::abs(mismatch), neutral_profile_residual);
                result.profile_change = macro_profile_change;
                result.boundary_change = boundary_change;
                result.minimum_population = minimum_population;
                result.profiles = evaluated_profiles;
                result.steady_backward_error =
                    steady_diagnostics.max_steady_backward_error;
                result.pseudo_time_backward_error =
                    last_plasma_trial.diagnostics.max_pseudo_time_backward_error;
                result.pseudo_time_profile_change = last_be_step_change;
                result.ion_trial = last_plasma_trial;
                result.ion_trial.amplitude = evaluated_amplitude;
                result.ion_trial.mismatch = mismatch;
                result.ion_trial.background = evaluated_profiles.background;
                result.ion_trial.upstream_boundary = evaluated_boundary;
                result.ion_trial.diagnostics = steady_diagnostics;
                if (fixed_amplitude_diagnostic) {
                    diagnostic_inner_merit_history.push_back(combined_merit);
                    const size_t window = static_cast<size_t>(
                        diagnostic_stagnation_window);
                    if (combined_merit >= 1.0 &&
                        diagnostic_inner_merit_history.size() >= window) {
                        const size_t first =
                            diagnostic_inner_merit_history.size() - window;
                        const double initial_window_merit =
                            diagnostic_inner_merit_history[first];
                        if (combined_merit >= 0.98 * initial_window_merit) {
                            result.failure = "inner_solve_stalled";
                            result.elapsed_seconds = elapsed_seconds();
                            std::cout << "NESTED_INNER_STATUS"
                                      << " status=inner_solve_stalled"
                                      << " macro=" << macro
                                      << " window=" << window
                                      << " first_merit="
                                      << initial_window_merit
                                      << " last_merit=" << combined_merit
                                      << " P_residual="
                                      << steady_diagnostics
                                          .max_steady_backward_error
                                      << " A_residual="
                                      << neutral_atom_residual
                                      << " M_residual="
                                      << neutral_molecule_residual
                                      << "\n";
                            return result;
                        }
                    }
                }
                const Vector scaled_current =
                    current_state.cwiseQuotient(anderson_scale);
                const bool anderson_map_stationary =
                    anderson_map_stationary_at_macro_start;
                if (use_neutral_pseudo_time &&
                    (!anderson_map_stationary || macro_had_retry)) {
                    history.clear();
                }
                if (anderson_map_stationary) {
                    history.push_back({
                        finite_ptc_scaled_current,
                        finite_ptc_joint_residual});
                    if (history.size() > 9) history.erase(history.begin());
                }

                const bool transient_neutral_mode = use_pseudo_time &&
                    (!anderson_map_stationary || macro_had_retry);
                Vector next_state;
                std::string update_method = use_neutral_pseudo_time
                    ? "neutral_be_map" : "fixed_point_relaxed";
                double update_alpha = macro_relaxation;
                if (fixed_amplitude_diagnostic) {
                    next_state = current_state;
                    next_state(0) = diagnostic_fixed_log_amplitude;
                    next_state.tail(base_recycled.size()) = evaluated_recycled;
                    update_method = "fixed_amplitude_finite_ptc_map";
                    update_alpha = 1.0;
                } else if (transient_neutral_mode) {
                    const double current_log_amplitude = current_state(0);
                    const double uncapped_raw_log_step =
                        -macro_relaxation * mismatch;
                    const double raw_log_step = std::clamp(
                        uncapped_raw_log_step, -0.05, 0.05);
                    const bool local_step_cap_active =
                        raw_log_step != uncapped_raw_log_step;
                    if (raw_log_step != 0.0 && checkpoint_context != nullptr) {
                        ContinuationCheckpoint provisional = *checkpoint_context;
                        provisional.provisional_valid = true;
                        provisional.provisional_temperature = config_.plasma.Te_eV;
                        provisional.provisional_amplitude = evaluated_amplitude;
                        provisional.provisional_profiles = evaluated_profiles;
                        provisional.provisional_upstream_boundary = evaluated_boundary;
                        if (!save_continuation_checkpoint(provisional)) {
                            throw std::runtime_error(
                                "Could not save pre-amplitude provisional checkpoint");
                        }
                    }
                    auto amplitude_merit = [&](double candidate_log_amplitude)
                        -> std::optional<double> {
                        const double candidate_amplitude =
                            std::exp(candidate_log_amplitude);
                        if (!amplitude_in_bounds(candidate_amplitude)) {
                            return std::nullopt;
                        }
                        Profiles candidate_profiles = evaluated_profiles;
                        SweepDiagnostics candidate_diagnostics;
                        try {
                            candidate_diagnostics = evaluate_residuals(
                                candidate_profiles,
                                make_upstream_ion_boundary(
                                    candidate_amplitude, composition));
                        } catch (const std::exception&) {
                            return std::nullopt;
                        }
                        if (!candidate_diagnostics.cells_converged ||
                            !block_all_finite(candidate_profiles.background) ||
                            profile_minimum(candidate_profiles) < 0.0) {
                            return std::nullopt;
                        }
                        Profiles neutral_image;
                        try {
                            neutral_image = sweep_recycled_neutrals(
                                candidate_profiles, lambda_recycling);
                        } catch (const std::exception&) {
                            return std::nullopt;
                        }
                        const double profile_residual =
                            recycled_fixed_point_residual(
                                pack_recycled_profiles(candidate_profiles),
                                pack_recycled_profiles(neutral_image));
                        return std::max({
                            std::abs(target_mismatch(candidate_profiles)) /
                                stage_tolerance,
                            candidate_diagnostics.max_steady_backward_error /
                                stage_tolerance,
                            profile_residual / stage_tolerance});
                    };
                    auto amplitude_step =
                        dcr::solver::backtrack_log_amplitude_step(
                            current_log_amplitude, raw_log_step,
                            amplitude_trust_bounds(), physical_amplitude_bounds,
                            combined_merit, amplitude_merit, 0.95, 0.5, 1.0e-8);
                    bool direction_reversed = false;
                    if (!amplitude_step.valid && raw_log_step != 0.0) {
                        amplitude_step =
                            dcr::solver::backtrack_log_amplitude_step(
                                current_log_amplitude, -raw_log_step,
                                amplitude_trust_bounds(),
                                physical_amplitude_bounds, combined_merit,
                                amplitude_merit, 0.95, 0.5, 1.0e-8);
                        direction_reversed = amplitude_step.valid;
                    }
                    const double accepted_log_step = amplitude_step.valid
                        ? amplitude_step.delta_log : 0.0;
                    next_state = current_state;
                    next_state(0) = current_log_amplitude + accepted_log_step;
                    next_state.tail(base_recycled.size()) = evaluated_recycled;
                    update_method = amplitude_step.valid
                        ? amplitude_step.reason
                        : "amplitude_no_feasible_descent_step";
                    update_alpha = amplitude_step.alpha_feasible;
                    if (amplitude_step.valid && accepted_log_step != 0.0) {
                        if (local_step_cap_active) {
                            ++local_step_cap_hits;
                            ++consecutive_local_step_caps;
                        } else {
                            consecutive_local_step_caps = 0;
                        }
                        if (amplitude_step.clipped) {
                            ++amplitude_bound_hits;
                            if (amplitude_step.clip_source ==
                                dcr::solver::AmplitudeClipSource::
                                    algorithmic_trust_region) {
                                ++trust_region_clip_hits;
                                ++consecutive_trust_region_clips;
                                consecutive_physical_bound_clips = 0;
                            } else if (amplitude_step.clip_source ==
                                dcr::solver::AmplitudeClipSource::physical_bound) {
                                ++physical_bound_clip_hits;
                                ++consecutive_physical_bound_clips;
                                consecutive_trust_region_clips = 0;
                            }
                        } else {
                            consecutive_trust_region_clips = 0;
                            consecutive_physical_bound_clips = 0;
                        }
                        amplitude_trust.accept(
                            next_state(0), amplitude_step.alpha_feasible == 1.0);
                    } else if (!amplitude_step.valid &&
                        anderson_map_stationary) {
                        amplitude_trust.reject();
                    }
                    const auto trust = amplitude_trust_bounds();
                    std::cout << "PSEUDO_TIME_AMPLITUDE_STEP"
                              << " macro=" << macro
                              << " method=" << update_method
                              << " log_amplitude=" << current_log_amplitude
                              << " mismatch=" << mismatch
                              << " raw_delta_log=" << raw_log_step
                              << " uncapped_delta_log="
                              << uncapped_raw_log_step
                              << " direction_reversed="
                              << (direction_reversed ? 1 : 0)
                              << " alpha_feasible="
                              << amplitude_step.alpha_feasible
                              << " accepted_delta_log=" << accepted_log_step
                              << " proposed_amplitude="
                              << std::exp(next_state(0))
                              << " trust_kind=algorithmic"
                              << " trust_center=" << amplitude_trust.center
                              << " trust_radius=" << amplitude_trust.radius
                              << " trust_lower_amplitude="
                              << std::exp(trust.lower)
                              << " trust_upper_amplitude="
                              << std::exp(trust.upper)
                              << " raw_candidate_feasible="
                              << (amplitude_step.raw_candidate_feasible ? 1 : 0)
                              << " trust_bound_active="
                              << (amplitude_step.trust_bound_active ? 1 : 0)
                              << " local_step_cap_active="
                              << (local_step_cap_active ? 1 : 0)
                              << " clip_source=" << std::quoted(
                                  local_step_cap_active
                                      ? dcr::solver::amplitude_clip_source_name(
                                          dcr::solver::AmplitudeClipSource::
                                              local_step_cap)
                                      : dcr::solver::amplitude_clip_source_name(
                                          amplitude_step.clip_source))
                              << " bound_hits=" << amplitude_bound_hits
                              << " local_cap_hits=" << local_step_cap_hits
                              << " trust_clip_hits=" << trust_region_clip_hits
                              << " physical_clip_hits="
                              << physical_bound_clip_hits
                              << " steady_image_role=diagnostic_only"
                              << "\n";
                } else {
                    if (use_neutral_pseudo_time) {
                        have_previous_amplitude_point = false;
                    }
                    next_state = physical_state(
                        scaled_current + joint_residual);
                    if (use_neutral_pseudo_time) {
                        next_state.tail(base_recycled.size()) =
                            evaluated_recycled;
                    }
                }
                const Vector fixed_relaxed_state = next_state;
                const double fixed_scaled_step =
                    finite_ptc_joint_residual.cwiseAbs().maxCoeff();
                if (anderson_map_stationary && anderson_phase_iterations == 1) {
                    anderson_merit.accept(combined_merit);
                }
                Profiles anderson_candidate_profiles;
                bool have_anderson_candidate_profiles = false;
                auto fixed_ptc_candidate_merit = [&] (
                    const Vector& candidate_state,
                    Profiles& candidate_profiles,
                    std::string& rejection_reason) {
                    candidate_profiles = evaluated_profiles;
                    double candidate_log_amplitude =
                        std::numeric_limits<double>::quiet_NaN();
                    try {
                        unpack_finite_ptc_state(
                            candidate_state, candidate_log_amplitude,
                            candidate_profiles);
                    } catch (const std::exception& error) {
                        rejection_reason = error.what();
                        return std::numeric_limits<double>::infinity();
                    }
                    const double candidate_amplitude =
                        std::exp(candidate_log_amplitude);
                    SweepDiagnostics candidate_diagnostics;
                    try {
                        candidate_diagnostics = evaluate_residuals(
                            candidate_profiles,
                            make_upstream_ion_boundary(
                                candidate_amplitude, composition));
                    } catch (const std::exception& error) {
                        rejection_reason = error.what();
                        return std::numeric_limits<double>::infinity();
                    }
                    if (!candidate_diagnostics.cells_converged ||
                        !block_all_finite(candidate_profiles.background) ||
                        profile_minimum(candidate_profiles) < 0.0) {
                        rejection_reason = "candidate_ptc_state_infeasible";
                        return std::numeric_limits<double>::infinity();
                    }
                    Profiles candidate_neutral_image;
                    try {
                        candidate_neutral_image = sweep_recycled_neutrals(
                            candidate_profiles, lambda_recycling);
                    } catch (const std::exception& error) {
                        rejection_reason = error.what();
                        return std::numeric_limits<double>::infinity();
                    }
                    const double candidate_profile_residual =
                        recycled_fixed_point_residual(
                            pack_recycled_profiles(candidate_profiles),
                            pack_recycled_profiles(candidate_neutral_image));
                    const double candidate_target_residual =
                        std::abs(target_mismatch(candidate_profiles));
                    return fixed_amplitude_diagnostic
                        ? std::max(
                            candidate_diagnostics.max_steady_backward_error /
                                stage_tolerance,
                            candidate_profile_residual / stage_tolerance)
                        : std::max({
                            candidate_target_residual / stage_tolerance,
                            candidate_diagnostics.max_steady_backward_error /
                                stage_tolerance,
                            candidate_profile_residual / stage_tolerance});
                };
                bool anderson_update_accepted = false;
                if (anderson_map_stationary &&
                    history.size() >= 2 &&
                    anderson_phase_iterations <=
                        config_.numerics.temperature_anderson_max_iterations) {
                    const int differences = std::min<int>(
                        std::max(1, config_.numerics.marching_anderson_depth),
                        static_cast<int>(history.size()) - 1);
                    const int first = static_cast<int>(history.size()) -
                        differences - 1;
                    Matrix delta_f(
                        finite_ptc_current_state.size(), differences);
                    Matrix delta_x(
                        finite_ptc_current_state.size(), differences);
                    for (int j = 0; j < differences; ++j) {
                        delta_f.col(j) =
                            history[static_cast<size_t>(first + j + 1)].residual -
                            history[static_cast<size_t>(first + j)].residual;
                        delta_x.col(j) =
                            history[static_cast<size_t>(first + j + 1)].state -
                            history[static_cast<size_t>(first + j)].state;
                    }
                    Matrix normal = delta_f.transpose() * delta_f;
                    normal.diagonal().array() += std::max(
                        0.0, config_.numerics.marching_anderson_regularization);
                    const Vector gamma = normal.ldlt().solve(
                        delta_f.transpose() * finite_ptc_joint_residual);
                    if (gamma.allFinite()) {
                        const double anderson_beta =
                            config_.numerics.marching_anderson_beta;
                        const Vector scaled_map = finite_ptc_scaled_current +
                            anderson_beta * finite_ptc_joint_residual;
                        Vector accelerated = (
                            scaled_map -
                            (delta_x + anderson_beta * delta_f) * gamma)
                                .cwiseProduct(finite_ptc_anderson_scale);
                        if (fixed_amplitude_diagnostic) {
                            accelerated(0) = diagnostic_fixed_log_amplitude;
                        }
                        if (fixed_amplitude_diagnostic) {
#ifdef DCR_NESTED_DIAGNOSTIC
                            ControllerChangeSummary anderson_change;
                            std::vector<ControllerActivityRow> anderson_rows;
                            int restored_trace_components = 0;
                            int nontrace_negative_components = 0;
                            const bool classify_anderson_controller_event =
                                fixed_gamma_activity_controller &&
                                accelerated.tail(accelerated.size() - 1)
                                        .minCoeff() < 0.0;
                            if (classify_anderson_controller_event) {
                                const ControllerActivitySnapshot activity =
                                    controller_activity_snapshot(
                                        evaluated_profiles, evaluated_boundary,
                                        lambda_recycling,
                                        activity_trace_protection_fraction);
                                anderson_rows.reserve(
                                    activity.P.size() + activity.A.size() +
                                    activity.M.size());
                                anderson_rows.insert(
                                    anderson_rows.end(),
                                    activity.P.begin(), activity.P.end());
                                anderson_rows.insert(
                                    anderson_rows.end(),
                                    activity.A.begin(), activity.A.end());
                                anderson_rows.insert(
                                    anderson_rows.end(),
                                    activity.M.begin(), activity.M.end());
                                anderson_change =
                                    controller_packed_change_summary(
                                        finite_ptc_mapped_state.tail(
                                            finite_ptc_mapped_state.size() - 1),
                                        accelerated.tail(
                                            accelerated.size() - 1),
                                        anderson_rows);
                                const auto restored = dcr::solver::
                                    restore_negative_trace_components(
                                        accelerated, finite_ptc_mapped_state,
                                        finite_ptc_trace_mask(activity), 1);
                                accelerated = restored.state;
                                restored_trace_components =
                                    restored.restored_trace_components;
                                nontrace_negative_components =
                                    restored.nontrace_negative_components;
                            }
#endif
                            const auto safe_blend =
                                dcr::solver::positivity_safe_blend(
                                    finite_ptc_mapped_state, accelerated, 1);
#ifdef DCR_NESTED_DIAGNOSTIC
                            if (classify_anderson_controller_event) {
                                report_activity_controller_event(
                                    "Anderson_positivity", "P_A_M", macro,
                                    anderson_phase_iterations,
                                    anderson_change, anderson_rows, nullptr,
                                    anderson_change.raw_maximum,
                                    safe_blend.positivity_limited
                                        ? "positivity_limited" : "unlimited",
                                    restored_trace_components,
                                    nontrace_negative_components);
                            }
#endif
                            double beta = safe_blend.beta;
                            bool merit_limited = false;
                            bool anderson_accepted = false;
                            int beta_backtracking_count = 0;
                            for (int backtrack = 0;
                                 beta > 0.0 && backtrack < 20; ++backtrack) {
                                const Vector candidate = finite_ptc_mapped_state +
                                    beta * (accelerated - finite_ptc_mapped_state);
                                std::string rejection_reason =
                                    finite_ptc_state_rejection_reason(candidate);
                                double candidate_merit =
                                    std::numeric_limits<double>::quiet_NaN();
                                Profiles candidate_profiles;
                                if (rejection_reason.empty()) {
                                    candidate_merit = fixed_ptc_candidate_merit(
                                        candidate, candidate_profiles,
                                        rejection_reason);
                                    if (rejection_reason.empty() &&
                                        !anderson_merit.accepts(
                                            candidate_merit, combined_merit)) {
                                        rejection_reason =
                                            "nonmonotone_merit_window_exceeded";
                                        merit_limited = true;
                                    }
                                }
                                if (rejection_reason.empty()) {
                                    anderson_candidate_profiles =
                                        std::move(candidate_profiles);
                                    double candidate_log_amplitude = 0.0;
                                    unpack_finite_ptc_state(
                                        candidate, candidate_log_amplitude,
                                        anderson_candidate_profiles);
                                    next_state = current_state;
                                    next_state(0) = candidate_log_amplitude;
                                    next_state.tail(base_recycled.size()) =
                                        pack_recycled_profiles(
                                            anderson_candidate_profiles);
                                    have_anderson_candidate_profiles = true;
                                    update_method = "anderson_ptc_blend";
                                    update_alpha = beta;
                                    ++result.anderson_accepted;
                                    anderson_merit.accept(candidate_merit);
                                    result.anderson_backtracks += backtrack;
                                    anderson_accepted = true;
                                    anderson_update_accepted = true;
                                    std::cout << "PSEUDO_TIME_ANDERSON_TRIAL"
                                              << " accepted=1 macro=" << macro
                                              << " beta_selected=" << beta
                                              << " beta_backtracking_count="
                                              << backtrack
                                              << " positivity_limited="
                                              << (safe_blend.positivity_limited ? 1 : 0)
                                              << " merit_limited="
                                              << (merit_limited ? 1 : 0)
                                              << " candidate_merit="
                                              << candidate_merit
                                              << "\n";
                                    break;
                                }
                                ++result.anderson_rejected;
                                std::cout << "PSEUDO_TIME_ANDERSON_TRIAL"
                                          << " accepted=0 macro=" << macro
                                          << " beta_selected=" << beta
                                          << " beta_backtracking_count="
                                          << backtrack
                                          << " positivity_limited="
                                          << (safe_blend.positivity_limited ? 1 : 0)
                                          << " merit_limited="
                                          << (rejection_reason ==
                                                  "nonmonotone_merit_window_exceeded"
                                              ? 1 : 0)
                                          << " candidate_merit="
                                          << candidate_merit
                                          << " reason="
                                          << std::quoted(rejection_reason)
                                          << "\n";
                                beta *= 0.5;
                                ++beta_backtracking_count;
                            }
                            if (!anderson_accepted) {
                                next_state = fixed_relaxed_state;
                                update_method = "finite_ptc_beta_zero";
                                update_alpha = 0.0;
                                anderson_merit.accept(combined_merit);
                                anderson_update_accepted = true;
                                std::cout << "PSEUDO_TIME_ANDERSON_TRIAL"
                                          << " accepted=0 macro=" << macro
                                          << " beta_selected=0"
                                          << " beta_backtracking_count="
                                          << beta_backtracking_count
                                          << " positivity_limited="
                                          << (safe_blend.positivity_limited ? 1 : 0)
                                          << " merit_limited="
                                          << (merit_limited ? 1 : 0)
                                          << " action=accept_plain_ptc"
                                          << "\n";
                            }
                        } else {
                            const auto amplitude_alpha =
                                dcr::solver::feasible_log_amplitude_step(
                                finite_ptc_current_state(0),
                                accelerated(0) - finite_ptc_current_state(0),
                                amplitude_trust_bounds(),
                                physical_amplitude_bounds);
                        const bool amplitude_bound_limited_update =
                            amplitude_alpha.valid && amplitude_alpha.clipped &&
                            amplitude_alpha.alpha_feasible < macro_relaxation;
                        if (amplitude_bound_limited_update) {
                            ++amplitude_bound_hits;
                            if (amplitude_alpha.clip_source ==
                                dcr::solver::AmplitudeClipSource::
                                    algorithmic_trust_region) {
                                ++trust_region_clip_hits;
                                ++consecutive_trust_region_clips;
                                consecutive_physical_bound_clips = 0;
                            } else {
                                ++physical_bound_clip_hits;
                                ++consecutive_physical_bound_clips;
                                consecutive_trust_region_clips = 0;
                            }
                            std::cout << "PSEUDO_TIME_AMPLITUDE_CLIP"
                                      << " macro=" << macro
                                      << " source=" << std::quoted(
                                          dcr::solver::
                                              amplitude_clip_source_name(
                                                  amplitude_alpha.clip_source))
                                      << " consecutive_trust="
                                      << consecutive_trust_region_clips
                                      << " consecutive_physical="
                                      << consecutive_physical_bound_clips
                                      << " alpha_feasible="
                                      << amplitude_alpha.alpha_feasible
                                      << "\n";
                        } else {
                            consecutive_trust_region_clips = 0;
                            consecutive_physical_bound_clips = 0;
                        }
                        double alpha = amplitude_alpha.valid
                            ? std::min(macro_relaxation,
                                amplitude_alpha.alpha_feasible)
                            : 0.0;
                        bool anderson_accepted = false;
                        for (int backtrack = 0;
                             alpha > 0.0 && backtrack < 12; ++backtrack) {
                            const Vector candidate = finite_ptc_current_state +
                                alpha * (accelerated -
                                    finite_ptc_current_state);
                            const double candidate_scaled_step = (
                                candidate.cwiseQuotient(
                                    finite_ptc_anderson_scale) -
                                finite_ptc_scaled_current)
                                    .cwiseAbs().maxCoeff();
                            std::string rejection_reason =
                                finite_ptc_state_rejection_reason(candidate);
                            if (rejection_reason.empty() &&
                                candidate_scaled_step > 2.0 * fixed_scaled_step) {
                                rejection_reason = "scaled_trust_region";
                            }
                            double candidate_merit =
                                std::numeric_limits<double>::quiet_NaN();
                            Profiles candidate_profiles;
                            if (rejection_reason.empty()) {
                                candidate_merit = fixed_ptc_candidate_merit(
                                    candidate, candidate_profiles,
                                    rejection_reason);
                                if (rejection_reason.empty() &&
                                    !anderson_merit.accepts(
                                        candidate_merit, combined_merit)) {
                                    rejection_reason =
                                        "nonmonotone_merit_window_exceeded";
                                }
                            }
                            if (rejection_reason.empty()) {
                                anderson_candidate_profiles =
                                    std::move(candidate_profiles);
                                double candidate_log_amplitude = 0.0;
                                unpack_finite_ptc_state(
                                    candidate, candidate_log_amplitude,
                                    anderson_candidate_profiles);
                                next_state = current_state;
                                next_state(0) = candidate_log_amplitude;
                                next_state.tail(base_recycled.size()) =
                                    pack_recycled_profiles(
                                        anderson_candidate_profiles);
                                have_anderson_candidate_profiles = true;
                                update_method = "anderson_relaxed";
                                update_alpha = alpha;
                                ++result.anderson_accepted;
                                anderson_merit.accept(candidate_merit);
                                result.anderson_backtracks += backtrack;
                                anderson_accepted = true;
                                anderson_update_accepted = true;
                                std::cout << "PSEUDO_TIME_ANDERSON_TRIAL"
                                          << " accepted=1 macro=" << macro
                                          << " alpha=" << alpha
                                          << " scaled_step="
                                          << candidate_scaled_step
                                           << " trust_limit="
                                           << 2.0 * fixed_scaled_step
                                           << " candidate_merit="
                                           << candidate_merit
                                           << "\n";
                                break;
                            }
                            ++result.anderson_rejected;
                            std::cout << "PSEUDO_TIME_ANDERSON_TRIAL"
                                      << " accepted=0 macro=" << macro
                                      << " alpha=" << alpha
                                      << " scaled_step="
                                      << candidate_scaled_step
                                       << " trust_limit="
                                       << 2.0 * fixed_scaled_step
                                       << " candidate_merit="
                                       << candidate_merit
                                       << " reason="
                                      << std::quoted(rejection_reason)
                                      << "\n";
                            alpha *= 0.5;
                        }
                        if (!anderson_accepted) {
                            std::cout << "PSEUDO_TIME_ANDERSON_FALLBACK"
                                      << " macro=" << macro
                                      << " reason=anderson_trials_rejected\n";
                        }
                        }
                    } else if (fixed_amplitude_diagnostic) {
                        ++result.anderson_rejected;
                        next_state = fixed_relaxed_state;
                        update_method = "finite_ptc_beta_zero";
                        update_alpha = 0.0;
                        anderson_merit.accept(combined_merit);
                        anderson_update_accepted = true;
                        std::cout << "PSEUDO_TIME_ANDERSON_TRIAL"
                                  << " accepted=0 macro=" << macro
                                  << " beta_selected=0"
                                  << " beta_backtracking_count=0"
                                  << " positivity_limited=0"
                                  << " merit_limited=0"
                                  << " reason=nonfinite_qr_coefficients"
                                  << " action=accept_plain_ptc\n";
                    } else {
                        ++result.anderson_rejected;
                        std::cout << "PSEUDO_TIME_ANDERSON_TRIAL"
                                  << " accepted=0 macro=" << macro
                                  << " alpha=0 reason=nonfinite_qr_coefficients\n";
                    }
                }
                if (anderson_map_stationary && !anderson_update_accepted) {
                    bool fallback_accepted = false;
                    double alpha = 1.0;
                    for (int backtrack = 0; backtrack < 12; ++backtrack) {
                        const Vector candidate = finite_ptc_current_state +
                            alpha * (finite_ptc_mapped_state -
                                finite_ptc_current_state);
                        std::string rejection_reason =
                            finite_ptc_state_rejection_reason(candidate);
                        Profiles candidate_profiles;
                        double candidate_merit =
                            std::numeric_limits<double>::infinity();
                        if (rejection_reason.empty()) {
                            candidate_merit = fixed_ptc_candidate_merit(
                                candidate, candidate_profiles, rejection_reason);
                            if (rejection_reason.empty() &&
                                !anderson_merit.accepts(
                                    candidate_merit, combined_merit)) {
                                rejection_reason =
                                    "nonmonotone_merit_window_exceeded";
                            }
                        }
                        std::cout << "PSEUDO_TIME_DAMPED_MAP_TRIAL"
                                  << " accepted="
                                  << (rejection_reason.empty() ? 1 : 0)
                                  << " macro=" << macro
                                  << " alpha=" << alpha
                                  << " candidate_merit=" << candidate_merit
                                  << " reason="
                                  << std::quoted(rejection_reason)
                                  << "\n";
                        if (rejection_reason.empty()) {
                            anderson_candidate_profiles =
                                std::move(candidate_profiles);
                            double candidate_log_amplitude = 0.0;
                            unpack_finite_ptc_state(
                                candidate, candidate_log_amplitude,
                                anderson_candidate_profiles);
                            next_state = current_state;
                            next_state(0) = candidate_log_amplitude;
                            next_state.tail(base_recycled.size()) =
                                pack_recycled_profiles(
                                    anderson_candidate_profiles);
                            have_anderson_candidate_profiles = true;
                            update_method = "damped_finite_ptc_map";
                            update_alpha = alpha;
                            ++result.relaxed_fallbacks;
                            anderson_merit.accept(candidate_merit);
                            fallback_accepted = true;
                            break;
                        }
                        alpha *= 0.5;
                    }
                    if (!fallback_accepted) {
                        const bool watchdog_due = anderson_merit.reject();
                        if (!watchdog_due) {
                            next_state = fixed_relaxed_state;
                            update_method = "finite_ptc_watchdog_step";
                            update_alpha = 1.0;
                            std::cout << "PSEUDO_TIME_ANDERSON_WATCHDOG"
                                      << " macro=" << macro
                                      << " count="
                                      << anderson_merit.rejection_count()
                                      << " limit=" << anderson_merit.watchdog()
                                      << " action=accept_unaccelerated_fixed_map"
                                      << "\n";
                        } else {
                            ++anderson_ptc_restarts;
                            profiles = previous_macro_profiles;
                            amplitude = std::exp(current_state(0));
                            history.clear();
                            anderson_merit.clear();
                            fixed_ptc_map_parameters.reset();
                            fixed_ptc_anderson_mode = false;
                            plasma_handoff_ready = false;
                            neutral_atom_handoff_ready =
                                !use_neutral_pseudo_time;
                            neutral_molecule_handoff_ready =
                                !use_neutral_pseudo_time;
                            steady_transition_streak = 0;
                            neutral_atom_steady_macro_streak = 0;
                            neutral_molecule_steady_macro_streak = 0;
                            successful_be_step_streak = 0;
                            neutral_atom_success_streak = 0;
                            neutral_molecule_success_streak = 0;
                            plasma_step_calibrated = false;
                            neutral_atom_step_calibrated = false;
                            neutral_molecule_step_calibrated =
                                fixed_amplitude_diagnostic &&
                                diagnostic_neutral_molecule_step_scale != 1.0;
                            ptc_true_merit_history.clear();
                            std::cout << "ANDERSON_RETURN_TO_PTC"
                                      << " macro=" << macro
                                      << " restart=" << anderson_ptc_restarts
                                      << " dtau_P=" << pseudo_time_step
                                      << " dtau_A=" << neutral_atom_step
                                      << " dtau_M=" << neutral_molecule_step
                                      << " reason=" << std::quoted(
                                          "nonmonotone_watchdog_expired")
                                      << "\n";
                            continue;
                        }
                    }
                }
                if (!joint_state_is_feasible(next_state)) {
                    if (use_neutral_pseudo_time && !transient_neutral_mode) {
                        next_state = fixed_relaxed_state;
                        update_method = "neutral_be_map";
                        update_alpha = macro_relaxation;
                    } else {
                        next_state = physical_state(
                            scaled_current + 0.5 * macro_relaxation * joint_residual);
                        update_method = "fixed_point_relaxed_backtracked";
                        update_alpha = 0.5 * macro_relaxation;
                    }
                }
                if (fixed_amplitude_diagnostic &&
                    next_state(0) != diagnostic_fixed_log_amplitude) {
                    result.failure =
                        "fixed-amplitude diagnostic attempted to mutate ell";
                    result.outer_iterations = macro;
                    result.elapsed_seconds = elapsed_seconds();
                    return result;
                }
                if (!joint_state_is_feasible(next_state)) {
                    const JointStateInfeasibility infeasibility =
                        joint_state_infeasibility(next_state);
                    result.infeasible = true;
                    result.outer_iterations = macro;
                    result.failure = "joint update infeasible: " +
                        infeasibility.reason;
                    result.attempted_amplitude = std::exp(next_state(0));
                    result.first_failing_cell = infeasibility.cell;
                    result.first_failing_species = infeasibility.gi;
                    std::cout << "PSEUDO_TIME_INFEASIBLE_CANDIDATE"
                              << " macro=" << macro
                              << " reason="
                              << std::quoted(infeasibility.reason)
                              << " block=" << infeasibility.block
                              << " cell=" << infeasibility.cell
                              << " gi=" << infeasibility.gi
                              << " species=" << std::quoted(
                                  infeasibility.gi >= 0
                                      ? species_label(infeasibility.gi)
                                      : "none")
                              << " value=" << infeasibility.value
                              << " current_log_amplitude=" << current_state(0)
                              << " proposed_log_amplitude=" << next_state(0)
                              << " proposed_delta_log="
                              << next_state(0) - current_state(0)
                              << " current_amplitude="
                              << std::exp(current_state(0))
                              << " proposed_amplitude="
                              << std::exp(next_state(0))
                              << " trust_center=" << amplitude_trust.center
                              << " trust_radius=" << amplitude_trust.radius
                              << " update_method=" << update_method
                              << " relaxation=" << update_alpha
                              << " fixed_ptc_map_used="
                              << (anderson_map_stationary ? 1 : 0)
                              << " map_role="
                              << (anderson_map_stationary
                                  ? "fixed_finite_ptc" : "adaptive_ptc")
                              << "\n";
                    result.elapsed_seconds = elapsed_seconds();
                    return result;
                }

                std::cout << "PSEUDO_TIME_MACRO_CYCLE"
                          << " macro=" << macro
                          << " plasma_steps=" << plasma_steps_this_macro
                          << " mode=" << (anderson_map_stationary
                              ? "fixed_ptc_anderson" : "adaptive_ptc")
                          << " dtau_next=" << pseudo_time_step;
                std::cout
                          << " phi=" << combined_merit
                          << " target_component="
                          << std::abs(mismatch) / stage_tolerance
                          << " steady_component="
                          << steady_diagnostics.max_steady_backward_error /
                              stage_tolerance
                          << " profile_component="
                          << neutral_profile_residual / stage_tolerance
                          << " macro_profile_change=" << macro_profile_change
                          << " last_be_change=" << last_be_step_change
                          << " F_BE="
                          << last_plasma_trial.diagnostics
                              .max_pseudo_time_backward_error
                          << " F_steady="
                          << steady_diagnostics.max_steady_backward_error
                          << " transient_update_ratio="
                          << last_transient_negligibility
                          << " state_change=" << macro_profile_change
                          << " fixed_ptc_map_residual="
                          << finite_ptc_joint_residual.cwiseAbs().maxCoeff()
                          << " P_raw_change="
                          << last_plasma_trial.diagnostics
                              .max_pseudo_time_state_change_norm
                          << " P_raw_transient="
                          << last_plasma_trial.diagnostics
                              .max_pseudo_time_transient_term_norm
                          << " P_raw_steady="
                          << last_plasma_trial.diagnostics
                              .max_steady_residual_norm
                          << " P_raw_BE="
                          << last_plasma_trial.diagnostics
                              .max_pseudo_time_residual_norm
                          << " P_D_BE_relative="
                          << last_plasma_trial.diagnostics
                              .max_be_identity_defect_relative
                          << " update=" << update_method
                          << " alpha=" << update_alpha
                          << " stable_macro_streak="
                          << stable_macro_cycle_streak
                          << " nonmonotone_window="
                          << anderson_merit.window()
                          << " nonmonotone_limit="
                          << anderson_merit.acceptance_limit(combined_merit)
                          << " watchdog_count="
                          << anderson_merit.rejection_count()
                          << " watchdog_limit=" << anderson_merit.watchdog();
                if (use_neutral_pseudo_time) {
                    std::cout << " neutral_mode_A=finite_ptc"
                              << " neutral_mode_M=finite_ptc"
                              << " neutral_dtau_A=" << neutral_atom_step
                              << " neutral_dtau_M=" << neutral_molecule_step
                              << " neutral_eta_A="
                              << last_neutral_sweep.atom.transient_update_ratio
                              << " neutral_eta_M="
                              << last_neutral_sweep.molecule.transient_update_ratio
                              << " neutral_residual_A=" << neutral_atom_residual
                              << " neutral_residual_M=" << neutral_molecule_residual
                              << " neutral_steady_A="
                              << last_neutral_sweep.atom.steady_backward_error
                              << " neutral_steady_M="
                              << last_neutral_sweep.molecule.steady_backward_error
                              << " A_raw_change="
                              << last_neutral_sweep.atom.state_change_norm
                              << " A_raw_transient="
                              << last_neutral_sweep.atom.transient_term_norm
                              << " A_raw_steady="
                              << last_neutral_sweep.atom.steady_residual_norm
                              << " A_raw_BE="
                              << last_neutral_sweep.atom.be_residual_norm
                              << " A_D_BE_relative="
                              << last_neutral_sweep.atom
                                  .be_identity_defect_relative
                              << " M_raw_change="
                              << last_neutral_sweep.molecule.state_change_norm
                              << " M_raw_transient="
                              << last_neutral_sweep.molecule.transient_term_norm
                              << " M_raw_steady="
                              << last_neutral_sweep.molecule.steady_residual_norm
                              << " M_raw_BE="
                              << last_neutral_sweep.molecule.be_residual_norm
                              << " M_D_BE_relative="
                              << last_neutral_sweep.molecule
                                  .be_identity_defect_relative;
                }
                std::cout
                          << "\n";

                const bool ready_for_audit =
                    anderson_map_stationary_at_macro_start &&
                    !macro_had_retry &&
                    combined_merit < 1.0 &&
                    fixed_scaled_step < stage_tolerance &&
                    macro_profile_change < stage_tolerance &&
                    boundary_change < stage_tolerance &&
                    std::isfinite(minimum_population) &&
                    minimum_population >= 0.0;
                if (ready_for_audit && strict_endpoint_audit != nullptr) {
                    ContinuationCheckpoint audit_checkpoint;
                    audit_checkpoint.lambda = 1.0;
                    audit_checkpoint.amplitude = evaluated_amplitude;
                    audit_checkpoint.profiles = evaluated_profiles;
                    audit_checkpoint.upstream_boundary = evaluated_boundary;
                    const std::uint64_t state_hash_before =
                        checkpoint_state_hash(audit_checkpoint);
                    EndpointAuditResult candidate_audit =
                        perform_lambda_one_endpoint_audit(audit_checkpoint);
                    EndpointAuditResult reproduced_audit =
                        perform_lambda_one_endpoint_audit(audit_checkpoint);
                    const std::uint64_t state_hash_after =
                        checkpoint_state_hash(audit_checkpoint);
                    const bool state_unchanged =
                        state_hash_before == state_hash_after;
                    const double reproduction_difference = std::max({
                        std::abs(candidate_audit.ion_trial.mismatch -
                            reproduced_audit.ion_trial.mismatch),
                        std::abs(candidate_audit.joint_residual -
                            reproduced_audit.joint_residual),
                        std::abs(candidate_audit.profile_change -
                            reproduced_audit.profile_change),
                        std::abs(candidate_audit.boundary_change -
                            reproduced_audit.boundary_change)});
                    const bool candidate_physical_audit_accepted =
                        fixed_amplitude_diagnostic
                        ? endpoint_inner_audit_accepted(candidate_audit)
                        : endpoint_audit_accepted(candidate_audit);
                    const bool reproduced_physical_audit_accepted =
                        fixed_amplitude_diagnostic
                        ? endpoint_inner_audit_accepted(reproduced_audit)
                        : endpoint_audit_accepted(reproduced_audit);
                    const bool audit_accepted =
                        candidate_physical_audit_accepted &&
                        reproduced_physical_audit_accepted &&
                        state_unchanged &&
                        reproduction_difference <= 1.0e-12;
                    const auto& audit_sweep =
                        candidate_audit.ion_trial.diagnostics;
                    std::string audit_failure_class = "none";
                    if (!state_unchanged) {
                        audit_failure_class = "audit_state_mutation_detected";
                    } else if (reproduction_difference > 1.0e-12) {
                        audit_failure_class = "steady_audit_inconsistency";
                    } else if (!audit_sweep.cells_converged) {
                        audit_failure_class = "audit_inner_tolerance_failure";
                    } else if (audit_sweep.max_steady_backward_error >=
                               kLocalResidualTolerance) {
                        audit_failure_class = "converged_map_not_steady";
                    } else if (audit_sweep.max_conservation_epsilon > 1.0e-6 ||
                               audit_sweep.max_hminus_epsilon >
                                   kNewtonResidualTolerance ||
                               audit_sweep.max_omitted_epsilon >
                                   kNewtonResidualTolerance) {
                        audit_failure_class = "physical_row_failure";
                    } else if (!audit_accepted) {
                        audit_failure_class =
                            "premature_convergence_or_be_cancellation";
                    }
                    std::cout << "TEMPERATURE_AUDIT_CANDIDATE"
                              << " iteration=" << macro
                              << " accepted=" << (audit_accepted ? 1 : 0)
                              << " failure_class="
                              << std::quoted(audit_failure_class)
                              << " fixed_amplitude_diagnostic="
                              << (fixed_amplitude_diagnostic ? 1 : 0)
                              << " target_mismatch_is_gate="
                              << (fixed_amplitude_diagnostic ? 0 : 1)
                              << " joint_residual="
                              << candidate_audit.joint_residual
                              << " profile_change="
                              << candidate_audit.profile_change
                              << " boundary_change="
                              << candidate_audit.boundary_change
                              << " reproduction_difference="
                              << reproduction_difference
                              << " state_hash_before=" << state_hash_before
                              << " state_hash_after=" << state_hash_after
                              << " state_unchanged="
                              << (state_unchanged ? 1 : 0)
                              << " amplitude_sensitivity_status="
                              << std::quoted(
                                  "insufficient_audited_fixed_amplitude_points")
                              << " audited_secant_slope=nan"
                              << " centered_slope=nan"
                              << " bound_hits=" << amplitude_bound_hits
                              << "\n";
                    if (audit_accepted) {
                        *strict_endpoint_audit = std::move(candidate_audit);
                        result.converged = true;
                        result.elapsed_seconds = elapsed_seconds();
                        return result;
                    }
                    history.clear();
                    anderson_merit.clear();
                    fixed_ptc_map_parameters.reset();
                    fixed_ptc_anderson_mode = false;
                    if (audit_failure_class ==
                            "audit_state_mutation_detected" ||
                        audit_failure_class ==
                            "steady_audit_inconsistency") {
                        result.failure = audit_failure_class;
                        result.outer_iterations = macro;
                        result.elapsed_seconds = elapsed_seconds();
                        return result;
                    }
                    plasma_handoff_ready = false;
                    neutral_atom_handoff_ready = !use_neutral_pseudo_time;
                    neutral_molecule_handoff_ready = !use_neutral_pseudo_time;
                    steady_transition_streak = 0;
                    neutral_atom_steady_macro_streak = 0;
                    neutral_molecule_steady_macro_streak = 0;
                    ptc_true_merit_history.clear();
                    next_state = fixed_relaxed_state;
                    have_anderson_candidate_profiles = false;
                    update_method = "audit_recovery_finite_ptc";
                    update_alpha = 1.0;
                    std::cout << "TEMPERATURE_AUDIT_RECOVERY"
                              << " iteration=" << macro
                              << " failure_class="
                              << std::quoted(audit_failure_class)
                              << " action=return_to_adaptive_ptc"
                              << " dtau_P=" << pseudo_time_step
                              << " dtau_A=" << neutral_atom_step
                              << " dtau_M=" << neutral_molecule_step
                              << "\n";
                }

                current_state = next_state;
                amplitude = std::exp(current_state(0));
                if (have_anderson_candidate_profiles) {
                    profiles = std::move(anderson_candidate_profiles);
                }
                unpack_recycled_profiles(
                    current_state.tail(base_recycled.size()), profiles);
                if (fixed_ptc_anderson_mode &&
                    !fixed_ptc_map_parameters.has_value()) {
                    fixed_ptc_map_parameters =
                        dcr::solver::FixedPtcMapParameters{
                            pseudo_time_step, neutral_atom_step,
                            neutral_molecule_step, amplitude_trust.center,
                            amplitude_trust.radius};
#ifdef DCR_NESTED_DIAGNOSTIC
                    if (use_local_plasma_pseudo_time) {
                        frozen_local_plasma_steps = local_plasma_steps;
                    }
#endif
                    std::cout << "PSEUDO_TIME_FIXED_MAP_FROZEN"
                              << " macro=" << macro
                              << " history_active_from_macro=" << macro + 1
                              << " dtau_P=" << pseudo_time_step
                              << " dtau_A=" << neutral_atom_step
                              << " dtau_M=" << neutral_molecule_step
                              << " trust_center=" << amplitude_trust.center
                              << " trust_radius=" << amplitude_trust.radius
#ifdef DCR_NESTED_DIAGNOSTIC
                              << " local_P_array="
                              << (use_local_plasma_pseudo_time ? 1 : 0)
                              << " local_P_hash="
                              << (use_local_plasma_pseudo_time
                                  ? local_plasma_hash() : 0)
#endif
                              << "\n";
                }
                result.update_type = update_method;

                if (anderson_map_stationary_at_macro_start &&
                    !macro_had_retry &&
                    macro_target_residual_history.size() >= 100 &&
                    macro % 250 == 0) {
                    const size_t count = macro_target_residual_history.size();
                    const size_t begin = count - 100;
                    std::vector<double> first_target(
                        macro_target_residual_history.begin() + begin,
                        macro_target_residual_history.begin() + begin + 20);
                    std::vector<double> last_target(
                        macro_target_residual_history.end() - 20,
                        macro_target_residual_history.end());
                    std::vector<double> first_profile(
                        macro_profile_residual_history.begin() + begin,
                        macro_profile_residual_history.begin() + begin + 20);
                    std::vector<double> last_profile(
                        macro_profile_residual_history.end() - 20,
                        macro_profile_residual_history.end());
                    std::vector<double> first_joint(20);
                    std::vector<double> last_joint(20);
                    for (size_t i = 0; i < 20; ++i) {
                        first_joint[i] = std::max(
                            first_target[i], first_profile[i]);
                        last_joint[i] = std::max(
                            last_target[i], last_profile[i]);
                    }
                    const double first_target_median = median(first_target);
                    const double last_target_median = median(last_target);
                    const double first_profile_median = median(first_profile);
                    const double last_profile_median = median(last_profile);
                    const double first_joint_median = median(first_joint);
                    const double last_joint_median = median(last_joint);
                    const double target_contraction =
                        last_target_median / first_target_median;
                    const double profile_contraction =
                        last_profile_median / first_profile_median;
                    const double joint_contraction =
                        last_joint_median / first_joint_median;
                    const bool target_progress =
                        target_contraction < 0.98 ||
                        last_target_median < stage_tolerance;
                    const bool slow_progress = fixed_amplitude_diagnostic
                        ? profile_contraction < 0.98
                        : (target_progress &&
                           joint_contraction < 0.98 &&
                           profile_contraction < 0.98);
                    std::cout << "PSEUDO_TIME_MACRO_BLOCK"
                              << " macro=" << macro
                              << " classification="
                              << (slow_progress ? "slow_progress" : "stagnation")
                              << " target_contraction=" << target_contraction
                              << " joint_contraction=" << joint_contraction
                              << " profile_contraction=" << profile_contraction
                              << " target_first20_median="
                              << first_target_median
                              << " target_last20_median=" << last_target_median
                              << " joint_first20_median=" << first_joint_median
                              << " joint_last20_median=" << last_joint_median
                              << " profile_first20_median="
                              << first_profile_median
                              << " profile_last20_median="
                              << last_profile_median
                              << "\n";
                    if (!slow_progress || macro == maximum_macro_cycles) {
                        DominantNeutralResidual dominant =
                            dominant_neutral_residual(
                                evaluated_recycled,
                                swept_recycled);
                        if (!fixed_amplitude_diagnostic &&
                            std::abs(mismatch) > dominant.residual) {
                            dominant.block = "target";
                            dominant.cell = 0;
                            dominant.gi = -1;
                            dominant.numerator = mismatch;
                            dominant.normalization = 1.0;
                            dominant.residual = std::abs(mismatch);
                        }
                        std::cout << "DOMINANT_JOINT_COMPONENT"
                                  << " macro=" << macro
                                  << " block=" << dominant.block
                                  << " cell=" << dominant.cell
                                  << " gi=" << dominant.gi
                                  << " species=" << std::quoted(
                                      dominant.gi >= 0
                                          ? species_label(dominant.gi)
                                          : "none")
                                  << " numerator=" << dominant.numerator
                                  << " normalization="
                                  << dominant.normalization
                                  << " residual=" << dominant.residual
                                  << " target_mismatch=" << mismatch
                                  << "\n";
                        result.failure = slow_progress
                            ? "slow_progress_extended_macro_limit"
                            : "neutral_coupling_stagnation";
                        result.elapsed_seconds = elapsed_seconds();
                        return result;
                    }
                }
            }

            result.outer_iterations = maximum_macro_cycles;
            result.failure = fixed_amplitude_diagnostic
                ? "inner_solve_stalled"
                : "slow_progress_extended_macro_limit";
            result.attempted_amplitude = amplitude;
            result.elapsed_seconds = elapsed_seconds();
            return result;
        }

        for (int outer = 1; outer <= maximum_outer_iterations; ++outer) {
            current_outer_iteration_ = outer;
            const Profiles previous = profiles;
            if (use_pseudo_time) {
                pseudo_time_old_background = profiles.background;
            }
            const Vector scaled_current = current_state.cwiseQuotient(anderson_scale);
            double baseline_target_mismatch =
                std::numeric_limits<double>::quiet_NaN();
            double baseline_profile_residual =
                std::numeric_limits<double>::quiet_NaN();
            SweepDiagnostics baseline_diagnostics;
            if (use_pseudo_time && outer == 1) {
                Profiles baseline_profiles = profiles;
                const Vector baseline_recycled =
                    current_state.tail(base_recycled.size());
                unpack_recycled_profiles(baseline_recycled, baseline_profiles);
                const UpstreamIonBoundary baseline_boundary =
                    make_upstream_ion_boundary(
                        std::exp(current_state(0)), composition);
                baseline_diagnostics =
                    evaluate_residuals(baseline_profiles, baseline_boundary);
                const double baseline_target_nuclei =
                    weighted_sum(
                        baseline_profiles.background.front(),
                        boundary_.P_indices, levels_) +
                    weighted_sum(
                        baseline_profiles.flowA.front(),
                        boundary_.A_indices, levels_) +
                    weighted_sum(
                        baseline_profiles.flowM.front(),
                        boundary_.M_indices, levels_);
                baseline_target_mismatch =
                    (baseline_target_nuclei - config_.plasma.total_density) /
                    config_.plasma.total_density;
                const Vector baseline_swept_recycled = pack_recycled_profiles(
                    sweep_recycled_neutrals(
                        baseline_profiles, lambda_recycling));
                baseline_profile_residual = recycled_fixed_point_residual(
                    baseline_recycled, baseline_swept_recycled);
            }
            TrialEvaluation current = evaluate_state(current_state);
            if (use_pseudo_time && outer == 1) {
                const double epsilon_f = stage_tolerance;
                const double epsilon_R = stage_tolerance;
                const double epsilon_p = stage_tolerance;
                const double baseline_target_component =
                    std::abs(baseline_target_mismatch) / epsilon_f;
                const double baseline_steady_component =
                    baseline_diagnostics.max_steady_backward_error / epsilon_R;
                const double baseline_profile_component =
                    baseline_profile_residual / epsilon_p;
                previous_accepted_combined_merit = std::max({
                    baseline_target_component,
                    baseline_steady_component,
                    baseline_profile_component});
                const double trial_target_component =
                    current.target_residual / epsilon_f;
                const double trial_steady_component =
                    current.ion_trial.diagnostics.max_steady_backward_error /
                    epsilon_R;
                const double trial_profile_component =
                    current.recycled_profile_residual / epsilon_p;
                auto weighted_transient_ratio_quantile = [&](int selected_gi) {
                    std::vector<std::pair<double, double>> samples;
                    double total_weight = 0.0;
                    for (size_t k = 0;
                         k < current.ion_trial.cell_diagnostics.size(); ++k) {
                        const auto& ratios = current.ion_trial
                            .cell_diagnostics[k]
                            .transient_to_steady_row_scale_ratios;
                        if (ratios.size() !=
                            static_cast<size_t>(pseudo_time_old_background[k].size())) {
                            continue;
                        }
                        const double local_nuclei = std::max(
                            kDensityFloor, total_nuclei(profiles, k));
                        for (int pi = 0; pi < pseudo_time_old_background[k].size(); ++pi) {
                            const int gi = boundary_.P_indices[
                                static_cast<size_t>(pi)];
                            if (selected_gi >= 0 && gi != selected_gi) continue;
                            const double weight = muP_(pi) * std::max(
                                0.0, pseudo_time_old_background[k](pi)) /
                                local_nuclei;
                            if (!(weight > 0.0)) continue;
                            samples.emplace_back(
                                ratios[static_cast<size_t>(pi)], weight);
                            total_weight += weight;
                        }
                    }
                    if (samples.empty() || !(total_weight > 0.0)) {
                        return std::numeric_limits<double>::quiet_NaN();
                    }
                    std::sort(samples.begin(), samples.end());
                    const double target_weight = 0.95 * total_weight;
                    double cumulative_weight = 0.0;
                    for (const auto& [ratio, weight] : samples) {
                        cumulative_weight += weight;
                        if (cumulative_weight >= target_weight) return ratio;
                    }
                    return samples.back().first;
                };
                const double dominant_nuclei_transient_ratio_q95 =
                    weighted_transient_ratio_quantile(-1);
                std::cout
                    << "PSEUDO_TIME_ITERATION1_DIAGNOSTIC"
                    << " dtau=" << pseudo_time_step
                    << " estimator_quantile=" << pseudo_time_estimate.quantile
                    << " quantile_rate=" << pseudo_time_estimate.quantile_rate
                    << " maximum_effective_rate="
                    << pseudo_time_estimate.maximum_rate
                    << " quantile_cell=" << pseudo_time_estimate.limiting_cell
                    << " quantile_pi=" << pseudo_time_estimate.limiting_pi
                    << " quantile_gi=" << pseudo_time_estimate.limiting_gi
                    << " quantile_species=" << std::quoted(
                        pseudo_time_estimate.limiting_gi >= 0
                            ? species_label(pseudo_time_estimate.limiting_gi)
                            : "none")
                    << " epsilon_f=" << epsilon_f
                    << " epsilon_R=" << epsilon_R
                    << " epsilon_p=" << epsilon_p
                    << " baseline_target_mismatch=" << baseline_target_mismatch
                    << " baseline_target_component=" << baseline_target_component
                    << " baseline_steady_residual="
                    << baseline_diagnostics.max_steady_backward_error
                    << " baseline_steady_component=" << baseline_steady_component
                    << " baseline_profile_residual=" << baseline_profile_residual
                    << " baseline_profile_component=" << baseline_profile_component
                    << " baseline_phi=" << previous_accepted_combined_merit
                    << " trial_target_mismatch=" << current.ion_trial.mismatch
                    << " trial_target_component=" << trial_target_component
                    << " trial_steady_residual="
                    << current.ion_trial.diagnostics.max_steady_backward_error
                    << " trial_steady_component=" << trial_steady_component
                    << " trial_profile_residual="
                    << current.recycled_profile_residual
                    << " trial_profile_component=" << trial_profile_component
                    << " trial_phi=" << std::max({
                        trial_target_component,
                        trial_steady_component,
                        trial_profile_component})
                    << " max_transient_to_steady_row_scale_ratio="
                    << current.ion_trial.diagnostics
                        .max_transient_to_steady_row_scale_ratio
                    << " transient_ratio_cell="
                    << current.ion_trial.diagnostics.max_transient_ratio_cell
                    << " transient_ratio_pi="
                    << current.ion_trial.diagnostics.max_transient_ratio_pi
                    << " transient_ratio_gi="
                    << current.ion_trial.diagnostics.max_transient_ratio_gi
                    << " dominant_nuclei_transient_ratio_q95="
                    << dominant_nuclei_transient_ratio_q95
                    << " population_update_norm="
                    << current.ion_trial.diagnostics.pseudo_time_profile_change
                    << " pseudo_time_modified_system_backward_error="
                    << current.ion_trial.diagnostics.max_pseudo_time_backward_error
                    << " pseudo_time_physical_residual_steady_scaled="
                    << current.ion_trial.diagnostics
                        .max_pseudo_time_physical_residual_steady_scaled
                    << " trial_valid=" << (current.valid ? 1 : 0)
                    << "\n";
                for (size_t rank = 0;
                     rank < pseudo_time_estimate.dominant_contributors.size();
                     ++rank) {
                    const auto& contributor =
                        pseudo_time_estimate.dominant_contributors[rank];
                    std::cout
                        << "PSEUDO_TIME_ESTIMATOR_DOMINANT_STATE"
                        << " rank=" << rank + 1
                        << " gi=" << contributor.gi
                        << " species=" << std::quoted(
                            species_label(contributor.gi))
                        << " nuclei_weight_fraction="
                        << contributor.nuclei_weight_fraction
                        << " weighted_mean_effective_rate="
                        << contributor.weighted_mean_rate
                        << " maximum_effective_rate="
                        << contributor.maximum_rate
                        << " transient_to_steady_ratio_q95="
                        << weighted_transient_ratio_quantile(contributor.gi)
                        << "\n";
                }
            }
            if (plain_backward_euler_once) {
                result.outer_iterations = 1;
                result.attempted_amplitude = current.ion_trial.amplitude;
                result.attempted_mismatch = current.ion_trial.mismatch;
                result.fixed_point_residual = current.fixed_point_residual;
                result.steady_backward_error =
                    current.ion_trial.diagnostics.max_steady_backward_error;
                result.pseudo_time_backward_error =
                    current.ion_trial.diagnostics.max_pseudo_time_backward_error;
                result.pseudo_time_profile_change =
                    current.ion_trial.diagnostics.pseudo_time_profile_change;
                result.ion_trial = current.ion_trial;
                result.profiles = current.profiles;
                result.update_type = "plain_backward_euler_diagnostic";
                result.failure = current.valid
                    ? "plain_backward_euler_diagnostic_complete"
                    : current.failure;
                result.elapsed_seconds = elapsed_seconds();
                return result;
            }
            if (!current.valid) {
                if (use_pseudo_time &&
                    pseudo_time_step > minimum_pseudo_time_step) {
                    pseudo_time_step = std::max(
                        minimum_pseudo_time_step, pseudo_time_step * 0.5);
                    result.pseudo_time_final_step = pseudo_time_step;
                    ++result.pseudo_time_rejected_steps;
                    accepted_macro_step_streak = 0;
                    history.clear();
                    std::cout << "PSEUDO_TIME_STEP accepted=0 iteration=" << outer
                              << " dtau=" << pseudo_time_step
                              << " reason=" << std::quoted(current.failure) << "\n";
                    continue;
                }
                result.infeasible = true;
                result.outer_iterations = outer;
                result.failure = current.failure;
                result.first_failing_cell = failure_cell(result.failure);
                result.first_failing_species = failure_species(result.failure);
                result.attempted_amplitude = amplitude;
                result.elapsed_seconds = elapsed_seconds();
                return result;
            }
            result.attempted_amplitude = current.ion_trial.amplitude;
            result.attempted_mismatch = current.ion_trial.mismatch;
            result.fixed_point_residual = current.fixed_point_residual;
            const double current_steady_backward_error =
                current.ion_trial.diagnostics.max_steady_backward_error;
            if (use_pseudo_time &&
                std::isfinite(previous_steady_backward_error) &&
                current_steady_backward_error >
                    1.05 * previous_steady_backward_error) {
                if (pseudo_time_step > minimum_pseudo_time_step) {
                    pseudo_time_step = std::max(
                        minimum_pseudo_time_step, pseudo_time_step * 0.5);
                    result.pseudo_time_final_step = pseudo_time_step;
                    ++result.pseudo_time_rejected_steps;
                    accepted_macro_step_streak = 0;
                    history.clear();
                    std::cout << "PSEUDO_TIME_STEP accepted=0 iteration=" << outer
                              << " dtau=" << pseudo_time_step
                              << " steady_residual="
                              << current_steady_backward_error
                              << " previous_steady_residual="
                              << previous_steady_backward_error
                              << " reason=pseudo_time_map_steady_residual_growth\n";
                    continue;
                }
                result.infeasible = true;
                result.outer_iterations = outer;
                result.failure =
                    "pseudo-time map grew the steady residual at the minimum step";
                result.amplitude = result.attempted_amplitude;
                result.mismatch = result.attempted_mismatch;
                result.update_type = "none";
                result.elapsed_seconds = elapsed_seconds();
                return result;
            }
            const Vector scaled_map = scaled_current + current.residual;
            const double residual_before = current.fixed_point_residual;
            const double merit_before = use_pseudo_time &&
                std::isfinite(previous_accepted_combined_merit)
                ? previous_accepted_combined_merit
                : current.combined_merit;
            auto trial_is_improving = [&](const TrialEvaluation& trial) {
                if (!trial.valid) return false;
                if (!use_pseudo_time) {
                    return trial.fixed_point_residual < residual_before;
                }
                const bool steady_safeguard =
                    !std::isfinite(previous_steady_backward_error) ||
                    trial.ion_trial.diagnostics.max_steady_backward_error <=
                        1.05 * previous_steady_backward_error;
                return steady_safeguard && trial.combined_merit < merit_before;
            };
            auto report_macro_trial = [&](const TrialEvaluation& trial,
                                          const char* method,
                                          double alpha,
                                          int substep,
                                          bool accepted_trial) {
                if (!use_pseudo_time) return;
                std::cout << "PSEUDO_TIME_MACRO_TRIAL"
                          << " accepted=" << (accepted_trial ? 1 : 0)
                          << " iteration=" << outer
                          << " dtau=" << pseudo_time_step
                          << " method=" << method
                          << " alpha=" << alpha
                          << " substep=" << substep
                          << " baseline_phi=" << merit_before
                          << " trial_phi=" << trial.combined_merit
                          << " target_component="
                          << trial.target_residual / stage_tolerance
                          << " steady_component="
                          << trial.ion_trial.diagnostics.max_steady_backward_error /
                              stage_tolerance
                          << " profile_component="
                          << trial.recycled_profile_residual / stage_tolerance
                          << " be_residual="
                          << trial.ion_trial.diagnostics
                              .max_pseudo_time_backward_error
                          << " valid=" << (trial.valid ? 1 : 0);
                if (!trial.failure.empty()) {
                    std::cout << " failure=" << std::quoted(trial.failure);
                }
                std::cout << "\n";
            };

            history.push_back({scaled_current, current.residual});
            if (history.size() > 9) history.erase(history.begin());

            TrialEvaluation accepted;
            const char* accepted_method = "relaxed";
            double accepted_alpha = 0.0;
            if (history.size() >= 2) {
                const int differences = std::min<int>(
                    8, static_cast<int>(history.size()) - 1);
                const int first = static_cast<int>(history.size()) - differences - 1;
                Matrix delta_f(current_state.size(), differences);
                Matrix delta_x(current_state.size(), differences);
                for (int j = 0; j < differences; ++j) {
                    delta_f.col(j) =
                        history[static_cast<size_t>(first + j + 1)].residual -
                        history[static_cast<size_t>(first + j)].residual;
                    delta_x.col(j) =
                        history[static_cast<size_t>(first + j + 1)].state -
                        history[static_cast<size_t>(first + j)].state;
                }
                const Vector gamma =
                    delta_f.colPivHouseholderQr().solve(current.residual);
                if (gamma.allFinite()) {
                    const Vector accelerated_scaled = scaled_map -
                        (delta_x + delta_f) * gamma;
                    const Vector accelerated = physical_state(accelerated_scaled);
                    double alpha = 1.0;
                    for (int backtrack = 0; backtrack < 16; ++backtrack) {
                        const Vector trial_state = current_state +
                            alpha * (accelerated - current_state);
                        TrialEvaluation trial = evaluate_state(trial_state);
                        const bool improving = trial_is_improving(trial);
                        report_macro_trial(
                            trial, "anderson", alpha, backtrack, improving);
                        if (improving) {
                            accepted = std::move(trial);
                            accepted_method = backtrack == 0
                                ? "anderson" : "anderson_backtracked";
                            accepted_alpha = alpha;
                            ++result.anderson_accepted;
                            result.anderson_backtracks += backtrack;
                            break;
                        }
                        ++result.rejected_macro_trials;
                        ++result.anderson_rejected;
                        alpha *= 0.5;
                    }
                }
            }

            if (!accepted.valid) {
                std::string fallback_failure;
                for (double relaxed_alpha : {0.2, 0.15, 0.1}) {
                    if (use_pseudo_time) {
                        double alpha = relaxed_alpha;
                        for (int backtrack = 0; backtrack < 16; ++backtrack) {
                            const Vector trial_scaled =
                                scaled_current + alpha * current.residual;
                            TrialEvaluation trial = evaluate_state(
                                physical_state(trial_scaled));
                            ++result.relaxed_fallbacks;
                            const bool improving = trial_is_improving(trial);
                            report_macro_trial(
                                trial, "fixed_point_relaxed", alpha,
                                backtrack, improving);
                            if (improving) {
                                accepted = std::move(trial);
                                accepted_method = backtrack == 0
                                    ? "fixed_point_relaxed"
                                    : "fixed_point_relaxed_backtracked";
                                accepted_alpha = alpha;
                                break;
                            }
                            ++result.rejected_macro_trials;
                            if (!trial.valid) fallback_failure = trial.failure;
                            alpha *= 0.5;
                        }
                    } else {
                        TrialEvaluation trial = current;
                        for (int substep = 0; substep < 256; ++substep) {
                            const Vector trial_scaled =
                                trial.state.cwiseQuotient(anderson_scale) +
                                relaxed_alpha * trial.residual;
                            trial = evaluate_state(physical_state(trial_scaled));
                            ++result.relaxed_fallbacks;
                            if (!trial.valid) {
                                fallback_failure = trial.failure;
                                break;
                            }
                            if (trial_is_improving(trial)) {
                                accepted = std::move(trial);
                                accepted_method = substep == 0
                                    ? "fixed_point_relaxed"
                                    : "fixed_point_relaxed_composite";
                                accepted_alpha = relaxed_alpha;
                                break;
                            }
                        }
                    }
                    if (accepted.valid) break;
                }
                if (!accepted.valid) {
                    if (use_pseudo_time &&
                        pseudo_time_step > minimum_pseudo_time_step) {
                        pseudo_time_step = std::max(
                            minimum_pseudo_time_step, pseudo_time_step * 0.5);
                        result.pseudo_time_final_step = pseudo_time_step;
                        ++result.pseudo_time_rejected_steps;
                        accepted_macro_step_streak = 0;
                        history.clear();
                        std::cout << "PSEUDO_TIME_STEP accepted=0 iteration=" << outer
                                  << " dtau=" << pseudo_time_step
                                  << " baseline_phi=" << merit_before
                                  << " reason=no_improving_complete_macro_step\n";
                        continue;
                    }
                    result.infeasible = true;
                    result.outer_iterations = outer;
                    result.failure = fallback_failure.empty()
                        ? (use_pseudo_time
                            ? "complete plasma-neutral macro-step trials did not "
                              "decrease combined merit "
                              "at the minimum pseudo-time step"
                            : "fixed-point relaxed trials did not reduce joint residual")
                        : fallback_failure;
                    result.first_failing_cell = failure_cell(result.failure);
                    result.first_failing_species = failure_species(result.failure);
                    result.amplitude = result.attempted_amplitude;
                    result.mismatch = result.attempted_mismatch;
                    result.update_type = "none";
                    result.elapsed_seconds = elapsed_seconds();
                    return result;
                }
            }

            const double accepted_steady_backward_error =
                accepted.ion_trial.diagnostics.max_steady_backward_error;
            profiles = std::move(accepted.profiles);
            current_state = accepted.state;
            amplitude = accepted.ion_trial.amplitude;
            const double profile_change = max_relative_change(profiles, previous);
            auto [physical_target_A, physical_target_M] =
                target_recycling(profiles.background.front());
            physical_target_A *= lambda_recycling;
            physical_target_M *= lambda_recycling;
            const double boundary_change = std::max(
                vector_relative_change(profiles.flowA.front(), physical_target_A),
                vector_relative_change(profiles.flowM.front(), physical_target_M));
            const double epsilon =
                accepted.ion_trial.diagnostics.max_conservation_epsilon;
            result.amplitude = amplitude;
            result.mismatch = accepted.ion_trial.mismatch;
            result.profile_change = profile_change;
            result.boundary_change = boundary_change;
            result.profiles = profiles;
            result.ion_trial = accepted.ion_trial;
            result.attempted_amplitude = amplitude;
            result.attempted_mismatch = accepted.ion_trial.mismatch;
            result.fixed_point_residual = accepted.fixed_point_residual;
            result.steady_backward_error = accepted_steady_backward_error;
            result.pseudo_time_backward_error =
                accepted.ion_trial.diagnostics.max_pseudo_time_backward_error;
            result.pseudo_time_profile_change =
                accepted.ion_trial.diagnostics.pseudo_time_profile_change;
            if (use_pseudo_time) {
                ++result.pseudo_time_accepted_steps;
                ++accepted_macro_step_streak;
                previous_steady_backward_error = accepted_steady_backward_error;
                previous_accepted_combined_merit = accepted.combined_merit;
                if (accepted_macro_step_streak >= 3) {
                    pseudo_time_step *= 1.5;
                    result.pseudo_time_final_step = pseudo_time_step;
                    accepted_macro_step_streak = 0;
                    history.clear();
                }
                std::cout << "PSEUDO_TIME_STEP accepted=1 iteration=" << outer
                          << " dtau=" << pseudo_time_step
                          << " baseline_phi=" << merit_before
                          << " accepted_phi=" << accepted.combined_merit
                          << " target_component="
                          << accepted.target_residual / stage_tolerance
                          << " steady_component="
                          << accepted_steady_backward_error / stage_tolerance
                          << " profile_component="
                          << accepted.recycled_profile_residual / stage_tolerance
                          << " steady_residual="
                          << result.steady_backward_error
                          << " pseudo_time_residual="
                          << result.pseudo_time_backward_error
                          << " pseudo_time_change="
                          << result.pseudo_time_profile_change
                          << " rejected_macro_trials_total="
                          << result.rejected_macro_trials << "\n";
            }
            result.minimum_population = std::numeric_limits<double>::infinity();
            for (const Vector& state : profiles.background) {
                if (state.size() > 0) {
                    result.minimum_population = std::min(
                        result.minimum_population, state.minCoeff());
                }
            }
            for (const Vector& state : profiles.flowA) {
                if (state.size() > 0) {
                    result.minimum_population = std::min(
                        result.minimum_population, state.minCoeff());
                }
            }
            for (const Vector& state : profiles.flowM) {
                if (state.size() > 0) {
                    result.minimum_population = std::min(
                        result.minimum_population, state.minCoeff());
                }
            }
            result.update_type = accepted_method;
            bool converged =
                std::abs(accepted.ion_trial.mismatch) < stage_tolerance &&
                accepted.fixed_point_residual < stage_tolerance &&
                profile_change < stage_tolerance &&
                boundary_change < stage_tolerance &&
                (!require_conservation || epsilon < 1.0e-6) &&
                accepted.ion_trial.diagnostics.max_hminus_epsilon <=
                    kNewtonResidualTolerance &&
                accepted.ion_trial.diagnostics.max_omitted_epsilon <=
                    kNewtonResidualTolerance &&
                (!use_pseudo_time ||
                 (result.steady_backward_error < stage_tolerance &&
                  result.pseudo_time_profile_change < stage_tolerance)) &&
                std::isfinite(result.minimum_population) &&
                result.minimum_population >= 0.0 &&
                accepted.ion_trial.diagnostics.cells_converged;
            if (converged && strict_endpoint_audit != nullptr) {
                ContinuationCheckpoint audit_checkpoint;
                audit_checkpoint.lambda = 1.0;
                audit_checkpoint.amplitude = amplitude;
                audit_checkpoint.profiles = profiles;
                audit_checkpoint.upstream_boundary =
                    accepted.ion_trial.upstream_boundary;
                EndpointAuditResult candidate_audit =
                    perform_lambda_one_endpoint_audit(audit_checkpoint);
                converged = endpoint_audit_accepted(candidate_audit);
                std::cout << "TEMPERATURE_AUDIT_CANDIDATE"
                          << " iteration=" << outer
                          << " accepted=" << (converged ? 1 : 0)
                          << " joint_residual=" << candidate_audit.joint_residual
                          << " profile_change=" << candidate_audit.profile_change
                          << "\n";
                if (converged) {
                    *strict_endpoint_audit = std::move(candidate_audit);
                }
            }
            if (outer == 1 || outer % 5 == 0 || converged) {
                std::cout << "AUTO_STAGE_PROGRESS lambda_rec=" << lambda_recycling
                          << " iteration=" << outer
                          << " amplitude=" << amplitude
                           << " mismatch=" << accepted.ion_trial.mismatch
                           << " tolerance=" << stage_tolerance
                           << " fixed_point_residual=" << accepted.fixed_point_residual
                          << " profile_change=" << profile_change
                          << " boundary_change=" << boundary_change
                          << " epsilon_Sigma_max=" << epsilon
                          << " epsilon_Hminus_max="
                          << accepted.ion_trial.diagnostics.max_hminus_epsilon
                           << " epsilon_omitted_max="
                           << accepted.ion_trial.diagnostics.max_omitted_epsilon
                           << " steady_residual=" << result.steady_backward_error
                           << " pseudo_time_residual="
                           << result.pseudo_time_backward_error
                           << " pseudo_time_change="
                           << result.pseudo_time_profile_change
                           << " dtau=" << pseudo_time_step
                          << " update=" << accepted_method
                          << " alpha=" << accepted_alpha
                          << " rejected_trials=" << result.anderson_rejected
                          << "\n";
            }

            if (converged) {
                result.converged = true;
                result.outer_iterations = outer;
                result.amplitude = amplitude;
                result.mismatch = accepted.ion_trial.mismatch;
                result.profile_change = profile_change;
                result.boundary_change = boundary_change;
                result.profiles = profiles;
                result.ion_trial = accepted.ion_trial;
                result.fixed_point_residual = accepted.fixed_point_residual;
                result.update_type = accepted_method;
                result.elapsed_seconds = elapsed_seconds();
                return result;
            }
        }
        result.outer_iterations = maximum_outer_iterations;
        result.failure = "outer_iteration_limit";
        result.attempted_amplitude = amplitude;
        result.amplitude = result.attempted_amplitude;
        result.mismatch = result.attempted_mismatch;
        result.elapsed_seconds = elapsed_seconds();
        return result;
    }

    int run_automatic_continuation(ContinuationCheckpoint checkpoint) {
        constexpr double maximum_step = 0.05;
        constexpr double minimum_step = 1.0e-4;
        const Vector composition =
            checkpoint.upstream_boundary.positive_ion_composition;
        if (checkpoint.last_step_rejected &&
            checkpoint.lambda < 1.0 - 1.0e-14 &&
            checkpoint.next_step < minimum_step) {
            report_minimum_population_locations(checkpoint.profiles);
            std::cout << "AUTOMATIC_CONTINUATION_RESULT converged=0"
                      << " accepted_lambda=" << checkpoint.lambda
                      << " next_step=" << checkpoint.next_step
                      << " failure=minimum_step_below_1e-4_checkpoint\n";
            return 2;
        }
        double step = std::clamp(
            checkpoint.next_step, minimum_step, maximum_step);

        while (checkpoint.lambda < 1.0 - 1.0e-14) {
            const double attempted_lambda = std::min(1.0, checkpoint.lambda + step);
            double predictor = checkpoint.amplitude;
            const double lambda_span = checkpoint.lambda - checkpoint.previous_lambda;
            if (lambda_span > 0.0) {
                predictor += (attempted_lambda - checkpoint.lambda) *
                    (checkpoint.amplitude - checkpoint.previous_amplitude) /
                    lambda_span;
            }
            predictor = std::max(1.0e-12, predictor);
            std::cout << "CONTINUATION_ATTEMPT from_lambda=" << checkpoint.lambda
                      << " lambda_rec=" << attempted_lambda
                      << " step=" << attempted_lambda - checkpoint.lambda
                      << " predictor=" << predictor
                      << " safe_endpoint=" << checkpoint.amplitude
                      << "\n";

            ContinuationPointResult point =
                solve_recycling_continuation_point_anderson(
                    checkpoint.profiles, composition, attempted_lambda,
                    predictor, checkpoint.amplitude);
            if (!point.converged) {
                const double old_step = step;
                step *= 0.5;
                append_stage_summary(
                    "rejected", checkpoint.lambda, attempted_lambda,
                    old_step, "rejected", point, nullptr);
                std::cout << "AUTO_STAGE_REJECTED lambda_rec=" << attempted_lambda
                          << " reason=" << std::quoted(point.failure)
                          << " first_failing_cell=" << point.first_failing_cell
                          << " attempted_amplitude=" << point.attempted_amplitude
                          << " attempted_mismatch=" << point.attempted_mismatch
                          << " outer_iterations=" << point.outer_iterations
                          << " old_step=" << old_step
                          << " reduced_step=" << step
                          << "\n";
                checkpoint.next_step = step;
                checkpoint.last_step_rejected = true;
                if (!save_continuation_checkpoint(checkpoint)) {
                    throw std::runtime_error(
                        "Could not save rejected continuation checkpoint");
                }
                if (step < minimum_step) {
                    std::cout << "AUTOMATIC_CONTINUATION_RESULT converged=0"
                              << " accepted_lambda=" << checkpoint.lambda
                              << " failed_lambda=" << attempted_lambda
                              << " next_step=" << step
                              << " failure=minimum_step_below_1e-4"
                              << " point_failure=" << std::quoted(point.failure)
                              << "\n";
                    return 2;
                }
                continue;
            }

            const ProfilePhysicalSummary physical = summarize_profiles(
                point.profiles, &point.ion_trial.upstream_boundary);
            const bool easy = point.outer_iterations <= 15;
            const bool moderate = point.outer_iterations <= 50;
            const char* difficulty = easy ? "easy" : (moderate ? "moderate" : "slow");
            append_stage_summary(
                "accepted", checkpoint.lambda, attempted_lambda,
                attempted_lambda - checkpoint.lambda, difficulty, point, &physical);
            std::cout << "AUTO_STAGE_ACCEPTED lambda_rec=" << attempted_lambda
                      << " step=" << attempted_lambda - checkpoint.lambda
                      << " difficulty=" << difficulty
                      << " coupled_iterations=" << point.outer_iterations
                      << " ion_sweep_evaluations=" << point.shooting_iterations
                      << " amplitude=" << point.amplitude
                      << " mismatch=" << point.mismatch
                      << " upstream_ion_nuclei_density="
                      << physical.upstream_ion_nuclei_density
                      << " target_atomic_flux=" << physical.target_atomic_flux
                      << " target_molecular_flux=" << physical.target_molecular_flux
                      << " profile_change=" << point.profile_change
                      << " minimum_population=" << physical.minimum_population
                      << " total_nuclei_min=" << physical.total_nuclei_min
                      << " total_nuclei_min_node=" << physical.total_nuclei_min_node
                      << " total_nuclei_max=" << physical.total_nuclei_max
                      << " total_nuclei_max_node=" << physical.total_nuclei_max_node
                      << " epsilon_Sigma_max="
                      << point.ion_trial.diagnostics.max_conservation_epsilon
                      << " epsilon_Sigma_cell="
                      << point.ion_trial.diagnostics.max_epsilon_cell
                      << " anderson_accepted=" << point.anderson_accepted
                      << " anderson_rejected=" << point.anderson_rejected
                      << " anderson_backtracks=" << point.anderson_backtracks
                      << " relaxed_fallbacks=" << point.relaxed_fallbacks
                      << " elapsed_seconds=" << point.elapsed_seconds
                      << "\n";

            const double accepted_lambda = checkpoint.lambda;
            const double accepted_amplitude = checkpoint.amplitude;
            checkpoint.previous_lambda = accepted_lambda;
            checkpoint.previous_amplitude = accepted_amplitude;
            checkpoint.lambda = attempted_lambda;
            checkpoint.amplitude = point.amplitude;
            checkpoint.profiles = point.profiles;
            checkpoint.upstream_boundary = point.ion_trial.upstream_boundary;
            if (easy) step = std::min(maximum_step, 1.5 * step);
            checkpoint.next_step = step;
            checkpoint.last_step_rejected = false;
            if (!save_continuation_checkpoint(checkpoint)) {
                std::cout << "AUTOMATIC_CONTINUATION_RESULT converged=0"
                          << " accepted_lambda=" << checkpoint.lambda
                          << " failure=checkpoint_write_failed\n";
                return 2;
            }
        }

        const IonTrial final_audit = evaluate_ion_trial(
            checkpoint.profiles, checkpoint.amplitude, composition);
        Profiles audited_profiles = checkpoint.profiles;
        audited_profiles.background = final_audit.background;
        const ProfilePhysicalSummary physical = summarize_profiles(
            audited_profiles, &final_audit.upstream_boundary);
        std::cout << "AUTOMATIC_CONTINUATION_RESULT converged=1 lambda_rec="
                  << checkpoint.lambda
                  << " amplitude=" << checkpoint.amplitude
                  << " target_mismatch=" << final_audit.mismatch
                  << " upstream_ion_nuclei_density="
                  << physical.upstream_ion_nuclei_density
                  << " target_atomic_flux=" << physical.target_atomic_flux
                  << " target_molecular_flux=" << physical.target_molecular_flux
                  << " background_min=" << physical.background_min
                  << " background_max=" << physical.background_max
                  << " flowA_min=" << physical.flowA_min
                  << " flowA_max=" << physical.flowA_max
                  << " flowM_min=" << physical.flowM_min
                  << " flowM_max=" << physical.flowM_max
                  << " total_nuclei_min=" << physical.total_nuclei_min
                  << " total_nuclei_min_node=" << physical.total_nuclei_min_node
                  << " total_nuclei_max=" << physical.total_nuclei_max
                  << " total_nuclei_max_node=" << physical.total_nuclei_max_node
                  << " epsilon_Sigma_max="
                  << final_audit.diagnostics.max_conservation_epsilon
                  << " epsilon_Sigma_cell="
                  << final_audit.diagnostics.max_epsilon_cell
                  << " checkpoint=" << std::quoted(continuation_checkpoint_path_)
                  << " stage_table=" << std::quoted(continuation_summary_path_)
                  << "\n";
        return run_lambda_one_endpoint_audit(std::move(checkpoint));
    }

    int run_reverse_cell_substep_diagnostic() {
        std::cout << std::scientific << std::setprecision(12);
        const double u_left = 2.0;
        const double u_right = 3.0;
        const double loss_rate = 0.4;
        const double dx = 0.5;
        const double n_right = 7.0;
        const double assembled = (u_right * n_right / dx) /
            (u_left / dx + loss_rate);
        const double expected = u_right * n_right / (u_left + loss_rate * dx);
        const double wrong_sign = u_right * n_right / (u_left - loss_rate * dx);
        const double sign_error = std::abs(assembled - expected) / expected;
        std::cout << "PURE_LOSS assembled_n_left=" << assembled
                  << " expected_n_left=" << expected
                  << " wrong_sign_n_left=" << wrong_sign
                  << " relative_error=" << sign_error
                  << " sign_correct=" << (sign_error < 1.0e-14 ? 1 : 0)
                  << "\n";
        if (!(assembled > 0.0) || sign_error >= 1.0e-14) {
            throw std::runtime_error("Reverse pure-loss transport sign is incorrect");
        }

        Profiles fixed = initial_profiles();
        const double amplitude = initial_shooting_amplitude(fixed.background.front());
        const Vector composition = upstream_ion_composition(fixed.background.front());
        const UpstreamIonBoundary upstream_face =
            make_upstream_ion_boundary(amplitude, composition);
        current_outer_iteration_ = 0;

        constexpr int failing_cell = 170;
        for (int k = static_cast<int>(fixed.background.size()) - 1;
             k > failing_cell; --k) {
            CellDiagnostics diagnostics;
            const Vector* upstream = k + 1 < static_cast<int>(fixed.background.size())
                ? &fixed.background[static_cast<size_t>(k + 1)] : nullptr;
            fixed.background[static_cast<size_t>(k)] = solve_reverse_cell(
                k,
                upstream,
                upstream == nullptr ? &upstream_face : nullptr,
                fixed.background[static_cast<size_t>(k)],
                fixed.flowA,
                fixed.flowM,
                amplitude,
                diagnostics);
        }

        const Vector frozen_upstream = fixed.background[static_cast<size_t>(failing_cell + 1)];
        const Vector frozen_initial = fixed.background[static_cast<size_t>(failing_cell)];
        std::cout << "RESIDUAL_ROW_DIAGNOSTIC cell=" << failing_cell
                  << " x_left=" << x_[failing_cell]
                  << " x_right=" << x_[failing_cell + 1]
                  << " shooting_amplitude=" << amplitude
                  << "\n";

        const Vector& flowA_left = fixed.flowA[static_cast<size_t>(failing_cell)];
        const Vector& flowA_right = fixed.flowA[static_cast<size_t>(failing_cell + 1)];
        const Vector& flowM_left = fixed.flowM[static_cast<size_t>(failing_cell)];
        const Vector& flowM_right = fixed.flowM[static_cast<size_t>(failing_cell + 1)];
        double selected_fraction = 0.0;
        const int selected_pi = select_conservation_replacement_row(
            frozen_initial, selected_fraction);
        for (int replacement_pi : {selected_pi, -1}) {
            const std::string mode = replacement_pi >= 0
                ? "weighted_conservation" : "all_physical_rows";
            const IntervalSolveResult result = solve_reverse_interval(
                x_[failing_cell], x_[failing_cell + 1], dx_[failing_cell],
                frozen_upstream, frozen_initial, flowA_left, flowA_right,
                flowM_left, flowM_right, replacement_pi);
            const Vector& audit_state = result.state;
            std::cout << "ROW_MODE_SUMMARY mode=" << mode
                      << " converged=" << (result.converged ? 1 : 0)
                      << " line_search_failed=" << (result.line_search_failed ? 1 : 0)
                      << " iterations=" << result.history.size()
                      << " final_residual_relative="
                       << result.final_physical_residual_relative
                       << " final_min_population=" << result.state.minCoeff()
                       << " audit_state=accepted_state"
                       << " audit_min_population=" << audit_state.minCoeff()
                       << "\n";
            report_residual_rows(
                mode, replacement_pi, audit_state, frozen_upstream,
                x_[failing_cell], x_[failing_cell + 1], dx_[failing_cell],
                flowA_left, flowA_right, flowM_left, flowM_right);
        }
        return 0;
    }

    static Vector interpolate(const Vector& left, const Vector& right, double fraction) {
        return (1.0 - fraction) * left + fraction * right;
    }

    IntervalSystem build_reverse_interval_system(
        const Vector& state,
        const Vector& upstream,
        double x_left,
        double x_right,
        double dx,
        const Vector& flowA_left,
        const Vector& flowA_right,
        const Vector& flowM_left,
        const Vector& flowM_right,
        int replacement_pi) const {
        const auto full = dcr::solver::make_background_full(state, boundary_, total_states_);
        const auto local = dcr::solver::assemble_local_system(
            config_, atomic_data_, plasma_, grid_, boundary_, full,
            flowA_left, flowM_left, x_left, false, &rate_cache_);
        const Matrix& rates = local.R_source_full;
        const Matrix Rpp = extract_block(rates, boundary_.P_indices, boundary_.P_indices);
        const auto chemistry = dcr::solver::assemble_local_chemistry_sources(
            local, boundary_, atomic_data_, state, flowA_left, flowM_left);
        const Vector source = chemistry.background - Rpp * state;
        IntervalSystem system;
        system.lhs = -Rpp;
        system.rhs = source;
        for (int pi = 0; pi < state.size(); ++pi) {
            if (is_positive_ion(pi)) {
                const int gi = boundary_.P_indices[static_cast<size_t>(pi)];
                system.lhs(pi, pi) += ion_speed(gi, x_left) / dx;
                system.rhs(pi) += ion_speed(gi, x_right) * upstream(pi) / dx;
            } else {
                system.lhs(pi, pi) += background_exhaust_rate(pi);
            }
        }
        const Vector original_residual = system.lhs * state - system.rhs;
        system.hminus_residual = original_residual(hminus_pi_);
        const Matrix original_lhs = system.lhs;
        const Vector original_rhs = system.rhs;
        const Vector exact_row = original_lhs.transpose() * muP_;
        const double exact_rhs = muP_.dot(original_rhs);
        if (replacement_pi >= 0) {
            system.lhs.row(replacement_pi) = exact_row.transpose();
            system.rhs(replacement_pi) = exact_rhs;
        }
        return system;
    }

    IntervalSolveResult solve_reverse_interval(
        double x_left,
        double x_right,
        double dx,
        const Vector& upstream,
        const Vector& initial,
        const Vector& flowA_left,
        const Vector& flowA_right,
        const Vector& flowM_left,
        const Vector& flowM_right,
        int replacement_pi) const {
        IntervalSolveResult result;
        const double initial_scale = std::max(kDensityFloor, initial.cwiseAbs().maxCoeff());
        result.interior_seed = std::max(kDensityFloor, 1.0e-12 * initial_scale);
        result.state = initial.cwiseMax(result.interior_seed);
        Vector log_state = result.state.unaryExpr([](double value) {
            return std::log(std::max(value, std::exp(kLogMinimum)));
        });
        auto decode = [](const Vector& encoded) {
            Vector decoded = encoded;
            for (int i = 0; i < decoded.size(); ++i) {
                decoded(i) = std::exp(std::clamp(encoded(i), kLogMinimum, kLogMaximum));
            }
            return decoded;
        };
        auto physical_residual = [&](const Vector& state) {
            const IntervalSystem system = build_reverse_interval_system(
                state, upstream, x_left, x_right, dx,
                flowA_left, flowA_right, flowM_left, flowM_right,
                replacement_pi);
            return system.lhs * state - system.rhs;
        };

        result.history.reserve(kMaxNewtonIterations);
        for (int iteration = 0; iteration < kMaxNewtonIterations; ++iteration) {
            const IntervalSystem system = build_reverse_interval_system(
                result.state, upstream, x_left, x_right, dx,
                flowA_left, flowA_right, flowM_left, flowM_right,
                replacement_pi);
            LocalIterationDiagnostics diagnostics;
            diagnostics.iteration = iteration;
            const Vector residual = system.lhs * result.state - system.rhs;
            diagnostics.physical_residual_norm = residual.norm();
            diagnostics.physical_residual_relative = componentwise_backward_error(
                system.lhs, result.state, system.rhs);
            diagnostics.minimum_population = result.state.minCoeff();

            const Vector lhs_value = system.lhs * result.state;
            Vector row_scale = Vector::Ones(residual.size());
            for (int i = 0; i < row_scale.size(); ++i) {
                row_scale(i) = std::max({1.0, std::abs(lhs_value(i)),
                                         std::abs(system.rhs(i))});
            }
            const Vector scaled_residual = residual.cwiseQuotient(row_scale);

            Eigen::FullPivLU<Matrix> lu(system.lhs);
            if (lu.rank() == system.lhs.rows()) {
                const Vector solved = lu.solve(system.rhs);
                if (solved.allFinite()) {
                    diagnostics.raw_solution_minimum = solved.minCoeff();
                    for (int i = 0; i < solved.size(); ++i) {
                        if (solved(i) <= 0.0) ++diagnostics.raw_nonpositive_count;
                    }
                }
            }

            for (int i = 0; i < result.state.size(); ++i) {
                if (result.state(i) <= kDensityFloor) {
                    ++diagnostics.floor_count;
                }
            }

            if (diagnostics.physical_residual_relative < kNewtonResidualTolerance) {
                result.converged = true;
                result.history.push_back(diagnostics);
                break;
            }

            Matrix jacobian = system.lhs;
            for (int row = 0; row < jacobian.rows(); ++row) {
                jacobian.row(row) /= row_scale(row);
            }
            for (int column = 0; column < jacobian.cols(); ++column) {
                jacobian.col(column) *= result.state(column);
            }
            Vector jacobian_column_scale = Vector::Ones(jacobian.cols());
            Matrix equilibrated_jacobian = jacobian;
            for (int column = 0; column < jacobian.cols(); ++column) {
                jacobian_column_scale(column) =
                    std::max(jacobian.col(column).norm(), 1.0e-30);
                equilibrated_jacobian.col(column) /= jacobian_column_scale(column);
            }

            Eigen::ColPivHouseholderQR<Matrix> qr(equilibrated_jacobian);
            qr.setThreshold(1.0e-14);
            diagnostics.jacobian_rank = qr.rank();
            result.matrix_singular = result.matrix_singular ||
                diagnostics.jacobian_rank < jacobian.cols();
            Vector newton_step = qr.solve(-scaled_residual)
                .cwiseQuotient(jacobian_column_scale);
            const bool finite_newton_step = newton_step.allFinite();
            if (!finite_newton_step) newton_step.setZero();
            const double unbounded_step_max = newton_step.cwiseAbs().maxCoeff();
            if (unbounded_step_max > 4.0) {
                newton_step *= 4.0 / unbounded_step_max;
            }
            diagnostics.newton_step_max = newton_step.cwiseAbs().maxCoeff();

            const double residual_norm = scaled_residual.norm();
            bool accepted = false;
            Vector accepted_log = log_state;
            Vector accepted_state = result.state;
            int attempted_backtracks = 0;
            auto try_direction = [&](const Vector& direction) {
                double step_length = 1.0;
                for (int backtrack = 0; backtrack <= kMaxNewtonBacktracks; ++backtrack) {
                    const Vector trial_log = log_state + step_length * direction;
                    const Vector trial_state = decode(trial_log);
                    const Vector trial_residual = physical_residual(trial_state);
                    const double trial_norm =
                        trial_residual.cwiseQuotient(row_scale).norm();
                    if (std::isfinite(trial_norm) &&
                        trial_norm < residual_norm * (1.0 - 1.0e-4 * step_length)) {
                        accepted = true;
                        accepted_log = trial_log;
                        accepted_state = trial_state;
                        diagnostics.line_search_backtracks = attempted_backtracks + backtrack;
                        diagnostics.accepted_step = step_length;
                        return;
                    }
                    step_length *= 0.5;
                }
                attempted_backtracks += kMaxNewtonBacktracks + 1;
            };
            if (finite_newton_step) try_direction(newton_step);

            if (!accepted) {
                Vector gradient_step =
                    -equilibrated_jacobian.transpose() * scaled_residual;
                gradient_step = gradient_step.cwiseQuotient(jacobian_column_scale);
                const double gradient_step_max = gradient_step.cwiseAbs().maxCoeff();
                if (gradient_step.allFinite() && gradient_step_max > 0.0) {
                    gradient_step *= std::min(1.0, 4.0 / gradient_step_max);
                    diagnostics.used_gradient_fallback = true;
                    try_direction(gradient_step);
                }
            }

            if (!accepted) {
                result.line_search_failed = true;
                diagnostics.line_search_backtracks = attempted_backtracks;
                result.history.push_back(diagnostics);
                break;
            }

            diagnostics.population_change = vector_relative_change(
                accepted_state, result.state);
            const IntervalSystem accepted_system = build_reverse_interval_system(
                accepted_state, upstream, x_left, x_right, dx,
                flowA_left, flowA_right, flowM_left, flowM_right,
                replacement_pi);
            diagnostics.linearized_post_residual_relative = componentwise_backward_error(
                accepted_system.lhs, accepted_state, accepted_system.rhs);
            result.history.push_back(diagnostics);
            log_state = accepted_log;
            result.state = accepted_state;
        }
        const IntervalSystem final_system = build_reverse_interval_system(
            result.state, upstream, x_left, x_right, dx,
            flowA_left, flowA_right, flowM_left, flowM_right,
            replacement_pi);
        const Vector final_residual = final_system.lhs * result.state - final_system.rhs;
        result.final_physical_residual_norm = final_residual.norm();
        result.final_physical_residual_relative = componentwise_backward_error(
            final_system.lhs, result.state, final_system.rhs);
        result.exact_residual = replacement_pi >= 0
            ? final_residual(replacement_pi)
            : std::numeric_limits<double>::quiet_NaN();
        result.hminus_residual = final_system.hminus_residual;
        return result;
    }

    std::string species_label(int global_index) const {
        if (global_index < 0 || global_index >= static_cast<int>(levels_.size())) {
            return "invalid";
        }
        const auto& level = levels_[static_cast<size_t>(global_index)];
        std::ostringstream label;
        label << level.label << "[g=" << global_index
              << ",q=" << level.charge
              << ",id=" << level.internal_id << "]";
        return label.str();
    }

    double quasineutral_density(const Vector& population) const {
        double density = 0.0;
        const int count = std::min<int>(population.size(), static_cast<int>(levels_.size()));
        for (int gi = 0; gi < count; ++gi) {
            density += static_cast<double>(levels_[static_cast<size_t>(gi)].charge) *
                std::max(population(gi), 0.0);
        }
        return std::max(0.0, density);
    }

    void redirect_diagnostic_dr_products(Matrix& rates) const {
        if (!config_.numerics.disable_h2plus_dr) return;
        int ground = -1;
        for (int gi = 0; gi < total_states_; ++gi) {
            const auto& level = levels_[static_cast<size_t>(gi)];
            if (level.type == dcr::atomic::SpeciesType::Atom &&
                level.charge == 0 && level.internal_id == 1) {
                ground = gi;
                break;
            }
        }
        if (ground < 0) return;
        for (int column = 0; column < total_states_; ++column) {
            const auto& donor = levels_[static_cast<size_t>(column)];
            if (donor.atomicity != 2 || donor.charge != 1 || !donor.is_background) continue;
            for (int row = 0; row < total_states_; ++row) {
                const auto& product = levels_[static_cast<size_t>(row)];
                if (product.type != dcr::atomic::SpeciesType::Atom ||
                    product.charge != 0 || product.internal_id <= 1) continue;
                rates(ground, column) += rates(row, column);
                rates(row, column) = 0.0;
            }
        }
    }

    SpeciesSourceBreakdown source_breakdown(
        int row_pi,
        const dcr::solver::LocalSystem& local,
        const Vector& background,
        const Vector& flowA,
        const Vector& flowM,
        double x_cm) const {
        SpeciesSourceBreakdown breakdown;
        const int row_gi = boundary_.P_indices[static_cast<size_t>(row_pi)];
        const auto& row_level = levels_[static_cast<size_t>(row_gi)];
        const bool charged_row = row_level.type == dcr::atomic::SpeciesType::Ion &&
            row_level.charge != 0;
        const bool atom_row = row_level.type == dcr::atomic::SpeciesType::Atom &&
            row_level.charge == 0;
        const bool molecule_row = row_level.type == dcr::atomic::SpeciesType::Molecule &&
            row_level.charge == 0;
        const auto temperatures = dcr::solver::evaluate_plasma_temperatures(config_, x_cm);
        const dcr::solver::LocalKineticContext context(
            plasma_, grid_, temperatures.electron_eV, temperatures.ion_eV,
            quasineutral_density(local.population_for_source));

        auto add_channel = [&](double contribution,
                               size_t process_index,
                               const dcr::ProcessBase& process,
                               const char* component,
                               int donor_gi) {
            if (contribution == 0.0 || !std::isfinite(contribution)) return;
            SourceChannel channel;
            channel.contribution = contribution;
            std::ostringstream label;
            label << process_kind(process) << "#" << process_index
                  << " donor=" << component << ":" << species_label(donor_gi);
            channel.label = label.str();
            breakdown.channels.push_back(std::move(channel));
            if (contribution > 0.0) breakdown.production += contribution;
            else breakdown.loss -= contribution;
        };

        const auto& processes = atomic_data_.get_processes();
        for (size_t process_index = 0; process_index < processes.size(); ++process_index) {
            const auto& process = processes[process_index];
            if (!process) continue;
            Matrix rates = Matrix::Zero(total_states_, total_states_);
            process->apply(
                context.plasma(), context.grid(), local.population_for_source, rates, nullptr);
            redirect_diagnostic_dr_products(rates);
            for (int pj = 0; pj < background.size(); ++pj) {
                const int donor_gi = boundary_.P_indices[static_cast<size_t>(pj)];
                add_channel(rates(row_gi, donor_gi) * background(pj),
                            process_index, *process, "P", donor_gi);
            }
            if (charged_row || molecule_row) {
                for (int ai = 0; ai < flowA.size(); ++ai) {
                    const int donor_gi = boundary_.A_indices[static_cast<size_t>(ai)];
                    add_channel(rates(row_gi, donor_gi) * flowA(ai),
                                process_index, *process, "A", donor_gi);
                }
            }
            if (charged_row || atom_row) {
                for (int mi = 0; mi < flowM.size(); ++mi) {
                    const int donor_gi = boundary_.M_indices[static_cast<size_t>(mi)];
                    add_channel(rates(row_gi, donor_gi) * flowM(mi),
                                process_index, *process, "M", donor_gi);
                }
            }
        }
        return breakdown;
    }

    void print_source_channels(const std::string& mode,
                               int row_pi,
                               const SpeciesSourceBreakdown& breakdown,
                               const std::string& prefix) const {
        std::vector<SourceChannel> sorted = breakdown.channels;
        std::sort(sorted.begin(), sorted.end(), [](const auto& a, const auto& b) {
            return std::abs(a.contribution) > std::abs(b.contribution);
        });
        for (size_t rank = 0; rank < sorted.size(); ++rank) {
            std::cout << prefix
                      << " mode=" << mode
                      << " row_pi=" << row_pi
                      << " rank=" << rank + 1
                      << " contribution=" << sorted[rank].contribution
                      << " sign=" << (sorted[rank].contribution >= 0.0 ? "production" : "loss")
                      << " channel=" << std::quoted(sorted[rank].label)
                      << "\n";
        }
    }

    void report_residual_rows(
        const std::string& mode,
        int replacement_pi,
        const Vector& state,
        const Vector& upstream,
        double x_left,
        double x_right,
        double dx,
        const Vector& flowA_left,
        const Vector& flowA_right,
        const Vector& flowM_left,
        const Vector& flowM_right) const {
        const auto full = dcr::solver::make_background_full(state, boundary_, total_states_);
        const auto local = dcr::solver::assemble_local_system(
            config_, atomic_data_, plasma_, grid_, boundary_, full,
            flowA_left, flowM_left, x_left, false, &rate_cache_);
        const auto chemistry = dcr::solver::assemble_local_chemistry_sources(
            local, boundary_, atomic_data_, state, flowA_left, flowM_left);
        Vector divergence = Vector::Zero(state.size());
        for (int pi = 0; pi < state.size(); ++pi) {
            if (is_positive_ion(pi)) {
                const int gi = boundary_.P_indices[static_cast<size_t>(pi)];
                divergence(pi) =
                    (ion_speed(gi, x_left) * state(pi) -
                     ion_speed(gi, x_right) * upstream(pi)) / dx;
            } else {
                divergence(pi) = background_exhaust_rate(pi) * state(pi);
            }
        }
        const Vector species_residual = divergence - chemistry.background;
        const IntervalSystem equation_system = build_reverse_interval_system(
            state, upstream, x_left, x_right, dx,
            flowA_left, flowA_right, flowM_left, flowM_right, replacement_pi);
        const Vector equation_residual = equation_system.lhs * state - equation_system.rhs;
        std::vector<int> order(static_cast<size_t>(state.size()));
        for (int pi = 0; pi < state.size(); ++pi) order[static_cast<size_t>(pi)] = pi;
        std::sort(order.begin(), order.end(), [&](int a, int b) {
            return std::abs(equation_residual(a)) > std::abs(equation_residual(b));
        });

        constexpr int maximum_rows = 8;
        for (int rank = 0; rank < std::min<int>(maximum_rows, order.size()); ++rank) {
            const int pi = order[static_cast<size_t>(rank)];
            const int gi = boundary_.P_indices[static_cast<size_t>(pi)];
            const SpeciesSourceBreakdown breakdown = source_breakdown(
                pi, local, state, flowA_left, flowM_left, x_left);
            const double channel_net = breakdown.production - breakdown.loss;
            std::cout << "RESIDUAL_ROW mode=" << mode
                      << " rank=" << rank + 1
                      << " row_pi=" << pi
                       << " species=" << std::quoted(species_label(gi))
                       << " row_kind="
                       << ((replacement_pi >= 0 && pi == replacement_pi)
                               ? "weighted_conservation" : "species")
                      << " R_s=" << equation_residual(pi)
                      << " species_R=" << species_residual(pi)
                      << " n_s=" << state(pi)
                      << " D_s=" << divergence(pi)
                      << " S_production=" << breakdown.production
                      << " S_loss=" << breakdown.loss
                      << " S_net_channels=" << channel_net
                      << " S_net_assembled=" << chemistry.background(pi)
                      << " channel_count=" << breakdown.channels.size()
                      << " channel_reconstruction_error="
                      << channel_net - chemistry.background(pi)
                      << "\n";
            print_source_channels(mode, pi, breakdown, "SOURCE_CHANNEL");
        }

        int pinned_count = 0;
        int violation_count = 0;
        int pinned_violation_count = 0;
        for (int pi = 0; pi < state.size(); ++pi) {
            const bool pinned = state(pi) <= kDensityFloor;
            if (pinned) ++pinned_count;
            Vector zero_state = state;
            zero_state(pi) = 0.0;
            const auto zero_full = dcr::solver::make_background_full(
                zero_state, boundary_, total_states_);
            const auto zero_local = dcr::solver::assemble_local_system(
                config_, atomic_data_, plasma_, grid_, boundary_, zero_full,
                flowA_left, flowM_left, x_left, false, &rate_cache_);
            const auto zero_chemistry = dcr::solver::assemble_local_chemistry_sources(
                zero_local, boundary_, atomic_data_, zero_state, flowA_left, flowM_left);
            const SpeciesSourceBreakdown zero_breakdown = source_breakdown(
                pi, zero_local, zero_state, flowA_left, flowM_left, x_left);
            const double source_at_zero = zero_chemistry.background(pi);
            const double scale = std::max({1.0, zero_breakdown.production,
                                           zero_breakdown.loss});
            const bool quasi_positive = source_at_zero >= -1.0e-12 * scale;
            if (!quasi_positive) ++violation_count;
            if (pinned && !quasi_positive) ++pinned_violation_count;
            const int gi = boundary_.P_indices[static_cast<size_t>(pi)];
            if (pinned || !quasi_positive) {
                std::cout << (pinned ? "ZERO_POPULATION" : "QUASIPOSITIVITY_VIOLATION")
                          << " mode=" << mode
                          << " row_pi=" << pi
                          << " species=" << std::quoted(species_label(gi))
                          << " n_s=" << state(pi)
                          << " S_at_zero=" << source_at_zero
                          << " S_production_at_zero=" << zero_breakdown.production
                          << " S_loss_at_zero=" << zero_breakdown.loss
                          << " quasi_positive=" << (quasi_positive ? 1 : 0)
                          << "\n";
            }
            if (!quasi_positive) {
                print_source_channels(
                    mode, pi, zero_breakdown, "QUASIPOSITIVITY_CHANNEL");
            }
        }
        std::cout << "QUASIPOSITIVITY_SUMMARY mode=" << mode
                  << " tested_count=" << state.size()
                  << " pinned_count=" << pinned_count
                  << " violation_count=" << violation_count
                  << " pinned_violation_count=" << pinned_violation_count
                  << "\n";
    }

    void initialize_layout() {
        const int Pn = static_cast<int>(boundary_.P_indices.size());
        p_position_.assign(static_cast<size_t>(total_states_), -1);
        muP_ = Vector::Ones(Pn);
        for (int pi = 0; pi < Pn; ++pi) {
            const int gi = boundary_.P_indices[static_cast<size_t>(pi)];
            p_position_[static_cast<size_t>(gi)] = pi;
            muP_(pi) = std::max(1, levels_[static_cast<size_t>(gi)].atomicity);
            const auto& level = levels_[static_cast<size_t>(gi)];
            if (level.type == dcr::atomic::SpeciesType::Ion && level.charge < 0 &&
                level.atomicity == 1) {
                if (hminus_pi_ >= 0) throw std::runtime_error("Expected one H- row");
                hminus_pi_ = pi;
            }
        }
        for (int gi : boundary_.ion_indices) {
            if (gi < 0 || gi >= total_states_) continue;
            const int pi = p_position_[static_cast<size_t>(gi)];
            if (pi >= 0) ion_p_indices_.push_back(pi);
        }
        if (hminus_pi_ < 0 || ion_p_indices_.empty()) {
            throw std::runtime_error("Alternating sweep requires positive ions and H-");
        }
        atom_ground_ai_ = compact_position(boundary_.A_indices, boundary_.atom_ground);
        molecule_ground_mi_ = compact_position(boundary_.M_indices, boundary_.molecule_ground);
        if (atom_ground_ai_ < 0 || molecule_ground_mi_ < 0) {
            throw std::runtime_error("Recycling ground states are unavailable");
        }
        atom_exhaust_rate_ = dcr::physics::calculate_thermal_speed(
            config_.plasma.neutral_atom_temperature_eV, boundary_.atom_mass_amu) /
            std::max(config_.grid.spatial_exhaust_width_cm, 1.0e-12);
        molecule_exhaust_rate_ = dcr::physics::calculate_thermal_speed(
            config_.plasma.neutral_molecule_temperature_eV, boundary_.molecule_mass_amu) /
            std::max(config_.grid.spatial_exhaust_width_cm, 1.0e-12);
    }

    void initialize_grid() {
        dx_ = dcr::solver::build_step_sizes_cm(config_);
        x_.assign(static_cast<size_t>(config_.grid.num_cells), 0.0);
        for (size_t k = 0; k < dx_.size(); ++k) x_[k + 1] = x_[k] + dx_[k];
    }

    void validate_case() const {
        auto close = [](double a, double b) {
            return std::abs(a - b) <= 1.0e-12 * std::max({1.0, std::abs(a), std::abs(b)});
        };
        const bool supported_temperature =
            close(config_.plasma.Te_eV, config_.plasma.Ti_eV) &&
            config_.plasma.Te_eV >= 1.0 && config_.plasma.Te_eV <= 5.0;
        if (!supported_temperature ||
            !close(config_.grid.length_cm, 50.0) || config_.grid.num_cells != 200 ||
            !close(config_.plasma.total_density, 1.0e16)) {
            throw std::runtime_error(
                "Prototype is restricted to equal Te=Ti in [1,5] eV, L=50, N=200 cases");
        }
        for (int gi : boundary_.ion_indices) {
            const double bohm = ion_bohm_speed(gi, x_.back());
            const double fraction = ion_speed(gi, x_.back()) / bohm;
            if (std::abs(fraction - 0.05) > 1.0e-8 &&
                std::abs(fraction - 0.1) > 1.0e-8 &&
                std::abs(fraction - 0.5) > 1.0e-8) {
                throw std::runtime_error(
                    "Configured upstream ion speed is not 0.05, 0.1, or 0.5 Bohm");
            }
        }
    }

    Profiles initial_profiles() const {
        Profiles profiles;
        const Vector p0 = compact_background_from_boundary(boundary_);
        const Vector a0 = compact_flow_from_boundary(
            boundary_, boundary_.A_indices, boundary_.flowA_last);
        const Vector m0 = compact_flow_from_boundary(
            boundary_, boundary_.M_indices, boundary_.flowM_last);
        profiles.background.assign(dx_.size(), p0);
        profiles.flowA.assign(x_.size(), a0);
        profiles.flowM.assign(x_.size(), m0);
        return profiles;
    }

    Vector upstream_ion_composition(const Vector& seed) const {
        Vector composition = Vector::Zero(seed.size());
        double sum = 0.0;
        for (const auto& initial : config_.plasma.initial_conditions) {
            if (initial.species_index < 0 || initial.species_index >= total_states_) continue;
            const int pi = p_position_[static_cast<size_t>(initial.species_index)];
            if (pi < 0 || !is_positive_ion(pi) || initial.fraction <= 0.0) continue;
            composition(pi) += initial.fraction;
            sum += initial.fraction;
        }
        if (!(sum > 0.0)) {
            for (int pi : ion_p_indices_) {
                const double weight = muP_(pi) * std::max(seed(pi), 0.0);
                composition(pi) = weight;
                sum += weight;
            }
        }
        if (!(sum > 0.0)) throw std::runtime_error("No upstream positive-ion composition");
        composition /= sum;
        return composition;
    }

    double initial_shooting_amplitude(const Vector& seed) const {
        double amplitude = ion_nuclei_flux(seed, 0.0);
        if (!(amplitude > 0.0)) {
            amplitude = config_.plasma.total_density * ion_speed(
                boundary_.ion_indices.front(), x_.back());
        }
        return amplitude;
    }

    UpstreamIonBoundary make_upstream_ion_boundary(
        double amplitude,
        const Vector& composition) const {
        if (!(amplitude > 0.0) || !std::isfinite(amplitude) ||
            composition.size() != static_cast<int>(boundary_.P_indices.size())) {
            throw std::runtime_error("Invalid upstream ion boundary request");
        }
        UpstreamIonBoundary result;
        result.position_cm = x_.back();
        result.positive_ion_nuclei_flux_cm2_s = Vector::Zero(composition.size());
        result.positive_ion_velocity_cm_s = Vector::Zero(composition.size());
        result.positive_ion_composition = Vector::Zero(composition.size());
        double composition_sum = 0.0;
        for (int pi : ion_p_indices_) {
            if (composition(pi) < 0.0 || !std::isfinite(composition(pi))) {
                throw std::runtime_error("Invalid upstream positive-ion composition");
            }
            composition_sum += composition(pi);
        }
        if (!(composition_sum > 0.0)) {
            throw std::runtime_error("Upstream positive-ion composition is empty");
        }
        for (int pi : ion_p_indices_) {
            const int gi = boundary_.P_indices[static_cast<size_t>(pi)];
            const double normalized = composition(pi) / composition_sum;
            result.positive_ion_composition(pi) = normalized;
            result.positive_ion_nuclei_flux_cm2_s(pi) = amplitude * normalized;
            result.positive_ion_velocity_cm_s(pi) = ion_speed(gi, x_.back());
        }
        return result;
    }

    bool valid_profile_layout(const Profiles& profiles) const {
        if (profiles.background.size() != dx_.size() ||
            profiles.flowA.size() != x_.size() ||
            profiles.flowM.size() != x_.size()) {
            return false;
        }
        auto valid_block = [](const std::vector<Vector>& block, size_t expected_size) {
            for (const Vector& vector : block) {
                if (vector.size() != static_cast<int>(expected_size) ||
                    !vector.allFinite() || vector.minCoeff() < 0.0) {
                    return false;
                }
            }
            return true;
        };
        return valid_block(profiles.background, boundary_.P_indices.size()) &&
            valid_block(profiles.flowA, boundary_.A_indices.size()) &&
            valid_block(profiles.flowM, boundary_.M_indices.size());
    }

    bool valid_upstream_ion_boundary(const UpstreamIonBoundary& upstream) const {
        const int expected = static_cast<int>(boundary_.P_indices.size());
        if (!std::isfinite(upstream.position_cm) ||
            std::abs(upstream.position_cm - x_.back()) > 1.0e-12 ||
            upstream.positive_ion_nuclei_flux_cm2_s.size() != expected ||
            upstream.positive_ion_velocity_cm_s.size() != expected ||
            upstream.positive_ion_composition.size() != expected ||
            !upstream.positive_ion_nuclei_flux_cm2_s.allFinite() ||
            !upstream.positive_ion_velocity_cm_s.allFinite() ||
            !upstream.positive_ion_composition.allFinite() ||
            upstream.positive_ion_nuclei_flux_cm2_s.minCoeff() < 0.0 ||
            upstream.positive_ion_velocity_cm_s.minCoeff() < 0.0 ||
            upstream.positive_ion_composition.minCoeff() < 0.0) {
            return false;
        }
        double composition_sum = 0.0;
        double flux_sum = 0.0;
        for (int pi = 0; pi < expected; ++pi) {
            if (is_positive_ion(pi)) {
                if (!(upstream.positive_ion_velocity_cm_s(pi) > 0.0)) return false;
                composition_sum += upstream.positive_ion_composition(pi);
                flux_sum += upstream.positive_ion_nuclei_flux_cm2_s(pi);
            } else if (upstream.positive_ion_nuclei_flux_cm2_s(pi) != 0.0 ||
                       upstream.positive_ion_velocity_cm_s(pi) != 0.0 ||
                       upstream.positive_ion_composition(pi) != 0.0) {
                return false;
            }
        }
        return flux_sum > 0.0 && std::abs(composition_sum - 1.0) <= 1.0e-12;
    }

    int select_conservation_replacement_row(
        const Vector& lagged_state,
        double& selected_population_fraction) const {
        int selected_pi = -1;
        double selected_score = -1.0;
        double total_nuclei = 0.0;
        for (int pi = 0; pi < lagged_state.size(); ++pi) {
            const int gi = boundary_.P_indices[static_cast<size_t>(pi)];
            const auto& level = levels_[static_cast<size_t>(gi)];
            const double mu = muP_(pi);
            if (!(mu > 0.0) || !level.is_background || level.is_recycling) continue;
            const double score = mu * std::max(lagged_state(pi), 0.0);
            total_nuclei += score;
            if (pi == hminus_pi_) continue;
            if (score > selected_score) {
                selected_score = score;
                selected_pi = pi;
            }
        }
        if (selected_pi < 0) {
            throw std::runtime_error(
                "No eligible physical P-block row for nuclei conservation");
        }
        selected_population_fraction = selected_score / std::max(total_nuclei, 1.0);
        return selected_pi;
    }

    double replacement_population_fraction(
        const Vector& lagged_state,
        int selected_pi) const {
        double selected_score = 0.0;
        double total_nuclei = 0.0;
        for (int pi = 0; pi < lagged_state.size(); ++pi) {
            const int gi = boundary_.P_indices[static_cast<size_t>(pi)];
            const auto& level = levels_[static_cast<size_t>(gi)];
            const double mu = muP_(pi);
            if (!(mu > 0.0) || !level.is_background || level.is_recycling) continue;
            const double score = mu * std::max(lagged_state(pi), 0.0);
            total_nuclei += score;
            if (pi == selected_pi) selected_score = score;
        }
        return selected_score / std::max(total_nuclei, 1.0);
    }

    std::vector<int> dominant_replacement_pattern(const Profiles& profiles) const {
        std::vector<int> pattern(x_.size() - 1, -1);
        for (size_t cell = 0; cell + 1 < x_.size(); ++cell) {
            double fraction = 0.0;
            pattern[cell] = select_conservation_replacement_row(
                profiles.background[cell], fraction);
        }
        return pattern;
    }

    PhysicalCellSystem assemble_physical_cell_system(
        int k,
        const Vector& state,
        const Vector* upstream,
        const UpstreamIonBoundary* upstream_face,
        const std::vector<Vector>& flowA,
        const std::vector<Vector>& flowM) const {
        const double dx = dx_[static_cast<size_t>(k)];
        const auto full = dcr::solver::make_background_full(state, boundary_, total_states_);
        const auto local = dcr::solver::assemble_local_system(
            config_, atomic_data_, plasma_, grid_, boundary_, full,
            flowA[static_cast<size_t>(k)], flowM[static_cast<size_t>(k)],
            x_[static_cast<size_t>(k)], false, &rate_cache_);
        const Matrix& rates = local.R_source_full;
        const Matrix Rpp = extract_block(
            rates, boundary_.P_indices, boundary_.P_indices);
        const auto chemistry = dcr::solver::assemble_local_chemistry_sources(
            local, boundary_, atomic_data_, state,
            flowA[static_cast<size_t>(k)], flowM[static_cast<size_t>(k)]);
        const Vector source = chemistry.background - Rpp * state;

        PhysicalCellSystem system;
        system.lhs = -Rpp;
        system.rhs = source;
        system.row_scales = Vector::Zero(state.size());
        for (int pi = 0; pi < state.size(); ++pi) {
            const int gi = boundary_.P_indices[static_cast<size_t>(pi)];
            for (int gj = 0; gj < rates.cols() &&
                 gj < local.population_for_source.size(); ++gj) {
                system.row_scales(pi) = std::max(
                    system.row_scales(pi),
                    std::abs(rates(gi, gj) * local.population_for_source(gj)));
            }
            system.row_scales(pi) = std::max(
                system.row_scales(pi), std::abs(source(pi)));
            if (is_positive_ion(pi)) {
                const double outgoing =
                    ion_speed(gi, x_[static_cast<size_t>(k)]) * state(pi) / dx;
                double incoming = 0.0;
                if (upstream != nullptr) {
                    incoming = ion_speed(gi, x_[static_cast<size_t>(k + 1)]) *
                        (*upstream)(pi) / dx;
                } else {
                    if (upstream_face == nullptr ||
                        k != static_cast<int>(dx_.size()) - 1) {
                        throw std::runtime_error(
                            "Missing upstream ion-face flux for final reverse cell");
                    }
                    incoming = dcr::solver::upstream_face_positive_ion_incoming_term(
                        upstream_face->positive_ion_nuclei_flux_cm2_s(pi),
                        muP_(pi), dx);
                }
                system.lhs(pi, pi) +=
                    ion_speed(gi, x_[static_cast<size_t>(k)]) / dx;
                system.rhs(pi) += incoming;
                system.row_scales(pi) = std::max(
                    system.row_scales(pi), std::abs(outgoing));
                system.row_scales(pi) = std::max(
                    system.row_scales(pi), std::abs(incoming));
            } else {
                const double exhaust = background_exhaust_rate(pi) * state(pi);
                system.lhs(pi, pi) += background_exhaust_rate(pi);
                system.row_scales(pi) = std::max(
                    system.row_scales(pi), std::abs(exhaust));
            }
        }
        return system;
    }

    Vector solve_reverse_cell(int k,
                              const Vector* upstream,
                              const UpstreamIonBoundary* upstream_face,
                              const Vector& initial,
                              const std::vector<Vector>& flowA,
                               const std::vector<Vector>& flowM,
                               double shooting_amplitude,
                               CellDiagnostics& diagnostics,
                               const Vector* pseudo_time_old = nullptr,
                               double inverse_pseudo_time_step = 0.0
#ifdef DCR_NESTED_DIAGNOSTIC
                               , const Vector* local_inverse_pseudo_time_steps = nullptr
#endif
                               ) const {
        Vector state = initial;
        if (state.minCoeff() < 0.0) {
            Eigen::Index failing_pi = -1;
            state.minCoeff(&failing_pi);
            const int failing_gi = boundary_.P_indices[
                static_cast<size_t>(failing_pi)];
            throw UnconvergedReverseCellError(
                "Reverse cell " + std::to_string(k) +
                " lagged state contains a negative population species_gi=" +
                std::to_string(failing_gi));
        }
        if (frozen_replacement_rows_.size() == x_.size() - 1) {
            diagnostics.replacement_pi = frozen_replacement_rows_[static_cast<size_t>(k)];
            diagnostics.replacement_population_fraction = replacement_population_fraction(
                initial, diagnostics.replacement_pi);
        } else {
            diagnostics.replacement_pi = select_conservation_replacement_row(
                initial, diagnostics.replacement_population_fraction);
        }
        diagnostics.replacement_gi = boundary_.P_indices[
            static_cast<size_t>(diagnostics.replacement_pi)];
        std::vector<LocalIterationDiagnostics> iteration_history;
        iteration_history.reserve(kMaxLocalIterations);
        for (int iteration = 0; iteration < kMaxLocalIterations; ++iteration) {
            const PhysicalCellSystem original = assemble_physical_cell_system(
                k, state, upstream, upstream_face, flowA, flowM);
            PhysicalCellSystem implicit = original;
            if (pseudo_time_old != nullptr &&
                (inverse_pseudo_time_step > 0.0
#ifdef DCR_NESTED_DIAGNOSTIC
                 || local_inverse_pseudo_time_steps != nullptr
#endif
                )) {
                for (int pi = 0; pi < state.size(); ++pi) {
                    const double inverse_step =
#ifdef DCR_NESTED_DIAGNOSTIC
                        local_inverse_pseudo_time_steps != nullptr
                        ? (*local_inverse_pseudo_time_steps)(pi) :
#endif
                        inverse_pseudo_time_step;
                    implicit.lhs(pi, pi) += inverse_step;
                    implicit.rhs(pi) += inverse_step *
                        (*pseudo_time_old)(pi);
                }
            }
            Matrix lhs = implicit.lhs;
            Vector rhs = implicit.rhs;
            const Vector exact_row = implicit.lhs.transpose() * muP_;
            const double exact_rhs = muP_.dot(implicit.rhs);
            lhs.row(diagnostics.replacement_pi) = exact_row.transpose();
            rhs(diagnostics.replacement_pi) = exact_rhs;

            LocalIterationDiagnostics iteration_diagnostics;
            iteration_diagnostics.iteration = iteration;
            const Vector physical_residual = lhs * state - rhs;
            iteration_diagnostics.physical_residual_norm = physical_residual.norm();
            iteration_diagnostics.physical_residual_relative =
                componentwise_backward_error(lhs, state, rhs);
            iteration_diagnostics.physical_residual_global_relative =
                physical_residual.norm() /
                std::max({1.0, (lhs * state).norm(), rhs.norm()});
            iteration_diagnostics.minimum_population = state.minCoeff();

            Vector column_scales = state.cwiseAbs().cwiseMax(kDensityFloor);
            Vector equation_scales = Vector::Zero(lhs.rows());
            Matrix equilibrated_lhs = lhs;
            Vector equilibrated_rhs = rhs;
            for (int row = 0; row < lhs.rows(); ++row) {
                equation_scales(row) = std::abs(rhs(row));
                for (int column = 0; column < lhs.cols(); ++column) {
                    equation_scales(row) += std::abs(lhs(row, column)) *
                        column_scales(column);
                }
                equation_scales(row) = std::max(
                    kPhysicalRowScaleFloor, equation_scales(row));
                equilibrated_rhs(row) /= equation_scales(row);
                for (int column = 0; column < lhs.cols(); ++column) {
                    equilibrated_lhs(row, column) *=
                        column_scales(column) / equation_scales(row);
                }
            }

            Eigen::FullPivLU<Matrix> lu(equilibrated_lhs);
            if (lu.rank() < lhs.rows()) {
                if (!conditioning_diagnostic_emitted_) {
                    emit_conditioning_diagnostic(
                        original.lhs, original.rhs, lhs, rhs,
                        k, iteration, shooting_amplitude);
                    conditioning_diagnostic_emitted_ = true;
                }
                throw UnconvergedReverseCellError(
                    "Reverse cell " + std::to_string(k) +
                    " matrix is singular species_gi=" +
                    std::to_string(diagnostics.replacement_gi));
            }
            Vector solved = column_scales.cwiseProduct(lu.solve(equilibrated_rhs));
            if (!solved.allFinite()) {
                throw UnconvergedReverseCellError(
                    "Reverse cell " + std::to_string(k) +
                    " solve is non-finite species_gi=" +
                    std::to_string(diagnostics.replacement_gi));
            }
            iteration_diagnostics.linear_solve_backward_error_before =
                componentwise_backward_error(lhs, solved, rhs);
            iteration_diagnostics.linear_solve_backward_error_after =
                iteration_diagnostics.linear_solve_backward_error_before;
            for (int refinement = 0; refinement < 4; ++refinement) {
                Vector residual(rhs.size());
                for (int row = 0; row < lhs.rows(); ++row) {
                    long double value = static_cast<long double>(rhs(row));
                    for (int column = 0; column < lhs.cols(); ++column) {
                        value -= static_cast<long double>(lhs(row, column)) *
                            static_cast<long double>(solved(column));
                    }
                    residual(row) = static_cast<double>(value);
                }
                const Vector correction = column_scales.cwiseProduct(
                    lu.solve(residual.cwiseQuotient(equation_scales)));
                if (!correction.allFinite()) break;
                const Vector candidate = solved + correction;
                const double candidate_error =
                    componentwise_backward_error(lhs, candidate, rhs);
                if (!std::isfinite(candidate_error) ||
                    candidate_error >=
                        iteration_diagnostics.linear_solve_backward_error_after) {
                    break;
                }
                solved = candidate;
                iteration_diagnostics.linear_solve_backward_error_after =
                    candidate_error;
                ++iteration_diagnostics.iterative_refinement_steps;
                if (candidate_error < kLocalResidualTolerance) break;
            }
            Eigen::Index raw_minimum_pi = -1;
            iteration_diagnostics.raw_solution_minimum = solved.minCoeff(&raw_minimum_pi);
            if (iteration_diagnostics.raw_solution_minimum <
                diagnostics.raw_minimum_population) {
                diagnostics.raw_minimum_population =
                    iteration_diagnostics.raw_solution_minimum;
                diagnostics.raw_minimum_pi = static_cast<int>(raw_minimum_pi);
            }
            for (int i = 0; i < solved.size(); ++i) {
                if (solved(i) <= 0.0) ++iteration_diagnostics.raw_nonpositive_count;
            }
            if (iteration_diagnostics.raw_nonpositive_count > 0) {
                CellDiagnostics raw_diagnostics = diagnostics;
                evaluate_cell_residual(
                    k, solved, upstream, upstream_face, flowA, flowM,
                    diagnostics.replacement_pi, raw_diagnostics);
                const int raw_gi = boundary_.P_indices[
                    static_cast<size_t>(raw_minimum_pi)];
                std::cout << "NONPOSITIVE_LOCAL_SOLVE"
                          << " cell=" << k
                          << " x=" << x_[static_cast<size_t>(k)]
                          << " selected_row_pi=" << diagnostics.replacement_pi
                          << " selected_species="
                          << std::quoted(species_label(diagnostics.replacement_gi))
                          << " raw_negative_pi=" << raw_minimum_pi
                          << " raw_negative_species="
                          << std::quoted(species_label(raw_gi))
                          << " raw_minimum="
                          << iteration_diagnostics.raw_solution_minimum
                          << " epsilon_Sigma="
                          << raw_diagnostics.conservation_epsilon
                          << " epsilon_Hminus="
                          << raw_diagnostics.hminus_epsilon
                          << " Hminus_active="
                          << (raw_diagnostics.hminus_active ? 1 : 0)
                          << " epsilon_omitted="
                          << raw_diagnostics.omitted_epsilon
                          << " omitted_active="
                          << (raw_diagnostics.omitted_active ? 1 : 0)
                          << "\n";
                throw UnconvergedReverseCellError(
                    "Reverse cell " + std::to_string(k) +
                    " unconstrained solve requested a nonpositive population species_gi=" +
                    std::to_string(raw_gi));
            }
            const Vector next = (1.0 - 0.7) * state + 0.7 * solved;
            const double change = vector_relative_change(next, state);
            iteration_diagnostics.population_change = change;
            const double global_change_scale = std::max(
                kPopulationChangeScaleFloor, state.cwiseAbs().maxCoeff());
            for (int i = 0; i < next.size(); ++i) {
                const double component_change =
                    std::abs(next(i) - state(i)) / global_change_scale;
                if (component_change >= change * (1.0 - 1.0e-12)) {
                    iteration_diagnostics.maximum_change_index = i;
                    iteration_diagnostics.maximum_change_previous = state(i);
                    iteration_diagnostics.maximum_change_next = next(i);
                    break;
                }
            }
            iteration_diagnostics.linearized_post_residual_relative =
                componentwise_backward_error(lhs, next, rhs);
            for (int i = 0; i < next.size(); ++i) {
                if (next(i) <= kDensityFloor * (1.0 + 1.0e-12)) {
                    ++iteration_diagnostics.floor_count;
                }
            }
            iteration_history.push_back(iteration_diagnostics);
            state = next;
            if (change < kLocalTolerance) {
                const PhysicalCellSystem next_original = assemble_physical_cell_system(
                    k, state, upstream, upstream_face, flowA, flowM);
                PhysicalCellSystem next_implicit = next_original;
                if (pseudo_time_old != nullptr &&
                    (inverse_pseudo_time_step > 0.0
#ifdef DCR_NESTED_DIAGNOSTIC
                     || local_inverse_pseudo_time_steps != nullptr
#endif
                    )) {
                    for (int pi = 0; pi < state.size(); ++pi) {
                        const double inverse_step =
#ifdef DCR_NESTED_DIAGNOSTIC
                            local_inverse_pseudo_time_steps != nullptr
                            ? (*local_inverse_pseudo_time_steps)(pi) :
#endif
                            inverse_pseudo_time_step;
                        next_implicit.lhs(pi, pi) += inverse_step;
                        next_implicit.rhs(pi) += inverse_step *
                            (*pseudo_time_old)(pi);
                    }
                }
                Matrix next_lhs = next_implicit.lhs;
                Vector next_rhs = next_implicit.rhs;
                const Vector next_exact_row =
                    next_implicit.lhs.transpose() * muP_;
                next_lhs.row(diagnostics.replacement_pi) =
                    next_exact_row.transpose();
                next_rhs(diagnostics.replacement_pi) =
                    muP_.dot(next_implicit.rhs);
                iteration_history.back().nonlinear_post_residual_relative =
                    componentwise_backward_error(next_lhs, state, next_rhs);
                if (iteration_history.back().nonlinear_post_residual_relative <
                    kLocalResidualTolerance) {
                    diagnostics.converged = true;
                    break;
                }
            }
        }
        evaluate_cell_residual(
            k, state, upstream, upstream_face, flowA, flowM,
            diagnostics.replacement_pi, diagnostics);
        if (pseudo_time_old != nullptr &&
            (inverse_pseudo_time_step > 0.0
#ifdef DCR_NESTED_DIAGNOSTIC
             || local_inverse_pseudo_time_steps != nullptr
#endif
            )) {
            const PhysicalCellSystem steady_system = assemble_physical_cell_system(
                k, state, upstream, upstream_face, flowA, flowM);
            PhysicalCellSystem pseudo_time_system = steady_system;
            diagnostics.transient_to_steady_row_scale_ratios.assign(
                static_cast<size_t>(state.size()), 0.0);
            Vector inverse_steps = Vector::Constant(
                state.size(), inverse_pseudo_time_step);
#ifdef DCR_NESTED_DIAGNOSTIC
            if (local_inverse_pseudo_time_steps != nullptr) {
                inverse_steps = *local_inverse_pseudo_time_steps;
            }
#endif
            for (int pi = 0; pi < state.size(); ++pi) {
                pseudo_time_system.lhs(pi, pi) += inverse_steps(pi);
                pseudo_time_system.rhs(pi) += inverse_steps(pi) *
                    (*pseudo_time_old)(pi);
            }
            const Vector pseudo_time_physical_residual =
                pseudo_time_system.lhs * state - pseudo_time_system.rhs;
            const Vector steady_residual =
                steady_system.lhs * state - steady_system.rhs;
            const Vector transient_term = inverse_steps.cwiseProduct(
                state - *pseudo_time_old);
            Vector be_identity_defect =
                pseudo_time_physical_residual -
                (transient_term + steady_residual);
#ifdef DCR_NESTED_DIAGNOSTIC
            if (local_inverse_pseudo_time_steps != nullptr) {
                for (int row = 0; row < state.size(); ++row) {
                    long double pseudo_residual =
                        -static_cast<long double>(pseudo_time_system.rhs(row));
                    long double physical_steady_residual =
                        -static_cast<long double>(steady_system.rhs(row));
                    for (int column = 0; column < state.size(); ++column) {
                        pseudo_residual += static_cast<long double>(
                            pseudo_time_system.lhs(row, column)) *
                            static_cast<long double>(state(column));
                        physical_steady_residual += static_cast<long double>(
                            steady_system.lhs(row, column)) *
                            static_cast<long double>(state(column));
                    }
                    const long double represented_transient =
                        static_cast<long double>(
                            pseudo_time_system.lhs(row, row) -
                            steady_system.lhs(row, row)) *
                            static_cast<long double>(state(row)) -
                        static_cast<long double>(
                            pseudo_time_system.rhs(row) -
                            steady_system.rhs(row));
                    be_identity_defect(row) = static_cast<double>(
                        pseudo_residual - physical_steady_residual -
                        represented_transient);
                }
            }
#endif
            diagnostics.pseudo_time_state_change_norm =
                (state - *pseudo_time_old).cwiseAbs().maxCoeff();
            diagnostics.pseudo_time_transient_term_norm =
                transient_term.cwiseAbs().maxCoeff();
            diagnostics.steady_residual_norm =
                steady_residual.cwiseAbs().maxCoeff();
            diagnostics.pseudo_time_residual_norm =
                pseudo_time_physical_residual.cwiseAbs().maxCoeff();
            diagnostics.be_identity_defect_norm =
                be_identity_defect.cwiseAbs().maxCoeff();
            for (int pi = 0; pi < state.size(); ++pi) {
                const double steady_row_scale = std::max(
                    kPhysicalRowScaleFloor, pseudo_time_system.row_scales(pi));
                if (pseudo_time_system.row_scales(pi) >
                    kPhysicalRowScaleFloor) {
                    const double transient_scale = inverse_steps(pi) *
                        std::max(
                            std::abs(state(pi)),
                            std::abs((*pseudo_time_old)(pi)));
                    const double transient_ratio =
                        transient_scale / steady_row_scale;
                    const double transient_update_ratio =
                        inverse_steps(pi) *
                        std::abs(state(pi) - (*pseudo_time_old)(pi)) /
                        steady_row_scale;
                    diagnostics.transient_to_steady_row_scale_ratios[
                        static_cast<size_t>(pi)] = transient_ratio;
                    diagnostics.transient_update_to_steady_row_scale_ratio =
                        std::max(
                            diagnostics
                                .transient_update_to_steady_row_scale_ratio,
                            transient_update_ratio);
                    if (transient_ratio >
                        diagnostics.transient_to_steady_row_scale_ratio) {
                        diagnostics.transient_to_steady_row_scale_ratio =
                            transient_ratio;
                        diagnostics.transient_to_steady_row_scale_pi = pi;
                    }
                }
                diagnostics.pseudo_time_physical_residual_steady_scaled =
                    std::max(
                        diagnostics.pseudo_time_physical_residual_steady_scaled,
                        std::abs(pseudo_time_physical_residual(pi)) /
                            steady_row_scale);
                const double identity_scale = std::max({
                    kPhysicalRowScaleFloor,
                    steady_row_scale,
                    std::abs(steady_residual(pi)),
                    std::abs(transient_term(pi)),
                    std::abs(pseudo_time_physical_residual(pi))});
                const double relative_identity_defect =
                    std::abs(be_identity_defect(pi)) / identity_scale;
                if (relative_identity_defect >
                    diagnostics.be_identity_defect_relative) {
                    diagnostics.be_identity_defect_relative =
                        relative_identity_defect;
                    diagnostics.be_identity_defect_pi = pi;
                }
            }
            const double projected_steady = muP_.dot(steady_residual);
            const double projected_transient = muP_.dot(transient_term);
            const double projected_be =
                muP_.dot(pseudo_time_physical_residual);
            diagnostics.conservation_be_identity_defect =
                projected_be - (projected_transient + projected_steady);
#ifdef DCR_NESTED_DIAGNOSTIC
            if (local_inverse_pseudo_time_steps != nullptr) {
                long double projected_identity_defect = 0.0L;
                for (int pi = 0; pi < be_identity_defect.size(); ++pi) {
                    projected_identity_defect +=
                        static_cast<long double>(muP_(pi)) *
                        static_cast<long double>(be_identity_defect(pi));
                }
                diagnostics.conservation_be_identity_defect =
                    static_cast<double>(projected_identity_defect);
            }
#endif
            diagnostics.conservation_be_identity_defect_relative =
                std::abs(diagnostics.conservation_be_identity_defect) /
                std::max({
                    kPhysicalRowScaleFloor,
                    std::abs(projected_be),
                    std::abs(projected_transient),
                    std::abs(projected_steady)});
#ifdef DCR_NESTED_DIAGNOSTIC
            if (local_inverse_pseudo_time_steps != nullptr) {
                double structural_relative_defect = 0.0;
                for (int pi = 0; pi < state.size(); ++pi) {
                    const double intended_inverse_step =
                        (*local_inverse_pseudo_time_steps)(pi);
                    const double represented_inverse_step =
                        pseudo_time_system.lhs(pi, pi) -
                        steady_system.lhs(pi, pi);
                    structural_relative_defect = std::max(
                        structural_relative_defect,
                        std::abs(represented_inverse_step -
                                 intended_inverse_step) /
                            std::max({1.0,
                                      std::abs(represented_inverse_step),
                                      std::abs(intended_inverse_step)}));
                    const double intended_rhs_increment =
                        intended_inverse_step * (*pseudo_time_old)(pi);
                    const double represented_rhs_increment =
                        pseudo_time_system.rhs(pi) - steady_system.rhs(pi);
                    structural_relative_defect = std::max(
                        structural_relative_defect,
                        std::abs(represented_rhs_increment -
                                 intended_rhs_increment) /
                            std::max({1.0,
                                      std::abs(represented_rhs_increment),
                                      std::abs(intended_rhs_increment)}));
                }
                diagnostics.conservation_be_identity_defect_relative =
                    structural_relative_defect;
            }
#endif
            const Vector pseudo_time_exact_row =
                pseudo_time_system.lhs.transpose() * muP_;
            pseudo_time_system.lhs.row(diagnostics.replacement_pi) =
                pseudo_time_exact_row.transpose();
            pseudo_time_system.rhs(diagnostics.replacement_pi) =
                muP_.dot(pseudo_time_system.rhs);
            diagnostics.pseudo_time_backward_error = componentwise_backward_error(
                pseudo_time_system.lhs, state, pseudo_time_system.rhs);
        } else {
            diagnostics.pseudo_time_backward_error =
                diagnostics.steady_backward_error;
        }
        if (!diagnostics.converged) {
            emit_backward_error_row_diagnostic(
                k, state, upstream, upstream_face, flowA, flowM,
                diagnostics.replacement_pi);
            emit_local_convergence_diagnostic(
                k, shooting_amplitude,
                upstream != nullptr ? *upstream : initial, initial,
                flowA[static_cast<size_t>(k)], flowM[static_cast<size_t>(k)],
                iteration_history, diagnostics);
            std::ostringstream message;
            message << "Reverse cell " << k << " failed local convergence at x="
                    << x_[static_cast<size_t>(k)]
                    << " species_gi=" << diagnostics.replacement_gi;
            throw UnconvergedReverseCellError(message.str());
        }
        return state;
    }

    void accumulate_cell_diagnostics(
        SweepDiagnostics& sweep,
        int cell_index,
        const CellDiagnostics& cell) const {
        if (sweep.replacement_species_counts.empty()) {
            sweep.replacement_species_counts.assign(
                static_cast<size_t>(total_states_), 0);
        }
        if (cell.replacement_gi >= 0 && cell.replacement_gi < total_states_) {
            ++sweep.replacement_species_counts[
                static_cast<size_t>(cell.replacement_gi)];
        }
        sweep.cells_converged = sweep.cells_converged && cell.converged;
        sweep.minimum_population = std::min(
            sweep.minimum_population, cell.final_minimum_population);
        if (cell.steady_backward_error > sweep.max_steady_backward_error) {
            sweep.max_steady_backward_error = cell.steady_backward_error;
            sweep.max_steady_backward_error_cell = cell_index;
        }
        sweep.max_pseudo_time_backward_error = std::max(
            sweep.max_pseudo_time_backward_error,
            cell.pseudo_time_backward_error);
        if (cell.transient_to_steady_row_scale_ratio >
            sweep.max_transient_to_steady_row_scale_ratio) {
            sweep.max_transient_to_steady_row_scale_ratio =
                cell.transient_to_steady_row_scale_ratio;
            sweep.max_transient_ratio_cell = cell_index;
            sweep.max_transient_ratio_pi =
                cell.transient_to_steady_row_scale_pi;
            sweep.max_transient_ratio_gi =
                cell.transient_to_steady_row_scale_pi >= 0
                ? boundary_.P_indices[static_cast<size_t>(
                    cell.transient_to_steady_row_scale_pi)]
                : -1;
        }
        sweep.max_pseudo_time_physical_residual_steady_scaled = std::max(
            sweep.max_pseudo_time_physical_residual_steady_scaled,
            cell.pseudo_time_physical_residual_steady_scaled);
        sweep.max_pseudo_time_state_change_norm = std::max(
            sweep.max_pseudo_time_state_change_norm,
            cell.pseudo_time_state_change_norm);
        sweep.max_pseudo_time_transient_term_norm = std::max(
            sweep.max_pseudo_time_transient_term_norm,
            cell.pseudo_time_transient_term_norm);
        sweep.max_steady_residual_norm = std::max(
            sweep.max_steady_residual_norm, cell.steady_residual_norm);
        sweep.max_pseudo_time_residual_norm = std::max(
            sweep.max_pseudo_time_residual_norm,
            cell.pseudo_time_residual_norm);
        sweep.max_be_identity_defect_norm = std::max(
            sweep.max_be_identity_defect_norm,
            cell.be_identity_defect_norm);
        if (cell.be_identity_defect_relative >
            sweep.max_be_identity_defect_relative) {
            sweep.max_be_identity_defect_relative =
                cell.be_identity_defect_relative;
            sweep.max_be_identity_defect_cell = cell_index;
            sweep.max_be_identity_defect_pi = cell.be_identity_defect_pi;
            sweep.max_be_identity_defect_gi =
                cell.be_identity_defect_pi >= 0
                ? boundary_.P_indices[static_cast<size_t>(
                    cell.be_identity_defect_pi)] : -1;
        }
        sweep.max_conservation_be_identity_defect = std::max(
            sweep.max_conservation_be_identity_defect,
            std::abs(cell.conservation_be_identity_defect));
        sweep.max_conservation_be_identity_defect_relative = std::max(
            sweep.max_conservation_be_identity_defect_relative,
            cell.conservation_be_identity_defect_relative);
        sweep.max_transient_update_to_steady_row_scale_ratio = std::max(
            sweep.max_transient_update_to_steady_row_scale_ratio,
            cell.transient_update_to_steady_row_scale_ratio);
        if (std::abs(cell.exact_residual) > std::abs(sweep.max_exact_residual)) {
            sweep.max_exact_residual = cell.exact_residual;
            sweep.max_exact_cell = cell_index;
            sweep.max_exact_normalization = cell.conservation_normalization;
            sweep.epsilon_at_max_exact = cell.conservation_epsilon;
        }
        if (std::abs(cell.hminus_residual) >
            std::abs(sweep.max_hminus_residual)) {
            sweep.max_hminus_residual = cell.hminus_residual;
            sweep.max_hminus_cell = cell_index;
        }
        if (std::isfinite(cell.conservation_epsilon) &&
            cell.conservation_epsilon > sweep.max_conservation_epsilon) {
            sweep.max_conservation_epsilon = cell.conservation_epsilon;
            sweep.residual_at_max_epsilon = cell.exact_residual;
            sweep.normalization_at_max_epsilon = cell.conservation_normalization;
            sweep.max_epsilon_cell = cell_index;
        }
        if (cell.hminus_active && std::isfinite(cell.hminus_epsilon) &&
            cell.hminus_epsilon > sweep.max_hminus_epsilon) {
            sweep.max_hminus_epsilon = cell.hminus_epsilon;
            sweep.max_hminus_epsilon_cell = cell_index;
        }
        if (cell.omitted_active && std::isfinite(cell.omitted_epsilon) &&
            cell.omitted_epsilon > sweep.max_omitted_epsilon) {
            sweep.max_omitted_epsilon = cell.omitted_epsilon;
            sweep.max_omitted_epsilon_cell = cell_index;
        }
    }

    IonTrial evaluate_ion_trial(const Profiles& fixed,
                                double amplitude,
                                const Vector& composition,
                                const std::vector<Vector>* pseudo_time_old = nullptr,
                                double inverse_pseudo_time_step = 0.0
#ifdef DCR_NESTED_DIAGNOSTIC
                                , const std::vector<Vector>*
                                    local_inverse_pseudo_time_steps = nullptr
#endif
                                ) const {
        IonTrial trial;
        trial.amplitude = amplitude;
        trial.background = fixed.background;
        trial.upstream_boundary = make_upstream_ion_boundary(amplitude, composition);
        trial.cell_diagnostics.resize(dx_.size());
        for (int k = static_cast<int>(trial.background.size()) - 1; k >= 0; --k) {
            CellDiagnostics cell;
            const Vector* upstream = k + 1 < static_cast<int>(trial.background.size())
                ? &trial.background[static_cast<size_t>(k + 1)] : nullptr;
            const UpstreamIonBoundary* upstream_face = upstream == nullptr
                ? &trial.upstream_boundary : nullptr;
            trial.background[static_cast<size_t>(k)] = solve_reverse_cell(
                k,
                upstream,
                upstream_face,
                fixed.background[static_cast<size_t>(k)],
                fixed.flowA,
                fixed.flowM,
                amplitude,
                cell,
                pseudo_time_old != nullptr
                    ? &(*pseudo_time_old)[static_cast<size_t>(k)] : nullptr,
                inverse_pseudo_time_step
#ifdef DCR_NESTED_DIAGNOSTIC
                , local_inverse_pseudo_time_steps != nullptr
                    ? &(*local_inverse_pseudo_time_steps)[static_cast<size_t>(k)]
                    : nullptr
#endif
                );
            trial.cell_diagnostics[static_cast<size_t>(k)] = cell;
            accumulate_cell_diagnostics(trial.diagnostics, k, cell);
        }
        if (pseudo_time_old != nullptr) {
            for (size_t k = 0; k < trial.background.size(); ++k) {
                for (int pi = 0; pi < trial.background[k].size(); ++pi) {
                    const double scale = std::max({
                        1.0,
                        std::abs(trial.background[k](pi)),
                        std::abs((*pseudo_time_old)[k](pi))});
                    const double change = std::abs(trial.background[k](pi) -
                        (*pseudo_time_old)[k](pi)) / scale;
                    if (change > trial.diagnostics.pseudo_time_profile_change) {
                        trial.diagnostics.pseudo_time_profile_change = change;
                        trial.diagnostics.max_pseudo_time_profile_change_cell =
                            static_cast<int>(k);
                        trial.diagnostics.max_pseudo_time_profile_change_pi = pi;
                        trial.diagnostics.max_pseudo_time_profile_change_gi =
                            boundary_.P_indices[static_cast<size_t>(pi)];
                        trial.diagnostics.max_pseudo_time_profile_change_previous =
                            (*pseudo_time_old)[k](pi);
                        trial.diagnostics.max_pseudo_time_profile_change_next =
                            trial.background[k](pi);
                    }
                }
            }
        }
        const double target_nuclei =
            weighted_sum(trial.background.front(), boundary_.P_indices, levels_) +
            weighted_sum(fixed.flowA.front(), boundary_.A_indices, levels_) +
            weighted_sum(fixed.flowM.front(), boundary_.M_indices, levels_);
        trial.mismatch = (target_nuclei - config_.plasma.total_density) /
            config_.plasma.total_density;
        return trial;
    }

    static MatrixDiagnostics matrix_diagnostics(const Matrix& matrix) {
        Eigen::JacobiSVD<Matrix> svd(matrix, Eigen::ComputeThinU | Eigen::ComputeThinV);
        MatrixDiagnostics result;
        result.rank = svd.rank();
        if (svd.singularValues().size() == 0) return result;
        const double sigma_max = svd.singularValues()(0);
        result.sigma_min = svd.singularValues()(svd.singularValues().size() - 1);
        if (result.sigma_min > 0.0) {
            result.condition_number = sigma_max / result.sigma_min;
        }
        return result;
    }

    static double componentwise_backward_error(const Matrix& matrix,
                                               const Vector& state,
                                               const Vector& rhs) {
        const Vector lhs = matrix * state;
        double maximum = 0.0;
        for (int row = 0; row < lhs.size(); ++row) {
            double scale = std::abs(rhs(row));
            for (int column = 0; column < matrix.cols(); ++column) {
                scale += std::abs(matrix(row, column)) * std::abs(state(column));
            }
            scale = std::max(kPhysicalRowScaleFloor, scale);
            maximum = std::max(maximum, std::abs(lhs(row) - rhs(row)) / scale);
        }
        return maximum;
    }

    void emit_backward_error_row_diagnostic(
        int cell,
        const Vector& state,
        const Vector* upstream,
        const UpstreamIonBoundary* upstream_face,
        const std::vector<Vector>& flowA,
        const std::vector<Vector>& flowM,
        int replacement_pi) const {
        const PhysicalCellSystem original = assemble_physical_cell_system(
            cell, state, upstream, upstream_face, flowA, flowM);
        Matrix lhs = original.lhs;
        Vector rhs = original.rhs;
        const Vector exact_row = original.lhs.transpose() * muP_;
        lhs.row(replacement_pi) = exact_row.transpose();
        rhs(replacement_pi) = muP_.dot(original.rhs);

        const Vector an = lhs * state;
        int worst_row = -1;
        double worst_numerator = 0.0;
        double worst_denominator = 0.0;
        double worst_backward_error = -1.0;
        double largest_cell_equation_scale = 0.0;
        for (int row = 0; row < lhs.rows(); ++row) {
            double denominator = std::abs(rhs(row));
            for (int column = 0; column < lhs.cols(); ++column) {
                denominator += std::abs(lhs(row, column)) *
                    std::abs(state(column));
            }
            denominator = std::max(kPhysicalRowScaleFloor, denominator);
            largest_cell_equation_scale = std::max(
                largest_cell_equation_scale, denominator);
            const double numerator = std::abs(an(row) - rhs(row));
            const double backward_error = numerator / denominator;
            if (backward_error > worst_backward_error) {
                worst_row = row;
                worst_numerator = numerator;
                worst_denominator = denominator;
                worst_backward_error = backward_error;
            }
        }

        double conservation_normalization = 0.0;
        for (int row = 0; row < original.row_scales.size(); ++row) {
            conservation_normalization = std::max(
                conservation_normalization,
                muP_(row) * original.row_scales(row));
        }
        const int worst_gi = boundary_.P_indices[static_cast<size_t>(worst_row)];
        std::cout << "BACKWARD_ERROR_WORST_ROW"
                  << " cell=" << cell
                  << " local_row=" << worst_row
                  << " global_index=" << worst_gi
                  << " species=" << std::quoted(species_label(worst_gi))
                  << " selected_conservation_row="
                  << (worst_row == replacement_pi ? 1 : 0)
                  << " population=" << state(worst_row)
                  << " numerator=" << worst_numerator
                  << " denominator=" << worst_denominator
                  << " epsilon_BE=" << worst_backward_error
                  << " physical_row_normalization="
                  << original.row_scales(worst_row)
                  << " conservation_normalization="
                  << conservation_normalization
                  << " An=" << an(worst_row)
                  << " b=" << rhs(worst_row)
                  << " row_norm=" << lhs.row(worst_row).norm()
                  << " largest_cell_equation_scale="
                  << largest_cell_equation_scale
                  << " largest_cell_physical_row_normalization="
                  << original.row_scales.maxCoeff()
                  << " machine_epsilon_cell_scale="
                  << std::numeric_limits<double>::epsilon() *
                      largest_cell_equation_scale
                  << "\n";

        std::vector<std::pair<double, int>> contributions;
        contributions.reserve(static_cast<size_t>(lhs.cols()));
        for (int column = 0; column < lhs.cols(); ++column) {
            contributions.emplace_back(
                std::abs(lhs(worst_row, column) * state(column)), column);
        }
        std::sort(contributions.begin(), contributions.end(),
                  [](const auto& left, const auto& right) {
                      return left.first > right.first;
                  });
        const int count = std::min<int>(8, contributions.size());
        for (int rank = 0; rank < count; ++rank) {
            const int column = contributions[static_cast<size_t>(rank)].second;
            const int gi = boundary_.P_indices[static_cast<size_t>(column)];
            const double contribution = lhs(worst_row, column) * state(column);
            std::cout << "BACKWARD_ERROR_CONTRIBUTION"
                      << " cell=" << cell
                      << " rank=" << rank + 1
                      << " local_column=" << column
                      << " global_index=" << gi
                      << " species=" << std::quoted(species_label(gi))
                      << " coefficient=" << lhs(worst_row, column)
                      << " population=" << state(column)
                      << " signed_A_n=" << contribution
                      << " abs_A_n=" << std::abs(contribution)
                      << "\n";
        }
    }

    void emit_local_convergence_diagnostic(
        int cell,
        double shooting_amplitude,
        const Vector& upstream,
        const Vector& initial,
        const Vector& fixed_flowA,
        const Vector& fixed_flowM,
        const std::vector<LocalIterationDiagnostics>& history,
        const CellDiagnostics& final_diagnostics) const {
        std::cout << "LOCAL_FAILURE first_unconverged_reverse_cell=1"
                  << " outer=" << current_outer_iteration_
                  << " cell=" << cell
                  << " x_left=" << x_[static_cast<size_t>(cell)]
                  << " x_right=" << x_[static_cast<size_t>(cell + 1)]
                  << " shooting_amplitude=" << shooting_amplitude
                  << " initial_min=" << initial.minCoeff()
                  << " initial_max=" << initial.maxCoeff()
                  << " upstream_min=" << upstream.minCoeff()
                  << " upstream_max=" << upstream.maxCoeff()
                  << " fixed_flowA_nuclei="
                  << weighted_sum(fixed_flowA, boundary_.A_indices, levels_)
                  << " fixed_flowM_nuclei="
                  << weighted_sum(fixed_flowM, boundary_.M_indices, levels_)
                  << "\n";
        for (const auto& iteration : history) {
            std::cout << "LOCAL_ITER iteration=" << iteration.iteration
                      << " physical_residual_norm="
                      << iteration.physical_residual_norm
                       << " physical_residual_relative="
                       << iteration.physical_residual_relative
                       << " physical_residual_global_relative="
                       << iteration.physical_residual_global_relative
                      << " linearized_post_residual_relative="
                      << iteration.linearized_post_residual_relative
                      << " nonlinear_post_residual_relative="
                      << iteration.nonlinear_post_residual_relative
                      << " population_change=" << iteration.population_change
                      << " linear_solve_BE_before="
                      << iteration.linear_solve_backward_error_before
                      << " linear_solve_BE_after="
                      << iteration.linear_solve_backward_error_after
                      << " refinement_steps="
                      << iteration.iterative_refinement_steps
                      << " minimum_population=" << iteration.minimum_population
                      << " raw_solution_minimum=" << iteration.raw_solution_minimum
                      << " raw_nonpositive_count=" << iteration.raw_nonpositive_count
                       << " floor_count=" << iteration.floor_count
                       << " maximum_change_pi=" << iteration.maximum_change_index
                       << " maximum_change_previous="
                       << iteration.maximum_change_previous
                       << " maximum_change_next=" << iteration.maximum_change_next
                       << "\n";
        }
        const double initial_residual = history.empty()
            ? std::numeric_limits<double>::quiet_NaN()
            : history.front().physical_residual_relative;
        const double final_residual = history.empty()
            ? std::numeric_limits<double>::quiet_NaN()
            : history.back().physical_residual_relative;
        const double final_change = history.empty()
            ? std::numeric_limits<double>::quiet_NaN()
            : history.back().population_change;
        int total_nonpositive = 0;
        int maximum_floor_count = 0;
        for (const auto& iteration : history) {
            total_nonpositive += iteration.raw_nonpositive_count;
            maximum_floor_count = std::max(maximum_floor_count, iteration.floor_count);
        }
        std::cout << "LOCAL_SUMMARY initial_physical_residual_relative=" << initial_residual
                  << " final_physical_residual_relative=" << final_residual
                  << " residual_ratio=" << final_residual / initial_residual
                  << " final_population_change=" << final_change
                  << " total_raw_nonpositive=" << total_nonpositive
                  << " maximum_floor_count=" << maximum_floor_count
                  << " exact_R_sigma=" << final_diagnostics.exact_residual
                   << " physical_Hminus=" << final_diagnostics.hminus_residual
                  << "\n";
    }

    static std::pair<Matrix, Vector> row_scaled_system(const Matrix& matrix,
                                                        const Vector& rhs) {
        Matrix scaled_matrix = matrix;
        Vector scaled_rhs = rhs;
        for (int row = 0; row < matrix.rows(); ++row) {
            const double norm = matrix.row(row).norm();
            if (!(norm > 0.0)) continue;
            scaled_matrix.row(row) /= norm;
            scaled_rhs(row) /= norm;
        }
        return {scaled_matrix, scaled_rhs};
    }

    static SolveDiagnostics attempt_solve(const Matrix& system_matrix,
                                          const Vector& system_rhs,
                                          const Matrix& unscaled_matrix,
                                          const Vector& unscaled_rhs) {
        Eigen::FullPivLU<Matrix> lu(system_matrix);
        const Vector solution = lu.solve(system_rhs);
        SolveDiagnostics result;
        result.rank = lu.rank();
        result.finite = solution.allFinite();
        if (!result.finite) return result;
        result.system_relative_residual =
            (system_matrix * solution - system_rhs).norm() /
            std::max(system_rhs.norm(), 1.0);
        result.unscaled_relative_residual =
            (unscaled_matrix * solution - unscaled_rhs).norm() /
            std::max(unscaled_rhs.norm(), 1.0);
        return result;
    }

    void emit_conditioning_diagnostic(const Matrix& original_matrix,
                                      const Vector& original_rhs,
                                      const Matrix& weighted_matrix,
                                      const Vector& weighted_rhs,
                                      int cell,
                                      int local_iteration,
                                      double shooting_amplitude) const {
        std::cout << "CONDITIONING first_failing_reverse_cell=1"
                  << " outer=" << current_outer_iteration_
                  << " cell=" << cell
                  << " x_left=" << x_[static_cast<size_t>(cell)]
                  << " x_right=" << x_[static_cast<size_t>(cell + 1)]
                  << " local_iteration=" << local_iteration
                  << " shooting_amplitude=" << shooting_amplitude << "\n";

        auto report_matrix = [&](const char* name,
                                 const Matrix& matrix,
                                 const Vector& rhs) {
            const auto raw_stats = matrix_diagnostics(matrix);
            const auto raw_solve = attempt_solve(matrix, rhs, matrix, rhs);
            std::cout << "CONDITIONING matrix=" << name
                      << " row_scaled=0"
                      << " rank=" << raw_stats.rank
                      << " sigma_min=" << raw_stats.sigma_min
                      << " kappa=" << raw_stats.condition_number
                      << " solve_rank=" << raw_solve.rank
                      << " solve_finite=" << (raw_solve.finite ? 1 : 0)
                      << " solve_relative_residual="
                      << raw_solve.system_relative_residual
                      << " unscaled_relative_residual="
                      << raw_solve.unscaled_relative_residual << "\n";

            const auto [scaled_matrix, scaled_rhs] = row_scaled_system(matrix, rhs);
            const auto scaled_stats = matrix_diagnostics(scaled_matrix);
            const auto scaled_solve = attempt_solve(
                scaled_matrix, scaled_rhs, matrix, rhs);
            std::cout << "CONDITIONING matrix=" << name
                      << " row_scaled=1"
                      << " rank=" << scaled_stats.rank
                      << " sigma_min=" << scaled_stats.sigma_min
                      << " kappa=" << scaled_stats.condition_number
                      << " solve_rank=" << scaled_solve.rank
                      << " solve_finite=" << (scaled_solve.finite ? 1 : 0)
                      << " solve_relative_residual="
                      << scaled_solve.system_relative_residual
                      << " unscaled_relative_residual="
                      << scaled_solve.unscaled_relative_residual << "\n";
        };

        report_matrix("original_Hminus", original_matrix, original_rhs);
        report_matrix("weighted_R_sigma", weighted_matrix, weighted_rhs);
    }

    IonTrial shoot_nearby_controlled(
        const Profiles& fixed,
        double initial_amplitude,
        const Vector& composition,
        double lambda_recycling,
        double safe_endpoint_amplitude,
        bool emit_probe_log = true) const {
        int evaluations = 0;
        auto evaluate = [&](double amplitude,
                            const char* kind,
                            const Profiles& initial_profiles) {
            IonTrial trial = evaluate_ion_trial(
                initial_profiles, amplitude, composition);
            ++evaluations;
            double minimum_population = std::numeric_limits<double>::infinity();
            for (const Vector& state : trial.background) {
                minimum_population = std::min(minimum_population, state.minCoeff());
            }
            double upstream_ion_nuclei_density = 0.0;
            for (int pi : ion_p_indices_) {
                upstream_ion_nuclei_density +=
                    trial.upstream_boundary.positive_ion_nuclei_flux_cm2_s(pi) /
                    trial.upstream_boundary.positive_ion_velocity_cm_s(pi);
            }
            const bool nonnegative = minimum_population >= 0.0;
            if (emit_probe_log) {
                std::cout << "CONTINUATION_SHOOT_PROBE lambda_rec=" << lambda_recycling
                          << " outer=" << current_outer_iteration_
                          << " kind=" << kind
                          << " evaluation=" << evaluations
                          << " amplitude=" << amplitude
                          << " mismatch=" << trial.mismatch
                          << " all_reverse_cells_converged="
                          << (trial.diagnostics.cells_converged ? 1 : 0)
                          << " nonnegative=" << (nonnegative ? 1 : 0)
                          << " minimum_population=" << minimum_population
                          << " upstream_ion_nuclei_density="
                          << upstream_ion_nuclei_density
                          << "\n";
            }
            if (!trial.diagnostics.cells_converged || !nonnegative) {
                throw std::runtime_error(
                    "Nearby continuation shooting lost reverse-cell feasibility");
            }
            trial.evaluations = evaluations;
            return trial;
        };

        const double previous_root =
            std::isfinite(safe_endpoint_amplitude) && safe_endpoint_amplitude > 0.0
            ? safe_endpoint_amplitude : initial_amplitude;
        const double minimum_amplitude = 0.1 * previous_root;
        const double maximum_amplitude = 4.0 * previous_root;
        const double predictor = std::clamp(
            initial_amplitude, minimum_amplitude, maximum_amplitude);

        std::vector<IonTrial> probes;
        IonTrial lower;
        IonTrial upper;
        bool bracketed = false;
        bool shooting_converged = false;
        IonTrial converged_trial;

        auto add_probe = [&](double amplitude, const char* kind) {
            amplitude = std::clamp(
                amplitude, minimum_amplitude, maximum_amplitude);
            for (size_t i = 0; i < probes.size(); ++i) {
                if (std::abs(probes[i].amplitude - amplitude) <=
                    1.0e-12 * std::max(probes[i].amplitude, amplitude)) {
                    return static_cast<int>(i);
                }
            }
            Profiles probe_initial = fixed;
            if (!probes.empty()) {
                size_t nearest = 0;
                double nearest_distance = std::numeric_limits<double>::infinity();
                for (size_t i = 0; i < probes.size(); ++i) {
                    const double distance = std::abs(
                        std::log(amplitude / probes[i].amplitude));
                    if (distance < nearest_distance) {
                        nearest = i;
                        nearest_distance = distance;
                    }
                }
                probe_initial.background = probes[nearest].background;
            }
            probes.push_back(evaluate(amplitude, kind, probe_initial));
            if (std::abs(probes.back().mismatch) < kOuterTolerance) {
                converged_trial = probes.back();
                shooting_converged = true;
            }
            return static_cast<int>(probes.size()) - 1;
        };

        auto find_tightest_bracket = [&]() {
            double tightest_width = std::numeric_limits<double>::infinity();
            int lower_index = -1;
            int upper_index = -1;
            for (size_t i = 0; i < probes.size(); ++i) {
                for (size_t j = i + 1; j < probes.size(); ++j) {
                    if (probes[i].mismatch * probes[j].mismatch >= 0.0) continue;
                    const int low = probes[i].amplitude < probes[j].amplitude
                        ? static_cast<int>(i) : static_cast<int>(j);
                    const int high = low == static_cast<int>(i)
                        ? static_cast<int>(j) : static_cast<int>(i);
                    const double width =
                        probes[static_cast<size_t>(high)].amplitude -
                        probes[static_cast<size_t>(low)].amplitude;
                    if (width < tightest_width) {
                        tightest_width = width;
                        lower_index = low;
                        upper_index = high;
                    }
                }
            }
            if (lower_index < 0) return false;
            lower = probes[static_cast<size_t>(lower_index)];
            upper = probes[static_cast<size_t>(upper_index)];
            return true;
        };

        const int predictor_index = add_probe(predictor, "secant_predictor");
        if (shooting_converged) {
            converged_trial.evaluations = evaluations;
            return converged_trial;
        }
        add_probe(previous_root, "previous_root_anchor");
        if (shooting_converged) {
            converged_trial.evaluations = evaluations;
            return converged_trial;
        }
        bracketed = find_tightest_bracket();

        int seed_index = predictor_index;
        for (size_t i = 0; i < probes.size(); ++i) {
            if (std::abs(probes[i].mismatch) <
                std::abs(probes[static_cast<size_t>(seed_index)].mismatch)) {
                seed_index = static_cast<int>(i);
            }
        }
        const bool mismatch_requests_increase =
            probes[static_cast<size_t>(seed_index)].mismatch < 0.0;
        constexpr double geometric_factor = 1.1;
        auto expand_to_bound = [&](bool increase) {
            int current_index = seed_index;
            while (!bracketed && !shooting_converged) {
                const double current =
                    probes[static_cast<size_t>(current_index)].amplitude;
                const double next = increase
                    ? std::min(maximum_amplitude, geometric_factor * current)
                    : std::max(minimum_amplitude, current / geometric_factor);
                if (next == current) break;
                current_index = add_probe(next, "controlled_geometric_bracket");
                bracketed = find_tightest_bracket();
                if (increase && next == maximum_amplitude) break;
                if (!increase && next == minimum_amplitude) break;
            }
        };
        if (!bracketed) expand_to_bound(mismatch_requests_increase);
        if (shooting_converged) {
            converged_trial.evaluations = evaluations;
            return converged_trial;
        }
        if (!bracketed) expand_to_bound(!mismatch_requests_increase);
        if (shooting_converged) {
            converged_trial.evaluations = evaluations;
            return converged_trial;
        }
        if (!bracketed) {
            const int minimum_index = add_probe(
                minimum_amplitude, "controlled_interval_minimum");
            const int maximum_index = add_probe(
                maximum_amplitude, "controlled_interval_maximum");
            bracketed = find_tightest_bracket();
            if (shooting_converged) {
                converged_trial.evaluations = evaluations;
                return converged_trial;
            }
            if (bracketed) {
                // Continue below with the bracket discovered at an interval endpoint.
            } else {
            std::ostringstream message;
                message << "No shooting root in controlled interval ["
                        << minimum_amplitude << ',' << maximum_amplitude
                        << "] anchored at previous accepted root " << previous_root
                        << "; endpoint mismatches=("
                        << probes[static_cast<size_t>(minimum_index)].mismatch << ','
                        << probes[static_cast<size_t>(maximum_index)].mismatch << ')';
            throw std::runtime_error(message.str());
            }
        }

        for (int iteration = 0; iteration < 24; ++iteration) {
            const double width = upper.amplitude - lower.amplitude;
            double amplitude =
                (lower.amplitude * upper.mismatch -
                 upper.amplitude * lower.mismatch) /
                (upper.mismatch - lower.mismatch);
            amplitude = std::clamp(
                amplitude, lower.amplitude + 0.1 * width,
                upper.amplitude - 0.1 * width);
            const double fraction = (amplitude - lower.amplitude) / width;
            Profiles root_initial = fixed;
            for (size_t k = 0; k < root_initial.background.size(); ++k) {
                root_initial.background[k] =
                    (1.0 - fraction) * lower.background[k] +
                    fraction * upper.background[k];
            }
            IonTrial trial = evaluate(amplitude, "nearby_root", root_initial);
            if (std::abs(trial.mismatch) < kOuterTolerance) {
                trial.evaluations = evaluations;
                return trial;
            }
            if (trial.mismatch * lower.mismatch < 0.0) {
                upper = std::move(trial);
            } else {
                lower = std::move(trial);
            }
        }
        throw std::runtime_error(
            "Nearby continuation shooting did not reach target mismatch tolerance");
    }

    IonTrial shoot_recycling_base_controlled(
        const Profiles& fixed,
        double initial_amplitude,
        const Vector& composition) const {
        int evaluations = 0;
        auto evaluate = [&](double amplitude,
                            const char* probe,
                            const Profiles& initial_profiles) {
            IonTrial trial = evaluate_ion_trial(initial_profiles, amplitude, composition);
            ++evaluations;
            double minimum_population = std::numeric_limits<double>::infinity();
            for (const Vector& state : trial.background) {
                minimum_population = std::min(minimum_population, state.minCoeff());
            }
            double upstream_ion_nuclei_density = 0.0;
            for (int pi : ion_p_indices_) {
                upstream_ion_nuclei_density +=
                    trial.upstream_boundary.positive_ion_nuclei_flux_cm2_s(pi) /
                    trial.upstream_boundary.positive_ion_velocity_cm_s(pi);
            }
            const bool nonnegative = minimum_population >= 0.0;
            std::cout << "CONTROLLED_SHOOT_PROBE kind=" << probe
                      << " evaluation=" << evaluations
                      << " amplitude=" << amplitude
                      << " mismatch=" << trial.mismatch
                      << " all_reverse_cells_converged="
                      << (trial.diagnostics.cells_converged ? 1 : 0)
                      << " nonnegative=" << (nonnegative ? 1 : 0)
                      << " minimum_population=" << minimum_population
                      << " upstream_ion_nuclei_density="
                      << upstream_ion_nuclei_density
                      << " upstream_positive_ion_nuclei_flux="
                      << trial.upstream_boundary.positive_ion_nuclei_flux_cm2_s.sum()
                      << "\n";
            if (!trial.diagnostics.cells_converged || !nonnegative) {
                throw std::runtime_error(
                    "Controlled shooting probe lost reverse-cell feasibility");
            }
            trial.evaluations = evaluations;
            return trial;
        };

        IonTrial anchor = evaluate(initial_amplitude, "initial_anchor", fixed);
        if (std::abs(anchor.mismatch) < kOuterTolerance) return anchor;
        IonTrial lower;
        IonTrial upper;
        IonTrial last = anchor;
        bool bracketed = false;
        for (int power = 1; power <= 6 && !bracketed; ++power) {
            const double factor = std::ldexp(1.0, power);
            const double amplitude = anchor.mismatch < 0.0
                ? initial_amplitude * factor : initial_amplitude / factor;
            Profiles probe_initial = fixed;
            probe_initial.background = last.background;
            IonTrial probe = evaluate(
                amplitude, "controlled_geometric_anchor", probe_initial);
            if (std::abs(probe.mismatch) < kOuterTolerance) return probe;
            if (anchor.mismatch * probe.mismatch < 0.0) {
                if (anchor.mismatch < 0.0) {
                    lower = anchor;
                    upper = std::move(probe);
                } else {
                    lower = std::move(probe);
                    upper = anchor;
                }
                bracketed = true;
                break;
            }
            last = std::move(probe);
        }
        if (!bracketed) {
            std::ostringstream message;
            message << "Controlled base shooting did not bracket the root in ["
                    << initial_amplitude / 64.0 << ',' << initial_amplitude * 64.0
                    << "]; anchor=(" << anchor.amplitude << ',' << anchor.mismatch
                    << ") last=(" << last.amplitude << ',' << last.mismatch << ')';
            throw std::runtime_error(message.str());
        }

        std::cout << "CONTROLLED_SHOOT_BRACKET lower_amplitude=" << lower.amplitude
                  << " lower_mismatch=" << lower.mismatch
                  << " upper_amplitude=" << upper.amplitude
                  << " upper_mismatch=" << upper.mismatch
                  << "\n";
        for (int iteration = 0; iteration < 24; ++iteration) {
            double amplitude =
                (lower.amplitude * upper.mismatch -
                 upper.amplitude * lower.mismatch) /
                (upper.mismatch - lower.mismatch);
            const double width = upper.amplitude - lower.amplitude;
            amplitude = std::clamp(
                amplitude, lower.amplitude + 0.1 * width,
                upper.amplitude - 0.1 * width);
            const double fraction = (amplitude - lower.amplitude) / width;
            Profiles trial_initial = fixed;
            for (size_t k = 0; k < trial_initial.background.size(); ++k) {
                trial_initial.background[k] =
                    (1.0 - fraction) * lower.background[k] +
                    fraction * upper.background[k];
            }
            IonTrial trial = evaluate(
                amplitude, "bracketed_root", trial_initial);
            if (std::abs(trial.mismatch) < kOuterTolerance) {
                trial.evaluations = evaluations;
                return trial;
            }
            if (trial.mismatch < 0.0) lower = std::move(trial);
            else upper = std::move(trial);
        }
        std::ostringstream message;
        message << "Controlled bracketed root did not reach tolerance; lower=("
                << lower.amplitude << "," << lower.mismatch << ") upper=("
                << upper.amplitude << "," << upper.mismatch << ")";
        throw std::runtime_error(message.str());
    }

    IonTrial shoot_ion_amplitude(const Profiles& fixed,
                                 double initial_amplitude,
                                 const Vector& composition) const {
        IonTrial negative;
        IonTrial positive;
        IonTrial best;
        bool have_negative = false;
        bool have_positive = false;
        bool have_best = false;
        int evaluations = 0;
        std::string last_failure;
        auto record = [&](IonTrial trial) {
            ++evaluations;
            if (!have_best || std::abs(trial.mismatch) < std::abs(best.mismatch)) {
                best = trial;
                have_best = true;
            }
            if (trial.mismatch < 0.0) {
                negative = std::move(trial);
                have_negative = true;
            } else {
                positive = std::move(trial);
                have_positive = true;
            }
        };
        for (int expansion = -1;
             expansion < kMaxShootingExpansionPowers &&
                 !(have_negative && have_positive);
             ++expansion) {
            const double factor = expansion < 0 ? 1.0 : std::ldexp(1.0, expansion + 1);
            const std::vector<double> probes = expansion < 0
                ? std::vector<double>{initial_amplitude}
                : std::vector<double>{initial_amplitude * factor, initial_amplitude / factor};
            for (double probe_amplitude : probes) {
                try {
                    IonTrial probe = evaluate_ion_trial(fixed, probe_amplitude, composition);
                    if (std::abs(probe.mismatch) < kOuterTolerance) {
                        probe.evaluations = evaluations + 1;
                        return probe;
                    }
                    record(std::move(probe));
                } catch (const UnconvergedReverseCellError&) {
                    throw;
                } catch (const std::exception& error) {
                    last_failure = error.what();
                }
                if (have_negative && have_positive) break;
            }
        }
        if (!(have_negative && have_positive)) {
            std::ostringstream message;
            message << "Unable to bracket the upstream ion-flux amplitude within ["
                    << initial_amplitude / std::ldexp(1.0, kMaxShootingExpansionPowers)
                    << ","
                    << initial_amplitude * std::ldexp(1.0, kMaxShootingExpansionPowers)
                    << "]";
            if (have_best) {
                message << "; best=(" << best.amplitude << "," << best.mismatch << ")";
            }
            if (!last_failure.empty()) message << "; last_failure=" << last_failure;
            throw std::runtime_error(message.str());
        }

        for (int iteration = 0; iteration < 24; ++iteration) {
            const double low_amplitude = std::min(negative.amplitude, positive.amplitude);
            const double high_amplitude = std::max(negative.amplitude, positive.amplitude);
            double next = (negative.amplitude * positive.mismatch -
                           positive.amplitude * negative.mismatch) /
                (positive.mismatch - negative.mismatch);
            next = std::clamp(next,
                              low_amplitude + 0.1 * (high_amplitude - low_amplitude),
                              high_amplitude - 0.1 * (high_amplitude - low_amplitude));
            IonTrial trial;
            try {
                trial = evaluate_ion_trial(fixed, next, composition);
            } catch (const UnconvergedReverseCellError&) {
                throw;
            } catch (const std::exception&) {
                next = 0.5 * (low_amplitude + high_amplitude);
                trial = evaluate_ion_trial(fixed, next, composition);
            }
            ++evaluations;
            if (std::abs(trial.mismatch) < std::abs(best.mismatch)) best = trial;
            if (std::abs(trial.mismatch) < kOuterTolerance) {
                trial.evaluations = evaluations;
                return trial;
            }
            if (trial.mismatch < 0.0) negative = std::move(trial);
            else positive = std::move(trial);
        }
        std::ostringstream message;
        message << "Shooting root did not reach tolerance; best=("
                << best.amplitude << "," << best.mismatch
                << ") evaluations=" << evaluations;
        throw std::runtime_error(message.str());
    }

    void evaluate_cell_residual(int k,
                                const Vector& state,
                                const Vector* upstream,
                                const UpstreamIonBoundary* upstream_face,
                                const std::vector<Vector>& flowA,
                                const std::vector<Vector>& flowM,
                                int replacement_pi,
                                CellDiagnostics& diagnostics) const {
        const PhysicalCellSystem original = assemble_physical_cell_system(
            k, state, upstream, upstream_face, flowA, flowM);
        const Vector residual = original.lhs * state - original.rhs;
        diagnostics.steady_backward_error = componentwise_backward_error(
            original.lhs, state, original.rhs);
        diagnostics.replacement_pi = replacement_pi;
        diagnostics.replacement_gi = replacement_pi >= 0
            ? boundary_.P_indices[static_cast<size_t>(replacement_pi)] : -1;
        diagnostics.hminus_residual = residual(hminus_pi_);
        diagnostics.omitted_residual = replacement_pi >= 0
            ? residual(replacement_pi)
            : std::numeric_limits<double>::quiet_NaN();
        diagnostics.exact_residual = muP_.dot(residual);
        diagnostics.conservation_normalization = 0.0;
        for (int pi = 0; pi < state.size(); ++pi) {
            diagnostics.conservation_normalization = std::max(
                diagnostics.conservation_normalization,
                muP_(pi) * original.row_scales(pi));
        }
        diagnostics.conservation_epsilon = diagnostics.conservation_normalization >
            kPhysicalRowScaleFloor
            ? std::abs(diagnostics.exact_residual) /
                diagnostics.conservation_normalization
            : std::numeric_limits<double>::quiet_NaN();
        diagnostics.hminus_normalization = original.row_scales(hminus_pi_);
        diagnostics.hminus_active =
            diagnostics.hminus_normalization > kPhysicalRowScaleFloor;
        diagnostics.hminus_epsilon = diagnostics.hminus_active
            ? std::abs(diagnostics.hminus_residual) /
                diagnostics.hminus_normalization
            : std::numeric_limits<double>::quiet_NaN();
        diagnostics.omitted_normalization = replacement_pi >= 0
            ? original.row_scales(replacement_pi) : 0.0;
        diagnostics.omitted_active =
            diagnostics.omitted_normalization > kPhysicalRowScaleFloor;
        diagnostics.omitted_epsilon = diagnostics.omitted_active
            ? std::abs(diagnostics.omitted_residual) /
                diagnostics.omitted_normalization
            : std::numeric_limits<double>::quiet_NaN();
        diagnostics.final_minimum_population = state.minCoeff();
    }

    double weighted_flow_residual(int k,
                                  const Vector& chemistryA,
                                  const Vector& chemistryM,
                                  const std::vector<Vector>& flowA,
                                  const std::vector<Vector>& flowM) const {
        const double dx = dx_[static_cast<size_t>(k)];
        const Vector residualA = boundary_.u_A *
            (flowA[static_cast<size_t>(k + 1)] - flowA[static_cast<size_t>(k)]) / dx +
            atom_exhaust_rate_ * flowA[static_cast<size_t>(k)] - chemistryA;
        const Vector residualM = boundary_.u_M *
            (flowM[static_cast<size_t>(k + 1)] - flowM[static_cast<size_t>(k)]) / dx +
            molecule_exhaust_rate_ * flowM[static_cast<size_t>(k)] - chemistryM;
        return weighted_sum(residualA, boundary_.A_indices, levels_) +
            weighted_sum(residualM, boundary_.M_indices, levels_);
    }

    std::pair<Vector, Vector> target_recycling(const Vector& target) const {
        Vector flowA = Vector::Zero(static_cast<int>(boundary_.A_indices.size()));
        Vector flowM = Vector::Zero(static_cast<int>(boundary_.M_indices.size()));
        double atomic_flux = 0.0;
        double molecular_flux = 0.0;
        const double acceptance = std::clamp(
            config_.numerics.boundary_molecular_flow_acceptance, 0.0, 1.0);
        for (int pi : ion_p_indices_) {
            const int gi = boundary_.P_indices[static_cast<size_t>(pi)];
            const double incident = ion_speed(gi, 0.0) * std::max(target(pi), 0.0);
            const int atomicity = std::max(1, levels_[static_cast<size_t>(gi)].atomicity);
            if (atomicity >= 2) {
                atomic_flux += atomicity * incident;
            } else {
                atomic_flux += wall_.alpha_atom * incident;
                molecular_flux += acceptance * wall_.alpha_molecule * incident / 2.0;
            }
        }
        flowA(atom_ground_ai_) = atomic_flux / boundary_.u_A;
        flowM(molecule_ground_mi_) = molecular_flux / boundary_.u_M;
        return {flowA, flowM};
    }

    Profiles sweep_recycled_neutrals(const Profiles& accepted,
                                     double lambda_recycling = 1.0) const {
        Profiles candidate = accepted;
        auto [target_A, target_M] = target_recycling(accepted.background.front());
        target_A *= std::clamp(lambda_recycling, 0.0, 1.0);
        target_M *= std::clamp(lambda_recycling, 0.0, 1.0);
        candidate.flowA.front() = target_A;
        candidate.flowM.front() = target_M;
        for (size_t k = 0; k + 1 < accepted.background.size(); ++k) {
            const auto background_full = dcr::solver::make_background_full(
                accepted.background[k + 1], boundary_, total_states_);
            const auto local = dcr::solver::assemble_local_system(
                config_, atomic_data_, plasma_, grid_, boundary_, background_full,
                candidate.flowA[k], candidate.flowM[k], x_[k + 1], false, &rate_cache_);
            const auto advanced = dcr::solver::advance_recycling_flow_one_step(
                config_, atomic_data_, boundary_, local,
                candidate.flowA[k], candidate.flowM[k], dx_[k]);
            candidate.flowA[k + 1] = advanced.flowA_next;
            candidate.flowM[k + 1] = advanced.flowM_next;
        }
        return candidate;
    }

    NeutralPseudoTimeSweepResult sweep_recycled_neutrals_pseudo_time(
        const Profiles& accepted,
        double inverse_dt_A,
        double inverse_dt_M,
        double lambda_recycling = 1.0,
        bool advance_atom = true,
        bool advance_molecule = true) const {
        NeutralPseudoTimeSweepResult result;
        result.profiles = accepted;
        auto [target_A, target_M] = target_recycling(
            accepted.background.front());
        target_A *= std::clamp(lambda_recycling, 0.0, 1.0);
        target_M *= std::clamp(lambda_recycling, 0.0, 1.0);
        if (advance_atom) result.profiles.flowA.front() = target_A;
        if (advance_molecule) result.profiles.flowM.front() = target_M;
        result.atom.minimum_population = result.profiles.flowA.front().minCoeff();
        result.molecule.minimum_population = result.profiles.flowM.front().minCoeff();

        for (size_t k = 0; k + 1 < accepted.background.size(); ++k) {
            const auto background_full = dcr::solver::make_background_full(
                accepted.background[k + 1], boundary_, total_states_);
            const auto local = dcr::solver::assemble_local_system(
                config_, atomic_data_, plasma_, grid_, boundary_, background_full,
                result.profiles.flowA[k], result.profiles.flowM[k],
                x_[k + 1], false, &rate_cache_);
            dcr::solver::RecyclingFlowPseudoTimeTerms terms;
            terms.old_flowA_next = &accepted.flowA[k + 1];
            terms.old_flowM_next = &accepted.flowM[k + 1];
            terms.inverse_dt_A = inverse_dt_A;
            terms.inverse_dt_M = inverse_dt_M;
            terms.advance_A = advance_atom;
            terms.advance_M = advance_molecule;
            const auto advanced = dcr::solver::advance_recycling_flow_one_step(
                config_, atomic_data_, boundary_, local,
                result.profiles.flowA[k], result.profiles.flowM[k], dx_[k], terms);

            if (advance_atom) {
                result.atom.modified_backward_error = std::max(
                    result.atom.modified_backward_error,
                    advanced.diagnosticsA.modified_backward_error);
                result.atom.steady_backward_error = std::max(
                    result.atom.steady_backward_error,
                    advanced.diagnosticsA.steady_backward_error);
                result.atom.transient_update_ratio = std::max(
                    result.atom.transient_update_ratio,
                    advanced.diagnosticsA.transient_update_ratio);
                result.atom.profile_change = std::max(
                    result.atom.profile_change,
                    vector_relative_change(
                        advanced.flowA_next, accepted.flowA[k + 1]));
                result.atom.minimum_population = std::min(
                    result.atom.minimum_population,
                    advanced.flowA_next.minCoeff());
                result.atom.state_change_norm = std::max(
                    result.atom.state_change_norm,
                    advanced.diagnosticsA.state_change_norm);
                result.atom.transient_term_norm = std::max(
                    result.atom.transient_term_norm,
                    advanced.diagnosticsA.transient_term_norm);
                result.atom.steady_residual_norm = std::max(
                    result.atom.steady_residual_norm,
                    advanced.diagnosticsA.steady_residual_norm);
                result.atom.be_residual_norm = std::max(
                    result.atom.be_residual_norm,
                    advanced.diagnosticsA.be_residual_norm);
                result.atom.be_identity_defect_norm = std::max(
                    result.atom.be_identity_defect_norm,
                    advanced.diagnosticsA.be_identity_defect_norm);
                if (advanced.diagnosticsA.be_identity_defect_relative >
                    result.atom.be_identity_defect_relative) {
                    result.atom.be_identity_defect_relative =
                        advanced.diagnosticsA.be_identity_defect_relative;
                    result.atom.be_identity_defect_cell =
                        static_cast<int>(k + 1);
                    const int row =
                        advanced.diagnosticsA.be_identity_defect_row;
                    result.atom.be_identity_defect_gi = row >= 0
                        ? boundary_.A_indices[static_cast<size_t>(row)] : -1;
                }
                result.profiles.flowA[k + 1] = advanced.flowA_next;
            }
            if (advance_molecule) {
                result.molecule.modified_backward_error = std::max(
                    result.molecule.modified_backward_error,
                    advanced.diagnosticsM.modified_backward_error);
                result.molecule.steady_backward_error = std::max(
                    result.molecule.steady_backward_error,
                    advanced.diagnosticsM.steady_backward_error);
                result.molecule.transient_update_ratio = std::max(
                    result.molecule.transient_update_ratio,
                    advanced.diagnosticsM.transient_update_ratio);
                result.molecule.profile_change = std::max(
                    result.molecule.profile_change,
                    vector_relative_change(
                        advanced.flowM_next, accepted.flowM[k + 1]));
                result.molecule.minimum_population = std::min(
                    result.molecule.minimum_population,
                    advanced.flowM_next.minCoeff());
                result.molecule.state_change_norm = std::max(
                    result.molecule.state_change_norm,
                    advanced.diagnosticsM.state_change_norm);
                result.molecule.transient_term_norm = std::max(
                    result.molecule.transient_term_norm,
                    advanced.diagnosticsM.transient_term_norm);
                result.molecule.steady_residual_norm = std::max(
                    result.molecule.steady_residual_norm,
                    advanced.diagnosticsM.steady_residual_norm);
                result.molecule.be_residual_norm = std::max(
                    result.molecule.be_residual_norm,
                    advanced.diagnosticsM.be_residual_norm);
                result.molecule.be_identity_defect_norm = std::max(
                    result.molecule.be_identity_defect_norm,
                    advanced.diagnosticsM.be_identity_defect_norm);
                if (advanced.diagnosticsM.be_identity_defect_relative >
                    result.molecule.be_identity_defect_relative) {
                    result.molecule.be_identity_defect_relative =
                        advanced.diagnosticsM.be_identity_defect_relative;
                    result.molecule.be_identity_defect_cell =
                        static_cast<int>(k + 1);
                    const int row =
                        advanced.diagnosticsM.be_identity_defect_row;
                    result.molecule.be_identity_defect_gi = row >= 0
                        ? boundary_.M_indices[static_cast<size_t>(row)] : -1;
                }
                result.profiles.flowM[k + 1] = advanced.flowM_next;
            }
        }
        return result;
    }

    NeutralPseudoTimeStepEstimate estimate_neutral_pseudo_time_steps(
        const Profiles& profiles,
        double lambda_recycling = 1.0) const {
        struct WeightedRate {
            double rate = 0.0;
            double weight = 0.0;
        };
        std::vector<WeightedRate> atom_samples;
        std::vector<WeightedRate> molecule_samples;
        Profiles spatial = profiles;
        auto [target_A, target_M] = target_recycling(
            profiles.background.front());
        target_A *= std::clamp(lambda_recycling, 0.0, 1.0);
        target_M *= std::clamp(lambda_recycling, 0.0, 1.0);
        spatial.flowA.front() = target_A;
        spatial.flowM.front() = target_M;

        for (size_t k = 0; k + 1 < profiles.background.size(); ++k) {
            const auto background_full = dcr::solver::make_background_full(
                profiles.background[k + 1], boundary_, total_states_);
            const auto local = dcr::solver::assemble_local_system(
                config_, atomic_data_, plasma_, grid_, boundary_, background_full,
                spatial.flowA[k], spatial.flowM[k], x_[k + 1], false, &rate_cache_);
            const auto advanced = dcr::solver::advance_recycling_flow_one_step(
                config_, atomic_data_, boundary_, local,
                spatial.flowA[k], spatial.flowM[k], dx_[k]);

            auto collect = [&](const Vector& state,
                               const dcr::solver::RecyclingBlockStepDiagnostics& diagnostics,
                               const std::vector<int>& indices,
                               std::vector<WeightedRate>& samples) {
                for (int local_index = 0; local_index < state.size(); ++local_index) {
                    double row_scale = std::abs(
                        diagnostics.steady_rhs(local_index));
                    for (int column = 0;
                         column < diagnostics.steady_lhs.cols(); ++column) {
                        row_scale += std::abs(
                            diagnostics.steady_lhs(local_index, column)) *
                            std::abs(state(column));
                    }
                    const double population = std::max(0.0, state(local_index));
                    const int gi = indices[static_cast<size_t>(local_index)];
                    const double weighted_population = std::max(
                        population, kDensityFloor);
                    const double weight = std::max(
                        1, levels_[static_cast<size_t>(gi)].atomicity) *
                        weighted_population;
                    if (std::isfinite(row_scale)) {
                        samples.push_back({
                            row_scale / weighted_population,
                            weight});
                    }
                }
            };
            collect(
                profiles.flowA[k + 1], advanced.diagnosticsA,
                boundary_.A_indices, atom_samples);
            collect(
                profiles.flowM[k + 1], advanced.diagnosticsM,
                boundary_.M_indices, molecule_samples);
            spatial.flowA[k + 1] = advanced.flowA_next;
            spatial.flowM[k + 1] = advanced.flowM_next;
        }

        auto weighted_q95 = [](std::vector<WeightedRate> samples) {
            if (samples.empty()) {
                throw std::runtime_error(
                    "Cannot estimate neutral pseudo-time step from an empty block");
            }
            std::sort(samples.begin(), samples.end(), [](const auto& lhs, const auto& rhs) {
                return lhs.rate < rhs.rate;
            });
            double total_weight = 0.0;
            for (const auto& sample : samples) total_weight += sample.weight;
            const double target = 0.95 * total_weight;
            double cumulative = 0.0;
            for (const auto& sample : samples) {
                cumulative += sample.weight;
                if (cumulative >= target) return sample.rate;
            }
            return samples.back().rate;
        };

        NeutralPseudoTimeStepEstimate estimate;
        estimate.atom_quantile_rate = weighted_q95(atom_samples);
        estimate.molecule_quantile_rate = weighted_q95(molecule_samples);
        estimate.atom_step = 0.1 / std::max(estimate.atom_quantile_rate, 1.0e-30);
        estimate.molecule_step = 0.1 /
            std::max(estimate.molecule_quantile_rate, 1.0e-30);
        return estimate;
    }

    static void relax_neutral_profiles(Profiles& destination,
                                       const Profiles& source,
                                       double relaxation) {
        auto relax = [&](std::vector<Vector>& dst, const std::vector<Vector>& src) {
            for (size_t k = 0; k < dst.size(); ++k) {
                dst[k] = ((1.0 - relaxation) * dst[k] +
                          relaxation * src[k]).cwiseMax(0.0);
            }
        };
        relax(destination.flowA, source.flowA);
        relax(destination.flowM, source.flowM);
    }

    SweepDiagnostics evaluate_residuals(
        const Profiles& profiles,
        const UpstreamIonBoundary& upstream_face) const {
        SweepDiagnostics result;
        for (int k = static_cast<int>(profiles.background.size()) - 1; k >= 0; --k) {
            CellDiagnostics cell;
            cell.replacement_pi = select_conservation_replacement_row(
                profiles.background[static_cast<size_t>(k)],
                cell.replacement_population_fraction);
            cell.replacement_gi = boundary_.P_indices[
                static_cast<size_t>(cell.replacement_pi)];
            cell.converged = true;
            const Vector* upstream = k + 1 < static_cast<int>(profiles.background.size())
                ? &profiles.background[static_cast<size_t>(k + 1)] : nullptr;
            evaluate_cell_residual(k,
                                   profiles.background[static_cast<size_t>(k)],
                                   upstream,
                                   upstream == nullptr ? &upstream_face : nullptr,
                                   profiles.flowA, profiles.flowM,
                                   cell.replacement_pi, cell);
            accumulate_cell_diagnostics(result, k, cell);
        }
        return result;
    }

    void report(const Profiles& profiles,
                bool converged,
                int iterations,
                double amplitude,
                double mismatch,
                double profile_change,
                const SweepDiagnostics& residuals,
                const std::string& failure_reason) const {
        double max_h = -1.0;
        double max_nuclei = -1.0;
        int max_h_node = -1;
        int max_nuclei_node = -1;
        for (size_t k = 0; k < profiles.background.size(); ++k) {
            const double h = atomic_hydrogen(profiles.background[k]);
            const double nuclei = total_nuclei(profiles, k);
            if (h > max_h) {
                max_h = h;
                max_h_node = static_cast<int>(k);
            }
            if (nuclei > max_nuclei) {
                max_nuclei = nuclei;
                max_nuclei_node = static_cast<int>(k);
            }
        }
        double near_point_one_max = 0.0;
        for (size_t k = 0; k < profiles.background.size(); ++k) {
            if (x_[k] >= 0.08 && x_[k] <= 0.12) {
                near_point_one_max = std::max(near_point_one_max, total_nuclei(profiles, k));
            }
        }
        std::cout << "RESULT converged=" << (converged ? 1 : 0) << "\n";
        if (!failure_reason.empty()) {
            std::cout << "RESULT failure_reason=" << failure_reason << "\n";
        }
        std::cout << "RESULT outer_iterations=" << iterations << "\n";
        std::cout << "RESULT upstream_shooting_amplitude=" << amplitude << "\n";
        std::cout << "RESULT target_ion_nuclei_flux="
                  << ion_nuclei_flux(profiles.background.front(), 0.0) << "\n";
        std::cout << "RESULT upstream_ion_nuclei_flux="
                  << amplitude << "\n";
        std::cout << "RESULT target_n_nuc=" << total_nuclei(profiles, 0) << "\n";
        std::cout << "RESULT target_density_mismatch=" << mismatch << "\n";
        std::cout << "RESULT max_profile_relative_change=" << profile_change << "\n";
        std::cout << "RESULT max_n_H=" << max_h << " x=" << x_[static_cast<size_t>(max_h_node)] << "\n";
        std::cout << "RESULT max_n_nuc=" << max_nuclei
                  << " x=" << x_[static_cast<size_t>(max_nuclei_node)] << "\n";
        std::cout << "RESULT max_n_nuc_near_0p1=" << near_point_one_max << "\n";
        std::cout << "RESULT previous_runaway_near_0p1_absent="
                  << (near_point_one_max < 2.0 * config_.plasma.total_density ? 1 : 0)
                  << "\n";
        const int exact_cell = std::max(residuals.max_exact_cell, 0);
        const int hminus_cell = std::max(residuals.max_hminus_cell, 0);
        std::cout << "RESULT max_exact_R_sigma=" << residuals.max_exact_residual
                  << " x=" << x_[static_cast<size_t>(exact_cell)] << "\n";
        std::cout << "RESULT max_physical_Hminus_residual=" << residuals.max_hminus_residual
                  << " x=" << x_[static_cast<size_t>(hminus_cell)] << "\n";
    }

    double total_nuclei(const Profiles& profiles, size_t node) const {
        return weighted_sum(profiles.background[node], boundary_.P_indices, levels_) +
            weighted_sum(profiles.flowA[node], boundary_.A_indices, levels_) +
            weighted_sum(profiles.flowM[node], boundary_.M_indices, levels_);
    }

    double atomic_hydrogen(const Vector& state) const {
        double result = 0.0;
        for (int pi = 0; pi < state.size(); ++pi) {
            const int gi = boundary_.P_indices[static_cast<size_t>(pi)];
            const auto& level = levels_[static_cast<size_t>(gi)];
            if (level.type == dcr::atomic::SpeciesType::Atom && level.charge == 0) {
                result += std::max(state(pi), 0.0);
            }
        }
        return result;
    }

    double ion_nuclei_flux(const Vector& state, double x) const {
        double result = 0.0;
        for (int pi : ion_p_indices_) {
            const int gi = boundary_.P_indices[static_cast<size_t>(pi)];
            result += muP_(pi) * ion_speed(gi, x) * std::max(state(pi), 0.0);
        }
        return result;
    }

    double ion_bohm_speed(int gi, double x) const {
        const auto temperatures = dcr::solver::evaluate_plasma_temperatures(config_, x);
        return dcr::physics::calculate_Bohm_speed(
            temperatures.electron_eV, temperatures.ion_eV,
            levels_[static_cast<size_t>(gi)].mass_amu);
    }

    double ion_speed(int gi, double x) const {
        const double length = config_.numerics.adaptive_recycling_domain.ion_velocity_length_cm > 0.0
            ? config_.numerics.adaptive_recycling_domain.ion_velocity_length_cm
            : config_.grid.length_cm;
        const double shape = std::max(
            config_.numerics.adaptive_recycling_domain.ion_velocity_floor_fraction,
            1.0 - x / std::max(length, 1.0e-12));
        return ion_bohm_speed(gi, x) * shape;
    }

    double upstream_velocity_fraction() const {
        const double length =
            config_.numerics.adaptive_recycling_domain.ion_velocity_length_cm > 0.0
            ? config_.numerics.adaptive_recycling_domain.ion_velocity_length_cm
            : config_.grid.length_cm;
        return std::max(
            config_.numerics.adaptive_recycling_domain.ion_velocity_floor_fraction,
            1.0 - config_.grid.length_cm / std::max(length, 1.0e-12));
    }

    bool is_positive_ion(int pi) const {
        const int gi = boundary_.P_indices[static_cast<size_t>(pi)];
        const auto& level = levels_[static_cast<size_t>(gi)];
        return level.type == dcr::atomic::SpeciesType::Ion && level.charge > 0;
    }

    double background_exhaust_rate(int pi) const {
        const int gi = boundary_.P_indices[static_cast<size_t>(pi)];
        const auto& level = levels_[static_cast<size_t>(gi)];
        if (level.type == dcr::atomic::SpeciesType::Atom && level.charge == 0) {
            return atom_exhaust_rate_;
        }
        if (level.type == dcr::atomic::SpeciesType::Molecule && level.charge == 0) {
            return molecule_exhaust_rate_;
        }
        return 0.0;
    }

    static int compact_position(const std::vector<int>& indices, int global_index) {
        const auto position = std::find(indices.begin(), indices.end(), global_index);
        return position == indices.end() ? -1 : static_cast<int>(position - indices.begin());
    }

    static double vector_relative_change(const Vector& next, const Vector& previous) {
        if (next.size() == 0 || previous.size() == 0) return 0.0;
        return (next - previous).cwiseAbs().maxCoeff() /
            std::max(kPopulationChangeScaleFloor, previous.cwiseAbs().maxCoeff());
    }

    static Vector positive_update(const Vector& current,
                                  const Vector& solved,
                                  double requested_relaxation) {
        return ((1.0 - requested_relaxation) * current +
                requested_relaxation * solved).cwiseMax(kDensityFloor);
    }

    dcr::io::Config config_;
    dcr::atomic::AtomicData& atomic_data_;
    dcr::state::PlasmaState& plasma_;
    const EEDFGridView& grid_;
    dcr::solver::BoundaryPhaseResult boundary_;
    dcr::physics::WallRecycling wall_;
    const std::vector<dcr::atomic::EnergyLevel>& levels_;
    int total_states_ = 0;
    double relaxation_ = 0.3;
    std::vector<double> dx_;
    std::vector<double> x_;
    std::vector<int> p_position_;
    std::vector<int> ion_p_indices_;
    std::vector<int> frozen_replacement_rows_;
    int hminus_pi_ = -1;
    int atom_ground_ai_ = -1;
    int molecule_ground_mi_ = -1;
    double atom_exhaust_rate_ = 0.0;
    double molecule_exhaust_rate_ = 0.0;
    Vector muP_;
    dcr::solver::LocalRateCache rate_cache_;
    int current_outer_iteration_ = 0;
    mutable bool conditioning_diagnostic_emitted_ = false;
    std::string checkpoint_input_path_;
    std::string reference_artifact_stem_;
    std::string continuation_checkpoint_path_;
    std::string continuation_checkpoint_temporary_path_;
    std::string continuation_summary_path_;
};

EEDFConfig make_eedf_config(const dcr::io::Config& config) {
    EEDFConfig result;
    result.type = config.plasma.eedf.type;
    result.power_p = config.plasma.eedf.power_p;
    result.hot_fraction = config.plasma.eedf.hot_fraction;
    result.hot_temperature_factor = config.plasma.eedf.hot_temperature_factor;
    return result;
}

} // namespace

int main(int argc, char** argv) {
    try {
        const std::string config_path = argc > 1
            ? argv[1]
            : std::string(DCR_SOURCE_DIR) +
                "/config/single_T1_nnuc1e16_L50_Nx200_individual_ion_flux_20260803.yaml";
        auto config = dcr::io::ConfigLoader::load(config_path);
        dcr::solver::normalize_input_roots(config, config_path);
        config.io.verbose_logging = false;

        double continuation_step_override =
            std::numeric_limits<double>::quiet_NaN();
        bool reconverge_current_lambda = false;
        bool audit_lambda_one_endpoint = false;
        bool solve_physical_reference = false;
        double checkpoint_velocity_fraction =
            std::numeric_limits<double>::quiet_NaN();
        bool temperature_continuation = false;
        bool temperature_pseudo_time = false;
        bool temperature_neutral_pseudo_time = false;
        bool temperature_plain_backward_euler_once = false;
#ifdef DCR_NESTED_DIAGNOSTIC
        bool temperature_fixed_amplitude_diagnostic = false;
        bool fixed_gamma_activity_controller = false;
        bool molecular_residual_postprocess = false;
        bool activity_trace_deweighting_scan = false;
        double activity_trace_protection_fraction = 1.0e-3;
        double diagnostic_gamma = std::numeric_limits<double>::quiet_NaN();
        int diagnostic_max_inner_macro_iterations = 1000;
        int diagnostic_stagnation_window = 100;
        double diagnostic_neutral_molecule_step_scale = 1.0;
#else
        constexpr bool temperature_fixed_amplitude_diagnostic = false;
        constexpr bool fixed_gamma_activity_controller = false;
        constexpr double activity_trace_protection_fraction = 1.0e-3;
        constexpr int diagnostic_max_inner_macro_iterations = 2000;
        constexpr int diagnostic_stagnation_window = 100;
        constexpr double diagnostic_neutral_molecule_step_scale = 1.0;
#endif
        double temperature_start = std::numeric_limits<double>::quiet_NaN();
        double single_temperature_stage =
            std::numeric_limits<double>::quiet_NaN();
        std::string checkpoint_input_path;
        std::string artifact_stem_override;
        for (int argument_index = 2; argument_index < argc; ++argument_index) {
            const std::string argument = argv[argument_index];
            const std::string continuation_prefix = "--continuation-step=";
            const std::string checkpoint_prefix = "--checkpoint-input=";
            const std::string artifact_prefix = "--artifact-stem=";
            const std::string checkpoint_velocity_prefix =
                "--checkpoint-u-over-uB=";
            const std::string temperature_start_prefix = "--temperature-start=";
            const std::string single_temperature_prefix =
                "--temperature-single-stage=";
#ifdef DCR_NESTED_DIAGNOSTIC
            const std::string diagnostic_gamma_prefix =
                "--diagnostic-gamma=";
            const std::string diagnostic_max_inner_prefix =
                "--max-inner-macro-iterations=";
            const std::string diagnostic_stagnation_prefix =
                "--inner-stagnation-window=";
            const std::string diagnostic_neutral_molecule_step_scale_prefix =
                "--neutral-molecule-pseudo-time-scale=";
            const std::string activity_trace_protection_prefix =
                "--activity-trace-protection-fraction=";
#endif
            if (argument.rfind(continuation_prefix, 0) == 0) {
                continuation_step_override = std::stod(
                    argument.substr(continuation_prefix.size()));
            } else if (argument == "--reconverge-current-lambda") {
                reconverge_current_lambda = true;
            } else if (argument == "--audit-lambda-one-endpoint") {
                audit_lambda_one_endpoint = true;
            } else if (argument == "--solve-physical-reference") {
                solve_physical_reference = true;
            } else if (argument == "--temperature-continuation") {
                temperature_continuation = true;
            } else if (argument == "--temperature-pseudo-time") {
                temperature_pseudo_time = true;
            } else if (argument == "--temperature-neutral-pseudo-time") {
                temperature_neutral_pseudo_time = true;
            } else if (argument == "--temperature-plain-be-once") {
                temperature_plain_backward_euler_once = true;
#ifdef DCR_NESTED_DIAGNOSTIC
            } else if (argument ==
                       "--temperature-fixed-amplitude-diagnostic") {
                temperature_fixed_amplitude_diagnostic = true;
            } else if (argument ==
                       "--fixed-gamma-activity-controller") {
                fixed_gamma_activity_controller = true;
            } else if (argument == "--molecular-residual-postprocess") {
                molecular_residual_postprocess = true;
            } else if (argument == "--activity-trace-deweighting-scan") {
                activity_trace_deweighting_scan = true;
            } else if (argument.rfind(
                           activity_trace_protection_prefix, 0) == 0) {
                activity_trace_protection_fraction = std::stod(
                    argument.substr(activity_trace_protection_prefix.size()));
            } else if (argument.rfind(diagnostic_gamma_prefix, 0) == 0) {
                diagnostic_gamma = std::stod(
                    argument.substr(diagnostic_gamma_prefix.size()));
            } else if (argument.rfind(diagnostic_max_inner_prefix, 0) == 0) {
                diagnostic_max_inner_macro_iterations = std::stoi(
                    argument.substr(diagnostic_max_inner_prefix.size()));
            } else if (argument.rfind(diagnostic_stagnation_prefix, 0) == 0) {
                diagnostic_stagnation_window = std::stoi(
                    argument.substr(diagnostic_stagnation_prefix.size()));
            } else if (argument.rfind(
                           diagnostic_neutral_molecule_step_scale_prefix, 0) == 0) {
                diagnostic_neutral_molecule_step_scale = std::stod(argument.substr(
                    diagnostic_neutral_molecule_step_scale_prefix.size()));
#endif
            } else if (argument.rfind(temperature_start_prefix, 0) == 0) {
                temperature_start = std::stod(
                    argument.substr(temperature_start_prefix.size()));
            } else if (argument.rfind(single_temperature_prefix, 0) == 0) {
                single_temperature_stage = std::stod(
                    argument.substr(single_temperature_prefix.size()));
            } else if (argument.rfind(checkpoint_prefix, 0) == 0) {
                checkpoint_input_path = argument.substr(checkpoint_prefix.size());
            } else if (argument.rfind(artifact_prefix, 0) == 0) {
                artifact_stem_override = argument.substr(artifact_prefix.size());
            } else if (argument.rfind(checkpoint_velocity_prefix, 0) == 0) {
                checkpoint_velocity_fraction = std::stod(
                    argument.substr(checkpoint_velocity_prefix.size()));
            } else {
                throw std::runtime_error("Unknown argument: " + argument);
            }
        }
        if ((checkpoint_input_path.empty() != artifact_stem_override.empty()) ||
            (solve_physical_reference &&
             (checkpoint_input_path.empty() || artifact_stem_override.empty()))) {
            throw std::runtime_error(
                "Checkpoint input and artifact stem must be provided together");
        }
        if (temperature_pseudo_time && !temperature_continuation) {
            throw std::runtime_error(
                "--temperature-pseudo-time requires --temperature-continuation");
        }
        if (temperature_plain_backward_euler_once &&
            (!temperature_continuation || !temperature_pseudo_time)) {
            throw std::runtime_error(
                "--temperature-plain-be-once requires temperature continuation "
                "and pseudo-time");
        }
        if (temperature_neutral_pseudo_time &&
            (!temperature_continuation || !temperature_pseudo_time)) {
            throw std::runtime_error(
                "--temperature-neutral-pseudo-time requires temperature "
                    "continuation and plasma pseudo-time");
        }
#ifdef DCR_NESTED_DIAGNOSTIC
        if (temperature_fixed_amplitude_diagnostic &&
            (!temperature_continuation || !temperature_pseudo_time ||
             !temperature_neutral_pseudo_time ||
             !std::isfinite(single_temperature_stage))) {
            throw std::runtime_error(
                "--temperature-fixed-amplitude-diagnostic requires single-stage "
                "temperature continuation with plasma and neutral pseudo-time");
        }
        if (fixed_gamma_activity_controller &&
            (!temperature_fixed_amplitude_diagnostic ||
             !temperature_continuation || !temperature_pseudo_time ||
             !temperature_neutral_pseudo_time ||
             !std::isfinite(single_temperature_stage))) {
            throw std::runtime_error(
                "--fixed-gamma-activity-controller requires the fixed-amplitude "
                "single-stage diagnostic with plasma and neutral pseudo-time");
        }
        if (temperature_fixed_amplitude_diagnostic &&
            (diagnostic_max_inner_macro_iterations <= 0 ||
             diagnostic_stagnation_window < 3 ||
             diagnostic_stagnation_window >
                 diagnostic_max_inner_macro_iterations)) {
            throw std::runtime_error(
                "Invalid nested diagnostic inner iteration limits");
        }
        if (std::isfinite(diagnostic_gamma) &&
            (!(diagnostic_gamma > 0.0) ||
             !temperature_fixed_amplitude_diagnostic)) {
            throw std::runtime_error(
                "--diagnostic-gamma requires a positive fixed-amplitude diagnostic");
        }
        if (!(diagnostic_neutral_molecule_step_scale > 0.0) ||
            !std::isfinite(diagnostic_neutral_molecule_step_scale) ||
            (diagnostic_neutral_molecule_step_scale != 1.0 &&
             !temperature_fixed_amplitude_diagnostic)) {
            throw std::runtime_error(
                "--neutral-molecule-pseudo-time-scale requires a positive "
                "fixed-amplitude diagnostic");
        }
        if (molecular_residual_postprocess && !temperature_continuation) {
            throw std::runtime_error(
                "--molecular-residual-postprocess requires temperature continuation "
                "checkpoint loading");
        }
        if (activity_trace_deweighting_scan && !temperature_continuation) {
            throw std::runtime_error(
                "--activity-trace-deweighting-scan requires temperature "
                "continuation checkpoint loading");
        }
        if (!(activity_trace_protection_fraction >= 0.0) ||
            !std::isfinite(activity_trace_protection_fraction)) {
            throw std::runtime_error(
                "Activity trace protection fraction must be finite and nonnegative");
        }
#endif

        const auto energy_grid = dcr::physics::make_default_energy_grid();
        std::vector<double> energies;
        std::vector<double> weights;
        energies.reserve(energy_grid.size());
        weights.reserve(energy_grid.size());
        for (const auto& point : energy_grid) {
            energies.push_back(point.energy_eV);
            weights.push_back(point.width_eV);
        }
        dcr::atomic::AtomicData atomic_data(config);

        if (temperature_continuation) {
            if (!std::isfinite(temperature_start) || !(temperature_start > 0.0)) {
                throw std::runtime_error(
                    "Temperature continuation requires --temperature-start=<eV>");
            }
            if (checkpoint_input_path.empty() || artifact_stem_override.empty()) {
                throw std::runtime_error(
                    "Temperature continuation requires checkpoint input and artifact stem");
            }
            if (reconverge_current_lambda || audit_lambda_one_endpoint ||
                solve_physical_reference ||
                std::isfinite(checkpoint_velocity_fraction)) {
                throw std::runtime_error(
                    "Temperature continuation cannot be combined with other solver modes");
            }
            const double configured_target_temperature = config.plasma.Te_eV;
            const double target_temperature = std::isfinite(single_temperature_stage)
                ? single_temperature_stage : configured_target_temperature;
            if (!std::isfinite(target_temperature) ||
                std::abs(config.plasma.Ti_eV - configured_target_temperature) > 1.0e-12 ||
                temperature_start > target_temperature) {
                throw std::runtime_error(
                    "Temperature continuation requires equal target Te=Ti at or above the start");
            }

            auto checkpoint_version = [](const std::string& path) {
                std::ifstream input(path, std::ios::binary);
                std::uint64_t magic = 0;
                std::uint32_t version = 0;
                input.read(reinterpret_cast<char*>(&magic), sizeof(magic));
                input.read(reinterpret_cast<char*>(&version), sizeof(version));
                return input && magic == 0x414c545245433031ULL ? version : 0U;
            };
            const std::string campaign_checkpoint =
                artifact_stem_override + "_checkpoint.bin";
            const std::uint32_t campaign_checkpoint_version =
                checkpoint_version(campaign_checkpoint);
            const bool resuming_campaign =
                campaign_checkpoint_version == 6 ||
                campaign_checkpoint_version == 7;
            const std::string resume_input = resuming_campaign
                ? campaign_checkpoint : checkpoint_input_path;
            const std::string temperature_summary_path =
                artifact_stem_override + "_temperature_summary.csv";

            auto stage_config = [&](double temperature) {
                dcr::io::Config stage = config;
                stage.plasma.Te_eV = temperature;
                stage.plasma.Ti_eV = temperature;
                stage.plasma.electron_temperature_profile.enabled = false;
                stage.plasma.ion_temperature_profile.enabled = false;
                stage.io.verbose_logging = false;
                return stage;
            };
            auto update_last_attempt = [](
                ContinuationCheckpoint& checkpoint,
                double attempted_temperature,
                bool accepted,
                const std::string& status,
                const ContinuationPointResult& point,
                const EndpointAuditResult* audit) {
                checkpoint.last_attempt_temperature = attempted_temperature;
                checkpoint.last_attempt_accepted = accepted;
                checkpoint.last_attempt_iterations = point.outer_iterations;
                checkpoint.last_attempt_amplitude = point.attempted_amplitude;
                checkpoint.last_attempt_mismatch = audit
                    ? audit->ion_trial.mismatch : point.attempted_mismatch;
                checkpoint.last_attempt_joint_residual = audit
                    ? audit->joint_residual : point.fixed_point_residual;
                checkpoint.last_attempt_profile_change = audit
                    ? audit->profile_change : point.profile_change;
                checkpoint.last_attempt_boundary_change = point.boundary_change;
                const bool point_diagnostics_valid =
                    !point.ion_trial.cell_diagnostics.empty();
                checkpoint.last_attempt_epsilon_sigma = audit
                    ? audit->ion_trial.diagnostics.max_conservation_epsilon
                    : (point_diagnostics_valid
                        ? point.ion_trial.diagnostics.max_conservation_epsilon
                        : std::numeric_limits<double>::quiet_NaN());
                checkpoint.last_attempt_epsilon_hminus = audit
                    ? audit->ion_trial.diagnostics.max_hminus_epsilon
                    : (point_diagnostics_valid
                        ? point.ion_trial.diagnostics.max_hminus_epsilon
                        : std::numeric_limits<double>::quiet_NaN());
                checkpoint.last_attempt_epsilon_omitted = audit
                    ? audit->ion_trial.diagnostics.max_omitted_epsilon
                    : (point_diagnostics_valid
                        ? point.ion_trial.diagnostics.max_omitted_epsilon
                        : std::numeric_limits<double>::quiet_NaN());
                checkpoint.last_attempt_minimum_population = audit
                    ? audit->minimum_population : point.minimum_population;
                checkpoint.last_attempt_failing_cell = point.first_failing_cell;
                checkpoint.last_attempt_failing_species = point.first_failing_species;
                checkpoint.last_attempt_rejected_anderson_trials =
                    point.anderson_rejected;
                checkpoint.last_attempt_update_type = point.update_type;
                checkpoint.last_attempt_status = status;
            };
            auto append_temperature_summary = [&temperature_summary_path](
                double from_temperature,
                double attempted_temperature,
                bool accepted,
                const std::string& status,
                const ContinuationPointResult& point,
                const EndpointAuditResult* audit) {
                std::ofstream output(temperature_summary_path, std::ios::app);
                if (!output) {
                    throw std::runtime_error(
                        "Cannot append temperature continuation summary");
                }
                const double mismatch = audit
                    ? audit->ion_trial.mismatch : point.attempted_mismatch;
                const double joint_residual = audit
                    ? audit->joint_residual : point.fixed_point_residual;
                const double profile_change = audit
                    ? audit->profile_change : point.profile_change;
                const SweepDiagnostics& diagnostics = audit
                    ? audit->ion_trial.diagnostics : point.ion_trial.diagnostics;
                output << std::scientific << std::setprecision(12)
                       << (accepted ? 1 : 0) << ','
                       << from_temperature << ','
                       << attempted_temperature << ','
                       << attempted_temperature - from_temperature << ',';
                write_csv_field(output, status);
                output << ',' << point.outer_iterations
                       << ',' << point.attempted_amplitude
                       << ',' << mismatch
                       << ',' << joint_residual
                       << ',' << profile_change
                       << ',' << point.boundary_change
                       << ',' << diagnostics.max_conservation_epsilon
                       << ',' << diagnostics.max_hminus_epsilon
                       << ',' << diagnostics.max_omitted_epsilon
                       << ',' << point.pseudo_time_initial_step
                       << ',' << point.pseudo_time_final_step
                       << ',' << point.pseudo_time_accepted_steps
                       << ',' << point.pseudo_time_rejected_steps
                       << ',' << point.neutral_atom_initial_step
                       << ',' << point.neutral_atom_final_step
                       << ',' << point.neutral_atom_accepted_steps
                       << ',' << point.neutral_atom_rejected_steps
                       << ',' << point.neutral_molecule_initial_step
                       << ',' << point.neutral_molecule_final_step
                       << ',' << point.neutral_molecule_accepted_steps
                       << ',' << point.neutral_molecule_rejected_steps
                       << ',' << point.steady_backward_error
                       << ',' << point.pseudo_time_backward_error
                       << ',' << point.pseudo_time_profile_change
                       << ',' << ((accepted && audit != nullptr) ? 1 : 0)
                       << '\n';
            };

            ContinuationCheckpoint checkpoint;
            {
                dcr::io::Config bootstrap_config = stage_config(temperature_start);
                dcr::state::PlasmaState plasma(
                    bootstrap_config.grid.num_cells, atomic_data.get_total_states());
                plasma.init_Te().setConstant(temperature_start);
                plasma.init_Ti().setConstant(temperature_start);
                plasma.init_ne().setConstant(bootstrap_config.plasma.total_density);
                EEDF eedf(temperature_start, make_eedf_config(bootstrap_config));
                eedf.normalize_on_grid(energies, weights);
                EEDFGridView grid(energies, weights, &eedf);
                const double ion_mass_amu = estimate_ion_mass_amu(bootstrap_config);
                const auto wall = dcr::physics::compute_wall_recycling(
                    bootstrap_config.wall.material, temperature_start, temperature_start,
                    ion_mass_amu, bootstrap_config.wall.sheath_potential_drop);
                auto boundary = dcr::solver::run_boundary_phase(
                    bootstrap_config, atomic_data, plasma, grid, ion_mass_amu, wall);
                if (!boundary.converged) {
                    throw std::runtime_error(
                        "Boundary initialization failed at temperature continuation start");
                }
                AlternatingSweepPrototype bootstrap(
                    std::move(bootstrap_config), atomic_data, plasma, grid,
                    std::move(boundary), wall, resume_input, artifact_stem_override);
                if (!bootstrap.load_temperature_continuation_checkpoint(checkpoint)) {
                    throw std::runtime_error(
                        "Could not load temperature continuation checkpoint: " +
                        resume_input);
                }
            }
#ifdef DCR_NESTED_DIAGNOSTIC
            if (molecular_residual_postprocess ||
                activity_trace_deweighting_scan) {
                if (!checkpoint.provisional_valid ||
                    !std::isfinite(checkpoint.provisional_temperature)) {
                    throw std::runtime_error(
                        "Checkpoint has no provisional state for molecular analysis");
                }
                dcr::io::Config analysis_config = stage_config(
                    checkpoint.provisional_temperature);
                dcr::state::PlasmaState analysis_plasma(
                    analysis_config.grid.num_cells, atomic_data.get_total_states());
                analysis_plasma.init_Te().setConstant(
                    checkpoint.provisional_temperature);
                analysis_plasma.init_Ti().setConstant(
                    checkpoint.provisional_temperature);
                analysis_plasma.init_ne().setConstant(
                    analysis_config.plasma.total_density);
                EEDF analysis_eedf(
                    checkpoint.provisional_temperature,
                    make_eedf_config(analysis_config));
                analysis_eedf.normalize_on_grid(energies, weights);
                EEDFGridView analysis_grid(
                    energies, weights, &analysis_eedf);
                const double analysis_ion_mass_amu =
                    estimate_ion_mass_amu(analysis_config);
                const auto analysis_wall = dcr::physics::compute_wall_recycling(
                    analysis_config.wall.material,
                    checkpoint.provisional_temperature,
                    checkpoint.provisional_temperature,
                    analysis_ion_mass_amu,
                    analysis_config.wall.sheath_potential_drop);
                auto analysis_boundary = dcr::solver::run_boundary_phase(
                    analysis_config, atomic_data, analysis_plasma,
                    analysis_grid, analysis_ion_mass_amu, analysis_wall);
                if (!analysis_boundary.converged) {
                    throw std::runtime_error(
                        "Boundary reconstruction failed for molecular analysis");
                }
                AlternatingSweepPrototype analysis(
                    std::move(analysis_config), atomic_data, analysis_plasma,
                    analysis_grid, std::move(analysis_boundary), analysis_wall,
                    resume_input, artifact_stem_override);
                const std::uint64_t hash_before =
                    checkpoint_state_hash(checkpoint);
                if (molecular_residual_postprocess) {
                    analysis.print_saved_molecular_residual_diagnostic(checkpoint);
                }
                analysis.print_saved_activity_spectrum_diagnostic(
                    checkpoint, activity_trace_deweighting_scan,
                    activity_trace_protection_fraction);
                const std::uint64_t hash_after =
                    checkpoint_state_hash(checkpoint);
                std::cout << "SAVED_STATE_POSTPROCESS_AUDIT"
                          << " mode=" << (activity_trace_deweighting_scan
                              ? "activity_trace_deweighting_scan"
                              : "molecular_residual")
                          << " state_hash_before=" << hash_before
                          << " state_hash_after=" << hash_after
                          << " state_unchanged="
                          << (hash_before == hash_after ? 1 : 0)
                          << " checkpoint_written=0"
                          << " solver_iterations=0"
                          << "\n";
                return hash_before == hash_after ? 0 : 1;
            }
#endif
            if (std::isfinite(continuation_step_override) && !resuming_campaign) {
                if (!(continuation_step_override > 0.0)) {
                    throw std::runtime_error(
                        "Temperature continuation step must be positive");
                }
                checkpoint.next_temperature_step = continuation_step_override;
            }
            if (!resuming_campaign) {
                std::ofstream output(temperature_summary_path, std::ios::trunc);
                if (!output) {
                    throw std::runtime_error(
                        "Cannot initialize temperature continuation summary");
                }
                output << "accepted,from_temperature,attempted_temperature,"
                          "temperature_step,status,macro_cycles,amplitude,mismatch,"
                          "joint_residual,profile_change,boundary_change,epsilon_sigma,"
                           "epsilon_hminus,epsilon_omitted,dtau_initial,dtau_final,"
                           "plasma_steps_accepted,plasma_steps_rejected,"
                           "neutral_dtau_A_initial,neutral_dtau_A_final,"
                           "neutral_A_steps_accepted,neutral_A_steps_rejected,"
                           "neutral_dtau_M_initial,neutral_dtau_M_final,"
                           "neutral_M_steps_accepted,neutral_M_steps_rejected,"
                           "steady_residual,"
                          "be_residual,pseudo_time_change,direct_steady_audit\n";
            }
            if (std::abs(checkpoint.lambda - 1.0) > 1.0e-14 ||
                !std::isfinite(checkpoint.current_temperature) ||
                checkpoint.current_temperature < temperature_start - 1.0e-12 ||
                checkpoint.current_temperature > target_temperature + 1.0e-12) {
                throw std::runtime_error(
                    "Temperature continuation checkpoint is not an accepted lambda=1 stage");
            }

            std::cout << std::scientific << std::setprecision(12)
                      << "TEMPERATURE_CONTINUATION_START accepted_T="
                      << checkpoint.current_temperature
                      << " target_T=" << target_temperature
                      << " next_dt=" << checkpoint.next_temperature_step
                      << " checkpoint_input=" << std::quoted(resume_input)
                      << " checkpoint_output=" << std::quoted(campaign_checkpoint)
                      << "\n";

            while (true) {
                const double accepted_temperature = checkpoint.current_temperature;
                const bool at_target = std::abs(
                    accepted_temperature - target_temperature) <= 1.0e-12;
                double proposal_temperature = accepted_temperature;
                double controlling_step = checkpoint.next_temperature_step;
                if (!(controlling_step > 0.0) || !std::isfinite(controlling_step)) {
                    throw std::runtime_error("Invalid temperature continuation step");
                }
                if (!at_target) {
                    if (std::isfinite(single_temperature_stage)) {
                        proposal_temperature = target_temperature;
                    } else {
                        proposal_temperature = std::min(
                            target_temperature,
                            accepted_temperature + controlling_step);
                    }
                }

                dcr::io::Config proposal_config = stage_config(proposal_temperature);
                dcr::state::PlasmaState plasma(
                    proposal_config.grid.num_cells, atomic_data.get_total_states());
                plasma.init_Te().setConstant(proposal_temperature);
                plasma.init_Ti().setConstant(proposal_temperature);
                plasma.init_ne().setConstant(proposal_config.plasma.total_density);
                EEDF eedf(proposal_temperature, make_eedf_config(proposal_config));
                eedf.normalize_on_grid(energies, weights);
                EEDFGridView grid(energies, weights, &eedf);
                const double ion_mass_amu = estimate_ion_mass_amu(proposal_config);
                const auto wall = dcr::physics::compute_wall_recycling(
                    proposal_config.wall.material, proposal_temperature,
                    proposal_temperature, ion_mass_amu,
                    proposal_config.wall.sheath_potential_drop);
                auto boundary = dcr::solver::run_boundary_phase(
                    proposal_config, atomic_data, plasma, grid, ion_mass_amu, wall);
                const bool proposal_boundary_converged = boundary.converged;
                AlternatingSweepPrototype prototype(
                    std::move(proposal_config), atomic_data, plasma, grid,
                    std::move(boundary), wall, resume_input, artifact_stem_override);

                if (at_target) {
                    EndpointAuditResult audit =
                        prototype.audit_temperature_checkpoint(checkpoint);
                    if (!prototype.temperature_audit_accepted(audit)) {
                        throw std::runtime_error(
                            "Accepted final temperature checkpoint failed endpoint audit");
                    }
                    prototype.export_temperature_endpoint(checkpoint, audit);
                    std::cout << "TEMPERATURE_CONTINUATION_FINAL accepted=1 T="
                              << checkpoint.current_temperature
                              << " amplitude=" << checkpoint.amplitude
                              << " mismatch=" << audit.ion_trial.mismatch
                              << " joint_residual=" << audit.joint_residual
                              << " checkpoint=" << std::quoted(campaign_checkpoint)
                              << "\n";
                    return 0;
                }

                ContinuationPointResult point;
                EndpointAuditResult audit;
                bool have_audit = false;
                bool accepted = false;
                std::string status = "solver_rejected";
                try {
                    if (!proposal_boundary_converged) {
                        throw std::runtime_error(
                            "Boundary initialization failed at proposed temperature");
                    }
                    ContinuationCheckpoint stage_checkpoint = checkpoint;
                    const bool resume_provisional = checkpoint.provisional_valid &&
                        std::abs(checkpoint.provisional_temperature -
                            proposal_temperature) <= 1.0e-12;
                    if (resume_provisional) {
                        stage_checkpoint.amplitude =
                            checkpoint.provisional_amplitude;
                        stage_checkpoint.profiles =
                            checkpoint.provisional_profiles;
                        stage_checkpoint.upstream_boundary =
                            checkpoint.provisional_upstream_boundary;
                        std::cout << "TEMPERATURE_PROVISIONAL_RESUME"
                                  << " T=" << proposal_temperature
                                  << " amplitude="
                                  << stage_checkpoint.amplitude
                                  << "\n";
                    }
#ifdef DCR_NESTED_DIAGNOSTIC
                    if (temperature_fixed_amplitude_diagnostic &&
                        std::isfinite(diagnostic_gamma)) {
                        stage_checkpoint.amplitude = diagnostic_gamma;
                        std::cout << "NESTED_DIAGNOSTIC_GAMMA_OVERRIDE"
                                  << " Gamma_up=" << diagnostic_gamma
                                  << " ell=" << std::log(diagnostic_gamma)
                                  << "\n";
                    }
#endif
                    const Vector predictor =
                        (resume_provisional ||
                         temperature_fixed_amplitude_diagnostic)
                        ? prototype.temperature_joint_state(stage_checkpoint)
                        : prototype.temperature_predictor(
                            checkpoint, proposal_temperature);
                    if (temperature_fixed_amplitude_diagnostic) {
                        prototype.print_temperature_cell34_molecular_diagnostic(
                            stage_checkpoint.profiles, "fixed_amplitude_initial");
                    }
                    point = prototype.solve_temperature_stage(
                        stage_checkpoint, predictor, audit, temperature_pseudo_time,
                        temperature_plain_backward_euler_once,
                        temperature_neutral_pseudo_time, &checkpoint,
                        temperature_fixed_amplitude_diagnostic,
                        diagnostic_max_inner_macro_iterations,
                        diagnostic_stagnation_window,
                        diagnostic_neutral_molecule_step_scale,
                        fixed_gamma_activity_controller,
                        activity_trace_protection_fraction);
                    if (point.converged) {
                        have_audit = true;
                        if (temperature_fixed_amplitude_diagnostic) {
                            const bool inner_audit_accepted = prototype
                                .temperature_inner_audit_accepted(audit);
                            accepted = false;
                            status = inner_audit_accepted
                                ? "inner_solve_converged_and_audited"
                                : "converged_map_but_audit_failed";
                            if (!inner_audit_accepted && point.failure.empty()) {
                                point.failure =
                                    "independent fixed-amplitude inner audit rejected stage";
                            }
                        } else {
                            accepted = prototype.temperature_audit_accepted(audit);
                            status = accepted
                                ? "accepted" : "endpoint_audit_rejected";
                            if (!accepted && point.failure.empty()) {
                                point.failure =
                                    "independent endpoint audit rejected stage";
                            }
                        }
                    } else if (!point.failure.empty()) {
                        if (temperature_fixed_amplitude_diagnostic) {
                            status = point.failure == "inner_solve_stalled"
                                ? "inner_solve_stalled"
                                : (point.failure ==
                                       "be_steady_operator_mismatch"
                                    ? "be_steady_operator_mismatch"
                                    : "inner_solve_failed");
                        } else {
                            status = point.failure;
                        }
                    }
                } catch (const std::exception& error) {
                    point.failure = error.what();
                    status = point.failure;
                    if (!std::isfinite(point.attempted_amplitude)) {
                        point.attempted_amplitude = checkpoint.amplitude;
                    }
                    const std::string cell_marker = "Reverse cell ";
                    const size_t cell_begin = point.failure.find(cell_marker);
                    if (cell_begin != std::string::npos) {
                        try {
                            point.first_failing_cell = std::stoi(
                                point.failure.substr(cell_begin + cell_marker.size()));
                        } catch (...) {
                            point.first_failing_cell = -1;
                        }
                    }
                    const std::string species_marker = "species_gi=";
                    const size_t species_begin = point.failure.find(species_marker);
                    if (species_begin != std::string::npos) {
                        try {
                            point.first_failing_species = std::stoi(
                                point.failure.substr(
                                    species_begin + species_marker.size()));
                        } catch (...) {
                            point.first_failing_species = -1;
                        }
                    }
                }

                const double mismatch = have_audit
                    ? audit.ion_trial.mismatch : point.attempted_mismatch;
                const double joint_residual = have_audit
                    ? audit.joint_residual : point.fixed_point_residual;
                const double profile_change = have_audit
                    ? audit.profile_change : point.profile_change;
                const SweepDiagnostics& diagnostics = have_audit
                    ? audit.ion_trial.diagnostics : point.ion_trial.diagnostics;
                const double minimum_population = have_audit
                    ? audit.minimum_population : point.minimum_population;
                update_last_attempt(
                    checkpoint, proposal_temperature, accepted, status,
                    point, have_audit ? &audit : nullptr);

                std::cout << "TEMPERATURE_STAGE"
                          << " accepted=" << (accepted ? 1 : 0)
                          << " from_T=" << accepted_temperature
                          << " attempted_T=" << proposal_temperature
                          << " dt=" << proposal_temperature - accepted_temperature
                          << " status=" << std::quoted(status)
                          << " failure_reason="
                          << std::quoted(point.failure)
                          << " coupled_iterations=" << point.outer_iterations
                          << " amplitude=" << point.attempted_amplitude
                          << " mismatch=" << mismatch
                          << " joint_residual=" << joint_residual
                          << " profile_change=" << profile_change
                          << " boundary_change=" << point.boundary_change
                          << " epsilon_Sigma="
                          << diagnostics.max_conservation_epsilon
                          << " epsilon_Hminus=" << diagnostics.max_hminus_epsilon
                          << " epsilon_omitted=" << diagnostics.max_omitted_epsilon
                          << " minimum_population=" << minimum_population
                          << " failing_cell=" << point.first_failing_cell
                          << " failing_species=" << std::quoted(
                              prototype.temperature_species_label(
                                  point.first_failing_species))
                          << " update_type=" << point.update_type
                          << " accepted_anderson_updates="
                          << point.anderson_accepted
                          << " rejected_anderson_trials="
                          << point.anderson_rejected
                          << " relaxed_fallbacks="
                          << point.relaxed_fallbacks
                          << " rejected_macro_trials="
                          << point.rejected_macro_trials
                          << " pseudo_time=" << (temperature_pseudo_time ? 1 : 0)
                          << " dtau_start=" << point.pseudo_time_initial_step
                          << " dtau_final=" << point.pseudo_time_final_step
                          << " pseudo_time_steps_accepted="
                          << point.pseudo_time_accepted_steps
                          << " pseudo_time_steps_rejected="
                          << point.pseudo_time_rejected_steps
                          << " neutral_pseudo_time="
                          << (temperature_neutral_pseudo_time ? 1 : 0)
                          << " neutral_dtau_A_start="
                          << point.neutral_atom_initial_step
                          << " neutral_dtau_A_final="
                          << point.neutral_atom_final_step
                          << " neutral_A_steps_accepted="
                          << point.neutral_atom_accepted_steps
                          << " neutral_A_steps_rejected="
                          << point.neutral_atom_rejected_steps
                          << " neutral_dtau_M_start="
                          << point.neutral_molecule_initial_step
                          << " neutral_dtau_M_final="
                          << point.neutral_molecule_final_step
                          << " neutral_M_steps_accepted="
                          << point.neutral_molecule_accepted_steps
                          << " neutral_M_steps_rejected="
                          << point.neutral_molecule_rejected_steps
                          << " steady_residual=" << point.steady_backward_error
                          << " pseudo_time_residual="
                          << point.pseudo_time_backward_error
                          << " pseudo_time_change="
                          << point.pseudo_time_profile_change
#ifdef DCR_NESTED_DIAGNOSTIC
                          << " local_dtau_P_min="
                          << point.local_plasma_step_minimum
                          << " local_dtau_P_q10="
                          << point.local_plasma_step_q10
                          << " local_dtau_P_median="
                          << point.local_plasma_step_median
                          << " local_dtau_P_q90="
                          << point.local_plasma_step_q90
                          << " local_dtau_P_max="
                          << point.local_plasma_step_maximum
                          << " local_dtau_P_reductions="
                          << point.local_plasma_reductions
#endif
                          << " direct_steady_audit="
                          << ((accepted && have_audit) ? 1 : 0)
                          << "\n";
                append_temperature_summary(
                    accepted_temperature, proposal_temperature, accepted, status,
                    point, have_audit ? &audit : nullptr);

                if (temperature_fixed_amplitude_diagnostic) {
                    std::cout << "NESTED_GAMMA_TRIAL"
                              << " Gamma_trial="
                              << point.attempted_amplitude
                              << " ell=" << (point.attempted_amplitude > 0.0
                                  ? std::log(point.attempted_amplitude)
                                  : std::numeric_limits<double>::quiet_NaN())
                              << " status=" << std::quoted(status)
                              << " H_valid="
                              << ((point.converged && have_audit &&
                                   prototype.temperature_inner_audit_accepted(
                                       audit)) ? 1 : 0)
                              << " H=" << ((point.converged && have_audit &&
                                   prototype.temperature_inner_audit_accepted(
                                       audit))
                                  ? audit.ion_trial.mismatch
                                  : std::numeric_limits<double>::quiet_NaN())
                              << " failure_reason="
                              << std::quoted(point.failure)
                              << "\n";
                }

                if (temperature_fixed_amplitude_diagnostic &&
                    point.converged && have_audit) {
                    const bool inner_audit_accepted =
                        prototype.temperature_inner_audit_accepted(audit);
                    prototype.print_temperature_cell34_molecular_diagnostic(
                        point.profiles, "fixed_amplitude_final");
                    std::cout << "FIXED_AMPLITUDE_DIAGNOSTIC_RESULT"
                              << " audited="
                              << (inner_audit_accepted ? 1 : 0)
                              << " T=" << proposal_temperature
                              << " ell=" << std::log(point.attempted_amplitude)
                              << " Gamma_up=" << point.attempted_amplitude
                              << " H=" << audit.ion_trial.mismatch
                              << " neutral_residual="
                              << audit.neutral_joint_residual
                              << " profile_change=" << audit.profile_change
                              << " boundary_change=" << audit.boundary_change
                              << " steady_residual="
                              << audit.ion_trial.diagnostics
                                  .max_steady_backward_error
                              << " epsilon_Sigma="
                              << audit.ion_trial.diagnostics
                                  .max_conservation_epsilon
                              << " epsilon_Hminus="
                              << audit.ion_trial.diagnostics
                                  .max_hminus_epsilon
                              << " epsilon_omitted="
                              << audit.ion_trial.diagnostics
                                  .max_omitted_epsilon
                              << " minimum_population="
                              << audit.minimum_population
                              << " production_checkpoint_written=0"
                              << "\n";
                    if (inner_audit_accepted) {
                        ContinuationCheckpoint diagnostic_checkpoint = checkpoint;
                        diagnostic_checkpoint.provisional_valid = true;
                        diagnostic_checkpoint.provisional_temperature =
                            proposal_temperature;
                        diagnostic_checkpoint.provisional_amplitude =
                            point.attempted_amplitude;
                        diagnostic_checkpoint.provisional_profiles = point.profiles;
                        diagnostic_checkpoint.provisional_upstream_boundary =
                            point.ion_trial.upstream_boundary;
                        diagnostic_checkpoint.last_attempt_temperature =
                            proposal_temperature;
                        diagnostic_checkpoint.last_attempt_accepted = false;
                        diagnostic_checkpoint.last_attempt_status =
                            "diagnostic_inner_audited_not_production";
                        prototype.save_temperature_continuation_checkpoint(
                            diagnostic_checkpoint);
                        std::cout << "NESTED_DIAGNOSTIC_STATE_SAVED"
                                  << " role=warm_start_only"
                                  << " accepted_temperature="
                                  << diagnostic_checkpoint.current_temperature
                                  << " provisional_temperature="
                                  << diagnostic_checkpoint
                                      .provisional_temperature
                                  << " production_checkpoint_written=0"
                                  << "\n";
                    }
                    return inner_audit_accepted ? 0 : 2;
                }

                if (!accepted) {
                    const bool resumable_iteration_boundary =
                        std::isfinite(point.attempted_amplitude) &&
                        point.attempted_amplitude > 0.0 &&
                        std::isfinite(point.minimum_population) &&
                        point.minimum_population >= 0.0 &&
                        !point.profiles.background.empty() &&
                        point.ion_trial.upstream_boundary
                            .positive_ion_composition.size() > 0;
                    checkpoint.provisional_valid = resumable_iteration_boundary;
                    if (resumable_iteration_boundary) {
                        checkpoint.provisional_temperature = proposal_temperature;
                        checkpoint.provisional_amplitude = point.attempted_amplitude;
                        checkpoint.provisional_profiles = point.profiles;
                        checkpoint.provisional_upstream_boundary =
                            point.ion_trial.upstream_boundary;
                        std::cout << "TEMPERATURE_PROVISIONAL_CHECKPOINT"
                                  << " T=" << proposal_temperature
                                  << " macro=" << point.outer_iterations
                                  << " amplitude=" << point.attempted_amplitude
                                  << " checkpoint="
                                  << std::quoted(campaign_checkpoint)
                                  << "\n";
                    }
                    if (std::isfinite(single_temperature_stage)) {
                        checkpoint.last_step_rejected = true;
                        prototype.save_temperature_continuation_checkpoint(checkpoint);
                        std::cout << "TEMPERATURE_CONTINUATION_STOP accepted_T="
                                  << checkpoint.current_temperature
                                  << " attempted_T=" << proposal_temperature
                                  << " next_dt=" << checkpoint.next_temperature_step
                                  << " status=single_stage_diagnostic"
                                  << " failure=" << std::quoted(point.failure)
                                  << "\n";
                        return 2;
                    }
                    const double failed_step =
                        proposal_temperature - accepted_temperature;
                    const bool failed_at_minimum_step =
                        failed_step <= 0.01 + 1.0e-12;
                    checkpoint.next_temperature_step = failed_at_minimum_step
                        ? 0.01
                        : std::max(0.01, 0.5 * failed_step);
                    checkpoint.last_step_rejected = true;
                    prototype.save_temperature_continuation_checkpoint(checkpoint);
                    if (failed_at_minimum_step) {
                        std::cout << "TEMPERATURE_CONTINUATION_STOP accepted_T="
                                  << checkpoint.current_temperature
                                  << " attempted_T=" << proposal_temperature
                                  << " next_dt=" << checkpoint.next_temperature_step
                                  << " status=minimum_temperature_step"
                                  << " failure=" << std::quoted(point.failure)
                                  << "\n";
                        return 2;
                    }
                    continue;
                }

                const Vector previous_joint_state =
                    prototype.temperature_joint_state(checkpoint);
                checkpoint.provisional_valid = false;
                checkpoint.previous_temperature = accepted_temperature;
                checkpoint.previous_joint_state = previous_joint_state;
                checkpoint.current_temperature = proposal_temperature;
                checkpoint.lambda = 1.0;
                checkpoint.previous_lambda = 1.0;
                checkpoint.amplitude = point.amplitude;
                checkpoint.previous_amplitude = point.amplitude;
                checkpoint.profiles = point.profiles;
                checkpoint.upstream_boundary = audit.ion_trial.upstream_boundary;
                checkpoint.last_step_rejected = false;
                const bool easy_temperature_stage = temperature_pseudo_time
                    ? point.outer_iterations <= 100 &&
                        point.pseudo_time_rejected_steps == 0
                    : point.outer_iterations <= 15;
                if (easy_temperature_stage) {
                    checkpoint.next_temperature_step = 1.5 * controlling_step;
                } else {
                    checkpoint.next_temperature_step = controlling_step;
                }
                prototype.save_temperature_continuation_checkpoint(checkpoint);
            }
        }

        dcr::state::PlasmaState plasma(config.grid.num_cells, atomic_data.get_total_states());
        plasma.init_Te().setConstant(config.plasma.Te_eV);
        plasma.init_Ti().setConstant(config.plasma.Ti_eV);
        plasma.init_ne().setConstant(config.plasma.total_density);
        EEDF eedf(config.plasma.Te_eV, make_eedf_config(config));
        eedf.normalize_on_grid(energies, weights);
        EEDFGridView grid(energies, weights, &eedf);
        const double ion_mass_amu = estimate_ion_mass_amu(config);
        const auto wall = dcr::physics::compute_wall_recycling(
            config.wall.material, config.plasma.Te_eV, config.plasma.Ti_eV,
            ion_mass_amu, config.wall.sheath_potential_drop);
        auto boundary = dcr::solver::run_boundary_phase(
            config, atomic_data, plasma, grid, ion_mass_amu, wall);
        if (!boundary.converged) throw std::runtime_error("Boundary initialization failed");

        AlternatingSweepPrototype prototype(
            std::move(config), atomic_data, plasma, grid, std::move(boundary), wall,
            std::move(checkpoint_input_path), std::move(artifact_stem_override));
        return prototype.run(
            continuation_step_override, reconverge_current_lambda,
            audit_lambda_one_endpoint, solve_physical_reference,
            checkpoint_velocity_fraction);
    } catch (const std::exception& error) {
        std::cerr << "DCR_AlternatingSweep: " << error.what() << "\n";
        return 1;
    }
}
