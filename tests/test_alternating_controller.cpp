#include <cmath>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <vector>

#include "solver/core/AlternatingController.hpp"

namespace {

using dcr::base::Vector;
using dcr::solver::AndersonSample;
using dcr::solver::AuditedAmplitudePoint;
using dcr::solver::AmplitudeClipSource;
using dcr::solver::FixedPtcMapParameters;
using dcr::solver::LogAmplitudeBoundKind;
using dcr::solver::LogAmplitudeBounds;
using dcr::solver::LogAmplitudeTrustRegion;

void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

void amplitude_clipping_regression() {
    const double current = std::log(1.408146739557e20);
    const double upper = std::log(1.410604033434e20);
    const LogAmplitudeBounds trust{
        current - 1.0, upper,
        LogAmplitudeBoundKind::algorithmic_trust_region};
    const LogAmplitudeBounds physical{
        std::log(std::numeric_limits<double>::min()),
        std::log(std::numeric_limits<double>::max()),
        LogAmplitudeBoundKind::physical_admissibility};
    const auto step = dcr::solver::backtrack_log_amplitude_step(
        current, 0.0470016098, trust, physical, 10.0,
        [](double candidate) -> std::optional<double> {
            return 9.0 + 0.01 * candidate;
        });
    require(step.valid, "Clipped amplitude step is invalid");
    require(!step.raw_candidate_feasible, "Raw amplitude candidate was not classified as infeasible");
    require(step.clipped, "Amplitude step was not clipped");
    require(step.clip_source == AmplitudeClipSource::algorithmic_trust_region,
        "Amplitude trust clipping source was not classified");
    require(step.alpha_feasible < 1.0, "Feasible alpha was not reduced");
    require(step.delta_log != 0.0470016098, "Controller retried the raw infeasible step");
    const double amplitude = std::exp(current + step.delta_log);
    require(std::isfinite(amplitude) && amplitude > 0.0, "Clipped amplitude is not finite and positive");
    require(amplitude < 1.410604033434e20, "Clipped amplitude is not strictly inside the bound");
}

void feasible_step_identity() {
    const LogAmplitudeBounds trust{-2.0, 2.0,
        LogAmplitudeBoundKind::algorithmic_trust_region};
    const LogAmplitudeBounds physical{-100.0, 100.0,
        LogAmplitudeBoundKind::physical_admissibility};
    const auto step = dcr::solver::feasible_log_amplitude_step(
        0.0, 0.25, trust, physical);
    require(step.valid && step.raw_candidate_feasible, "Feasible step was rejected");
    require(step.alpha_feasible == 1.0, "Feasible step alpha changed");
    require(step.delta_log == 0.25, "Feasible proposal changed before merit backtracking");
}

void trust_region_recenter() {
    LogAmplitudeTrustRegion trust;
    trust.center = 1.0;
    trust.radius = 0.5;
    const LogAmplitudeBounds physical{-10.0, 10.0,
        LogAmplitudeBoundKind::physical_admissibility};
    trust.accept(1.4, true);
    require(trust.center == 1.4, "Algorithmic trust center was not recentered");
    require(trust.radius > 0.5, "Successful trust radius did not expand");
    require(physical.lower == -10.0 && physical.upper == 10.0, "Physical bounds were mutated");
    const double accepted_radius = trust.radius;
    trust.reject();
    require(trust.center == 1.4, "Trust rejection moved the center");
    require(trust.radius < accepted_radius, "Trust rejection did not shrink the radius");
}

void unstable_finite_ptc_map_anderson_endgame() {
    constexpr double delta_tau = 0.1;
    auto finite_ptc_map = [](double old_state) {
        // Backward Euler for dx/dtau = 11 (x - 1). Its finite-step map is
        // unstable at this delta_tau, while its fixed point remains x=1.
        return (old_state - 11.0 * delta_tau) /
            (1.0 - 11.0 * delta_tau);
    };
    auto residual = [&](double x) { return finite_ptc_map(x) - x; };
    std::vector<AndersonSample> history;
    for (double x : {0.9, 0.8}) {
        Vector state(1);
        Vector r(1);
        state(0) = x;
        r(0) = residual(x);
        history.push_back({state, r});
    }
    Vector current(1);
    Vector current_residual(1);
    current(0) = 0.8;
    current_residual(0) = residual(current(0));
    const double plain = current(0) + current_residual(0);
    require(std::abs(plain - 1.0) > std::abs(current(0) - 1.0),
        "Synthetic finite-PTC map is not unstable");
    const auto candidate = dcr::solver::anderson_candidate(
        history, current, current_residual, 5, 1.0, 1.0e-12);
    require(candidate.has_value(), "Anderson candidate was unavailable");
    require(std::abs((*candidate)(0) - 1.0) < 1.0e-9, "Anderson did not reach the synthetic fixed point");
}

void fixed_map_and_nonmonotone_watchdog() {
    const FixedPtcMapParameters frozen{1.0, 2.0, 3.0, 4.0, 5.0};
    FixedPtcMapParameters changed = frozen;
    require(dcr::solver::same_fixed_ptc_map(frozen, changed),
        "Identical finite-PTC maps were classified as changed");
    changed.atom_step *= 0.5;
    require(!dcr::solver::same_fixed_ptc_map(frozen, changed),
        "A timestep change did not invalidate the finite-PTC map");

    dcr::solver::NonmonotoneMeritWatchdog merit(5, 12);
    for (double value : {10.0, 9.0, 8.0, 7.0, 6.0}) merit.accept(value);
    require(merit.accepts(9.5, 6.0),
        "Non-monotone merit rejected a value within the five-step window");
    require(!merit.accepts(10.5, 6.0),
        "Non-monotone merit accepted growth beyond its window");
    for (int rejection = 1; rejection < 12; ++rejection) {
        require(!merit.reject(), "Watchdog fired before twelve rejections");
    }
    require(merit.reject(), "Watchdog did not fire on rejection twelve");

    const auto increased = dcr::solver::increased_pseudo_time_step(
        1.0, 100.0);
    require(increased.has_value() && *increased == 10.0,
        "Pseudo-time calibration did not increase an interior step");
    require(!dcr::solver::increased_pseudo_time_step(100.0, 100.0).has_value(),
        "Pseudo-time calibration retried an unchanged ceiling step");
}

void positivity_safe_ptc_anderson_blend() {
    Vector ptc(4);
    Vector anderson(4);
    ptc << 3.0, 2.0, 4.0, 1.0;
    anderson << 3.0, -2.0, 5.0, 0.5;
    const auto limited = dcr::solver::positivity_safe_blend(
        ptc, anderson, 1);
    require(limited.positivity_limited,
        "Negative Anderson population did not limit the blend");
    require(std::abs(limited.beta - 0.475) < 1.0e-14,
        "Positivity blend did not apply the fraction-to-boundary safety factor");
    const Vector candidate = ptc + limited.beta * (anderson - ptc);
    require(candidate.tail(3).minCoeff() > 0.0,
        "Positivity-limited candidate is not strictly positive");

    anderson << 3.0, 3.0, 5.0, 2.0;
    const auto unrestricted = dcr::solver::positivity_safe_blend(
        ptc, anderson, 1);
    require(!unrestricted.positivity_limited,
        "Positive Anderson candidate was classified as positivity-limited");
    require(std::abs(unrestricted.beta - 0.95) < 1.0e-14,
        "Unrestricted Anderson blend did not apply the safety factor");
}

void activity_trace_controller_behavior() {
    const auto negligible = dcr::solver::classify_activity_trace_row(
        1.0, 100.0, {0.0}, {1000.0}, 1.0e-3);
    require(negligible.trace && !negligible.protected_pathway,
        "Negligible unprotected activity was not classified trace");

    Vector previous(2);
    Vector next(2);
    previous << 1.0, 100.0;
    next << 2.0, 101.0;
    const auto change = dcr::solver::maximum_nontrace_relative_change(
        previous, next, {true, false});
    require(change.index == 1 && std::abs(change.maximum - 1.0 / 101.0) < 1.0e-14,
        "A trace row controlled the masked state-change limiter");

    const auto protected_sparse = dcr::solver::classify_activity_trace_row(
        1.0, 100.0, {2.0}, {1000.0}, 1.0e-3);
    require(!protected_sparse.trace && protected_sparse.protected_pathway,
        "A sparse pathway-important row was de-weighted");

    Vector ptc(3);
    Vector candidate(3);
    ptc << 5.0, 2.0, 3.0;
    candidate << 5.0, -1.0, 4.0;
    const auto restored = dcr::solver::restore_negative_trace_components(
        candidate, ptc, {true, false}, 1);
    require(!restored.nontrace_negative &&
            restored.restored_trace_components == 1 &&
            restored.nontrace_negative_components == 0 &&
            restored.state(1) == ptc(1),
        "Negative trace Anderson component did not recover the PTC value");
    candidate << 5.0, 1.0, -1.0;
    const auto rejected = dcr::solver::restore_negative_trace_components(
        candidate, ptc, {true, false}, 1);
    require(rejected.nontrace_negative &&
            rejected.nontrace_negative_components == 1,
        "Negative non-trace Anderson component bypassed positivity handling");

    Vector residual(3);
    Vector activity(3);
    Vector floor(3);
    residual << 1.0, 2.0, 3.0;
    activity << 1.0, 2.0, 3.0;
    floor << 10.0, 10.0, 10.0;
    const auto audit = dcr::solver::evaluate_complete_hybrid_audit(
        residual, activity, floor, {true, false, true});
    require(audit.total_rows == 3 && audit.trace_rows == 2 &&
            audit.nontrace_rows == 1,
        "Hybrid audit failed to evaluate every trace and non-trace row");

    const std::vector<double> cancellation_contributions{60.0, -59.0};
    const double signed_total = cancellation_contributions[0] +
        cancellation_contributions[1];
    const double gross_total = std::abs(cancellation_contributions[0]) +
        std::abs(cancellation_contributions[1]);
    const auto cancellation_protected =
        dcr::solver::classify_activity_trace_row(
            1.0, 100.0, {cancellation_contributions[0]},
            {gross_total}, 0.4);
    require(std::abs(signed_total) == 1.0 &&
            cancellation_protected.protected_pathway,
        "Pathway protection used signed cancellation instead of gross throughput");
}

void fold_diagnostic() {
    AuditedAmplitudePoint lower{5.0, -0.01, -1.0e-4, 1.0e-8, true};
    AuditedAmplitudePoint upper{5.0, 0.01, -1.0e-4, 1.0e-8, true};
    const auto secant = dcr::solver::audited_secant_slope(lower, upper);
    require(secant.has_value(), "Audited secant slope was unavailable");
    require(std::abs(*secant) < 1.0e-12, "Audited secant slope is incorrect");
    const auto indicators = dcr::solver::fold_indicators(
        {1.0, 0.05, -0.04, 0.03}, 1.0, 0.005, 0.1, 20.0, 1.0);
    require(indicators.derivative_collapsed, "Fold derivative collapse was not detected");
    require(indicators.derivative_sign_changed, "Fold derivative sign change was not detected");
    require(indicators.possible_fold(), "Combined fold indicators were not detected");
    const auto healthy = dcr::solver::fold_indicators(
        {1.0, 0.9, 0.8}, 1.0, 0.1, 0.1, 1.0, 1.0);
    require(!healthy.possible_fold(), "Healthy synthetic branch was classified as a fold");
}

} // namespace

int main() {
    amplitude_clipping_regression();
    feasible_step_identity();
    trust_region_recenter();
    unstable_finite_ptc_map_anderson_endgame();
    fixed_map_and_nonmonotone_watchdog();
    positivity_safe_ptc_anderson_blend();
    activity_trace_controller_behavior();
    fold_diagnostic();
    std::cout << "[PASS] Alternating controller checks.\n";
    return 0;
}
