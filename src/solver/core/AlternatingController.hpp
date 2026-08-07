#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <optional>
#include <string>
#include <vector>

#include "../../base/Types.hpp"

namespace dcr::solver {

enum class LogAmplitudeBoundKind {
    algorithmic_trust_region,
    physical_admissibility
};

enum class AmplitudeClipSource {
    none,
    local_step_cap,
    algorithmic_trust_region,
    physical_bound
};

inline const char* amplitude_clip_source_name(AmplitudeClipSource source) {
    switch (source) {
    case AmplitudeClipSource::none: return "none";
    case AmplitudeClipSource::local_step_cap: return "local_step_cap";
    case AmplitudeClipSource::algorithmic_trust_region:
        return "algorithmic_trust_region";
    case AmplitudeClipSource::physical_bound: return "physical_bound";
    }
    return "unknown";
}

struct LogAmplitudeBounds {
    double lower = -std::numeric_limits<double>::infinity();
    double upper = std::numeric_limits<double>::infinity();
    LogAmplitudeBoundKind kind =
        LogAmplitudeBoundKind::physical_admissibility;
};

struct LogAmplitudeTrustRegion {
    double center = 0.0;
    double radius = std::log(4.0);
    double minimum_radius = 1.0e-6;
    double maximum_radius = std::log(64.0);

    LogAmplitudeBounds bounds() const {
        return {
            center - radius,
            center + radius,
            LogAmplitudeBoundKind::algorithmic_trust_region};
    }

    void accept(double accepted_log_amplitude, bool well_predicted) {
        center = accepted_log_amplitude;
        if (well_predicted) {
            radius = std::min(maximum_radius, 1.5 * radius);
        }
    }

    void reject() {
        radius = std::max(minimum_radius, 0.5 * radius);
    }
};

struct FeasibleLogAmplitudeStep {
    bool valid = false;
    bool raw_candidate_feasible = false;
    bool clipped = false;
    bool trust_bound_active = false;
    AmplitudeClipSource clip_source = AmplitudeClipSource::none;
    double alpha_feasible = 0.0;
    double delta_log = 0.0;
    std::string reason;
};

inline FeasibleLogAmplitudeStep feasible_log_amplitude_step(
    double current_log_amplitude,
    double raw_delta_log,
    const LogAmplitudeBounds& trust_bounds,
    const LogAmplitudeBounds& physical_bounds,
    double theta = 0.95) {
    FeasibleLogAmplitudeStep result;
    if (!std::isfinite(current_log_amplitude) ||
        !std::isfinite(raw_delta_log) ||
        !(theta > 0.0 && theta < 1.0)) {
        result.reason = "amplitude_raw_step_outside_trust_region";
        return result;
    }
    const double lower = std::max(trust_bounds.lower, physical_bounds.lower);
    const double upper = std::min(trust_bounds.upper, physical_bounds.upper);
    if (!(lower < current_log_amplitude && current_log_amplitude < upper)) {
        result.reason = "amplitude_trust_bound_active";
        result.trust_bound_active = true;
        return result;
    }
    if (raw_delta_log == 0.0) {
        result.valid = true;
        result.raw_candidate_feasible = true;
        result.alpha_feasible = 1.0;
        return result;
    }

    const double raw_candidate = current_log_amplitude + raw_delta_log;
    result.raw_candidate_feasible =
        std::isfinite(raw_candidate) && raw_candidate > lower &&
        raw_candidate < upper;
    double alpha = 1.0;
    double trust_alpha = 1.0;
    double physical_alpha = 1.0;
    if (raw_delta_log > 0.0) {
        trust_alpha = theta *
            (trust_bounds.upper - current_log_amplitude) / raw_delta_log;
        physical_alpha = theta *
            (physical_bounds.upper - current_log_amplitude) / raw_delta_log;
    } else {
        trust_alpha = theta *
            (trust_bounds.lower - current_log_amplitude) / raw_delta_log;
        physical_alpha = theta *
            (physical_bounds.lower - current_log_amplitude) / raw_delta_log;
    }
    alpha = std::min({1.0, trust_alpha, physical_alpha});
    if (!(alpha > 0.0) || !std::isfinite(alpha)) {
        result.reason = "amplitude_no_feasible_descent_step";
        return result;
    }
    result.delta_log = alpha * raw_delta_log;
    const double candidate = current_log_amplitude + result.delta_log;
    if (!std::isfinite(candidate) || !(candidate > lower && candidate < upper)) {
        result.reason = "amplitude_no_feasible_descent_step";
        return result;
    }
    result.valid = true;
    result.alpha_feasible = alpha;
    result.clipped = alpha < 1.0;
    if (result.clipped && trust_alpha <= physical_alpha) {
        result.clip_source = AmplitudeClipSource::algorithmic_trust_region;
        result.trust_bound_active = true;
    } else if (result.clipped) {
        result.clip_source = AmplitudeClipSource::physical_bound;
    }
    result.reason = result.clipped
        ? "amplitude_step_clipped" : "amplitude_step_feasible";
    return result;
}

template <typename MeritEvaluator>
FeasibleLogAmplitudeStep backtrack_log_amplitude_step(
    double current_log_amplitude,
    double raw_delta_log,
    const LogAmplitudeBounds& trust_bounds,
    const LogAmplitudeBounds& physical_bounds,
    double current_merit,
    MeritEvaluator&& evaluate_merit,
    double theta = 0.95,
    double beta = 0.5,
    double minimum_alpha = 1.0e-8) {
    FeasibleLogAmplitudeStep result = feasible_log_amplitude_step(
        current_log_amplitude, raw_delta_log, trust_bounds,
        physical_bounds, theta);
    if (!result.valid) return result;

    double alpha = result.alpha_feasible;
    const double initial_alpha = alpha;
    while (alpha >= minimum_alpha) {
        const double candidate = current_log_amplitude + alpha * raw_delta_log;
        const std::optional<double> merit = evaluate_merit(candidate);
        if (merit.has_value() && std::isfinite(*merit) &&
            (*merit < current_merit || *merit < 1.0)) {
            result.alpha_feasible = alpha;
            result.delta_log = alpha * raw_delta_log;
            result.clipped = initial_alpha < 1.0;
            result.reason = alpha < initial_alpha
                ? "amplitude_step_backtracked"
                : (result.clipped
                    ? "amplitude_step_clipped" : "amplitude_step_feasible");
            return result;
        }
        alpha *= beta;
    }
    result.valid = false;
    result.alpha_feasible = 0.0;
    result.delta_log = 0.0;
    result.reason = "amplitude_no_feasible_descent_step";
    return result;
}

struct AndersonSample {
    dcr::base::Vector state;
    dcr::base::Vector residual;
};

struct PositivitySafeBlend {
    double beta = 0.0;
    bool positivity_limited = false;
};

struct ActivityTraceClassification {
    bool trace = false;
    bool protected_pathway = false;
    std::vector<int> protected_pathways;
};

inline ActivityTraceClassification classify_activity_trace_row(
    double activity,
    double activity_floor,
    const std::vector<double>& signed_pathway_contributions,
    const std::vector<double>& gross_pathway_contributions,
    double protection_fraction = 1.0e-3) {
    ActivityTraceClassification result;
    if (!(activity >= 0.0) || !(activity_floor >= 0.0) ||
        !(protection_fraction >= 0.0) ||
        !std::isfinite(activity) || !std::isfinite(activity_floor) ||
        !std::isfinite(protection_fraction) ||
        signed_pathway_contributions.size() !=
            gross_pathway_contributions.size()) {
        return result;
    }
    for (size_t pathway = 0;
         pathway < signed_pathway_contributions.size(); ++pathway) {
        const double contribution = signed_pathway_contributions[pathway];
        const double gross = gross_pathway_contributions[pathway];
        if (!std::isfinite(contribution) || !(gross >= 0.0) ||
            !std::isfinite(gross)) {
            return ActivityTraceClassification{};
        }
        if (gross > 0.0 &&
            std::abs(contribution) > protection_fraction * gross) {
            result.protected_pathways.push_back(
                static_cast<int>(pathway));
        }
    }
    result.protected_pathway = !result.protected_pathways.empty();
    result.trace = activity < activity_floor && !result.protected_pathway;
    return result;
}

struct MaskedStateChange {
    double maximum = 0.0;
    int index = -1;
    int evaluated_rows = 0;
};

inline MaskedStateChange maximum_nontrace_relative_change(
    const dcr::base::Vector& previous,
    const dcr::base::Vector& next,
    const std::vector<bool>& trace_mask) {
    MaskedStateChange result;
    if (previous.size() != next.size() ||
        static_cast<size_t>(previous.size()) != trace_mask.size() ||
        !previous.allFinite() || !next.allFinite()) {
        return result;
    }
    for (int i = 0; i < previous.size(); ++i) {
        if (trace_mask[static_cast<size_t>(i)]) continue;
        ++result.evaluated_rows;
        const double scale = std::max(
            {1.0, std::abs(previous(i)), std::abs(next(i))});
        const double change = std::abs(next(i) - previous(i)) / scale;
        if (change > result.maximum) {
            result.maximum = change;
            result.index = i;
        }
    }
    return result;
}

struct TraceAwareAndersonCandidate {
    dcr::base::Vector state;
    bool nontrace_negative = false;
    int restored_trace_components = 0;
    int nontrace_negative_components = 0;
};

inline TraceAwareAndersonCandidate restore_negative_trace_components(
    const dcr::base::Vector& candidate,
    const dcr::base::Vector& ptc_state,
    const std::vector<bool>& trace_mask,
    int first_population_index = 0) {
    TraceAwareAndersonCandidate result;
    result.state = candidate;
    if (candidate.size() != ptc_state.size() ||
        first_population_index < 0 ||
        first_population_index > candidate.size() ||
        static_cast<size_t>(candidate.size() - first_population_index) !=
            trace_mask.size() ||
        !candidate.allFinite() || !ptc_state.allFinite()) {
        result.nontrace_negative = true;
        return result;
    }
    for (int i = first_population_index; i < candidate.size(); ++i) {
        if (result.state(i) >= 0.0) continue;
        const bool trace = trace_mask[
            static_cast<size_t>(i - first_population_index)];
        if (trace && ptc_state(i) >= 0.0) {
            result.state(i) = ptc_state(i);
            ++result.restored_trace_components;
        } else {
            result.nontrace_negative = true;
            if (!trace) ++result.nontrace_negative_components;
        }
    }
    return result;
}

struct HybridAuditStatistics {
    int total_rows = 0;
    int trace_rows = 0;
    int nontrace_rows = 0;
    double maximum_all = 0.0;
    double maximum_trace = 0.0;
    double maximum_nontrace = 0.0;
};

inline HybridAuditStatistics evaluate_complete_hybrid_audit(
    const dcr::base::Vector& residual,
    const dcr::base::Vector& activity,
    const dcr::base::Vector& activity_floor,
    const std::vector<bool>& trace_mask) {
    HybridAuditStatistics result;
    if (residual.size() != activity.size() ||
        residual.size() != activity_floor.size() ||
        static_cast<size_t>(residual.size()) != trace_mask.size()) {
        return result;
    }
    for (int i = 0; i < residual.size(); ++i) {
        const double denominator = trace_mask[static_cast<size_t>(i)]
            ? std::max(activity(i), activity_floor(i))
            : activity(i);
        const double scaled = std::abs(residual(i)) /
            std::max(denominator, std::numeric_limits<double>::min());
        ++result.total_rows;
        result.maximum_all = std::max(result.maximum_all, scaled);
        if (trace_mask[static_cast<size_t>(i)]) {
            ++result.trace_rows;
            result.maximum_trace = std::max(result.maximum_trace, scaled);
        } else {
            ++result.nontrace_rows;
            result.maximum_nontrace = std::max(
                result.maximum_nontrace, scaled);
        }
    }
    return result;
}

inline PositivitySafeBlend positivity_safe_blend(
    const dcr::base::Vector& ptc_state,
    const dcr::base::Vector& anderson_state,
    int first_population_index = 0,
    double safety = 0.95) {
    PositivitySafeBlend result;
    if (ptc_state.size() != anderson_state.size() ||
        first_population_index < 0 ||
        first_population_index > ptc_state.size() ||
        !(safety > 0.0 && safety < 1.0) ||
        !ptc_state.allFinite() || !anderson_state.allFinite()) {
        return result;
    }

    double beta_max = 1.0;
    for (int i = first_population_index; i < ptc_state.size(); ++i) {
        if (ptc_state(i) < 0.0) return result;
        const double direction = anderson_state(i) - ptc_state(i);
        if (direction < 0.0) {
            beta_max = std::min(beta_max, ptc_state(i) / -direction);
        }
    }
    result.positivity_limited = beta_max < 1.0;
    if (!(beta_max > 0.0) || !std::isfinite(beta_max)) return result;
    result.beta = safety * std::min(1.0, beta_max);
    return result;
}

class NonmonotoneMeritWatchdog {
public:
    NonmonotoneMeritWatchdog(int window = 5, int watchdog = 12)
        : window_(std::max(1, window)),
          watchdog_(std::max(2 * window_, watchdog)) {}

    double acceptance_limit(double fallback) const {
        if (accepted_merits_.empty()) return fallback;
        return *std::max_element(accepted_merits_.begin(), accepted_merits_.end());
    }

    bool accepts(double merit, double fallback) const {
        return std::isfinite(merit) &&
            (merit <= acceptance_limit(fallback) || merit < 1.0);
    }

    void accept(double merit) {
        if (!std::isfinite(merit)) return;
        accepted_merits_.push_back(merit);
        if (static_cast<int>(accepted_merits_.size()) > window_) {
            accepted_merits_.erase(accepted_merits_.begin());
        }
        consecutive_rejections_ = 0;
    }

    bool reject() {
        return ++consecutive_rejections_ >= watchdog_;
    }

    void clear() {
        accepted_merits_.clear();
        consecutive_rejections_ = 0;
    }

    int rejection_count() const { return consecutive_rejections_; }
    int window() const { return window_; }
    int watchdog() const { return watchdog_; }

private:
    int window_;
    int watchdog_;
    int consecutive_rejections_ = 0;
    std::vector<double> accepted_merits_;
};

struct FixedPtcMapParameters {
    double plasma_step = 0.0;
    double atom_step = 0.0;
    double molecule_step = 0.0;
    double trust_center = 0.0;
    double trust_radius = 0.0;
};

inline bool same_fixed_ptc_map(
    const FixedPtcMapParameters& left,
    const FixedPtcMapParameters& right) {
    return left.plasma_step == right.plasma_step &&
        left.atom_step == right.atom_step &&
        left.molecule_step == right.molecule_step &&
        left.trust_center == right.trust_center &&
        left.trust_radius == right.trust_radius;
}

inline std::optional<double> increased_pseudo_time_step(
    double current,
    double maximum,
    double factor = 10.0) {
    if (!(current > 0.0) || !(maximum >= current) || !(factor > 1.0) ||
        !std::isfinite(current) || !std::isfinite(maximum) ||
        !std::isfinite(factor)) {
        return std::nullopt;
    }
    const double increased = std::min(maximum, factor * current);
    return increased > current
        ? std::optional<double>(increased) : std::nullopt;
}

inline std::optional<dcr::base::Vector> anderson_candidate(
    const std::vector<AndersonSample>& history,
    const dcr::base::Vector& current_state,
    const dcr::base::Vector& current_residual,
    int requested_depth,
    double beta,
    double regularization) {
    using dcr::base::Matrix;
    using dcr::base::Vector;
    if (history.size() < 2 || current_state.size() == 0 ||
        current_state.size() != current_residual.size()) {
        return std::nullopt;
    }
    const int differences = std::min<int>(
        std::max(1, requested_depth), static_cast<int>(history.size()) - 1);
    const int first = static_cast<int>(history.size()) - differences - 1;
    Matrix delta_f(current_state.size(), differences);
    Matrix delta_x(current_state.size(), differences);
    for (int j = 0; j < differences; ++j) {
        const AndersonSample& previous = history[static_cast<size_t>(first + j)];
        const AndersonSample& next = history[static_cast<size_t>(first + j + 1)];
        if (previous.state.size() != current_state.size() ||
            previous.residual.size() != current_state.size() ||
            next.state.size() != current_state.size() ||
            next.residual.size() != current_state.size()) {
            return std::nullopt;
        }
        delta_f.col(j) = next.residual - previous.residual;
        delta_x.col(j) = next.state - previous.state;
    }
    Matrix normal = delta_f.transpose() * delta_f;
    normal.diagonal().array() += std::max(0.0, regularization);
    const Vector rhs = delta_f.transpose() * current_residual;
    const Vector gamma = normal.ldlt().solve(rhs);
    if (!gamma.allFinite()) return std::nullopt;
    const Vector candidate = current_state + beta * current_residual -
        (delta_x + beta * delta_f) * gamma;
    return candidate.allFinite()
        ? std::optional<Vector>(candidate) : std::nullopt;
}

struct AuditedAmplitudePoint {
    double temperature = std::numeric_limits<double>::quiet_NaN();
    double log_amplitude = std::numeric_limits<double>::quiet_NaN();
    double mismatch = std::numeric_limits<double>::quiet_NaN();
    double true_steady_residual = std::numeric_limits<double>::infinity();
    bool audit_passed = false;
};

inline std::optional<double> audited_secant_slope(
    const AuditedAmplitudePoint& previous,
    const AuditedAmplitudePoint& current) {
    const double delta = current.log_amplitude - previous.log_amplitude;
    if (!previous.audit_passed || !current.audit_passed ||
        !std::isfinite(previous.mismatch) || !std::isfinite(current.mismatch) ||
        !std::isfinite(delta) || std::abs(delta) <= 1.0e-12 ||
        std::abs(current.temperature - previous.temperature) > 1.0e-12) {
        return std::nullopt;
    }
    return (current.mismatch - previous.mismatch) / delta;
}

struct FoldIndicators {
    bool derivative_collapsed = false;
    bool continuation_steps_collapsed = false;
    bool amplitude_response_large = false;
    bool derivative_sign_changed = false;

    bool possible_fold() const {
        const int evidence = static_cast<int>(derivative_collapsed) +
            static_cast<int>(continuation_steps_collapsed) +
            static_cast<int>(amplitude_response_large) +
            static_cast<int>(derivative_sign_changed);
        return derivative_collapsed && evidence >= 3;
    }
};

inline FoldIndicators fold_indicators(
    const std::vector<double>& audited_slopes,
    double reference_slope,
    double continuation_step,
    double reference_continuation_step,
    double amplitude_response,
    double reference_amplitude_response) {
    FoldIndicators result;
    if (audited_slopes.size() < 3 || !(std::abs(reference_slope) > 0.0)) {
        return result;
    }
    result.derivative_collapsed = std::all_of(
        audited_slopes.end() - 3, audited_slopes.end(),
        [&](double slope) {
            return std::isfinite(slope) &&
                std::abs(slope) <= 0.1 * std::abs(reference_slope);
        });
    result.continuation_steps_collapsed =
        reference_continuation_step > 0.0 && continuation_step > 0.0 &&
        continuation_step <= 0.1 * reference_continuation_step;
    result.amplitude_response_large =
        reference_amplitude_response > 0.0 &&
        amplitude_response >= 10.0 * reference_amplitude_response;
    for (size_t i = 1; i < audited_slopes.size(); ++i) {
        if (audited_slopes[i - 1] * audited_slopes[i] < 0.0) {
            result.derivative_sign_changed = true;
            break;
        }
    }
    return result;
}

} // namespace dcr::solver
