#include "MarchingDriver.hpp"

#include "CellSolve.hpp"
#include "MarchingDiagnostics.hpp"

#include <algorithm>
#include <cstdlib>
#include <cmath>
#include <iostream>
#include <string>

namespace dcr::solver {

namespace {

// Sum_{i=0}^{n-1} first * ratio^i
double geometric_sum(double first, double ratio, int n_terms) {
    if (n_terms <= 0) return 0.0;
    if (std::abs(ratio - 1.0) < 1e-12) return first * static_cast<double>(n_terms);
    return first * (std::pow(ratio, n_terms) - 1.0) / (ratio - 1.0);
}

// Build per-cell dx from the configured mesh settings.
std::vector<double> build_step_sizes_cm(const dcr::io::Config& config) {
    const int n_steps = std::max(0, config.grid.num_cells - 1);
    std::vector<double> dx(static_cast<size_t>(n_steps), 0.0);
    if (n_steps == 0) return dx;

    const double L = std::max(0.0, config.grid.length_cm);
    if (L <= 0.0) return dx;

    if (config.grid.type == "log" && config.grid.first_cell_cm > 0.0) {
        const double d0 = config.grid.first_cell_cm;
        if (n_steps == 1) {
            dx[0] = L;
            return dx;
        }
        // If requested first cell is too large for a geometric progression over full length, fallback linear.
        if (d0 * static_cast<double>(n_steps) >= L) {
            const double dlin = L / static_cast<double>(n_steps);
            std::fill(dx.begin(), dx.end(), dlin);
            return dx;
        }

        // Solve for geometric ratio r so sum_i d0 r^i = L.
        double lo = 1.0;
        double hi = 2.0;
        while (geometric_sum(d0, hi, n_steps) < L && hi < 1.0e6) {
            hi *= 2.0;
        }
        if (geometric_sum(d0, hi, n_steps) < L) {
            const double dlin = L / static_cast<double>(n_steps);
            std::fill(dx.begin(), dx.end(), dlin);
            return dx;
        }
        for (int it = 0; it < 100; ++it) {
            const double mid = 0.5 * (lo + hi);
            if (geometric_sum(d0, mid, n_steps) < L) lo = mid;
            else hi = mid;
        }
        const double r = 0.5 * (lo + hi);
        double s = 0.0;
        for (int i = 0; i < n_steps; ++i) {
            dx[static_cast<size_t>(i)] = d0 * std::pow(r, i);
            s += dx[static_cast<size_t>(i)];
        }
        // Force exact integral length by adjusting the last cell.
        dx.back() += (L - s);
        if (dx.back() < 0.0) dx.back() = 0.0;
        return dx;
    }

    const double dlin = L / static_cast<double>(n_steps);
    std::fill(dx.begin(), dx.end(), dlin);
    return dx;
}

} // namespace

MarchingHistory run_full_marching(
    const dcr::io::Config& config,
    const dcr::atomic::AtomicData& atomic_data,
    const dcr::state::PlasmaState& plasma,
    const EEDFGridView& grid,
    const BoundaryPhaseResult& boundary) {

    MarchingHistory history;

    // Build mesh spacing and x coordinates.
    const auto dx = build_step_sizes_cm(config);
    const int n_nodes = static_cast<int>(dx.size()) + 1;
    if (n_nodes <= 0) return history;

    history.x_cm.assign(static_cast<size_t>(n_nodes), 0.0);
    for (int i = 1; i < n_nodes; ++i) {
        history.x_cm[static_cast<size_t>(i)] = history.x_cm[static_cast<size_t>(i - 1)] + dx[static_cast<size_t>(i - 1)];
    }

    const auto& levels = atomic_data.get_levels();
    const int total_states = atomic_data.get_total_states();
    const int Pn = static_cast<int>(boundary.P_indices.size());
    if (Pn == 0) {
        if (config.io.verbose_logging) {
            std::cout << "[DCR_Solver] Marching skipped: no background states in P block.\n";
        }
        return history;
    }

    if (config.io.verbose_logging) {
        std::cout << "[DCR_Solver] Marching mode: fixed-point (constant nuclei closure)\n";
    }

    // Initialize compact background state from boundary solution.
    dcr::base::Vector nP = dcr::base::Vector::Zero(Pn);
    for (int i = 0; i < Pn; ++i) {
        const int gi = boundary.P_indices[static_cast<size_t>(i)];
        if (gi >= 0 && gi < boundary.population.size()) {
            nP(i) = std::max(boundary.population(gi), 0.0);
        }
    }

    // Initialize recycling flows from boundary output.
    dcr::base::Vector flowA = dcr::base::Vector::Zero(static_cast<int>(boundary.A_indices.size()));
    dcr::base::Vector flowM = dcr::base::Vector::Zero(static_cast<int>(boundary.M_indices.size()));
    if (boundary.explicit_recycling) {
        for (size_t i = 0; i < boundary.A_indices.size(); ++i) {
            const int gi = boundary.A_indices[i];
            if (gi >= 0 && gi < boundary.population.size()) {
                flowA(static_cast<int>(i)) = std::max(boundary.population(gi), 0.0);
            }
        }
        for (size_t i = 0; i < boundary.M_indices.size(); ++i) {
            const int gi = boundary.M_indices[i];
            if (gi >= 0 && gi < boundary.population.size()) {
                flowM(static_cast<int>(i)) = std::max(boundary.population(gi), 0.0);
            }
        }
    } else if (boundary.have_flow_last) {
        flowA = boundary.flowA_last;
        flowM = boundary.flowM_last;
    }

    const dcr::base::Vector nP_initial = nP;
    const dcr::base::Vector flowA_initial = flowA;
    const dcr::base::Vector flowM_initial = flowM;

    history.background_full.reserve(static_cast<size_t>(n_nodes));
    history.flowA.reserve(static_cast<size_t>(n_nodes));
    history.flowM.reserve(static_cast<size_t>(n_nodes));
    history.background_full.push_back(make_background_full(nP, boundary, total_states));
    history.flowA.push_back(flowA);
    history.flowM.push_back(flowM);

    // Optional per-cell debug logging control:
    // - default detailed iteration log remains cell 1 only.
    // - set env DCR_DEBUG_CELL=<cell_index> to additionally enable detailed logs
    //   for a specific marching cell (1-based).
    int debug_cell = -1;
    if (const char* env = std::getenv("DCR_DEBUG_CELL")) {
        debug_cell = std::atoi(env);
    }

    // March cell-by-cell.
    for (size_t k = 0; k < dx.size(); ++k) {
        const dcr::base::Vector flowA_before = flowA;
        const dcr::base::Vector flowM_before = flowM;
        const dcr::base::Vector nP_before = nP;

        // Keep full iteration log for the first cell by default, and optionally for
        // one selected cell via DCR_DEBUG_CELL.
        const int cell_index = static_cast<int>(k + 1);
        const bool detailed_log_cell = (k == 0) || (debug_cell > 0 && cell_index == debug_cell);
        const double x_left = history.x_cm[k];
        const double x_right = history.x_cm[k + 1];

        // Retry strategy at fixed dx:
        // - If a cell is slow (iterations > threshold) or fails, retry with
        //   an alternative initial guess (no local dx reduction).
        const int slow_iter_threshold = std::max(1, config.numerics.marching_slow_iter_threshold);
        const int max_guess_retries = std::max(1, config.numerics.marching_guess_retries);
        bool accepted = false;
        CellImplicitResult step;
        for (int retry = 0; retry < max_guess_retries && !accepted; ++retry) {
            // Retry seeds:
            // 1) current-cell initial guess
            // 2) marching initial (boundary) guess
            // 3) midpoint of (1) and (2)
            dcr::base::Vector nP_seed = nP_before;
            dcr::base::Vector flowA_seed = flowA_before;
            dcr::base::Vector flowM_seed = flowM_before;
            if (retry == 1) {
                nP_seed = nP_initial;
                flowA_seed = flowA_initial;
                flowM_seed = flowM_initial;
            } else if (retry == 2) {
                nP_seed = 0.5 * (nP_before + nP_initial);
                flowA_seed = 0.5 * (flowA_before + flowA_initial);
                flowM_seed = 0.5 * (flowM_before + flowM_initial);
            }

            step = solve_cell_implicit(
                config,
                atomic_data,
                plasma,
                grid,
                boundary,
                levels,
                nP_before,
                flowA_before,
                flowM_before,
                dx[k],
                x_left,
                x_right,
                cell_index,
                detailed_log_cell,
                true,
                &nP_seed,
                &flowA_seed,
                &flowM_seed
            );

            if (step.converged && step.iterations <= slow_iter_threshold) {
                accepted = true;
                nP = step.nP_new;
                flowA = step.flowA_new;
                flowM = step.flowM_new;
                break;
            }

            if (retry + 1 < max_guess_retries) {
                if (config.io.verbose_logging) {
                    std::cout << "[DCR_Solver] Marching cell " << cell_index
                              << ": retrying with alternative initial guess (attempt "
                              << (retry + 2) << "/" << max_guess_retries
                              << ", previous iters=" << step.iterations
                              << ", converged=" << (step.converged ? "yes" : "no")
                              << ").\n";
                }
            } else if (step.converged) {
                // Accept final retry even if iteration count is still high.
                accepted = true;
                nP = step.nP_new;
                flowA = step.flowA_new;
                flowM = step.flowM_new;
            }
        }

        if (!accepted) {
            // Keep last iterate and proceed, while making the failure explicit in logs.
            nP = step.nP_new;
            flowA = step.flowA_new;
            flowM = step.flowM_new;
            if (config.io.verbose_logging) {
                std::cout << "[DCR_Solver] Marching cell " << cell_index
                          << ": failed to converge after fixed-dx retries.\n";
            }
        }

        const dcr::base::Vector bg_full = make_background_full(nP, boundary, total_states);
        history.background_full.push_back(bg_full);
        history.flowA.push_back(flowA);
        history.flowM.push_back(flowM);

        // Preserve detailed first-cell diagnostics for model verification.
        if (k == 0) {
            log_single_step_marching(
                config,
                atomic_data,
                boundary,
                bg_full,
                flowA_before,
                flowM_before,
                step.local_final,
                step.flow_final,
                x_left,
                x_right
            );
        }
    }

    return history;
}

} // namespace dcr::solver
