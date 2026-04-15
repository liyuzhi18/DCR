#include "MarchingDriver.hpp"

#include "CellSolve.hpp"
#include "MarchingDiagnostics.hpp"

#include <H5Cpp.h>

#include <algorithm>
#include <cstdlib>
#include <cmath>
#include <iostream>
#include <limits>
#include <string>

namespace dcr::solver {

namespace {

struct RestartProfileSeed {
    bool valid = false;
    std::string path;
    std::vector<double> x_cm;
    std::vector<dcr::base::Vector> nP_nodes;
    std::vector<dcr::base::Vector> flowA_nodes;
    std::vector<dcr::base::Vector> flowM_nodes;
};

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

int max_index(const dcr::base::Vector& values) {
    int best = -1;
    double best_value = -1.0;
    for (int i = 0; i < values.size(); ++i) {
        if (!std::isfinite(values(i))) continue;
        if (values(i) > best_value) {
            best_value = values(i);
            best = i;
        }
    }
    return best;
}

void log_rate_snapshot(double x_cm,
                       const RateDiagnosticSnapshot& snapshot,
                       const std::vector<dcr::atomic::EnergyLevel>& levels) {
    (void)levels;
    if (!snapshot.atomic_effective.valid) return;
    std::cout << "[DCR_Solver][Rates][atomic] x=" << x_cm
              << " cm SCD=" << snapshot.atomic_effective.scd_cm3_s
              << " ACD=" << snapshot.atomic_effective.acd_cm3_s
              << "\n";
}

std::vector<double> read_vector_double(H5::H5File& file, const std::string& name) {
    H5::DataSet ds = file.openDataSet(name);
    H5::DataSpace space = ds.getSpace();
    if (space.getSimpleExtentNdims() != 1) {
        throw std::runtime_error("Expected rank-1 dataset: " + name);
    }
    hsize_t dims[1] = {0};
    space.getSimpleExtentDims(dims, nullptr);
    std::vector<double> data(static_cast<size_t>(dims[0]), 0.0);
    if (!data.empty()) ds.read(data.data(), H5::PredType::NATIVE_DOUBLE);
    return data;
}

std::vector<int> read_vector_int(H5::H5File& file, const std::string& name) {
    H5::DataSet ds = file.openDataSet(name);
    H5::DataSpace space = ds.getSpace();
    if (space.getSimpleExtentNdims() != 1) {
        throw std::runtime_error("Expected rank-1 dataset: " + name);
    }
    hsize_t dims[1] = {0};
    space.getSimpleExtentDims(dims, nullptr);
    std::vector<int> data(static_cast<size_t>(dims[0]), 0);
    if (!data.empty()) ds.read(data.data(), H5::PredType::NATIVE_INT);
    return data;
}

std::vector<dcr::base::Vector> read_matrix_double(H5::H5File& file, const std::string& name) {
    H5::DataSet ds = file.openDataSet(name);
    H5::DataSpace space = ds.getSpace();
    if (space.getSimpleExtentNdims() != 2) {
        throw std::runtime_error("Expected rank-2 dataset: " + name);
    }
    hsize_t dims[2] = {0, 0};
    space.getSimpleExtentDims(dims, nullptr);
    const size_t rows = static_cast<size_t>(dims[0]);
    const size_t cols = static_cast<size_t>(dims[1]);
    std::vector<double> flat(rows * cols, 0.0);
    if (!flat.empty()) ds.read(flat.data(), H5::PredType::NATIVE_DOUBLE);
    std::vector<dcr::base::Vector> out;
    out.reserve(rows);
    for (size_t i = 0; i < rows; ++i) {
        dcr::base::Vector row = dcr::base::Vector::Zero(static_cast<int>(cols));
        for (size_t j = 0; j < cols; ++j) {
            row(static_cast<int>(j)) = flat[i * cols + j];
        }
        out.push_back(std::move(row));
    }
    return out;
}

bool equal_indices(const std::vector<int>& lhs, const std::vector<int>& rhs) {
    if (lhs.size() != rhs.size()) return false;
    for (size_t i = 0; i < lhs.size(); ++i) {
        if (lhs[i] != rhs[i]) return false;
    }
    return true;
}

RestartProfileSeed load_restart_profile_seed(
    const dcr::io::Config& config,
    const BoundaryPhaseResult& boundary,
    int total_states) {

    RestartProfileSeed out;
    out.path = config.io.restart_profile_h5;
    if (out.path.empty()) return out;

    try {
        H5::H5File file(out.path, H5F_ACC_RDONLY);
        out.x_cm = read_vector_double(file, "/grid/x_cm");
        const auto p_indices = read_vector_int(file, "/states/P_indices");
        const auto a_indices = read_vector_int(file, "/states/A_indices");
        const auto m_indices = read_vector_int(file, "/states/M_indices");
        if (!equal_indices(p_indices, boundary.P_indices) ||
            !equal_indices(a_indices, boundary.A_indices) ||
            !equal_indices(m_indices, boundary.M_indices)) {
            throw std::runtime_error("restart profile state block indices do not match current model");
        }

        const auto bg_full = read_matrix_double(file, "/population/background_full");
        const auto flowA = read_matrix_double(file, "/population/flowA");
        const auto flowM = read_matrix_double(file, "/population/flowM");

        if (out.x_cm.empty() ||
            bg_full.size() != out.x_cm.size() ||
            flowA.size() != out.x_cm.size() ||
            flowM.size() != out.x_cm.size()) {
            throw std::runtime_error("restart profile node counts do not match");
        }

        out.nP_nodes.reserve(bg_full.size());
        out.flowA_nodes.reserve(flowA.size());
        out.flowM_nodes.reserve(flowM.size());
        for (size_t i = 0; i < bg_full.size(); ++i) {
            const auto& bg_row = bg_full[i];
            if (bg_row.size() != total_states) {
                throw std::runtime_error("restart background_full width does not match total_states");
            }
            dcr::base::Vector nP = dcr::base::Vector::Zero(static_cast<int>(boundary.P_indices.size()));
            for (size_t j = 0; j < boundary.P_indices.size(); ++j) {
                const int gi = boundary.P_indices[j];
                if (gi < 0 || gi >= bg_row.size()) continue;
                nP(static_cast<int>(j)) = std::max(bg_row(gi), 0.0);
            }
            out.nP_nodes.push_back(std::move(nP));

            if (flowA[i].size() != static_cast<int>(boundary.A_indices.size()) ||
                flowM[i].size() != static_cast<int>(boundary.M_indices.size())) {
                throw std::runtime_error("restart flow block widths do not match current model");
            }
            out.flowA_nodes.push_back(flowA[i].cwiseMax(0.0));
            out.flowM_nodes.push_back(flowM[i].cwiseMax(0.0));
        }

        out.valid = true;
        return out;
    } catch (const H5::Exception& ex) {
        if (config.io.verbose_logging) {
            std::cout << "[DCR_Solver] Restart profile disabled: failed to read "
                      << out.path << " (" << ex.getDetailMsg() << ")\n";
        }
        return RestartProfileSeed{};
    } catch (const std::exception& ex) {
        if (config.io.verbose_logging) {
            std::cout << "[DCR_Solver] Restart profile disabled: "
                      << ex.what() << "\n";
        }
        return RestartProfileSeed{};
    }
}

dcr::base::Vector interpolate_restart_vector(
    const std::vector<double>& x_nodes,
    const std::vector<dcr::base::Vector>& values,
    double x_target) {

    if (x_nodes.empty() || values.empty()) return {};
    if (values.size() != x_nodes.size()) return {};
    if (x_target <= x_nodes.front()) return values.front();
    if (x_target >= x_nodes.back()) return values.back();

    const auto it = std::upper_bound(x_nodes.begin(), x_nodes.end(), x_target);
    const size_t i1 = static_cast<size_t>(std::distance(x_nodes.begin(), it));
    const size_t i0 = i1 - 1;
    const double x0 = x_nodes[i0];
    const double x1 = x_nodes[i1];
    if (!(x1 > x0)) return values[i0];
    const double alpha = (x_target - x0) / (x1 - x0);
    return (1.0 - alpha) * values[i0] + alpha * values[i1];
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
    history.boundary_elapsed_seconds = boundary.elapsed_seconds;
    history.boundary_iterations = boundary.iterations;

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

    const AtomicRateCalculator rate_calculator(atomic_data);
    const RestartProfileSeed restart_seed =
        load_restart_profile_seed(config, boundary, total_states);
    if (config.io.verbose_logging && restart_seed.valid) {
        std::cout << "[DCR_Solver] Marching restart seed loaded from "
                  << restart_seed.path << " with " << restart_seed.x_cm.size()
                  << " nodes.\n";
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
    history.rate_diagnostics.reserve(static_cast<size_t>(n_nodes));
    history.cell_elapsed_seconds.reserve(dx.size());
    history.cell_iterations.reserve(dx.size());
    history.cell_converged.reserve(dx.size());

    const dcr::base::Vector bg_full_boundary = make_background_full(nP, boundary, total_states);
    history.background_full.push_back(bg_full_boundary);
    history.flowA.push_back(flowA);
    history.flowM.push_back(flowM);
    const auto boundary_local = assemble_local_system(
        config, atomic_data, plasma, grid, boundary, bg_full_boundary, flowA, flowM, 0.0
    );
    history.rate_diagnostics.push_back(
        rate_calculator.evaluate(config, boundary, boundary_local, bg_full_boundary, 0.0)
    );
    if (config.io.verbose_logging) {
        log_rate_snapshot(0.0, history.rate_diagnostics.back(), levels);
    }

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
        dcr::base::Vector nP_restart;
        dcr::base::Vector flowA_restart;
        dcr::base::Vector flowM_restart;
        const dcr::base::Vector* nP_init_ptr = nullptr;
        const dcr::base::Vector* flowA_init_ptr = nullptr;
        const dcr::base::Vector* flowM_init_ptr = nullptr;
        if (restart_seed.valid) {
            nP_restart = interpolate_restart_vector(restart_seed.x_cm, restart_seed.nP_nodes, x_right);
            flowA_restart = interpolate_restart_vector(restart_seed.x_cm, restart_seed.flowA_nodes, x_right);
            flowM_restart = interpolate_restart_vector(restart_seed.x_cm, restart_seed.flowM_nodes, x_right);
            if (nP_restart.size() == nP_before.size()) nP_init_ptr = &nP_restart;
            if (flowA_restart.size() == flowA_before.size()) flowA_init_ptr = &flowA_restart;
            if (flowM_restart.size() == flowM_before.size()) flowM_init_ptr = &flowM_restart;
        }

        const int slow_iter_threshold = std::max(1, config.numerics.marching_slow_iter_threshold);
        CellImplicitResult step = solve_cell_implicit(
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
            nP_init_ptr,
            flowA_init_ptr,
            flowM_init_ptr
        );

        nP = step.nP_new;
        flowA = step.flowA_new;
        flowM = step.flowM_new;
        history.cell_elapsed_seconds.push_back(step.elapsed_seconds);
        history.cell_iterations.push_back(step.iterations);
        history.cell_converged.push_back(step.converged ? 1 : 0);

        if (config.io.verbose_logging) {
            if (!step.converged) {
                std::cout << "[DCR_Solver] Marching cell " << cell_index
                          << ": failed to converge in a single fixed-dx solve.\n";
            } else if (step.iterations > slow_iter_threshold) {
                std::cout << "[DCR_Solver] Marching cell " << cell_index
                          << ": converged, but required more than "
                          << slow_iter_threshold << " iterations.\n";
            }
        }
        if (!step.converged && config.numerics.abort_on_marching_nonconvergence) {
            throw std::runtime_error(
                "Marching cell " + std::to_string(cell_index) +
                " reached marching_max_iterations without convergence."
            );
        }

        const dcr::base::Vector bg_full = make_background_full(nP, boundary, total_states);
        history.background_full.push_back(bg_full);
        history.flowA.push_back(flowA);
        history.flowM.push_back(flowM);
        history.rate_diagnostics.push_back(
            rate_calculator.evaluate(config, boundary, step.local_final, bg_full, x_right)
        );
        if (config.io.verbose_logging) {
            log_rate_snapshot(x_right, history.rate_diagnostics.back(), levels);
        }

        // Preserve the detailed one-step marching dump only when explicitly
        // requested for cell 1. The unconditional verbose dump makes long runs
        // look stuck before cell 2 even starts.
        if (k == 0 && debug_cell == 1) {
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
