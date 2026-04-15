#include "QSSGrid.hpp"

#include <algorithm>
#include <cmath>

namespace dcr::solver {

namespace {
double geometric_sum(double first, double ratio, int n_terms) {
    if (n_terms <= 0) return 0.0;
    if (std::abs(ratio - 1.0) < 1e-12) return first * static_cast<double>(n_terms);
    return first * (std::pow(ratio, n_terms) - 1.0) / (ratio - 1.0);
}
}

std::vector<double> build_step_sizes_cm(const dcr::io::Config& config) {
    const int n_steps = std::max(0, config.grid.num_cells - 1);
    std::vector<double> dx(static_cast<size_t>(n_steps), 0.0);
    const double L = std::max(0.0, config.grid.length_cm);
    if (n_steps == 0 || L <= 0.0) return dx;
    if (config.grid.type != "log" || config.grid.first_cell_cm <= 0.0) {
        std::fill(dx.begin(), dx.end(), L / static_cast<double>(n_steps));
        return dx;
    }
    const double d0 = config.grid.first_cell_cm;
    if (n_steps == 1 || d0 * static_cast<double>(n_steps) >= L) {
        std::fill(dx.begin(), dx.end(), L / static_cast<double>(n_steps));
        return dx;
    }
    double lo = 1.0, hi = 2.0;
    while (geometric_sum(d0, hi, n_steps) < L && hi < 1.0e6) hi *= 2.0;
    if (geometric_sum(d0, hi, n_steps) < L) {
        std::fill(dx.begin(), dx.end(), L / static_cast<double>(n_steps));
        return dx;
    }
    for (int it = 0; it < 100; ++it) {
        const double mid = 0.5 * (lo + hi);
        if (geometric_sum(d0, mid, n_steps) < L) lo = mid; else hi = mid;
    }
    const double r = 0.5 * (lo + hi);
    double s = 0.0;
    for (int i = 0; i < n_steps; ++i) { dx[static_cast<size_t>(i)] = d0 * std::pow(r, i); s += dx[static_cast<size_t>(i)]; }
    dx.back() = std::max(0.0, dx.back() + (L - s));
    return dx;
}

} // namespace dcr::solver
