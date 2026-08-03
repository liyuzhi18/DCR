#include "GlobalBVP.hpp"

#include "CellSolve.hpp"
#include "LocalSystemAssembler.hpp"
#include "../core/TemperatureProfile.hpp"
#include "../qss/internal/QSSGrid.hpp"
#include "../../physics/Sheath.hpp"

#include <Eigen/SparseCore>
#include <Eigen/SparseLU>
#include <Eigen/LU>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <vector>

namespace dcr::solver {
namespace {

using SparseMatrix = Eigen::SparseMatrix<double, Eigen::ColMajor, int>;
using Triplet = Eigen::Triplet<double>;
using Clock = std::chrono::steady_clock;

double elapsed_seconds(Clock::time_point start) {
    return std::chrono::duration<double>(Clock::now() - start).count();
}

struct ScopedTimer {
    double* total = nullptr;
    Clock::time_point start = Clock::now();

    explicit ScopedTimer(double* destination) : total(destination) {}
    ~ScopedTimer() {
        if (total) *total += elapsed_seconds(start);
    }
};

struct Layout {
    int nodes = 0;
    int p = 0;
    int a = 0;
    int m = 0;
    int block = 0;
    int q_index = 0;
    int proton_pi = -1;
    std::vector<int> p_pos;
    std::vector<unsigned char> positive_ion;
};

struct State {
    std::vector<dcr::base::Vector> p;
    std::vector<dcr::base::Vector> a;
    std::vector<dcr::base::Vector> m;
    double q_up = 0.0;
};

struct NodeData {
    LocalSystem local;
    dcr::base::Vector chem_p;
    dcr::base::Vector pump_p;
    dcr::base::Vector chem_a;
    dcr::base::Vector pump_a;
    dcr::base::Vector chem_m;
    dcr::base::Vector pump_m;
};

int offset(const Layout& layout, int node) {
    return node * layout.block;
}

double safe_exp(double y) {
    return std::exp(std::clamp(y, -138.0, 50.0));
}

std::vector<double> coordinates(const dcr::io::Config& config) {
    const auto dx = build_step_sizes_cm(config);
    std::vector<double> x(dx.size() + 1, 0.0);
    for (size_t k = 0; k < dx.size(); ++k) x[k + 1] = x[k] + dx[k];
    return x;
}

struct SpatialRateCaches {
    std::vector<LocalRateCache> cells;
    LocalRateCache upstream;
};

SpatialRateCaches build_spatial_rate_caches(
    const dcr::io::Config& config,
    const dcr::atomic::AtomicData& atomic_data,
    const dcr::state::PlasmaState& plasma,
    const EEDFGridView& grid,
    const std::vector<double>& x,
    double density_reference_cm3) {
    SpatialRateCaches out;
    std::vector<PlasmaTemperatures> cached_temperatures;
    std::vector<LocalRateCache> unique_caches;
    auto cache_at = [&](double position) {
        const auto temperatures = evaluate_plasma_temperatures(config, position);
        for (size_t i = 0; i < cached_temperatures.size(); ++i) {
            if (std::abs(cached_temperatures[i].electron_eV - temperatures.electron_eV) < 1.0e-14 &&
                std::abs(cached_temperatures[i].ion_eV - temperatures.ion_eV) < 1.0e-14) {
                return unique_caches[i];
            }
        }
        cached_temperatures.push_back(temperatures);
        unique_caches.push_back(build_local_rate_cache(
            config, atomic_data, plasma, grid, position, density_reference_cm3));
        return unique_caches.back();
    };
    out.cells.reserve(x.size() > 0 ? x.size() - 1 : 0);
    for (size_t k = 0; k + 1 < x.size(); ++k) {
        out.cells.push_back(cache_at(0.5 * (x[k] + x[k + 1])));
    }
    out.upstream = cache_at(x.back());
    return out;
}

Layout make_layout(const BoundaryPhaseResult& boundary, int nodes, int total_states) {
    Layout out;
    out.nodes = nodes;
    out.p = static_cast<int>(boundary.P_indices.size());
    out.a = static_cast<int>(boundary.A_indices.size());
    out.m = static_cast<int>(boundary.M_indices.size());
    // Molecular states are recovered by the forward transport operator. They
    // are metadata here, never global unknowns or residual rows.
    out.block = out.p + out.a;
    out.q_index = nodes * out.block;
    out.p_pos.assign(static_cast<size_t>(total_states), -1);
    out.positive_ion.assign(static_cast<size_t>(out.p), 0);
    for (int pi = 0; pi < out.p; ++pi) {
        const int gi = boundary.P_indices[static_cast<size_t>(pi)];
        if (gi >= 0 && gi < total_states) out.p_pos[static_cast<size_t>(gi)] = pi;
    }
    for (int gi : boundary.ion_indices) {
        if (gi < 0 || gi >= total_states) continue;
        const int pi = out.p_pos[static_cast<size_t>(gi)];
        if (pi >= 0) out.positive_ion[static_cast<size_t>(pi)] = 1;
    }
    return out;
}

int select_proton_pi(
    const Layout& layout,
    const BoundaryPhaseResult& boundary,
    const std::vector<dcr::atomic::EnergyLevel>& levels) {
    int selected = -1;
    int candidates = 0;
    for (int gi : boundary.ion_indices) {
        if (gi < 0 || gi >= static_cast<int>(levels.size())) continue;
        const auto& level = levels[static_cast<size_t>(gi)];
        if (level.charge <= 0 || level.atomicity != 1 ||
            level.type != dcr::atomic::SpeciesType::Ion) continue;
        const int pi = layout.p_pos[static_cast<size_t>(gi)];
        if (pi < 0) continue;
        selected = pi;
        ++candidates;
    }
    if (candidates != 1) {
        throw std::runtime_error(
            "Global BVP requires exactly one monatomic positive-ion proton carrier.");
    }
    return selected;
}

dcr::base::Vector compact_boundary_p(const BoundaryPhaseResult& boundary) {
    dcr::base::Vector out = dcr::base::Vector::Zero(static_cast<int>(boundary.P_indices.size()));
    for (int pi = 0; pi < out.size(); ++pi) {
        const int gi = boundary.P_indices[static_cast<size_t>(pi)];
        if (gi >= 0 && gi < boundary.population.size()) out(pi) = std::max(boundary.population(gi), 0.0);
    }
    return out;
}

State decode(const dcr::base::Vector& y, const Layout& layout, double n_ref, double q_ref) {
    State out;
    out.p.reserve(static_cast<size_t>(layout.nodes));
    out.a.reserve(static_cast<size_t>(layout.nodes));
    out.m.reserve(static_cast<size_t>(layout.nodes));
    for (int k = 0; k < layout.nodes; ++k) {
        const int base = offset(layout, k);
        dcr::base::Vector p = dcr::base::Vector::Zero(layout.p);
        dcr::base::Vector a = dcr::base::Vector::Zero(layout.a);
        for (int i = 0; i < layout.p; ++i) p(i) = n_ref * safe_exp(y(base + i));
        for (int i = 0; i < layout.a; ++i) a(i) = n_ref * safe_exp(y(base + layout.p + i));
        out.p.push_back(std::move(p));
        out.a.push_back(std::move(a));
        out.m.push_back(dcr::base::Vector::Zero(layout.m));
    }
    out.q_up = q_ref * safe_exp(y(layout.q_index));
    return out;
}

NodeData evaluate_node(
    const dcr::io::Config& config,
    const dcr::atomic::AtomicData& atomic_data,
    const dcr::state::PlasmaState& plasma,
    const EEDFGridView& grid,
    const BoundaryPhaseResult& boundary,
    const dcr::base::Vector& p,
    const dcr::base::Vector& a,
    const dcr::base::Vector& m,
    double x_cm,
    bool assemble_rate_derivatives = false,
    const LocalRateCache* rate_cache = nullptr) {
    NodeData out;
    const auto& levels = atomic_data.get_levels();
    const auto full = make_background_full(p, boundary, atomic_data.get_total_states());
    out.local = assemble_local_system(
        config, atomic_data, plasma, grid, boundary, full, a, m, x_cm,
        assemble_rate_derivatives, rate_cache);
    const auto chemistry = assemble_local_chemistry_sources(
        out.local, boundary, atomic_data, p, a, m);
    out.chem_p = chemistry.background;
    out.pump_p = dcr::base::Vector::Zero(p.size());
    const auto temperatures = evaluate_plasma_temperatures(config, x_cm);
    const double width = std::max(config.grid.spatial_exhaust_width_cm, 1.0e-12);
    const double atom_pump = dcr::physics::calculate_thermal_speed(
        temperatures.ion_eV, boundary.atom_mass_amu) / width;
    const double molecule_pump = dcr::physics::calculate_thermal_speed(
        config.plasma.neutral_molecule_temperature_eV, boundary.molecule_mass_amu) / width;
    for (int gi : boundary.atom_bg_indices) {
        if (gi >= 0 && gi < static_cast<int>(boundary.P_indices.size())) {}
        const int pi = (gi >= 0 && gi < static_cast<int>(levels.size()))
            ? std::find(boundary.P_indices.begin(), boundary.P_indices.end(), gi) - boundary.P_indices.begin()
            : p.size();
        if (pi >= 0 && pi < p.size()) out.pump_p(pi) = atom_pump * p(pi);
    }
    for (int gi : boundary.mol_bg_indices) {
        const int pi = (gi >= 0 && gi < static_cast<int>(levels.size()))
            ? std::find(boundary.P_indices.begin(), boundary.P_indices.end(), gi) - boundary.P_indices.begin()
            : p.size();
        if (pi >= 0 && pi < p.size()) out.pump_p(pi) = molecule_pump * p(pi);
    }
    out.chem_a = chemistry.flowA;
    out.chem_m = chemistry.flowM;
    out.pump_a = a * (dcr::physics::calculate_thermal_speed(
        config.plasma.neutral_atom_temperature_eV, boundary.atom_mass_amu) / width);
    out.pump_m = m * molecule_pump;
    return out;
}

struct MolecularRecovery {
    std::vector<dcr::base::Vector> density;
    double expected_target_ground_flux = 0.0;
    double maximum_scaled_balance = 0.0;
    bool valid = false;
};

MolecularRecovery recover_molecular_flow(
    const dcr::io::Config& config,
    const dcr::atomic::AtomicData& atomic_data,
    const dcr::state::PlasmaState& plasma,
    const EEDFGridView& grid,
    const BoundaryPhaseResult& boundary,
    const dcr::physics::WallRecycling& wall,
    const Layout& layout,
    const std::vector<double>& x,
    const std::vector<LocalRateCache>& cell_rate_caches,
    const State& reduced_state,
    double chemistry_fraction,
    double rate_ref,
    GlobalBVPSolveDiagnostics* diagnostics = nullptr) {
    MolecularRecovery out;
    out.density.assign(static_cast<size_t>(layout.nodes),
        dcr::base::Vector::Zero(layout.m));
    if (layout.m == 0) {
        out.valid = true;
        return out;
    }
    if (!(boundary.u_M > 0.0) || reduced_state.p.size() != static_cast<size_t>(layout.nodes)) {
        return out;
    }

    const auto& levels = atomic_data.get_levels();
    int ground_mi = -1;
    for (int mi = 0; mi < layout.m; ++mi) {
        if (boundary.M_indices[static_cast<size_t>(mi)] == boundary.molecule_ground) {
            ground_mi = mi;
            break;
        }
    }
    if (ground_mi < 0) return out;
    for (int gi : boundary.ion_indices) {
        if (levels[static_cast<size_t>(gi)].atomicity >= 2) continue;
        const int pi = layout.p_pos[static_cast<size_t>(gi)];
        if (pi < 0) continue;
        const double incident_flux = -global_ion_velocity_cm_s(
            config, levels[static_cast<size_t>(gi)], 0.0) * reduced_state.p[0](pi);
        out.expected_target_ground_flux +=
            config.numerics.boundary_molecular_flow_acceptance *
            wall.alpha_molecule * incident_flux / 2.0;
    }
    if (!std::isfinite(out.expected_target_ground_flux) ||
        out.expected_target_ground_flux < 0.0) return out;
    out.density[0](ground_mi) = out.expected_target_ground_flux / boundary.u_M;

    const double molecule_pump_rate = dcr::physics::calculate_thermal_speed(
        config.plasma.neutral_molecule_temperature_eV, boundary.molecule_mass_amu) /
        std::max(config.grid.spatial_exhaust_width_cm, 1.0e-12);
    for (int k = 0; k + 1 < layout.nodes; ++k) {
        const double dx = x[static_cast<size_t>(k + 1)] - x[static_cast<size_t>(k)];
        if (!(dx > 0.0)) return out;
        const auto& left = out.density[static_cast<size_t>(k)];
        dcr::base::Vector right = left;
        const dcr::base::Vector p_mid = 0.5 *
            (reduced_state.p[static_cast<size_t>(k)] +
             reduced_state.p[static_cast<size_t>(k + 1)]);
        const dcr::base::Vector a_mid = 0.5 *
            (reduced_state.a[static_cast<size_t>(k)] +
             reduced_state.a[static_cast<size_t>(k + 1)]);
        const double x_mid = 0.5 *
            (x[static_cast<size_t>(k)] + x[static_cast<size_t>(k + 1)]);
        bool converged = false;
        for (int iteration = 0; iteration < 80; ++iteration) {
            const dcr::base::Vector m_mid = 0.5 * (left + right);
            const NodeData node = evaluate_node(
                config, atomic_data, plasma, grid, boundary, p_mid, a_mid, m_mid,
                x_mid, false, &cell_rate_caches[static_cast<size_t>(k)]);
            dcr::base::Matrix rates = dcr::base::Matrix::Zero(layout.m, layout.m);
            for (int i = 0; i < layout.m; ++i) {
                const int gi = boundary.M_indices[static_cast<size_t>(i)];
                for (int j = 0; j < layout.m; ++j) {
                    const int gj = boundary.M_indices[static_cast<size_t>(j)];
                    rates(i, j) = node.local.R_source_full(gi, gj);
                }
            }
            const double transport = boundary.u_M / dx;
            dcr::base::Matrix lhs = transport *
                dcr::base::Matrix::Identity(layout.m, layout.m) -
                0.5 * chemistry_fraction * rates;
            lhs.diagonal().array() += 0.5 * molecule_pump_rate;
            dcr::base::Matrix rhs_matrix = transport *
                dcr::base::Matrix::Identity(layout.m, layout.m) +
                0.5 * chemistry_fraction * rates;
            rhs_matrix.diagonal().array() -= 0.5 * molecule_pump_rate;
            Eigen::FullPivLU<dcr::base::Matrix> lu(lhs);
            if (diagnostics) ++diagnostics->molecular_interval_factorizations;
            if (lu.rank() != layout.m) return out;
            const dcr::base::Vector candidate = lu.solve(rhs_matrix * left);
            if (!candidate.allFinite() || (candidate.array() < 0.0).any()) return out;
            const double change = (candidate - right).lpNorm<Eigen::Infinity>() /
                std::max(1.0, candidate.lpNorm<Eigen::Infinity>());
            right = candidate;
            if (change < 1.0e-15) {
                converged = true;
                break;
            }
        }
        if (!converged) return out;

        const dcr::base::Vector m_mid = 0.5 * (left + right);
        const NodeData node = evaluate_node(
            config, atomic_data, plasma, grid, boundary, p_mid, a_mid, m_mid,
            x_mid, false, &cell_rate_caches[static_cast<size_t>(k)]);
        const dcr::base::Vector balance = boundary.u_M * (right - left) / dx -
            chemistry_fraction * node.chem_m + node.pump_m;
        out.maximum_scaled_balance = std::max(out.maximum_scaled_balance,
            balance.cwiseAbs().maxCoeff() / std::max(1.0, rate_ref));
        if (!balance.allFinite() || out.maximum_scaled_balance > 1.0e-10) return out;
        out.density[static_cast<size_t>(k + 1)] = std::move(right);
    }
    out.valid = true;
    return out;
}

struct RateDerivativeSources {
    std::vector<LocalChemistrySources> by_global_state;
    std::vector<unsigned char> active;
};

RateDerivativeSources assemble_rate_derivative_sources(
    const NodeData& node,
    const BoundaryPhaseResult& boundary,
    const dcr::atomic::AtomicData& atomic_data,
    const dcr::base::Vector& p,
    const dcr::base::Vector& a,
    const dcr::base::Vector& m) {
    const int total_states = atomic_data.get_total_states();
    const auto& levels = atomic_data.get_levels();
    RateDerivativeSources out;
    out.by_global_state.resize(static_cast<size_t>(total_states));
    out.active.assign(static_cast<size_t>(total_states), 0);
    for (int gi = 0; gi < total_states; ++gi) {
        dcr::base::Matrix derivative = dcr::base::Matrix::Zero(total_states, total_states);
        bool active = false;
        const int charge = levels[static_cast<size_t>(gi)].charge;
        if (charge != 0 && node.local.dR_source_dne.size() > 0) {
            derivative += static_cast<double>(charge) * node.local.dR_source_dne;
            active = true;
        }
        for (size_t j = 0; j < node.local.rate_population_derivative_indices.size(); ++j) {
            if (node.local.rate_population_derivative_indices[j] != gi) continue;
            derivative += node.local.dR_source_dpopulation[j];
            active = true;
        }
        if (!active) continue;
        out.by_global_state[static_cast<size_t>(gi)] =
            assemble_local_chemistry_sources_from_rate_matrix(
                derivative, boundary, atomic_data, p, a, m);
        out.active[static_cast<size_t>(gi)] = 1;
    }
    return out;
}

struct Evaluator {
    const dcr::io::Config& config;
    const dcr::atomic::AtomicData& atomic_data;
    const dcr::state::PlasmaState& plasma;
    const EEDFGridView& grid;
    const BoundaryPhaseResult& boundary;
    const dcr::physics::WallRecycling& wall;
    const Layout& layout;
    const std::vector<double>& x;
    const std::vector<LocalRateCache>& cell_rate_caches;
    const LocalRateCache& upstream_rate_cache;
    double n_ref;
    double q_ref;
    double rate_ref;
    double local_rate_ref;
    double chemistry_fraction;
    GlobalBVPSolveDiagnostics* diagnostics = nullptr;
    mutable dcr::base::Vector cached_y;
    mutable State cached_state;
    mutable MolecularRecovery cached_molecular;
    mutable dcr::base::Vector cached_residual;
    mutable bool molecular_cache_valid = false;
    mutable bool residual_cache_valid = false;

    bool cache_matches(const dcr::base::Vector& y) const {
        return molecular_cache_valid && cached_y.size() == y.size() &&
            (cached_y.array() == y.array()).all();
    }

    bool recover(const dcr::base::Vector& y, State& state,
                  MolecularRecovery* diagnostics = nullptr) const {
        if (cache_matches(y)) {
            state = cached_state;
            if (diagnostics) *diagnostics = cached_molecular;
            if (this->diagnostics) ++this->diagnostics->molecular_transport_cache_hits;
            return cached_molecular.valid;
        }
        if (this->diagnostics) ++this->diagnostics->molecular_transport_evaluations;
        ScopedTimer timer(this->diagnostics
            ? &this->diagnostics->molecular_transport_seconds : nullptr);
        state = decode(y, layout, n_ref, q_ref);
        const MolecularRecovery recovered = recover_molecular_flow(
            config, atomic_data, plasma, grid, boundary, wall, layout, x,
            cell_rate_caches, state, chemistry_fraction, rate_ref, this->diagnostics);
        cached_y = y;
        cached_state = state;
        cached_molecular = recovered;
        molecular_cache_valid = true;
        residual_cache_valid = false;
        if (!recovered.valid) return false;
        state.m = recovered.density;
        cached_state = state;
        if (diagnostics) *diagnostics = recovered;
        return true;
    }

    dcr::base::Vector operator()(const dcr::base::Vector& y) const {
        if (diagnostics) ++diagnostics->residual_evaluations;
        ScopedTimer timer(diagnostics ? &diagnostics->residual_seconds : nullptr);
        if (residual_cache_valid && cache_matches(y)) {
            if (diagnostics) ++diagnostics->residual_cache_hits;
            return cached_residual;
        }
        State state;
        if (!recover(y, state)) {
            return dcr::base::Vector::Constant(
                layout.q_index + 1, std::numeric_limits<double>::quiet_NaN());
        }
        const NodeData upstream_node = evaluate_node(
            config, atomic_data, plasma, grid, boundary,
            state.p.back(), state.a.back(), state.m.back(), x.back(), false,
            &upstream_rate_cache);
        std::vector<NodeData> cells;
        cells.reserve(static_cast<size_t>(std::max(0, layout.nodes - 1)));
        for (int k = 0; k + 1 < layout.nodes; ++k) {
            const dcr::base::Vector p_mid = 0.5 *
                (state.p[static_cast<size_t>(k)] + state.p[static_cast<size_t>(k + 1)]);
            const dcr::base::Vector a_mid = 0.5 *
                (state.a[static_cast<size_t>(k)] + state.a[static_cast<size_t>(k + 1)]);
            const dcr::base::Vector m_mid = 0.5 *
                (state.m[static_cast<size_t>(k)] + state.m[static_cast<size_t>(k + 1)]);
            cells.push_back(evaluate_node(
                config, atomic_data, plasma, grid, boundary, p_mid, a_mid, m_mid,
                0.5 * (x[static_cast<size_t>(k)] + x[static_cast<size_t>(k + 1)]),
                false, &cell_rate_caches[static_cast<size_t>(k)]));
        }
        dcr::base::Vector residual = dcr::base::Vector::Zero(layout.q_index + 1);
        const auto& levels = atomic_data.get_levels();
        for (int k = 0; k < layout.nodes; ++k) {
            const int base = offset(layout, k);
            const auto& p_node = k + 1 < layout.nodes
                ? cells[static_cast<size_t>(k)] : upstream_node;
            for (int pi = 0; pi < layout.p; ++pi) {
                const int gi = boundary.P_indices[static_cast<size_t>(pi)];
                if (layout.positive_ion[static_cast<size_t>(pi)] && k + 1 < layout.nodes) {
                    const double dx = x[static_cast<size_t>(k + 1)] - x[static_cast<size_t>(k)];
                    const double velocity_right = global_ion_velocity_cm_s(
                        config, levels[static_cast<size_t>(gi)], x[static_cast<size_t>(k + 1)]);
                    const double velocity_left = global_ion_velocity_cm_s(
                        config, levels[static_cast<size_t>(gi)], x[static_cast<size_t>(k)]);
                    residual(base + pi) = (conservative_flux_divergence_cm3_s(
                        velocity_left, state.p[static_cast<size_t>(k)](pi),
                        velocity_right, state.p[static_cast<size_t>(k + 1)](pi), dx) -
                        chemistry_fraction * p_node.chem_p(pi)) / rate_ref;
                } else if (layout.positive_ion[static_cast<size_t>(pi)]) {
                    const double flux = global_ion_velocity_cm_s(
                        config, levels[static_cast<size_t>(gi)], x.back()) * state.p.back()(pi);
                    residual(base + pi) = (pi == layout.proton_pi)
                        ? (flux + state.q_up) / q_ref
                        : flux / q_ref;
                } else {
                    residual(base + pi) =
                        (-chemistry_fraction * p_node.chem_p(pi) + p_node.pump_p(pi)) /
                        local_rate_ref;
                }
            }
            for (int ai = 0; ai < layout.a; ++ai) {
                const int row = base + layout.p + ai;
                if (k == 0) {
                    double launched = 0.0;
                    if (boundary.A_indices[static_cast<size_t>(ai)] == boundary.atom_ground) {
                        for (int gi : boundary.ion_indices) {
                            const int pi = layout.p_pos[static_cast<size_t>(gi)];
                            const double gamma = -global_ion_velocity_cm_s(
                                config, levels[static_cast<size_t>(gi)], 0.0) * state.p[0](pi);
                            launched += (levels[static_cast<size_t>(gi)].atomicity >= 2)
                                ? levels[static_cast<size_t>(gi)].atomicity * gamma
                                : wall.alpha_atom * gamma;
                        }
                    }
                    residual(row) = (boundary.u_A * state.a[0](ai) - launched) / q_ref;
                } else {
                    const auto& cell = cells[static_cast<size_t>(k - 1)];
                    const double dx = x[static_cast<size_t>(k)] - x[static_cast<size_t>(k - 1)];
                    residual(row) = ((boundary.u_A * state.a[static_cast<size_t>(k)](ai) -
                        boundary.u_A * state.a[static_cast<size_t>(k - 1)](ai)) / dx -
                        chemistry_fraction * cell.chem_a(ai) + cell.pump_a(ai)) / rate_ref;
                }
            }
        }
        double target_nuclei = 0.0;
        for (int pi = 0; pi < layout.p; ++pi) {
            const int gi = boundary.P_indices[static_cast<size_t>(pi)];
            target_nuclei += std::max(1, levels[static_cast<size_t>(gi)].atomicity) * state.p[0](pi);
        }
        for (int ai = 0; ai < layout.a; ++ai) {
            const int gi = boundary.A_indices[static_cast<size_t>(ai)];
            target_nuclei += std::max(1, levels[static_cast<size_t>(gi)].atomicity) * state.a[0](ai);
        }
        for (int mi = 0; mi < layout.m; ++mi) {
            const int gi = boundary.M_indices[static_cast<size_t>(mi)];
            target_nuclei += std::max(1, levels[static_cast<size_t>(gi)].atomicity) * state.m[0](mi);
        }
        residual(layout.q_index) = (target_nuclei - config.plasma.total_density) / n_ref;
        cached_residual = residual;
        residual_cache_valid = true;
        return residual;
    }
};

dcr::base::Vector initial_guess(
    const Layout& layout,
    const BoundaryPhaseResult& boundary,
    const std::vector<double>& x,
    const dcr::io::Config& config,
    const std::vector<dcr::atomic::EnergyLevel>& levels,
    double n_ref,
    double q_ref) {
    constexpr double relative_seed = 1.0e-12;
    const double density_seed = n_ref * relative_seed;
    const double flux_seed = q_ref * relative_seed;
    dcr::base::Vector y = dcr::base::Vector::Constant(
        layout.q_index + 1, std::log(relative_seed));
    const auto p0 = compact_boundary_p(boundary);
    const dcr::base::Vector a0 = boundary.have_flow_last
        ? boundary.flowA_last : dcr::base::Vector::Constant(layout.a, 1.0e-60);
    for (int k = 0; k < layout.nodes; ++k) {
        const int base = offset(layout, k);
        for (int i = 0; i < layout.p; ++i) {
            y(base + i) = std::log(std::max(p0(i), density_seed) / n_ref);
        }
        const double decay = std::exp(-x[static_cast<size_t>(k)] / std::max(config.grid.length_cm, 1.0e-12));
        for (int i = 0; i < layout.a; ++i) y(base + layout.p + i) =
            std::log(std::max(a0(i) * decay, density_seed) / n_ref);
    }
    const int proton_gi = boundary.P_indices[static_cast<size_t>(layout.proton_pi)];
    const double q0 = -global_ion_velocity_cm_s(
        config, levels[static_cast<size_t>(proton_gi)], x.back()) * p0(layout.proton_pi);
    y(layout.q_index) = std::log(std::max(q0, flux_seed) / q_ref);
    return y;
}

dcr::base::Vector interpolate_initial_guess(
    const GlobalBVPResult& coarse,
    const Layout& layout,
    const BoundaryPhaseResult& boundary,
    const std::vector<double>& x,
    const dcr::io::Config& config,
    const std::vector<dcr::atomic::EnergyLevel>& levels,
    double n_ref,
    double q_ref) {
    if (coarse.history.x_cm.size() < 2 ||
        coarse.history.background_full.size() != coarse.history.x_cm.size() ||
        coarse.history.flowA.size() != coarse.history.x_cm.size()) {
        throw std::runtime_error("Invalid coarse global-BVP solution for interpolation.");
    }
    constexpr double relative_floor = 1.0e-12;
    dcr::base::Vector y = dcr::base::Vector::Constant(
        layout.q_index + 1, std::log(relative_floor));
    for (int k = 0; k < layout.nodes; ++k) {
        const double position = x[static_cast<size_t>(k)];
        const bool extending_upstream = position > coarse.history.x_cm.back();
        const auto upper_iterator = std::upper_bound(
            coarse.history.x_cm.begin(), coarse.history.x_cm.end(), position);
        size_t upper = static_cast<size_t>(
            std::distance(coarse.history.x_cm.begin(), upper_iterator));
        if (upper == 0) upper = 1;
        if (upper >= coarse.history.x_cm.size()) upper = coarse.history.x_cm.size() - 1;
        const size_t lower = upper - 1;
        const double span = coarse.history.x_cm[upper] - coarse.history.x_cm[lower];
        const double fraction = span > 0.0
            ? std::clamp((position - coarse.history.x_cm[lower]) / span, 0.0, 1.0)
            : 0.0;
        const int base = offset(layout, k);
        for (int pi = 0; pi < layout.p; ++pi) {
            const int gi = boundary.P_indices[static_cast<size_t>(pi)];
            double density = (1.0 - fraction) * coarse.history.background_full[lower](gi) +
                fraction * coarse.history.background_full[upper](gi);
            if (extending_upstream && layout.positive_ion[static_cast<size_t>(pi)]) {
                const double old_velocity = global_ion_velocity_cm_s(
                    config, levels[static_cast<size_t>(gi)], coarse.history.x_cm.back());
                const double new_velocity = global_ion_velocity_cm_s(
                    config, levels[static_cast<size_t>(gi)], position);
                if (std::abs(new_velocity) > 0.0) {
                    density = coarse.history.background_full.back()(gi) *
                        old_velocity / new_velocity;
                }
            }
            y(base + pi) = std::log(std::max(density / n_ref, relative_floor));
        }
        auto extrapolated_flow_density = [&](const std::vector<dcr::base::Vector>& flow, int i) {
            if (!extending_upstream) {
                return (1.0 - fraction) * flow[lower](i) + fraction * flow[upper](i);
            }
            const size_t last = flow.size() - 1;
            const size_t previous = last - 1;
            const double dx = coarse.history.x_cm[last] - coarse.history.x_cm[previous];
            double slope = 0.0;
            if (dx > 0.0) {
                slope = (std::log(std::max(flow[last](i), n_ref * relative_floor)) -
                    std::log(std::max(flow[previous](i), n_ref * relative_floor))) / dx;
            }
            slope = std::clamp(slope, -100.0, 0.0);
            return flow[last](i) * std::exp(
                slope * (position - coarse.history.x_cm.back()));
        };
        for (int ai = 0; ai < layout.a; ++ai) {
            const double density = extrapolated_flow_density(coarse.history.flowA, ai);
            y(base + layout.p + ai) =
                std::log(std::max(density / n_ref, relative_floor));
        }
    }
    const bool same_domain = std::abs(x.back() - coarse.history.x_cm.back()) <=
        1.0e-12 * std::max(1.0, std::abs(x.back()));
    if (same_domain) {
        auto trapezoid = [](const std::vector<double>& positions, auto value_at) {
            double integral = 0.0;
            for (size_t k = 0; k + 1 < positions.size(); ++k) {
                integral += 0.5 * (positions[k + 1] - positions[k]) *
                    (value_at(k) + value_at(k + 1));
            }
            return integral;
        };
        for (int pi = 0; pi < layout.p; ++pi) {
            const int gi = boundary.P_indices[static_cast<size_t>(pi)];
            const double old_integral = trapezoid(coarse.history.x_cm,
                [&](size_t k) { return coarse.history.background_full[k](gi); });
            const double new_integral = trapezoid(x, [&](size_t k) {
                return n_ref * safe_exp(y(offset(layout, static_cast<int>(k)) + pi));
            });
            if (old_integral > 0.0 && new_integral > 0.0) {
                const double log_scale = std::log(old_integral / new_integral);
                for (int k = 0; k < layout.nodes; ++k) {
                    y(offset(layout, k) + pi) += log_scale;
                }
            }
        }
        for (int ai = 0; ai < layout.a; ++ai) {
            const double old_integral = trapezoid(coarse.history.x_cm,
                [&](size_t k) { return coarse.history.flowA[k](ai); });
            const double new_integral = trapezoid(x, [&](size_t k) {
                return n_ref * safe_exp(
                    y(offset(layout, static_cast<int>(k)) + layout.p + ai));
            });
            if (old_integral > 0.0 && new_integral > 0.0) {
                const double log_scale = std::log(old_integral / new_integral);
                for (int k = 0; k < layout.nodes; ++k) {
                    y(offset(layout, k) + layout.p + ai) += log_scale;
                }
            }
        }
    }
    y(layout.q_index) = std::log(std::max(
        coarse.upstream_proton_flux_cm2_s / q_ref, relative_floor));
    return y;
}

SparseMatrix assemble_sparse_jacobian(
    const Evaluator& evaluator,
    const dcr::base::Vector& y) {
    State state;
    if (!evaluator.recover(y, state)) {
        throw std::runtime_error("Reduced preconditioner molecular recovery failed.");
    }
    const auto& layout = evaluator.layout;
    const auto& boundary = evaluator.boundary;
    const auto& levels = evaluator.atomic_data.get_levels();
    std::vector<Triplet> triplets;
    triplets.reserve(static_cast<size_t>(y.size()) * static_cast<size_t>(layout.block + 4));
    auto add = [&](int row, int col, double value) {
        if (std::isfinite(value) && std::abs(value) > 1.0e-18) {
            triplets.emplace_back(row, col, value);
        }
    };

    // Every interval uses one midpoint chemistry state. The same chemistry
    // vector feeds P row k and counterstreaming A rows k+1. Derived M is held
    // fixed here; the matrix-free operator supplies all through-M coupling.
    for (int k = 0; k + 1 < layout.nodes; ++k) {
        const int left = offset(layout, k);
        const int right = offset(layout, k + 1);
        const dcr::base::Vector p_mid = 0.5 *
            (state.p[static_cast<size_t>(k)] + state.p[static_cast<size_t>(k + 1)]);
        const dcr::base::Vector a_mid = 0.5 *
            (state.a[static_cast<size_t>(k)] + state.a[static_cast<size_t>(k + 1)]);
        const dcr::base::Vector m_mid = 0.5 *
            (state.m[static_cast<size_t>(k)] + state.m[static_cast<size_t>(k + 1)]);
        const auto cell = evaluate_node(
            evaluator.config, evaluator.atomic_data, evaluator.plasma, evaluator.grid,
            boundary, p_mid, a_mid, m_mid,
            0.5 * (evaluator.x[static_cast<size_t>(k)] + evaluator.x[static_cast<size_t>(k + 1)]),
            true, &evaluator.cell_rate_caches[static_cast<size_t>(k)]);
        const auto rate_derivatives = assemble_rate_derivative_sources(
            cell, boundary, evaluator.atomic_data, p_mid, a_mid, m_mid);

        for (int pi = 0; pi < layout.p; ++pi) {
            const int row = left + pi;
            const int gi = boundary.P_indices[static_cast<size_t>(pi)];
            const double scale = layout.positive_ion[static_cast<size_t>(pi)]
                ? evaluator.rate_ref : evaluator.local_rate_ref;
            for (int pj = 0; pj < layout.p; ++pj) {
                const int gj = boundary.P_indices[static_cast<size_t>(pj)];
                double coefficient = -evaluator.chemistry_fraction *
                    cell.local.R_source_full(gi, gj);
                if (!layout.positive_ion[static_cast<size_t>(pi)] && pi == pj &&
                    cell.pump_p(pi) > 0.0 && p_mid(pi) > 0.0) {
                    coefficient += cell.pump_p(pi) / p_mid(pi);
                }
                add(row, left + pj, coefficient * 0.5 * state.p[static_cast<size_t>(k)](pj) / scale);
                add(row, right + pj, coefficient * 0.5 * state.p[static_cast<size_t>(k + 1)](pj) / scale);
            }
            const auto& level = levels[static_cast<size_t>(gi)];
            const bool charged = level.type == dcr::atomic::SpeciesType::Ion && level.charge != 0;
            const bool atom = level.type == dcr::atomic::SpeciesType::Atom && level.charge == 0;
            const bool molecule = level.type == dcr::atomic::SpeciesType::Molecule && level.charge == 0;
            if (charged || molecule) {
                for (int ai = 0; ai < layout.a; ++ai) {
                    const int gj = boundary.A_indices[static_cast<size_t>(ai)];
                    const double coefficient = -evaluator.chemistry_fraction *
                        cell.local.R_source_full(gi, gj) / scale;
                    add(row, left + layout.p + ai,
                        coefficient * 0.5 * state.a[static_cast<size_t>(k)](ai));
                    add(row, right + layout.p + ai,
                        coefficient * 0.5 * state.a[static_cast<size_t>(k + 1)](ai));
                }
            }
            (void)charged;
            (void)atom;
        }

        for (int ai = 0; ai < layout.a; ++ai) {
            const int row = right + layout.p + ai;
            const int gi = boundary.A_indices[static_cast<size_t>(ai)];
            for (int aj = 0; aj < layout.a; ++aj) {
                const int gj = boundary.A_indices[static_cast<size_t>(aj)];
                double coefficient = -evaluator.chemistry_fraction *
                    cell.local.R_source_full(gi, gj);
                if (ai == aj && cell.pump_a(ai) > 0.0 && a_mid(ai) > 0.0) {
                    coefficient += cell.pump_a(ai) / a_mid(ai);
                }
                add(row, left + layout.p + aj,
                    coefficient * 0.5 * state.a[static_cast<size_t>(k)](aj) / evaluator.rate_ref);
                add(row, right + layout.p + aj,
                    coefficient * 0.5 * state.a[static_cast<size_t>(k + 1)](aj) / evaluator.rate_ref);
            }
        }
        auto add_rate_column = [&](int column, int global_index, double population_derivative) {
            if (global_index < 0 || global_index >= static_cast<int>(rate_derivatives.active.size()) ||
                !rate_derivatives.active[static_cast<size_t>(global_index)]) return;
            const auto& source = rate_derivatives.by_global_state[static_cast<size_t>(global_index)];
            for (int pi = 0; pi < layout.p; ++pi) {
                const double scale = layout.positive_ion[static_cast<size_t>(pi)]
                    ? evaluator.rate_ref : evaluator.local_rate_ref;
                add(left + pi, column,
                    -evaluator.chemistry_fraction * source.background(pi) *
                    population_derivative / scale);
            }
            for (int ai = 0; ai < layout.a; ++ai) {
                add(right + layout.p + ai, column,
                    -evaluator.chemistry_fraction * source.flowA(ai) *
                    population_derivative / evaluator.rate_ref);
            }
        };
        for (int pj = 0; pj < layout.p; ++pj) {
            const int gi = boundary.P_indices[static_cast<size_t>(pj)];
            add_rate_column(left + pj, gi, 0.5 * state.p[static_cast<size_t>(k)](pj));
            add_rate_column(right + pj, gi, 0.5 * state.p[static_cast<size_t>(k + 1)](pj));
        }
        for (int ai = 0; ai < layout.a; ++ai) {
            const int gi = boundary.A_indices[static_cast<size_t>(ai)];
            add_rate_column(left + layout.p + ai, gi, 0.5 * state.a[static_cast<size_t>(k)](ai));
            add_rate_column(right + layout.p + ai, gi, 0.5 * state.a[static_cast<size_t>(k + 1)](ai));
        }
    }

    // The upstream node retains algebraic equations for non-transported P states.
    const int last_node = layout.nodes - 1;
    const int last = offset(layout, last_node);
    const auto upstream = evaluate_node(
        evaluator.config, evaluator.atomic_data, evaluator.plasma, evaluator.grid,
        boundary, state.p.back(), state.a.back(), state.m.back(), evaluator.x.back(), true,
        &evaluator.upstream_rate_cache);
    const auto upstream_rate_derivatives = assemble_rate_derivative_sources(
        upstream, boundary, evaluator.atomic_data, state.p.back(), state.a.back(), state.m.back());
    for (int pi = 0; pi < layout.p; ++pi) {
        if (layout.positive_ion[static_cast<size_t>(pi)]) continue;
        const int row = last + pi;
        const int gi = boundary.P_indices[static_cast<size_t>(pi)];
        for (int pj = 0; pj < layout.p; ++pj) {
            const int gj = boundary.P_indices[static_cast<size_t>(pj)];
            double coefficient = -evaluator.chemistry_fraction * upstream.local.R_source_full(gi, gj);
            if (pi == pj && upstream.pump_p(pi) > 0.0 && state.p.back()(pi) > 0.0) {
                coefficient += upstream.pump_p(pi) / state.p.back()(pi);
            }
            add(row, last + pj,
                coefficient * state.p.back()(pj) / evaluator.local_rate_ref);
        }
        const auto& level = levels[static_cast<size_t>(gi)];
        const bool charged = level.type == dcr::atomic::SpeciesType::Ion && level.charge != 0;
        const bool atom = level.type == dcr::atomic::SpeciesType::Atom && level.charge == 0;
        const bool molecule = level.type == dcr::atomic::SpeciesType::Molecule && level.charge == 0;
        if (charged || molecule) {
            for (int ai = 0; ai < layout.a; ++ai) {
                const int gj = boundary.A_indices[static_cast<size_t>(ai)];
                add(row, last + layout.p + ai,
                    -evaluator.chemistry_fraction * upstream.local.R_source_full(gi, gj) *
                    state.a.back()(ai) / evaluator.local_rate_ref);
            }
        }
        (void)atom;
    }
    auto add_upstream_rate_column = [&](int column, int global_index, double population_derivative) {
        if (global_index < 0 || global_index >= static_cast<int>(upstream_rate_derivatives.active.size()) ||
            !upstream_rate_derivatives.active[static_cast<size_t>(global_index)]) return;
        const auto& source = upstream_rate_derivatives.by_global_state[static_cast<size_t>(global_index)];
        for (int pi = 0; pi < layout.p; ++pi) {
            if (layout.positive_ion[static_cast<size_t>(pi)]) continue;
            add(last + pi, column,
                -evaluator.chemistry_fraction * source.background(pi) *
                population_derivative / evaluator.local_rate_ref);
        }
    };
    for (int pj = 0; pj < layout.p; ++pj) {
        add_upstream_rate_column(last + pj, boundary.P_indices[static_cast<size_t>(pj)],
            state.p.back()(pj));
    }
    for (int ai = 0; ai < layout.a; ++ai) {
        add_upstream_rate_column(last + layout.p + ai,
            boundary.A_indices[static_cast<size_t>(ai)], state.a.back()(ai));
    }

    // Signed face-flux and inflow-boundary derivatives.
    for (int k = 0; k < layout.nodes; ++k) {
        const int base = offset(layout, k);
        if (k + 1 < layout.nodes) {
            const double dx = evaluator.x[static_cast<size_t>(k + 1)] - evaluator.x[static_cast<size_t>(k)];
            for (int pi = 0; pi < layout.p; ++pi) {
                if (!layout.positive_ion[static_cast<size_t>(pi)]) continue;
                const int gi = boundary.P_indices[static_cast<size_t>(pi)];
                add(base + pi, base + pi,
                    -global_ion_velocity_cm_s(evaluator.config, levels[static_cast<size_t>(gi)], evaluator.x[static_cast<size_t>(k)]) *
                    state.p[static_cast<size_t>(k)](pi) / (dx * evaluator.rate_ref));
                add(base + pi, offset(layout, k + 1) + pi,
                    global_ion_velocity_cm_s(evaluator.config, levels[static_cast<size_t>(gi)], evaluator.x[static_cast<size_t>(k + 1)]) *
                    state.p[static_cast<size_t>(k + 1)](pi) / (dx * evaluator.rate_ref));
            }
        } else {
            for (int pi = 0; pi < layout.p; ++pi) {
                if (!layout.positive_ion[static_cast<size_t>(pi)]) continue;
                const int gi = boundary.P_indices[static_cast<size_t>(pi)];
                add(base + pi, base + pi,
                    global_ion_velocity_cm_s(evaluator.config, levels[static_cast<size_t>(gi)], evaluator.x.back()) *
                    state.p.back()(pi) / evaluator.q_ref);
                if (pi == layout.proton_pi) add(base + pi, layout.q_index, state.q_up / evaluator.q_ref);
            }
        }

        for (int ai = 0; ai < layout.a; ++ai) {
            const int row = base + layout.p + ai;
            if (k == 0) {
                add(row, row, boundary.u_A * state.a[0](ai) / evaluator.q_ref);
                if (boundary.A_indices[static_cast<size_t>(ai)] == boundary.atom_ground) {
                    for (int gi : boundary.ion_indices) {
                        const int pi = layout.p_pos[static_cast<size_t>(gi)];
                        const double yield = levels[static_cast<size_t>(gi)].atomicity >= 2
                            ? levels[static_cast<size_t>(gi)].atomicity : evaluator.wall.alpha_atom;
                        add(row, pi, yield * global_ion_velocity_cm_s(
                            evaluator.config, levels[static_cast<size_t>(gi)], 0.0) *
                            state.p[0](pi) / evaluator.q_ref);
                    }
                }
            } else {
                const double dx = evaluator.x[static_cast<size_t>(k)] - evaluator.x[static_cast<size_t>(k - 1)];
                add(row, row, boundary.u_A * state.a[static_cast<size_t>(k)](ai) /
                    (dx * evaluator.rate_ref));
                add(row, offset(layout, k - 1) + layout.p + ai,
                    -boundary.u_A * state.a[static_cast<size_t>(k - 1)](ai) /
                    (dx * evaluator.rate_ref));
            }
        }
    }

    // Target nuclei closure.
    for (int pi = 0; pi < layout.p; ++pi) {
        const int gi = boundary.P_indices[static_cast<size_t>(pi)];
        add(layout.q_index, pi,
            std::max(1, levels[static_cast<size_t>(gi)].atomicity) * state.p[0](pi) /
            evaluator.n_ref);
    }
    for (int ai = 0; ai < layout.a; ++ai) {
        const int gi = boundary.A_indices[static_cast<size_t>(ai)];
        add(layout.q_index, layout.p + ai,
            std::max(1, levels[static_cast<size_t>(gi)].atomicity) * state.a[0](ai) /
            evaluator.n_ref);
    }
    SparseMatrix jacobian(y.size(), y.size());
    jacobian.setFromTriplets(triplets.begin(), triplets.end());
    return jacobian;
}

SparseMatrix assemble_finite_difference_jacobian(
    const Evaluator& evaluator,
    const dcr::base::Vector& y,
    const dcr::base::Vector& residual) {
    std::vector<Triplet> triplets;
    const double epsilon = 1.0e-6;
    for (int column = 0; column < y.size(); ++column) {
        dcr::base::Vector perturbed = y;
        perturbed(column) += epsilon;
        const dcr::base::Vector derivative = (evaluator(perturbed) - residual) / epsilon;
        for (int row = 0; row < derivative.size(); ++row) {
            if (std::isfinite(derivative(row)) && std::abs(derivative(row)) > 1.0e-14) {
                triplets.emplace_back(row, column, derivative(row));
            }
        }
    }
    SparseMatrix jacobian(y.size(), y.size());
    jacobian.setFromTriplets(triplets.begin(), triplets.end());
    return jacobian;
}

dcr::base::Vector matrix_free_jacobian_action(
    const Evaluator& evaluator,
    const dcr::base::Vector& y,
    const dcr::base::Vector& direction,
    double normalized_step = 1.0e-6) {
    if (evaluator.diagnostics) ++evaluator.diagnostics->jacobian_vector_products;
    const double norm = direction.norm();
    if (norm == 0.0) return dcr::base::Vector::Zero(y.size());
    const dcr::base::Vector unit = direction / norm;
    const dcr::base::Vector plus = evaluator(y + normalized_step * unit);
    const dcr::base::Vector minus = evaluator(y - normalized_step * unit);
    if (!plus.allFinite() || !minus.allFinite()) {
        const dcr::base::Vector center = evaluator(y);
        if (!center.allFinite() || !plus.allFinite()) {
            return dcr::base::Vector::Constant(
                y.size(), std::numeric_limits<double>::quiet_NaN());
        }
        return norm * (plus - center) / normalized_step;
    }
    return norm * (plus - minus) / (2.0 * normalized_step);
}

struct GMRESResult {
    dcr::base::Vector solution;
    double relative_residual = std::numeric_limits<double>::infinity();
    int iterations = 0;
    bool converged = false;
};

template <typename Operator>
GMRESResult restarted_gmres(
    const Operator& apply,
    const dcr::base::Vector& rhs,
    int restart,
    int maximum_iterations,
    double tolerance) {
    GMRESResult out;
    const int n = rhs.size();
    out.solution = dcr::base::Vector::Zero(n);
    const double rhs_norm = rhs.norm();
    if (rhs_norm == 0.0) {
        out.relative_residual = 0.0;
        out.converged = true;
        return out;
    }
    restart = std::max(1, std::min(restart, n));
    dcr::base::Vector residual = rhs;
    while (out.iterations < maximum_iterations) {
        const double beta = residual.norm();
        out.relative_residual = beta / rhs_norm;
        if (out.relative_residual < tolerance) {
            out.converged = true;
            break;
        }
        const int cycle = std::min(restart, maximum_iterations - out.iterations);
        dcr::base::Matrix basis = dcr::base::Matrix::Zero(n, cycle + 1);
        dcr::base::Matrix hessenberg = dcr::base::Matrix::Zero(cycle + 1, cycle);
        basis.col(0) = residual / beta;
        int used = 0;
        dcr::base::Vector cycle_solution;
        for (int j = 0; j < cycle; ++j) {
            dcr::base::Vector image = apply(basis.col(j));
            if (!image.allFinite()) return out;
            for (int i = 0; i <= j; ++i) {
                hessenberg(i, j) = basis.col(i).dot(image);
                image -= hessenberg(i, j) * basis.col(i);
            }
            // A second pass prevents loss of orthogonality in stiff log systems.
            for (int i = 0; i <= j; ++i) {
                const double correction = basis.col(i).dot(image);
                hessenberg(i, j) += correction;
                image -= correction * basis.col(i);
            }
            hessenberg(j + 1, j) = image.norm();
            if (hessenberg(j + 1, j) > 1.0e-14) {
                basis.col(j + 1) = image / hessenberg(j + 1, j);
            }
            dcr::base::Vector projected_rhs = dcr::base::Vector::Zero(j + 2);
            projected_rhs(0) = beta;
            cycle_solution = hessenberg.topLeftCorner(j + 2, j + 1)
                .colPivHouseholderQr().solve(projected_rhs);
            used = j + 1;
            ++out.iterations;
            const double projected_residual =
                (projected_rhs - hessenberg.topLeftCorner(j + 2, j + 1) *
                    cycle_solution).norm() / rhs_norm;
            if (projected_residual < tolerance ||
                hessenberg(j + 1, j) <= 1.0e-14) break;
        }
        if (used == 0 || !cycle_solution.allFinite()) return out;
        out.solution += basis.leftCols(used) * cycle_solution;
        residual = rhs - apply(out.solution);
        if (!residual.allFinite()) return out;
    }
    out.relative_residual = (rhs - apply(out.solution)).norm() / rhs_norm;
    out.converged = out.relative_residual < tolerance;
    return out;
}

struct ResidualBlockNorms {
    double ion = 0.0;
    double flow_a = 0.0;
    double flow_m = 0.0;
    double local = 0.0;
    double boundary = 0.0;
};

ResidualBlockNorms residual_block_norms(
    const dcr::base::Vector& residual,
    const Layout& layout) {
    ResidualBlockNorms out;
    for (int k = 0; k < layout.nodes; ++k) {
        const int base = offset(layout, k);
        for (int pi = 0; pi < layout.p; ++pi) {
            const double value = std::abs(residual(base + pi));
            if (layout.positive_ion[static_cast<size_t>(pi)]) {
                if (k + 1 == layout.nodes) out.boundary = std::max(out.boundary, value);
                else out.ion = std::max(out.ion, value);
            } else {
                out.local = std::max(out.local, value);
            }
        }
        for (int ai = 0; ai < layout.a; ++ai) {
            const double value = std::abs(residual(base + layout.p + ai));
            if (k == 0) out.boundary = std::max(out.boundary, value);
            else out.flow_a = std::max(out.flow_a, value);
        }
    }
    out.boundary = std::max(out.boundary, std::abs(residual(layout.q_index)));
    return out;
}

std::pair<const char*, double> largest_residual_block(
    const ResidualBlockNorms& blocks) {
    std::pair<const char*, double> largest{"ion", blocks.ion};
    for (const auto& block : std::array<std::pair<const char*, double>, 4>{
             std::pair<const char*, double>{"A", blocks.flow_a},
             {"M", blocks.flow_m}, {"local", blocks.local},
             {"boundary", blocks.boundary}}) {
        if (block.second > largest.second) largest = block;
    }
    return largest;
}

double maximum_nuclei_density(
    const State& state,
    const Layout& layout,
    const BoundaryPhaseResult& boundary,
    const std::vector<dcr::atomic::EnergyLevel>& levels) {
    double maximum = 0.0;
    for (int k = 0; k < layout.nodes; ++k) {
        double nuclei = 0.0;
        for (int pi = 0; pi < layout.p; ++pi) {
            const int gi = boundary.P_indices[static_cast<size_t>(pi)];
            nuclei += levels[static_cast<size_t>(gi)].atomicity * state.p[k](pi);
        }
        for (int ai = 0; ai < layout.a; ++ai) {
            const int gi = boundary.A_indices[static_cast<size_t>(ai)];
            nuclei += levels[static_cast<size_t>(gi)].atomicity * state.a[k](ai);
        }
        for (int mi = 0; mi < layout.m; ++mi) {
            const int gi = boundary.M_indices[static_cast<size_t>(mi)];
            nuclei += levels[static_cast<size_t>(gi)].atomicity * state.m[k](mi);
        }
        maximum = std::max(maximum, nuclei);
    }
    return maximum;
}

double sparse_one_norm(const SparseMatrix& matrix) {
    double norm = 0.0;
    for (int column = 0; column < matrix.outerSize(); ++column) {
        double sum = 0.0;
        for (SparseMatrix::InnerIterator entry(matrix, column); entry; ++entry) {
            sum += std::abs(entry.value());
        }
        norm = std::max(norm, sum);
    }
    return norm;
}

template <typename Solver>
double inverse_one_norm_estimate(Solver& solver, int size);

double estimate_reduced_preconditioner_condition(
    const Evaluator& evaluator,
    const dcr::base::Vector& y,
    GlobalBVPSolveDiagnostics& diagnostics) {
    ScopedTimer timer(&diagnostics.condition_seconds);
    SparseMatrix jacobian = assemble_sparse_jacobian(evaluator, y);
    jacobian.makeCompressed();
    Eigen::SparseLU<SparseMatrix, Eigen::COLAMDOrdering<int>> solver;
    solver.analyzePattern(jacobian);
    solver.factorize(jacobian);
    if (solver.info() != Eigen::Success) {
        return std::numeric_limits<double>::infinity();
    }
    return sparse_one_norm(jacobian) *
        inverse_one_norm_estimate(solver, jacobian.rows());
}

template <typename Solver>
dcr::base::Vector estimate_smallest_right_singular_vector(
    Solver& solver,
    const SparseMatrix& matrix) {
    dcr::base::Vector direction(matrix.rows());
    for (int i = 0; i < direction.size(); ++i) {
        direction(i) = std::sin(0.73 * (i + 1));
    }
    direction.normalize();
    for (int iteration = 0; iteration < 10; ++iteration) {
        const dcr::base::Vector transpose_solution =
            solver.transpose().solve(direction);
        const dcr::base::Vector next = solver.solve(transpose_solution);
        if (!next.allFinite() || next.norm() == 0.0) break;
        direction = next.normalized();
    }
    return direction;
}

void print_smallest_mode(
    const Evaluator& evaluator,
    const dcr::base::Vector& y,
    const SparseMatrix& jacobian,
    Eigen::SparseLU<SparseMatrix, Eigen::COLAMDOrdering<int>>& solver) {
    const dcr::base::Vector mode = estimate_smallest_right_singular_vector(
        solver, jacobian);
    const double singular_value_estimate = (jacobian * mode).norm();
    const auto state = decode(
        y, evaluator.layout, evaluator.n_ref, evaluator.q_ref);
    std::vector<int> indices(static_cast<size_t>(mode.size()));
    for (int i = 0; i < mode.size(); ++i) indices[static_cast<size_t>(i)] = i;
    std::sort(indices.begin(), indices.end(), [&](int left, int right) {
        return std::abs(mode(left)) > std::abs(mode(right));
    });
    const auto& layout = evaluator.layout;
    const auto& levels = evaluator.atomic_data.get_levels();
    for (size_t rank = 0; rank < std::min<size_t>(20, indices.size()); ++rank) {
        const int index = indices[rank];
        if (index == layout.q_index) {
            std::cout << "[DCR_Solver][global-bvp-smallest-mode] rank=" << (rank + 1)
                      << " singular_value_estimate=" << singular_value_estimate
                      << " variable=q_up component=" << mode(index)
                      << " value=" << state.q_up << " floor_hit=false" << std::endl;
            continue;
        }
        const int node = index / layout.block;
        const int local = index - offset(layout, node);
        const char* block = "P";
        int state_index = local;
        int gi = evaluator.boundary.P_indices[static_cast<size_t>(state_index)];
        double density = state.p[node](state_index);
        if (local >= layout.p + layout.a) {
            block = "M";
            state_index = local - layout.p - layout.a;
            gi = evaluator.boundary.M_indices[static_cast<size_t>(state_index)];
            density = state.m[node](state_index);
        } else if (local >= layout.p) {
            block = "A";
            state_index = local - layout.p;
            gi = evaluator.boundary.A_indices[static_cast<size_t>(state_index)];
            density = state.a[node](state_index);
        }
        std::cout << "[DCR_Solver][global-bvp-smallest-mode] rank=" << (rank + 1)
                  << " singular_value_estimate=" << singular_value_estimate
                  << " node=" << node
                  << " x_cm=" << evaluator.x[static_cast<size_t>(node)]
                  << " block=" << block
                  << " state_index=" << state_index
                  << " species='" << levels[static_cast<size_t>(gi)].label << "'"
                  << " component=" << mode(index)
                  << " density=" << density
                  << " floor_hit="
                  << (density <= 1.0001e-12 * evaluator.n_ref)
                  << std::endl;
    }
}

template <typename Solver>
double inverse_one_norm_estimate(Solver& solver, int size) {
    dcr::base::Vector x = dcr::base::Vector::Constant(size, 1.0 / std::max(1, size));
    double estimate = 0.0;
    for (int iteration = 0; iteration < 6; ++iteration) {
        const dcr::base::Vector y = solver.solve(x);
        if (solver.info() != Eigen::Success || !y.allFinite()) {
            return std::numeric_limits<double>::infinity();
        }
        estimate = std::max(estimate, y.lpNorm<1>());
        dcr::base::Vector signs = y.unaryExpr([](double value) {
            return value >= 0.0 ? 1.0 : -1.0;
        });
        const dcr::base::Vector z = solver.transpose().solve(signs);
        if (solver.info() != Eigen::Success || !z.allFinite()) {
            return std::numeric_limits<double>::infinity();
        }
        Eigen::Index index = 0;
        z.cwiseAbs().maxCoeff(&index);
        dcr::base::Vector next = dcr::base::Vector::Zero(size);
        next(index) = 1.0;
        if ((next - x).squaredNorm() == 0.0) break;
        x = next;
    }
    return estimate;
}

std::vector<unsigned char> inactive_target_m_variables(
    const Evaluator& evaluator,
    const dcr::base::Vector& y,
    const dcr::base::Vector& residual,
    double tolerance,
    bool log_selection);

void print_condition_estimates(
    const Evaluator& evaluator,
    const dcr::base::Vector& y) {
    SparseMatrix jacobian = assemble_sparse_jacobian(evaluator, y);
    jacobian.makeCompressed();
    Eigen::SparseLU<SparseMatrix, Eigen::COLAMDOrdering<int>> solver;
    solver.analyzePattern(jacobian);
    solver.factorize(jacobian);
    if (solver.info() == Eigen::Success) {
        const double inverse_norm = inverse_one_norm_estimate(solver, jacobian.rows());
        const double condition = sparse_one_norm(jacobian) * inverse_norm;
        std::cout << "[DCR_Solver][global-bvp-condition] block=global"
                  << " condition_1_estimate=" << condition
                  << " reciprocal_condition_estimate="
                  << (condition > 0.0 ? 1.0 / condition : 0.0) << std::endl;
        print_smallest_mode(evaluator, y, jacobian, solver);
    } else {
        std::cout << "[DCR_Solver][global-bvp-condition] block=global factorization_failed=true"
                  << std::endl;
    }

    const dcr::base::Vector residual = evaluator(y);
    const double tolerance = 5.0e-9 /
        std::max(1.0, evaluator.config.grid.length_cm);
    const auto inactive = inactive_target_m_variables(
        evaluator, y, residual, tolerance, true);
    std::vector<int> active_position(static_cast<size_t>(jacobian.rows()), -1);
    int active_size = 0;
    for (int index = 0; index < jacobian.rows(); ++index) {
        if (!inactive[static_cast<size_t>(index)]) {
            active_position[static_cast<size_t>(index)] = active_size++;
        }
    }
    if (active_size < jacobian.rows()) {
        std::vector<Triplet> active_triplets;
        for (int column = 0; column < jacobian.outerSize(); ++column) {
            const int active_column = active_position[static_cast<size_t>(column)];
            if (active_column < 0) continue;
            for (SparseMatrix::InnerIterator entry(jacobian, column); entry; ++entry) {
                const int active_row = active_position[static_cast<size_t>(entry.row())];
                if (active_row >= 0) {
                    active_triplets.emplace_back(
                        active_row, active_column, entry.value());
                }
            }
        }
        SparseMatrix active_jacobian(active_size, active_size);
        active_jacobian.setFromTriplets(active_triplets.begin(), active_triplets.end());
        active_jacobian.makeCompressed();
        Eigen::SparseLU<SparseMatrix, Eigen::COLAMDOrdering<int>> active_solver;
        active_solver.analyzePattern(active_jacobian);
        active_solver.factorize(active_jacobian);
        if (active_solver.info() == Eigen::Success) {
            const double inverse_norm = inverse_one_norm_estimate(
                active_solver, active_size);
            const double condition = sparse_one_norm(active_jacobian) * inverse_norm;
            std::cout << "[DCR_Solver][global-bvp-condition] block=active_global"
                      << " eliminated_variables=" << (jacobian.rows() - active_size)
                      << " condition_1_estimate=" << condition
                      << " reciprocal_condition_estimate="
                      << (condition > 0.0 ? 1.0 / condition : 0.0)
                      << std::endl;
        }
    }

    const auto& layout = evaluator.layout;
    const int m_size = layout.nodes * layout.m;
    std::vector<int> m_position(static_cast<size_t>(jacobian.rows()), -1);
    for (int k = 0; k < layout.nodes; ++k) {
        for (int mi = 0; mi < layout.m; ++mi) {
            m_position[static_cast<size_t>(offset(layout, k) + layout.p + layout.a + mi)] =
                k * layout.m + mi;
        }
    }
    std::vector<Triplet> triplets;
    for (int column = 0; column < jacobian.outerSize(); ++column) {
        const int m_column = m_position[static_cast<size_t>(column)];
        if (m_column < 0) continue;
        for (SparseMatrix::InnerIterator entry(jacobian, column); entry; ++entry) {
            const int m_row = m_position[static_cast<size_t>(entry.row())];
            if (m_row >= 0) triplets.emplace_back(m_row, m_column, entry.value());
        }
    }
    SparseMatrix m_block(m_size, m_size);
    m_block.setFromTriplets(triplets.begin(), triplets.end());
    m_block.makeCompressed();
    Eigen::SparseLU<SparseMatrix, Eigen::COLAMDOrdering<int>> m_solver;
    m_solver.analyzePattern(m_block);
    m_solver.factorize(m_block);
    if (m_solver.info() == Eigen::Success) {
        const double inverse_norm = inverse_one_norm_estimate(m_solver, m_size);
        const double condition = sparse_one_norm(m_block) * inverse_norm;
        std::cout << "[DCR_Solver][global-bvp-condition] block=M_principal"
                  << " condition_1_estimate=" << condition
                  << " reciprocal_condition_estimate="
                  << (condition > 0.0 ? 1.0 / condition : 0.0) << std::endl;
    } else {
        std::cout << "[DCR_Solver][global-bvp-condition] block=M_principal factorization_failed=true"
                  << std::endl;
    }
}

int worst_m_row(const dcr::base::Vector& residual, const Layout& layout) {
    int worst = layout.p + layout.a;
    double maximum = -1.0;
    for (int k = 0; k < layout.nodes; ++k) {
        for (int mi = 0; mi < layout.m; ++mi) {
            const int row = offset(layout, k) + layout.p + layout.a + mi;
            if (std::abs(residual(row)) > maximum) {
                maximum = std::abs(residual(row));
                worst = row;
            }
        }
    }
    return worst;
}

void print_state_jacobian_checks(
    const char* state_label,
    const Evaluator& evaluator,
    const dcr::base::Vector& y,
    const dcr::base::Vector& residual,
    const dcr::base::Vector& newton_delta) {
    const auto& layout = evaluator.layout;
    const int worst_row = worst_m_row(residual, layout);
    const int worst_node = worst_row / layout.block;
    const int worst_mi = worst_row - offset(layout, worst_node) - layout.p - layout.a;
    SparseMatrix jacobian = assemble_sparse_jacobian(evaluator, y);
    std::vector<std::pair<std::string, dcr::base::Vector>> directions;
    dcr::base::Vector worst = dcr::base::Vector::Zero(y.size());
    worst(worst_row) = 1.0;
    directions.emplace_back("worst_M", worst);
    dcr::base::Vector neighbor = dcr::base::Vector::Zero(y.size());
    const int neighbor_node = worst_node > 0 ? worst_node - 1 : std::min(1, layout.nodes - 1);
    neighbor(offset(layout, neighbor_node) + layout.p + layout.a + worst_mi) = 1.0;
    directions.emplace_back("neighbor_M", neighbor);
    dcr::base::Vector all_m = dcr::base::Vector::Zero(y.size());
    for (int k = 0; k < layout.nodes; ++k) {
        for (int mi = 0; mi < layout.m; ++mi) {
            all_m(offset(layout, k) + layout.p + layout.a + mi) = 1.0;
        }
    }
    directions.emplace_back("all_M", all_m);
    dcr::base::Vector mixed(y.size());
    for (int i = 0; i < mixed.size(); ++i) mixed(i) = std::sin(0.37 * (i + 1));
    directions.emplace_back("mixed", mixed);
    if (newton_delta.size() == y.size() && newton_delta.norm() > 0.0) {
        directions.emplace_back("newton", newton_delta);
    }

    for (auto& [direction_label, direction] : directions) {
        direction /= direction.norm();
        const dcr::base::Vector analytic = jacobian * direction;
        for (double step : {1.0e-3, 3.0e-4, 1.0e-4, 3.0e-5}) {
            const dcr::base::Vector centered =
                (evaluator(y + step * direction) - evaluator(y - step * direction)) /
                (2.0 * step);
            const double full_error = (centered - analytic).norm() /
                std::max({1.0e-12, centered.norm(), analytic.norm()});
            double m_error_squared = 0.0;
            double m_scale_squared = 0.0;
            for (int k = 0; k < layout.nodes; ++k) {
                for (int mi = 0; mi < layout.m; ++mi) {
                    const int row = offset(layout, k) + layout.p + layout.a + mi;
                    m_error_squared += std::pow(centered(row) - analytic(row), 2);
                    m_scale_squared += std::max(
                        std::pow(centered(row), 2), std::pow(analytic(row), 2));
                }
            }
            const double m_error = std::sqrt(m_error_squared /
                std::max(1.0e-24, m_scale_squared));
            const double row_error = std::abs(centered(worst_row) - analytic(worst_row)) /
                std::max({1.0e-12, std::abs(centered(worst_row)),
                    std::abs(analytic(worst_row))});
            const dcr::base::Vector difference = centered - analytic;
            Eigen::Index maximum_error_row = 0;
            difference.cwiseAbs().maxCoeff(&maximum_error_row);
            const int maximum_row = static_cast<int>(maximum_error_row);
            const int maximum_node = maximum_row == layout.q_index
                ? -1 : maximum_row / layout.block;
            const int maximum_local = maximum_node >= 0
                ? maximum_row - offset(layout, maximum_node) : -1;
            const char* maximum_block = maximum_row == layout.q_index ? "target" :
                (maximum_local < layout.p ? "P" :
                    (maximum_local < layout.p + layout.a ? "A" : "M"));
            std::cout << "[DCR_Solver][global-bvp-state-jv] state=" << state_label
                      << " direction=" << direction_label
                      << " step=" << step
                      << " relative_error_2=" << full_error
                      << " M_relative_error_2=" << m_error
                      << " worst_M_row_relative_error=" << row_error
                      << " maximum_error_block=" << maximum_block
                      << " maximum_error_node=" << maximum_node
                      << " maximum_error_row=" << maximum_row
                      << " maximum_absolute_error=" << std::abs(difference(maximum_row))
                      << std::endl;
        }
    }
}

void print_m_residual_rows(
    const Evaluator& evaluator,
    const dcr::base::Vector& y,
    const dcr::base::Vector& residual) {
    struct Row {
        int node = 0;
        int mi = 0;
        int gi = 0;
        double scaled = 0.0;
        double scale = 0.0;
        double incoming = 0.0;
        double outgoing = 0.0;
        double divergence = 0.0;
        double chemistry = 0.0;
        double pump = 0.0;
        double midpoint_population = 0.0;
        double midpoint_flux = 0.0;
    };
    const auto state = decode(y, evaluator.layout, evaluator.n_ref, evaluator.q_ref);
    const auto& layout = evaluator.layout;
    const auto& levels = evaluator.atomic_data.get_levels();
    std::vector<NodeData> cells;
    for (int k = 0; k + 1 < layout.nodes; ++k) {
        const dcr::base::Vector p_mid = 0.5 * (state.p[k] + state.p[k + 1]);
        const dcr::base::Vector a_mid = 0.5 * (state.a[k] + state.a[k + 1]);
        const dcr::base::Vector m_mid = 0.5 * (state.m[k] + state.m[k + 1]);
        cells.push_back(evaluate_node(
            evaluator.config, evaluator.atomic_data, evaluator.plasma, evaluator.grid,
            evaluator.boundary, p_mid, a_mid, m_mid,
            0.5 * (evaluator.x[k] + evaluator.x[k + 1]), false,
            &evaluator.cell_rate_caches[k]));
    }
    std::vector<Row> rows;
    for (int k = 0; k < layout.nodes; ++k) {
        for (int mi = 0; mi < layout.m; ++mi) {
            Row row;
            row.node = k;
            row.mi = mi;
            row.gi = evaluator.boundary.M_indices[static_cast<size_t>(mi)];
            const int residual_row = offset(layout, k) + layout.p + layout.a + mi;
            row.scaled = residual(residual_row);
            row.scale = k == 0 ? evaluator.q_ref : evaluator.rate_ref;
            if (k == 0) {
                row.outgoing = evaluator.boundary.u_M * state.m[0](mi);
                if (row.gi == evaluator.boundary.molecule_ground) {
                    for (int ion_gi : evaluator.boundary.ion_indices) {
                        if (levels[static_cast<size_t>(ion_gi)].atomicity >= 2) continue;
                        const int pi = layout.p_pos[static_cast<size_t>(ion_gi)];
                        const double gamma = -global_ion_velocity_cm_s(
                            evaluator.config, levels[static_cast<size_t>(ion_gi)], 0.0) *
                            state.p[0](pi);
                        row.incoming += evaluator.config.numerics.boundary_molecular_flow_acceptance *
                            evaluator.wall.alpha_molecule * gamma / 2.0;
                    }
                }
                row.divergence = row.outgoing - row.incoming;
                row.midpoint_population = state.m[0](mi);
                row.midpoint_flux = row.outgoing;
            } else {
                const double dx = evaluator.x[k] - evaluator.x[k - 1];
                row.incoming = evaluator.boundary.u_M * state.m[k - 1](mi);
                row.outgoing = evaluator.boundary.u_M * state.m[k](mi);
                row.divergence = (row.outgoing - row.incoming) / dx;
                row.chemistry = evaluator.chemistry_fraction * cells[k - 1].chem_m(mi);
                row.pump = cells[k - 1].pump_m(mi);
                row.midpoint_population = 0.5 * (state.m[k - 1](mi) + state.m[k](mi));
                row.midpoint_flux = evaluator.boundary.u_M * row.midpoint_population;
            }
            rows.push_back(row);
        }
    }
    std::sort(rows.begin(), rows.end(), [](const Row& left, const Row& right) {
        return std::abs(left.scaled) > std::abs(right.scaled);
    });
    for (size_t rank = 0; rank < std::min<size_t>(20, rows.size()); ++rank) {
        const auto& row = rows[rank];
        std::cout << "[DCR_Solver][global-bvp-M-row] rank=" << (rank + 1)
                  << " node=" << row.node
                  << " x_cm=" << evaluator.x[static_cast<size_t>(row.node)]
                  << " state_index=" << row.mi
                  << " species='" << levels[static_cast<size_t>(row.gi)].label << "'"
                  << " absolute_residual=" << std::abs(row.scaled * row.scale)
                  << " scaled_residual=" << row.scaled
                  << " residual_scale=" << row.scale
                  << " incoming_face_flux=" << row.incoming
                  << " outgoing_face_flux=" << row.outgoing
                  << " flux_divergence=" << row.divergence
                  << " midpoint_chemistry=" << row.chemistry
                  << " midpoint_pump=" << row.pump
                  << " midpoint_population=" << row.midpoint_population
                  << " transported_molecular_flux=" << row.midpoint_flux
                  << std::endl;
    }
}

void write_state_snapshot(
    const dcr::io::Config& config,
    const char* label,
    double lambda,
    const dcr::base::Vector& y,
    const dcr::base::Vector& residual) {
    std::filesystem::path directory = config.io.output_dir.empty()
        ? std::filesystem::path("output") : std::filesystem::path(config.io.output_dir);
    std::filesystem::create_directories(directory);
    std::ostringstream length;
    length << config.grid.length_cm;
    std::string length_label = length.str();
    std::replace(length_label.begin(), length_label.end(), '.', 'p');
    const auto path = directory /
        ("global_bvp_L" + length_label + "_" + label + ".tsv");
    std::ofstream output(path);
    output << std::setprecision(17);
    output << "lambda\t" << lambda << "\n";
    output << "index\tlog_state\tresidual\n";
    for (int i = 0; i < y.size(); ++i) {
        output << i << '\t' << y(i) << '\t'
               << (i < residual.size() ? residual(i) : 0.0) << '\n';
    }
    std::cout << "[DCR_Solver][global-bvp-snapshot] label=" << label
              << " path=" << path.string() << std::endl;
}

std::vector<unsigned char> inactive_target_m_variables(
    const Evaluator& evaluator,
    const dcr::base::Vector& y,
    const dcr::base::Vector& residual,
    double tolerance,
    bool log_selection) {
    const auto state = decode(
        y, evaluator.layout, evaluator.n_ref, evaluator.q_ref);
    std::vector<unsigned char> inactive(static_cast<size_t>(y.size()), 0);
    for (int mi = 0; mi < evaluator.layout.m; ++mi) {
        const int gi = evaluator.boundary.M_indices[static_cast<size_t>(mi)];
        if (gi == evaluator.boundary.molecule_ground) continue;
        const int index = evaluator.layout.p + evaluator.layout.a + mi;
        if (state.m[0](mi) > 1.0001e-12 * evaluator.n_ref ||
            std::abs(residual(index)) >= tolerance) {
            continue;
        }
        dcr::base::Vector removed = y;
        removed(index) = -138.0;
        const double maximum_contribution =
            (evaluator(removed) - residual).cwiseAbs().maxCoeff();
        const bool negligible = maximum_contribution < 0.1 * tolerance;
        if (negligible) inactive[static_cast<size_t>(index)] = 1;
        if (log_selection) {
            std::cout << "[DCR_Solver][global-bvp-inactive-M]"
                      << " node=0 state_index=" << mi
                      << " global_index=" << gi
                      << " density=" << state.m[0](mi)
                      << " row_residual=" << residual(index)
                      << " maximum_residual_contribution=" << maximum_contribution
                      << " threshold=" << 0.1 * tolerance
                      << " frozen=" << negligible << std::endl;
        }
    }
    return inactive;
}

} // namespace

double global_ion_velocity_cm_s(
    const dcr::io::Config& config,
    const dcr::atomic::EnergyLevel& level,
    double x_cm) {
    const double length = config.global_bvp.ion_velocity_transition_length_cm;
    const double upstream_fraction = config.global_bvp.ion_upstream_speed_fraction;
    if (!(length > 0.0)) {
        throw std::runtime_error(
            "ion_velocity_transition_length_cm must be positive");
    }
    if (!(upstream_fraction > 0.0 && upstream_fraction <= 1.0)) {
        throw std::runtime_error(
            "ion_upstream_speed_fraction must be in (0, 1]");
    }
    const double xi = std::clamp(x_cm / length, 0.0, 1.0);
    const double shape = 1.0 - (1.0 - upstream_fraction) * xi;
    const auto temperatures = evaluate_plasma_temperatures(config, x_cm);
    return -dcr::physics::calculate_Bohm_speed(
        temperatures.electron_eV,
        temperatures.ion_eV,
        std::max(level.mass_amu, 1.0e-12)) * shape;
}

double conservative_flux_divergence_cm3_s(
    double velocity_left_cm_s,
    double density_left_cm3,
    double velocity_right_cm_s,
    double density_right_cm3,
    double dx_cm) {
    if (!(dx_cm > 0.0)) throw std::invalid_argument("Flux divergence requires dx_cm > 0.");
    return (velocity_right_cm_s * density_right_cm3 -
        velocity_left_cm_s * density_left_cm3) / dx_cm;
}

void diagnose_global_bvp_log_state(
    const dcr::io::Config& config,
    const dcr::atomic::AtomicData& atomic_data,
    const dcr::state::PlasmaState& plasma,
    const EEDFGridView& grid,
    const BoundaryPhaseResult& initial_boundary,
    const dcr::physics::WallRecycling& wall,
    const dcr::base::Vector& log_state,
    double chemistry_fraction) {
    const auto x = coordinates(config);
    Layout layout = make_layout(
        initial_boundary, static_cast<int>(x.size()), atomic_data.get_total_states());
    const auto& levels = atomic_data.get_levels();
    layout.proton_pi = select_proton_pi(layout, initial_boundary, levels);
    if (log_state.size() != layout.q_index + 1) {
        throw std::runtime_error("Global-BVP diagnostic state has the wrong size.");
    }
    const double n_ref = std::max(config.plasma.total_density, 1.0);
    const int proton_gi = initial_boundary.P_indices[static_cast<size_t>(layout.proton_pi)];
    const double u_ref = std::max(1.0, -global_ion_velocity_cm_s(
        config, levels[static_cast<size_t>(proton_gi)], 0.0));
    const double q_ref = n_ref * u_ref;
    const double rate_ref = q_ref / std::max(config.grid.length_cm, 1.0e-6);
    const double local_rate_ref = q_ref /
        std::max(config.grid.spatial_exhaust_width_cm, 1.0e-6);
    const auto caches = build_spatial_rate_caches(
        config, atomic_data, plasma, grid, x, n_ref);
    Evaluator evaluator{
        config, atomic_data, plasma, grid, initial_boundary, wall, layout, x,
        caches.cells, caches.upstream, n_ref, q_ref, rate_ref, local_rate_ref,
        chemistry_fraction};
    const dcr::base::Vector residual = evaluator(log_state);
    State state;
    MolecularRecovery molecular;
    const bool recovered = evaluator.recover(log_state, state, &molecular);
    std::cout << "[DCR_Solver][global-bvp-reduced-diagnostic] unknowns="
              << log_state.size() << " residual_norm="
              << (residual.allFinite() ? residual.cwiseAbs().maxCoeff()
                                       : std::numeric_limits<double>::infinity())
              << " molecular_recovery_valid=" << recovered
              << " molecular_balance=" << molecular.maximum_scaled_balance
              << std::endl;
}

GlobalBVPJacobianDiagnostics check_global_bvp_initial_jacobian(
    const dcr::io::Config& config,
    const dcr::atomic::AtomicData& atomic_data,
    const dcr::state::PlasmaState& plasma,
    const EEDFGridView& grid,
    const BoundaryPhaseResult& initial_boundary,
    const dcr::physics::WallRecycling& wall) {
    const auto x = coordinates(config);
    Layout layout = make_layout(
        initial_boundary, static_cast<int>(x.size()), atomic_data.get_total_states());
    const auto& levels = atomic_data.get_levels();
    layout.proton_pi = select_proton_pi(layout, initial_boundary, levels);
    const double n_ref = std::max(config.plasma.total_density, 1.0);
    const int proton_gi = initial_boundary.P_indices[static_cast<size_t>(layout.proton_pi)];
    const double u_ref = std::max(1.0, -global_ion_velocity_cm_s(
        config, levels[static_cast<size_t>(proton_gi)], 0.0));
    const double q_ref = n_ref * u_ref;
    const double rate_ref = q_ref / std::max(config.grid.length_cm, 1.0e-6);
    const double local_rate_ref = q_ref /
        std::max(config.grid.spatial_exhaust_width_cm, 1.0e-6);
    const auto spatial_rate_caches = build_spatial_rate_caches(
        config, atomic_data, plasma, grid, x, n_ref);
    dcr::base::Vector y = initial_guess(
        layout, initial_boundary, x, config, levels, n_ref, q_ref);
    y.array() += 1.0;
    Evaluator evaluator{
        config, atomic_data, plasma, grid, initial_boundary, wall, layout, x,
        spatial_rate_caches.cells, spatial_rate_caches.upstream,
        n_ref, q_ref, rate_ref, local_rate_ref, 1.0};
    GlobalBVPJacobianDiagnostics diagnostics;
    diagnostics.labels = {"P", "A", "target", "upstream", "neighbor"};
    std::vector<dcr::base::Vector> directions(
        diagnostics.labels.size(), dcr::base::Vector::Zero(y.size()));
    for (int k = 0; k < layout.nodes; ++k) {
        const int base = offset(layout, k);
        for (int i = 0; i < layout.p; ++i) directions[0](base + i) = 0.5 + 0.01 * i;
        for (int i = 0; i < layout.a; ++i) directions[1](base + layout.p + i) = 0.7 + 0.02 * i;
        const double sign = (k % 2 == 0) ? 1.0 : -1.0;
        for (int i = 0; i < layout.block; ++i) directions[4](base + i) = sign;
    }
    for (int i = 0; i < layout.block; ++i) directions[2](i) = 1.0;
    const int upstream_base = offset(layout, layout.nodes - 1);
    for (int i = 0; i < layout.block; ++i) directions[3](upstream_base + i) = 1.0;
    directions[3](layout.q_index) = 1.0;

    for (auto& direction : directions) {
        const double norm = direction.norm();
        if (norm > 0.0) direction /= norm;
        const dcr::base::Vector product = matrix_free_jacobian_action(evaluator, y, direction);
        const std::array<double, 4> steps{{1.0e-2, 5.0e-3, 2.5e-3, 1.25e-3}};
        std::array<double, 4> errors{};
        for (size_t step_index = 0; step_index < steps.size(); ++step_index) {
            const double epsilon = steps[step_index];
            dcr::base::Vector plus = y + epsilon * direction;
            dcr::base::Vector minus = y - epsilon * direction;
            const dcr::base::Vector finite_difference =
                (evaluator(plus) - evaluator(minus)) / (2.0 * epsilon);
            errors[step_index] = (finite_difference - product).norm() /
                std::max({1.0e-12, finite_difference.norm(), product.norm()});
        }
        diagnostics.relative_errors.push_back(*std::min_element(errors.begin(), errors.end()));
        double best_order = -std::numeric_limits<double>::infinity();
        for (size_t i = 0; i + 1 < errors.size(); ++i) {
            if (errors[i] > 0.0 && errors[i + 1] > 0.0) {
                best_order = std::max(best_order, std::log(errors[i] / errors[i + 1]) / std::log(2.0));
            }
        }
        diagnostics.observed_orders.push_back(best_order);
    }
    return diagnostics;
}

GlobalBVPCountDiagnostics global_bvp_count_diagnostics(
    const dcr::io::Config& config,
    const dcr::atomic::AtomicData& atomic_data,
    const BoundaryPhaseResult& boundary) {
    GlobalBVPCountDiagnostics out;
    const auto x = coordinates(config);
    Layout layout = make_layout(
        boundary, static_cast<int>(x.size()), atomic_data.get_total_states());
    layout.proton_pi = select_proton_pi(layout, boundary, atomic_data.get_levels());
    out.nodes = layout.nodes;
    out.intervals = std::max(0, layout.nodes - 1);
    out.background_states = layout.p;
    out.recycling_atom_states = layout.a;
    out.recycling_molecule_states = layout.m;
    out.positive_ions = static_cast<int>(std::count(
        layout.positive_ion.begin(), layout.positive_ion.end(), 1));
    out.nontransported_background_states = layout.p - out.positive_ions;
    out.unknown_count = layout.q_index + 1;
    out.molecular_unknown_count = 0;
    out.molecular_residual_rows = 0;
    out.ion_interval_rows = out.intervals * out.positive_ions;
    out.ion_upstream_boundary_rows = out.positive_ions;
    out.local_background_rows = layout.nodes * out.nontransported_background_states;
    out.recycling_interval_rows = out.intervals * layout.a;
    out.recycling_target_boundary_rows = layout.a;
    out.target_density_rows = 1;
    out.spatial_density_rows = 0;
    out.legacy_ion_closure_rows = 0;
    out.q_up_coupled_rows = 1;
    out.residual_count = out.ion_interval_rows + out.ion_upstream_boundary_rows +
        out.local_background_rows + out.recycling_interval_rows +
        out.recycling_target_boundary_rows + out.target_density_rows;
    out.proton_global_index = boundary.P_indices[static_cast<size_t>(layout.proton_pi)];
    return out;
}

GlobalBVPForwardMTransportDiagnostics check_global_bvp_forward_m_transport(
    const dcr::io::Config& config,
    const dcr::atomic::AtomicData& atomic_data,
    const dcr::state::PlasmaState& plasma,
    const EEDFGridView& grid,
    const BoundaryPhaseResult& initial_boundary,
    const dcr::physics::WallRecycling& wall) {
    const auto x = coordinates(config);
    Layout layout = make_layout(
        initial_boundary, static_cast<int>(x.size()), atomic_data.get_total_states());
    const auto& levels = atomic_data.get_levels();
    layout.proton_pi = select_proton_pi(layout, initial_boundary, levels);
    const double n_ref = std::max(config.plasma.total_density, 1.0);
    const int proton_gi = initial_boundary.P_indices[static_cast<size_t>(layout.proton_pi)];
    const double q_ref = n_ref * std::max(1.0, -global_ion_velocity_cm_s(
        config, levels[static_cast<size_t>(proton_gi)], 0.0));
    const double rate_ref = q_ref / std::max(config.grid.length_cm, 1.0e-6);
    const double local_rate_ref = q_ref /
        std::max(config.grid.spatial_exhaust_width_cm, 1.0e-6);
    const auto caches = build_spatial_rate_caches(
        config, atomic_data, plasma, grid, x, n_ref);
    const dcr::base::Vector y = initial_guess(
        layout, initial_boundary, x, config, levels, n_ref, q_ref);
    Evaluator evaluator{
        config, atomic_data, plasma, grid, initial_boundary, wall, layout, x,
        caches.cells, caches.upstream, n_ref, q_ref, rate_ref, local_rate_ref, 1.0};
    State state;
    MolecularRecovery recovery;
    if (!evaluator.recover(y, state, &recovery)) {
        throw std::runtime_error("Forward molecular transport diagnostic recovery failed.");
    }

    GlobalBVPForwardMTransportDiagnostics out;
    out.density = state.m;
    out.expected_target_ground_flux = recovery.expected_target_ground_flux;
    out.maximum_scaled_midpoint_balance = recovery.maximum_scaled_balance;
    out.minimum_density = std::numeric_limits<double>::infinity();
    out.finite_nonnegative = true;
    for (const auto& density : state.m) {
        out.flux.push_back(initial_boundary.u_M * density);
        for (int mi = 0; mi < density.size(); ++mi) {
            out.minimum_density = std::min(out.minimum_density, density(mi));
            out.finite_nonnegative = out.finite_nonnegative &&
                std::isfinite(density(mi)) && density(mi) >= 0.0;
        }
    }
    if (state.m.empty()) out.minimum_density = 0.0;
    for (int mi = 0; mi < layout.m; ++mi) {
        const double target_flux = initial_boundary.u_M * state.m[0](mi);
        if (initial_boundary.M_indices[static_cast<size_t>(mi)] ==
            initial_boundary.molecule_ground) {
            out.target_ground_flux = target_flux;
        } else {
            out.maximum_excited_target_flux = std::max(
                out.maximum_excited_target_flux, std::abs(target_flux));
        }
        out.target_total_flux += target_flux;
        out.upstream_total_flux += initial_boundary.u_M * state.m.back()(mi);
    }
    return out;
}

GlobalBVPFDCouplingDiagnostics check_global_bvp_fd_m_coupling(
    const dcr::io::Config& config,
    const dcr::atomic::AtomicData& atomic_data,
    const dcr::state::PlasmaState& plasma,
    const EEDFGridView& grid,
    const BoundaryPhaseResult& initial_boundary,
    const dcr::physics::WallRecycling& wall) {
    const auto x = coordinates(config);
    Layout layout = make_layout(
        initial_boundary, static_cast<int>(x.size()), atomic_data.get_total_states());
    const auto& levels = atomic_data.get_levels();
    layout.proton_pi = select_proton_pi(layout, initial_boundary, levels);
    const double n_ref = std::max(config.plasma.total_density, 1.0);
    const int proton_gi = initial_boundary.P_indices[static_cast<size_t>(layout.proton_pi)];
    const double q_ref = n_ref * std::max(1.0, -global_ion_velocity_cm_s(
        config, levels[static_cast<size_t>(proton_gi)], 0.0));
    const double rate_ref = q_ref / std::max(config.grid.length_cm, 1.0e-6);
    const double local_rate_ref = q_ref /
        std::max(config.grid.spatial_exhaust_width_cm, 1.0e-6);
    const auto caches = build_spatial_rate_caches(
        config, atomic_data, plasma, grid, x, n_ref);
    const dcr::base::Vector y = initial_guess(
        layout, initial_boundary, x, config, levels, n_ref, q_ref);
    Evaluator evaluator{
        config, atomic_data, plasma, grid, initial_boundary, wall, layout, x,
        caches.cells, caches.upstream, n_ref, q_ref, rate_ref, local_rate_ref, 1.0};
    dcr::base::Vector direction = dcr::base::Vector::Zero(y.size());
    direction(layout.proton_pi) = 1.0;
    if (layout.nodes > 1) direction(offset(layout, 1) + layout.proton_pi) = -0.37;
    direction.normalize();
    const dcr::base::Vector implemented = matrix_free_jacobian_action(
        evaluator, y, direction, 1.0e-6);
    constexpr double reference_step = 5.0e-6;
    const dcr::base::Vector reference =
        (evaluator(y + reference_step * direction) -
         evaluator(y - reference_step * direction)) / (2.0 * reference_step);
    GlobalBVPFDCouplingDiagnostics out;
    out.relative_error = (implemented - reference).norm() /
        std::max({1.0e-12, implemented.norm(), reference.norm()});
    State plus;
    State minus;
    if (!evaluator.recover(y + reference_step * direction, plus) ||
        !evaluator.recover(y - reference_step * direction, minus)) {
        throw std::runtime_error("FD molecular coupling diagnostic recovery failed.");
    }
    double difference = 0.0;
    double scale = 1.0;
    for (int k = 0; k < layout.nodes; ++k) {
        difference += (plus.m[static_cast<size_t>(k)] -
            minus.m[static_cast<size_t>(k)]).squaredNorm();
        scale += plus.m[static_cast<size_t>(k)].squaredNorm();
    }
    out.molecular_profile_relative_change = std::sqrt(difference / scale);
    return out;
}

GlobalBVPResult solve_global_target_conditioned_bvp(
    const dcr::io::Config& config,
    const dcr::atomic::AtomicData& atomic_data,
    const dcr::state::PlasmaState& plasma,
    const EEDFGridView& grid,
    const BoundaryPhaseResult& initial_boundary,
    const dcr::physics::WallRecycling& wall,
    const GlobalBVPResult* coarse_solution,
    double initial_log_shift) {
    const Clock::time_point solve_start = Clock::now();
    GlobalBVPResult out;
    out.diagnostics.initialization_source = coarse_solution
        ? "conservative_interpolation" : "boundary_seed";
    out.boundary = initial_boundary;
    out.velocity_profile_type = "fixed_scale_linear_bohm";
    out.ion_velocity_transition_length_cm =
        config.global_bvp.ion_velocity_transition_length_cm;
    out.ion_upstream_speed_fraction = config.global_bvp.ion_upstream_speed_fraction;
    const auto target_temperatures = evaluate_plasma_temperatures(config, 0.0);
    out.target_electron_temperature_eV = target_temperatures.electron_eV;
    out.target_ion_temperature_eV = target_temperatures.ion_eV;
    out.target_nuclei_density_cm3 = config.plasma.total_density;
    out.boundary_exhaust_width_cm = config.grid.boundary_poloidal_width_cm;
    out.spatial_exhaust_width_cm = config.grid.spatial_exhaust_width_cm;
    if (coarse_solution) {
        const bool compatible =
            coarse_solution->velocity_profile_type == out.velocity_profile_type &&
            coarse_solution->ion_velocity_transition_length_cm ==
                out.ion_velocity_transition_length_cm &&
            coarse_solution->ion_upstream_speed_fraction ==
                out.ion_upstream_speed_fraction &&
            coarse_solution->target_electron_temperature_eV ==
                out.target_electron_temperature_eV &&
            coarse_solution->target_ion_temperature_eV == out.target_ion_temperature_eV &&
            coarse_solution->target_nuclei_density_cm3 == out.target_nuclei_density_cm3 &&
            coarse_solution->boundary_exhaust_width_cm == out.boundary_exhaust_width_cm &&
            coarse_solution->spatial_exhaust_width_cm == out.spatial_exhaust_width_cm;
        if (!compatible) {
            throw std::runtime_error(
                "Incompatible global-BVP coarse solution metadata.");
        }
    }
    const auto x = coordinates(config);
    Layout layout = make_layout(initial_boundary, static_cast<int>(x.size()), atomic_data.get_total_states());
    out.diagnostics.unknown_count = layout.q_index + 1;
    const auto& levels = atomic_data.get_levels();
    layout.proton_pi = select_proton_pi(layout, initial_boundary, levels);
    if (!(initial_boundary.u_A > 0.0) || !(initial_boundary.u_M > 0.0)) {
        throw std::runtime_error("Global BVP requires positive recycling-flow speeds.");
    }
    const double n_ref = std::max(config.plasma.total_density, 1.0);
    const int proton_gi = initial_boundary.P_indices[static_cast<size_t>(layout.proton_pi)];
    const double u_ref = std::max(1.0, -global_ion_velocity_cm_s(
        config, levels[static_cast<size_t>(proton_gi)], 0.0));
    const double q_ref = n_ref * u_ref;
    const double rate_ref = q_ref / std::max(config.grid.length_cm, 1.0e-6);
    const double local_rate_ref = q_ref /
        std::max(config.grid.spatial_exhaust_width_cm, 1.0e-6);
    const auto spatial_rate_caches = build_spatial_rate_caches(
        config, atomic_data, plasma, grid, x, n_ref);
    dcr::base::Vector y = coarse_solution
        ? interpolate_initial_guess(
            *coarse_solution, layout, initial_boundary, x, config, levels, n_ref, q_ref)
        : initial_guess(layout, initial_boundary, x, config, levels, n_ref, q_ref);
    y.array() += initial_log_shift;
    const int max_iterations = config.numerics.marching_max_iterations > 0
        ? config.numerics.marching_max_iterations
        : std::max(1, config.numerics.max_iterations);
    const double requested_tolerance = config.numerics.marching_tolerance > 0.0
        ? config.numerics.marching_tolerance : config.numerics.tolerance;
    const double domain_scaled_tolerance = layout.nodes <= 2
        ? std::min(2.0e-11, 5.0e-9 / std::max(1.0, config.grid.length_cm))
        : 5.0e-9 / std::max(1.0, config.grid.length_cm);
    const double tolerance = std::min(
        {1.0e-8, std::max(requested_tolerance, 1.0e-12), domain_scaled_tolerance});
    int total_iterations = 0;
    double final_norm = std::numeric_limits<double>::infinity();
    double completed_fraction = 0.0;
    bool direct_attempt_pending = coarse_solution != nullptr;
    double continuation_step = direct_attempt_pending ? 1.0 : 0.1;
    constexpr double minimum_continuation_step = 1.0e-4;
    dcr::base::Vector last_accepted_y = y;
    dcr::base::Vector last_accepted_residual;
    double last_accepted_fraction = 0.0;
    bool have_converged_lambda_checkpoint = false;
    dcr::base::Vector last_rejected_trial_y;
    dcr::base::Vector last_rejected_trial_residual;
    dcr::base::Vector last_newton_delta;

    while (completed_fraction < 1.0 - 1.0e-14) {
        const double fraction = std::min(1.0, completed_fraction + continuation_step);
        const bool direct_attempt = direct_attempt_pending &&
            completed_fraction == 0.0 && fraction == 1.0;
        if (direct_attempt) {
            out.diagnostics.direct_full_physics_attempted = true;
        } else {
            ++out.diagnostics.continuation_stage_attempts;
        }
        const dcr::base::Vector stage_start = y;
        Evaluator evaluator{
            config, atomic_data, plasma, grid, initial_boundary, wall, layout, x,
            spatial_rate_caches.cells, spatial_rate_caches.upstream,
            n_ref, q_ref, rate_ref, local_rate_ref, fraction, &out.diagnostics};
        bool stage_converged = false;
        double regularization = 0.0;
        int rejected_steps = 0;
        double last_line_search_factor = 1.0;
        ResidualBlockNorms last_blocks;
        const char* failure_mode = "iteration_limit";
        for (int iteration = 0; iteration < max_iterations; ++iteration) {
            ++total_iterations;
            ++out.diagnostics.newton_iterations;
            const dcr::base::Vector residual = evaluator(y);
            final_norm = residual.cwiseAbs().maxCoeff();
            last_blocks = residual_block_norms(residual, layout);
            if (config.io.verbose_logging) {
                const Clock::time_point logging_start = Clock::now();
                Eigen::Index max_index = 0;
                residual.cwiseAbs().maxCoeff(&max_index);
                const auto blocks = residual_block_norms(residual, layout);
                std::cout << "[DCR_Solver][global-bvp] lambda=" << fraction
                          << " iter=" << (iteration + 1)
                          << " max_residual=" << final_norm
                          << " max_index=" << max_index
                          << " value=" << residual(max_index)
                          << " blocks{ion=" << blocks.ion
                          << ",A=" << blocks.flow_a
                          << ",M=" << blocks.flow_m
                          << ",local=" << blocks.local
                          << ",boundary=" << blocks.boundary << "}"
                           << " regularization=" << regularization << std::endl;
                out.diagnostics.logging_seconds += elapsed_seconds(logging_start);
            }
            if (std::isfinite(final_norm) && final_norm < tolerance) {
                stage_converged = true;
                last_accepted_y = y;
                last_accepted_residual = residual;
                last_accepted_fraction = fraction;
                have_converged_lambda_checkpoint = true;
                break;
            }
            if (!residual.allFinite()) throw std::runtime_error("Global BVP residual became non-finite.");

            auto linear_operator = [&](const dcr::base::Vector& vector) {
                dcr::base::Vector image = matrix_free_jacobian_action(evaluator, y, vector);
                if (regularization > 0.0) image += regularization * vector;
                return image;
            };
            const bool dense_fd_preconditioner = y.size() <= 150;
            dcr::base::Matrix dense_preconditioner;
            Eigen::FullPivLU<dcr::base::Matrix> dense_preconditioner_solver;
            SparseMatrix preconditioner;
            Eigen::SparseLU<SparseMatrix, Eigen::COLAMDOrdering<int>> preconditioner_solver;
            bool preconditioner_valid = false;
            ++out.diagnostics.jacobian_assemblies;
            if (dense_fd_preconditioner) {
                {
                    ScopedTimer timer(&out.diagnostics.jacobian_seconds);
                    dense_preconditioner.resize(y.size(), y.size());
                    for (int column = 0; column < y.size(); ++column) {
                        dcr::base::Vector basis = dcr::base::Vector::Zero(y.size());
                        basis(column) = 1.0;
                        dense_preconditioner.col(column) =
                            matrix_free_jacobian_action(evaluator, y, basis);
                    }
                    dense_preconditioner.diagonal().array() += regularization;
                }
                {
                    ScopedTimer timer(&out.diagnostics.factorization_seconds);
                    dense_preconditioner_solver.compute(dense_preconditioner);
                }
                preconditioner_valid = dense_preconditioner_solver.isInvertible();
                out.diagnostics.jacobian_storage = "dense";
                out.diagnostics.factorization_type = "Eigen::FullPivLU";
                out.diagnostics.jacobian_rows = y.size();
                out.diagnostics.jacobian_columns = y.size();
                out.diagnostics.jacobian_nonzeros =
                    static_cast<long long>(y.size()) * y.size();
                out.diagnostics.factorization_nonzeros = out.diagnostics.jacobian_nonzeros;
                out.diagnostics.factorization_memory_bytes =
                    2.0 * out.diagnostics.jacobian_nonzeros * sizeof(double);
            } else {
                {
                    ScopedTimer timer(&out.diagnostics.jacobian_seconds);
                    preconditioner = assemble_sparse_jacobian(evaluator, y);
                    if (regularization > 0.0) {
                        for (int diagonal = 0; diagonal < preconditioner.rows(); ++diagonal) {
                            preconditioner.coeffRef(diagonal, diagonal) += regularization;
                        }
                    }
                    preconditioner.makeCompressed();
                }
                {
                    ScopedTimer timer(&out.diagnostics.factorization_seconds);
                    preconditioner_solver.analyzePattern(preconditioner);
                    preconditioner_solver.factorize(preconditioner);
                }
                preconditioner_valid = preconditioner_solver.info() == Eigen::Success;
                out.diagnostics.jacobian_storage = "sparse";
                out.diagnostics.factorization_type =
                    "Eigen::SparseLU<COLAMD>";
                out.diagnostics.jacobian_rows = preconditioner.rows();
                out.diagnostics.jacobian_columns = preconditioner.cols();
                out.diagnostics.jacobian_nonzeros = preconditioner.nonZeros();
                if (preconditioner_valid) {
                    out.diagnostics.factorization_nonzeros =
                        preconditioner_solver.nnzL() + preconditioner_solver.nnzU();
                    out.diagnostics.factorization_memory_bytes =
                        static_cast<double>(out.diagnostics.factorization_nonzeros) *
                        (sizeof(double) + sizeof(int));
                }
            }
            out.diagnostics.jacobian_nonzeros_per_row =
                static_cast<double>(out.diagnostics.jacobian_nonzeros) /
                std::max(1, out.diagnostics.jacobian_rows);
            out.diagnostics.factorization_fill_ratio =
                static_cast<double>(out.diagnostics.factorization_nonzeros) /
                std::max<long long>(1, out.diagnostics.jacobian_nonzeros);
            if (!preconditioner_valid) {
                regularization = regularization > 0.0 ? 10.0 * regularization : 1.0e-8;
                if (++rejected_steps >= 8) {
                    failure_mode = "preconditioner_rejection";
                    break;
                }
                --total_iterations;
                --iteration;
                continue;
            }
            auto apply_preconditioner = [&](const dcr::base::Vector& vector) {
                dcr::base::Vector result;
                if (dense_fd_preconditioner) {
                    result = dense_preconditioner_solver.solve(vector);
                } else {
                    result = preconditioner_solver.solve(vector);
                }
                return result;
            };
            auto right_preconditioned_operator = [&](const dcr::base::Vector& vector) {
                if (dense_fd_preconditioner) {
                    dcr::base::Vector result =
                        dense_preconditioner * apply_preconditioner(vector);
                    return result;
                }
                return linear_operator(apply_preconditioner(vector));
            };
            ++out.diagnostics.linear_solves;
            GMRESResult linear;
            {
                ScopedTimer timer(&out.diagnostics.linear_solve_seconds);
                linear = restarted_gmres(
                    right_preconditioned_operator, -residual,
                    std::min(30, static_cast<int>(y.size())),
                    std::max(100, 2 * static_cast<int>(y.size())), 1.0e-6);
            }
            out.diagnostics.linear_iterations += linear.iterations;
            if (!linear.converged || !linear.solution.allFinite()) {
                regularization = regularization > 0.0 ? 10.0 * regularization : 1.0e-8;
                if (++rejected_steps >= 8) {
                    failure_mode = "gmres_rejection";
                    break;
                }
                --total_iterations;
                --iteration;
                continue;
            }
            const dcr::base::Vector delta = apply_preconditioner(linear.solution);
            last_newton_delta = delta;
            const dcr::base::Vector linear_image = dense_fd_preconditioner
                ? dense_preconditioner * delta : linear_operator(delta);
            const double linear_relative_residual =
                (linear_image + residual).norm() /
                std::max(1.0e-30, residual.norm());
            const double linear_acceptance = 2.0e-6;
            if (!(linear_relative_residual < linear_acceptance)) {
                regularization = regularization > 0.0 ? 10.0 * regularization : 1.0e-8;
                if (++rejected_steps >= 8) {
                    failure_mode = "linear_validation_rejection";
                    break;
                }
                --total_iterations;
                --iteration;
                continue;
            }
            if (config.global_bvp.diagnostic_logging) {
                const Clock::time_point logging_start = Clock::now();
                std::cout << "[DCR_Solver][global-bvp-linear-solve]"
                          << " L_cm=" << config.grid.length_cm
                          << " lambda=" << fraction
                          << " linear_relative_residual=" << linear_relative_residual
                          << " regularization=" << regularization
                          << std::endl;
                out.diagnostics.logging_seconds += elapsed_seconds(logging_start);
            }
            double step = 1.0;
            bool accepted = false;
            while (step >= 1.0e-5) {
                last_line_search_factor = step;
                ++out.diagnostics.line_search_trial_evaluations;
                dcr::base::Vector trial = y + step * delta;
                const dcr::base::Vector trial_residual = evaluator(trial);
                const dcr::base::Vector applied_step = trial - y;
                const dcr::base::Vector unregularized_model_residual = residual +
                    matrix_free_jacobian_action(evaluator, y, applied_step);
                const double merit = 0.5 * residual.squaredNorm();
                const double predicted_reduction = merit -
                    0.5 * unregularized_model_residual.squaredNorm();
                const double actual_reduction = merit -
                    0.5 * trial_residual.squaredNorm();
                if (config.global_bvp.diagnostic_logging && step == 1.0) {
                    const Clock::time_point logging_start = Clock::now();
                    std::cout << "[DCR_Solver][global-bvp-merit]"
                              << " L_cm=" << config.grid.length_cm
                              << " lambda=" << fraction
                              << " predicted_reduction=" << predicted_reduction
                              << " actual_reduction=" << actual_reduction
                              << " actual_to_predicted="
                              << (predicted_reduction != 0.0
                                    ? actual_reduction / predicted_reduction : 0.0)
                               << std::endl;
                    out.diagnostics.logging_seconds += elapsed_seconds(logging_start);
                }
                if (trial_residual.allFinite() &&
                    trial_residual.squaredNorm() < residual.squaredNorm()) {
                    y = trial;
                    accepted = true;
                    out.minimum_line_search_factor = std::min(
                        out.minimum_line_search_factor, step);
                    rejected_steps = 0;
                    regularization *= 0.1;
                    if (regularization < 1.0e-14) regularization = 0.0;
                    break;
                }
                last_rejected_trial_y = trial;
                last_rejected_trial_residual = trial_residual;
                step *= 0.5;
            }
            if (!accepted) {
                regularization = regularization > 0.0 ? 10.0 * regularization : 1.0e-8;
                if (++rejected_steps >= 8) {
                    failure_mode = "line_search_rejection";
                    break;
                }
                --total_iterations;
                --iteration;
            }
        }
        if (!stage_converged) {
            const dcr::base::Vector failed_y = y;
            const dcr::base::Vector failed_residual = evaluator(failed_y);
            final_norm = failed_residual.cwiseAbs().maxCoeff();
            last_blocks = residual_block_norms(failed_residual, layout);
            State failed_state;
            evaluator.recover(failed_y, failed_state);
            const auto largest_block = largest_residual_block(last_blocks);
            const Clock::time_point logging_start = Clock::now();
            std::cout << "[DCR_Solver][global-bvp-stage-failure]"
                      << " L_cm=" << config.grid.length_cm
                      << " lambda=" << fraction
                      << " max_residual=" << final_norm
                      << " largest_residual_block=" << largest_block.first
                      << " largest_block_residual=" << largest_block.second
                      << " line_search_factor=" << last_line_search_factor
                      << " q_up=" << failed_state.q_up
                      << " max_n_nuc=" << maximum_nuclei_density(
                            failed_state, layout, initial_boundary, levels)
                      << " failure_mode=" << failure_mode
                      << " iteration_limit=" << max_iterations
                      << std::endl;
            out.diagnostics.logging_seconds += elapsed_seconds(logging_start);
            y = stage_start;
            if (direct_attempt) {
                direct_attempt_pending = false;
                out.diagnostics.continuation_fallback_used = true;
                out.diagnostics.solve_path = "direct_lambda1_then_continuation";
                continuation_step = 0.9;
                continue;
            }
            continuation_step *= 0.5;
            if (continuation_step < minimum_continuation_step) {
                if (config.global_bvp.diagnostic_logging) {
                    const char* accepted_label = have_converged_lambda_checkpoint
                        ? "last_accepted_lambda" : "unvalidated_lambda_seed";
                    if (last_accepted_residual.size() == 0) {
                        Evaluator accepted_evaluator{
                            config, atomic_data, plasma, grid, initial_boundary, wall,
                            layout, x, spatial_rate_caches.cells,
                            spatial_rate_caches.upstream, n_ref, q_ref, rate_ref,
                            local_rate_ref, last_accepted_fraction, &out.diagnostics};
                        last_accepted_residual = accepted_evaluator(last_accepted_y);
                    }
                    write_state_snapshot(config, accepted_label, last_accepted_fraction,
                        last_accepted_y, last_accepted_residual);
                    write_state_snapshot(config, "failed_state", fraction,
                        failed_y, failed_residual);
                    if (last_rejected_trial_y.size() == y.size()) {
                        write_state_snapshot(config, "last_rejected_trial", fraction,
                            last_rejected_trial_y, last_rejected_trial_residual);
                    }
                }
                out.diagnostics.reduced_condition_estimate =
                    estimate_reduced_preconditioner_condition(
                        evaluator, failed_y, out.diagnostics);
                throw std::runtime_error(
                    "Global BVP continuation failed below the minimum continuation step.");
            }
            if (config.io.verbose_logging) {
                const Clock::time_point retry_logging_start = Clock::now();
                std::cout << "[DCR_Solver][global-bvp] retry lambda=" << fraction
                          << " with continuation_step=" << continuation_step << std::endl;
                out.diagnostics.logging_seconds += elapsed_seconds(retry_logging_start);
            }
            continue;
        }
        if (direct_attempt) {
            direct_attempt_pending = false;
            out.diagnostics.direct_full_physics_succeeded = true;
            out.diagnostics.solve_path = "direct_lambda1";
        } else {
            ++out.diagnostics.continuation_stages;
            if (out.diagnostics.solve_path.empty()) {
                out.diagnostics.solve_path = "continuation";
            }
        }
        completed_fraction = fraction;
        continuation_step = std::min(0.25, 1.5 * continuation_step);
    }

    Evaluator final_evaluator{
        config, atomic_data, plasma, grid, initial_boundary, wall, layout, x,
        spatial_rate_caches.cells, spatial_rate_caches.upstream,
        n_ref, q_ref, rate_ref, local_rate_ref, 1.0, &out.diagnostics};
    auto integrated_equation_defect = [&](const dcr::base::Vector& residual) {
        double defect = 0.0;
        for (int k = 0; k + 1 < layout.nodes; ++k) {
            const double dx = x[static_cast<size_t>(k + 1)] - x[static_cast<size_t>(k)];
            for (int pi = 0; pi < layout.p; ++pi) {
                const int gi = initial_boundary.P_indices[static_cast<size_t>(pi)];
                const double scale = layout.positive_ion[static_cast<size_t>(pi)]
                    ? rate_ref : local_rate_ref;
                defect += dx * levels[static_cast<size_t>(gi)].atomicity * scale *
                    residual(offset(layout, k) + pi);
            }
            for (int ai = 0; ai < layout.a; ++ai) {
                const int gi = initial_boundary.A_indices[static_cast<size_t>(ai)];
                defect += dx * levels[static_cast<size_t>(gi)].atomicity * rate_ref *
                    residual(offset(layout, k + 1) + layout.p + ai);
            }
        }
        return defect;
    };
    // A converged infinity-norm residual can still add coherently across a grid.
    // Project that aggregate error onto a smooth proton-flux ramp while retaining
    // the original per-equation acceptance criterion.
    if (layout.nodes > 2) {
        dcr::base::Vector direction = dcr::base::Vector::Zero(y.size());
        const double length = std::max(x.back(), 1.0e-30);
        for (int k = 0; k < layout.nodes; ++k) {
            direction(offset(layout, k) + layout.proton_pi) =
                x[static_cast<size_t>(k)] / length;
        }
        direction(layout.q_index) = 1.0;
        for (int iteration = 0; iteration < 4; ++iteration) {
            const dcr::base::Vector residual = final_evaluator(y);
            const double defect = integrated_equation_defect(residual);
            if (std::abs(defect) < 1.0e-10 * q_ref) break;
            constexpr double difference_step = 1.0e-5;
            const double plus_defect = integrated_equation_defect(
                final_evaluator(y + difference_step * direction));
            const double minus_defect = integrated_equation_defect(
                final_evaluator(y - difference_step * direction));
            const double derivative =
                (plus_defect - minus_defect) / (2.0 * difference_step);
            if (!std::isfinite(derivative) || std::abs(derivative) < 1.0) break;
            double step = std::clamp(-defect / derivative, -1.0e-3, 1.0e-3);
            bool accepted = false;
            for (int line_search = 0; line_search < 12; ++line_search) {
                ++out.diagnostics.line_search_trial_evaluations;
                const dcr::base::Vector trial = y + step * direction;
                const dcr::base::Vector trial_residual = final_evaluator(trial);
                if (trial_residual.allFinite() &&
                    trial_residual.cwiseAbs().maxCoeff() < 1.0e-8 &&
                    std::abs(integrated_equation_defect(trial_residual)) < std::abs(defect)) {
                    y = trial;
                    accepted = true;
                    break;
                }
                step *= 0.5;
            }
            if (!accepted) break;
        }
    }
    State state;
    MolecularRecovery final_molecular;
    if (!final_evaluator.recover(y, state, &final_molecular)) {
        throw std::runtime_error("Global BVP final molecular recovery failed.");
    }
    const dcr::base::Vector final_residual = final_evaluator(y);
    final_norm = final_residual.cwiseAbs().maxCoeff();
    const auto final_blocks = residual_block_norms(final_residual, layout);
    out.iterations = total_iterations;
    out.residual_norm = final_norm;
    out.final_ion_residual = final_blocks.ion;
    out.final_flow_a_residual = final_blocks.flow_a;
    out.final_flow_m_residual = final_molecular.maximum_scaled_balance;
    out.final_local_residual = final_blocks.local;
    out.final_boundary_residual = final_blocks.boundary;
    out.upstream_proton_flux_cm2_s = state.q_up;
    double integrated_pump_nuclei = 0.0;
    double integrated_ion_chemistry = 0.0;
    for (int k = 0; k + 1 < layout.nodes; ++k) {
        const double dx = x[static_cast<size_t>(k + 1)] - x[static_cast<size_t>(k)];
        const dcr::base::Vector p_mid = 0.5 *
            (state.p[static_cast<size_t>(k)] + state.p[static_cast<size_t>(k + 1)]);
        const dcr::base::Vector a_mid = 0.5 *
            (state.a[static_cast<size_t>(k)] + state.a[static_cast<size_t>(k + 1)]);
        const dcr::base::Vector m_mid = 0.5 *
            (state.m[static_cast<size_t>(k)] + state.m[static_cast<size_t>(k + 1)]);
        const auto cell = evaluate_node(
            config, atomic_data, plasma, grid, initial_boundary,
            p_mid, a_mid, m_mid,
            0.5 * (x[static_cast<size_t>(k)] + x[static_cast<size_t>(k + 1)]),
            false, &spatial_rate_caches.cells[static_cast<size_t>(k)]);
        double chemistry_nuclei = 0.0;
        double chemistry_absolute = 0.0;
        for (int pi = 0; pi < layout.p; ++pi) {
            const int gi = initial_boundary.P_indices[static_cast<size_t>(pi)];
            const double weighted = levels[static_cast<size_t>(gi)].atomicity * cell.chem_p(pi);
            chemistry_nuclei += weighted;
            chemistry_absolute += std::abs(weighted);
            if (layout.positive_ion[static_cast<size_t>(pi)]) {
                integrated_ion_chemistry += dx * weighted;
            } else {
                integrated_pump_nuclei += dx *
                    levels[static_cast<size_t>(gi)].atomicity * cell.pump_p(pi);
            }
        }
        for (int ai = 0; ai < layout.a; ++ai) {
            const int gi = initial_boundary.A_indices[static_cast<size_t>(ai)];
            const double weighted = levels[static_cast<size_t>(gi)].atomicity * cell.chem_a(ai);
            chemistry_nuclei += weighted;
            chemistry_absolute += std::abs(weighted);
            integrated_pump_nuclei += dx *
                levels[static_cast<size_t>(gi)].atomicity * cell.pump_a(ai);
        }
        for (int mi = 0; mi < layout.m; ++mi) {
            const int gi = initial_boundary.M_indices[static_cast<size_t>(mi)];
            const double weighted = levels[static_cast<size_t>(gi)].atomicity * cell.chem_m(mi);
            chemistry_nuclei += weighted;
            chemistry_absolute += std::abs(weighted);
            integrated_pump_nuclei += dx *
                levels[static_cast<size_t>(gi)].atomicity * cell.pump_m(mi);
        }
        out.max_chemistry_nuclei_relative_error = std::max(
            out.max_chemistry_nuclei_relative_error,
            std::abs(chemistry_nuclei) / std::max(1.0, chemistry_absolute));
    }
    auto nuclei_flux = [&](int node, bool ions_only) {
        double flux = 0.0;
        for (int pi = 0; pi < layout.p; ++pi) {
            if (!layout.positive_ion[static_cast<size_t>(pi)]) continue;
            const int gi = initial_boundary.P_indices[static_cast<size_t>(pi)];
            flux += levels[static_cast<size_t>(gi)].atomicity *
                global_ion_velocity_cm_s(config, levels[static_cast<size_t>(gi)],
                    x[static_cast<size_t>(node)]) * state.p[static_cast<size_t>(node)](pi);
        }
        if (ions_only) return flux;
        for (int ai = 0; ai < layout.a; ++ai) {
            const int gi = initial_boundary.A_indices[static_cast<size_t>(ai)];
            flux += levels[static_cast<size_t>(gi)].atomicity *
                initial_boundary.u_A * state.a[static_cast<size_t>(node)](ai);
        }
        for (int mi = 0; mi < layout.m; ++mi) {
            const int gi = initial_boundary.M_indices[static_cast<size_t>(mi)];
            flux += levels[static_cast<size_t>(gi)].atomicity *
                initial_boundary.u_M * state.m[static_cast<size_t>(node)](mi);
        }
        return flux;
    };
    const double total_flux_start = nuclei_flux(0, false);
    const double total_flux_end = nuclei_flux(layout.nodes - 1, false);
    out.integrated_nuclei_balance_relative_error = std::abs(
        total_flux_end - total_flux_start + integrated_pump_nuclei) /
        std::max({1.0, std::abs(total_flux_start), std::abs(total_flux_end),
            std::abs(integrated_pump_nuclei)});
    const double ion_flux_start = nuclei_flux(0, true);
    const double ion_flux_end = nuclei_flux(layout.nodes - 1, true);
    out.ion_flux_chemistry_relative_error = std::abs(
        ion_flux_end - ion_flux_start - integrated_ion_chemistry) /
        std::max({1.0, std::abs(ion_flux_start), std::abs(ion_flux_end),
            std::abs(integrated_ion_chemistry)});
    if (!(out.residual_norm < 1.0e-8)) {
        throw std::runtime_error("Global BVP failed the scaled residual acceptance criterion.");
    }
    if (!(out.max_chemistry_nuclei_relative_error < 1.0e-10)) {
        throw std::runtime_error("Global BVP failed cell-local chemistry nuclei conservation.");
    }
    const double coarse_grid_balance_tolerance = std::max(
        1.0e-7,
        1.0e-6 / std::pow(static_cast<double>(std::max(1, layout.nodes - 1)), 2.0));
    if (!(out.integrated_nuclei_balance_relative_error < coarse_grid_balance_tolerance)) {
        const Clock::time_point logging_start = Clock::now();
        std::cout << std::setprecision(17)
                  << "[DCR_Solver][global-bvp-balance-failure] residual="
                  << out.residual_norm
                  << " ion=" << out.final_ion_residual
                  << " A=" << out.final_flow_a_residual
                  << " local=" << out.final_local_residual
                  << " M_forward=" << out.final_flow_m_residual
                  << " nuclei=" << out.integrated_nuclei_balance_relative_error
                  << " integrated_equation_defect="
                  << integrated_equation_defect(final_residual)
                  << " flux_balance_absolute="
                  << (total_flux_end - total_flux_start + integrated_pump_nuclei)
                  << std::endl;
        out.diagnostics.logging_seconds += elapsed_seconds(logging_start);
        throw std::runtime_error(
            "Global BVP failed integrated nuclei balance: relative_error=" +
            std::to_string(out.integrated_nuclei_balance_relative_error) +
            ", tolerance=" + std::to_string(coarse_grid_balance_tolerance) +
            ", residual=" + std::to_string(out.residual_norm) +
            ", ion_block=" + std::to_string(out.final_ion_residual) +
            ", A_block=" + std::to_string(out.final_flow_a_residual) +
            ", local_block=" + std::to_string(out.final_local_residual) +
            ", M_forward=" + std::to_string(out.final_flow_m_residual));
    }
    if (!(out.ion_flux_chemistry_relative_error < 1.0e-7)) {
        throw std::runtime_error("Global BVP failed ion flux-chemistry balance.");
    }
    for (int k = 0; k < layout.nodes; ++k) {
        auto require_positive = [](const dcr::base::Vector& values) {
            for (int i = 0; i < values.size(); ++i) {
                if (!std::isfinite(values(i)) || !(values(i) > 0.0)) return false;
            }
            return true;
        };
        auto require_nonnegative = [](const dcr::base::Vector& values) {
            return values.allFinite() && (values.array() >= 0.0).all();
        };
        if (!require_positive(state.p[static_cast<size_t>(k)]) ||
            !require_positive(state.a[static_cast<size_t>(k)]) ||
            !require_nonnegative(state.m[static_cast<size_t>(k)])) {
            throw std::runtime_error("Global BVP produced a non-positive or non-finite population.");
        }
    }
    out.diagnostics.reduced_condition_estimate =
        estimate_reduced_preconditioner_condition(final_evaluator, y, out.diagnostics);
    const Clock::time_point output_start = Clock::now();
    out.converged = true;
    out.history.x_cm = x;
    out.history.boundary_elapsed_seconds = initial_boundary.elapsed_seconds;
    out.history.boundary_iterations = initial_boundary.iterations;
    AtomicRateCalculator rate_calculator(atomic_data);
    for (int k = 0; k < layout.nodes; ++k) {
        const auto full = make_background_full(state.p[static_cast<size_t>(k)], initial_boundary, atomic_data.get_total_states());
        out.history.background_full.push_back(full);
        out.history.flowA.push_back(state.a[static_cast<size_t>(k)]);
        out.history.flowM.push_back(state.m[static_cast<size_t>(k)]);
        const auto local = assemble_local_system(
            config, atomic_data, plasma, grid, initial_boundary, full,
            state.a[static_cast<size_t>(k)], state.m[static_cast<size_t>(k)], x[static_cast<size_t>(k)]);
        out.history.rate_diagnostics.push_back(rate_calculator.evaluate(
            config, initial_boundary, local, full, x[static_cast<size_t>(k)], plasma, grid));
        if (k > 0) {
            out.history.cell_elapsed_seconds.push_back(0.0);
            out.history.cell_iterations.push_back(total_iterations);
            out.history.cell_converged.push_back(1);
        }
    }
    for (int pi = 0; pi < layout.p; ++pi) {
        const int gi = initial_boundary.P_indices[static_cast<size_t>(pi)];
        if (gi >= 0 && gi < out.boundary.population.size()) out.boundary.population(gi) = state.p[0](pi);
    }
    out.boundary.flowA_last = state.a[0];
    out.boundary.flowM_last = state.m[0];
    out.boundary.have_flow_last = true;
    out.boundary.converged = true;
    out.diagnostics.logging_seconds += elapsed_seconds(output_start);
    out.diagnostics.total_seconds = elapsed_seconds(solve_start);
    return out;
}

} // namespace dcr::solver
