#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>

#include "../src/atomic/AtomicData.hpp"
#include "../src/physics/WallBoundary.hpp"
#include "../src/solver/boundary/BoundaryPhase.hpp"
#include "../src/solver/marching/LocalSystemAssembler.hpp"
#include "TestDCRSetup.hpp"

namespace {

std::pair<double, double> weighted_sum(
    const dcr::solver::LocalChemistrySources& chemistry,
    const dcr::solver::BoundaryPhaseResult& boundary,
    const std::vector<dcr::atomic::EnergyLevel>& levels) {
    double total = 0.0;
    double absolute = 0.0;
    auto accumulate = [&](const dcr::base::Vector& source, const std::vector<int>& indices) {
        for (int i = 0; i < source.size(); ++i) {
            const int gi = indices[static_cast<size_t>(i)];
            const double value = levels[static_cast<size_t>(gi)].atomicity * source(i);
            total += value;
            absolute += std::abs(value);
        }
    };
    accumulate(chemistry.background, boundary.P_indices);
    accumulate(chemistry.flowA, boundary.A_indices);
    accumulate(chemistry.flowM, boundary.M_indices);
    return {total, absolute};
}

} // namespace

int main() {
    auto config = test_dcr::load_reference_config(2, false);
    dcr::atomic::AtomicData atomic_data(config);
    auto plasma = test_dcr::make_plasma_state(config, atomic_data.get_total_states());
    const auto temperatures = test_dcr::plasma_temperatures_at(config, 0.0);
    test_dcr::EEDFContext eedf(temperatures.electron_eV);
    const double ion_mass_amu = test_dcr::estimate_ion_mass_amu(config);
    const auto wall = dcr::physics::compute_wall_recycling(
        config.wall.material,
        temperatures.electron_eV,
        temperatures.ion_eV,
        ion_mass_amu,
        config.wall.sheath_potential_drop);
    const auto boundary = dcr::solver::run_boundary_phase(
        config, atomic_data, plasma, eedf.grid, ion_mass_amu, wall);
    if (!boundary.converged) throw std::runtime_error("Boundary phase did not converge.");

    const auto& levels = atomic_data.get_levels();
    for (int sample = 0; sample < 4; ++sample) {
        dcr::base::Vector p = test_dcr::compact_background_from_boundary(boundary);
        dcr::base::Vector a = test_dcr::flowA_from_boundary(boundary);
        dcr::base::Vector m = test_dcr::flowM_from_boundary(boundary);
        for (int i = 0; i < p.size(); ++i) {
            p(i) = std::max(1.0e4, p(i) * (0.7 + 0.1 * ((i + sample) % 5)));
        }
        for (int i = 0; i < a.size(); ++i) {
            a(i) = std::max(1.0e4, a(i) * (0.6 + 0.1 * ((2 * i + sample) % 6)));
        }
        for (int i = 0; i < m.size(); ++i) {
            m(i) = std::max(1.0e4, m(i) * (0.5 + 0.1 * ((3 * i + sample) % 7)));
        }

        const auto full = test_dcr::full_background_from_compact(
            p, boundary, atomic_data.get_total_states());
        const auto local = dcr::solver::assemble_local_system(
            config, atomic_data, plasma, eedf.grid, boundary, full, a, m, 0.1);
        const auto chemistry = dcr::solver::assemble_local_chemistry_sources(
            local, boundary, atomic_data, p, a, m);

        const auto [nuclei_sum, absolute_sum] = weighted_sum(chemistry, boundary, levels);
        const double relative_error = std::abs(nuclei_sum) / std::max(1.0, absolute_sum);
        if (!(relative_error < 1.0e-10)) {
            double largest_column_contribution = 0.0;
            int largest_column = -1;
            for (int gj = 0; gj < local.R_source_full.cols(); ++gj) {
                double column_defect = 0.0;
                for (int gi = 0; gi < local.R_source_full.rows(); ++gi) {
                    column_defect += levels[static_cast<size_t>(gi)].atomicity *
                        local.R_source_full(gi, gj);
                }
                double donor_population = full(gj);
                for (int ai = 0; ai < a.size(); ++ai) {
                    if (boundary.A_indices[static_cast<size_t>(ai)] == gj) donor_population += a(ai);
                }
                for (int mi = 0; mi < m.size(); ++mi) {
                    if (boundary.M_indices[static_cast<size_t>(mi)] == gj) donor_population += m(mi);
                }
                const double contribution = column_defect * donor_population;
                if (std::abs(contribution) > std::abs(largest_column_contribution)) {
                    largest_column_contribution = contribution;
                    largest_column = gj;
                }
            }
            throw std::runtime_error(
                "Cell-local chemistry nuclei cancellation failed for sample " +
                std::to_string(sample) + ": relative_error=" +
                std::to_string(relative_error) + ", largest_column=" +
                std::to_string(largest_column) + ", label=" +
                (largest_column >= 0 ? levels[static_cast<size_t>(largest_column)].label : "none") +
                ", contribution=" + std::to_string(largest_column_contribution));
        }

        constexpr double epsilon = 1.0e-4;
        dcr::base::Vector p_plus = p;
        dcr::base::Vector p_minus = p;
        dcr::base::Vector a_plus = a;
        dcr::base::Vector a_minus = a;
        dcr::base::Vector m_plus = m;
        dcr::base::Vector m_minus = m;
        for (int i = 0; i < p.size(); ++i) {
            const double direction = 0.3 + 0.01 * ((i + sample) % 17);
            p_plus(i) *= std::exp(epsilon * direction);
            p_minus(i) *= std::exp(-epsilon * direction);
        }
        for (int i = 0; i < a.size(); ++i) {
            const double direction = 0.4 + 0.02 * ((i + sample) % 11);
            a_plus(i) *= std::exp(epsilon * direction);
            a_minus(i) *= std::exp(-epsilon * direction);
        }
        for (int i = 0; i < m.size(); ++i) {
            const double direction = 0.2 + 0.015 * ((i + sample) % 13);
            m_plus(i) *= std::exp(epsilon * direction);
            m_minus(i) *= std::exp(-epsilon * direction);
        }
        auto evaluate = [&](const dcr::base::Vector& p_state,
                            const dcr::base::Vector& a_state,
                            const dcr::base::Vector& m_state) {
            const auto full_state = test_dcr::full_background_from_compact(
                p_state, boundary, atomic_data.get_total_states());
            const auto local_state = dcr::solver::assemble_local_system(
                config, atomic_data, plasma, eedf.grid, boundary,
                full_state, a_state, m_state, 0.1);
            return dcr::solver::assemble_local_chemistry_sources(
                local_state, boundary, atomic_data, p_state, a_state, m_state);
        };
        const auto plus = evaluate(p_plus, a_plus, m_plus);
        const auto minus = evaluate(p_minus, a_minus, m_minus);
        dcr::solver::LocalChemistrySources derivative{
            (plus.background - minus.background) / (2.0 * epsilon),
            (plus.flowA - minus.flowA) / (2.0 * epsilon),
            (plus.flowM - minus.flowM) / (2.0 * epsilon)};
        const auto [jacobian_sum, jacobian_absolute] = weighted_sum(derivative, boundary, levels);
        const double jacobian_error = std::abs(jacobian_sum) /
            std::max(1.0, jacobian_absolute);
        if (!(jacobian_error < 1.0e-10)) {
            throw std::runtime_error(
                "Chemistry Jacobian nuclei cancellation failed for sample " +
                std::to_string(sample) + ": relative_error=" +
                std::to_string(jacobian_error));
        }
    }

    std::cout << "[PASS] Cell-local chemistry and its Jacobian conserve nuclei.\n";
    return 0;
}
