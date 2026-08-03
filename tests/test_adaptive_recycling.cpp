#include <cassert>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>

#include "../src/atomic/EnergyLevel.hpp"
#include "../src/base/Constants.hpp"
#include "../src/solver/marching/AdaptiveRecycling.hpp"

namespace {

void print_compare(const char* label, double value, double expected) {
    std::cout << "  " << label << ": got=" << value << " expected=" << expected
              << " diff=" << (value - expected) << "\n";
}

void require_close(const char* label, double value, double expected, double tol = 1e-12) {
    print_compare(label, value, expected);
    if (std::abs(value - expected) > tol) {
        throw std::runtime_error(std::string(label) + " mismatch");
    }
}

void require_relative(const char* label, double value, double expected, double rel_tol = 1e-12) {
    print_compare(label, value, expected);
    const double scale = std::max(1.0, std::abs(expected));
    if (std::abs(value - expected) / scale > rel_tol) {
        throw std::runtime_error(std::string(label) + " mismatch");
    }
}

dcr::base::Vector vec2(double a, double b) {
    dcr::base::Vector v(2);
    v << a, b;
    return v;
}

dcr::base::Vector full_state(double ion_a, double ion_b) {
    dcr::base::Vector v(3);
    v << ion_a, 0.0, ion_b;
    return v;
}

dcr::base::Vector neutral_state(double atom_a, double atom_b, double mol_a) {
    dcr::base::Vector v(4);
    v << 0.0, atom_a, atom_b, mol_a;
    return v;
}

} // namespace

int main() {
    std::cout << "--- Testing AdaptiveRecycling ---\n";

    const std::vector<double> x = {0.0, 0.1, 0.2, 0.3, 0.4};
    const std::vector<dcr::base::Vector> flowA = {
        vec2(8.0, 2.0),
        vec2(4.0, 1.0),
        vec2(0.8, 0.2),
        vec2(0.08, 0.01),
        vec2(0.0, 0.0),
    };

    const auto frac = dcr::solver::normalized_atomic_flow_fraction(flowA);
    require_close("F_A[0]", frac[0], 1.0, 1e-14);
    require_close("F_A[1]", frac[1], 0.5, 1e-14);
    require_close("F_A[2]", frac[2], 0.1, 1e-14);
    require_close("F_A[3]", frac[3], 0.009, 1e-14);
    require_close("F_A[4]", frac[4], 0.0, 1e-14);
    assert(frac.size() == flowA.size());
    assert(std::abs(frac[0] - 1.0) < 1e-14);
    assert(std::abs(frac[1] - 0.5) < 1e-14);
    assert(std::abs(frac[2] - 0.1) < 1e-14);
    assert(std::abs(frac[3] - 0.009) < 1e-14);
    assert(frac[4] == 0.0);

    const auto estimate = dcr::solver::estimate_recycling_layer_from_atomic_flow(
        x, flowA, 1e-2
    );
    const auto generic_estimate = dcr::solver::estimate_flow_depletion_layer(
        x, flowA, 1e-2
    );
    std::cout << "  L1 estimate: found=" << estimate.found << " index=" << estimate.index
              << " L1_cm=" << estimate.L1_cm << " expected_index=3 expected_L1_cm=0.3\n";
    assert(estimate.found);
    assert(estimate.index == 3);
    assert(std::abs(estimate.L1_cm - 0.3) < 1e-14);
    assert(generic_estimate.found);
    assert(generic_estimate.index == estimate.index);
    assert(std::abs(generic_estimate.length_cm - estimate.L1_cm) < 1e-14);

    const auto not_found = dcr::solver::estimate_recycling_layer_from_atomic_flow(
        x, flowA, 1e-4
    );
    assert(!not_found.found);
    assert(not_found.index == -1);
    assert(not_found.L1_cm == 0.0);

    const std::vector<dcr::base::Vector> zero_initial = {vec2(0.0, 0.0), vec2(1.0, 0.0)};
    const auto zero_frac = dcr::solver::normalized_atomic_flow_fraction(zero_initial);
    assert(zero_frac.size() == zero_initial.size());
    assert(zero_frac[0] == 0.0);
    assert(zero_frac[1] == 0.0);

    const std::vector<double> x_uniform = {0.0, 0.25, 0.5, 0.75, 1.0};
    const auto u = dcr::solver::linear_presheath_velocity_profile_cm_s(
        x_uniform, 1.0, 10.0, 0.1
    );
    require_close("u[0]", u[0], 10.0, 1e-14);
    require_close("u[1]", u[1], 7.5, 1e-14);
    require_close("u[2]", u[2], 5.0, 1e-14);
    require_close("u[3]", u[3], 2.5, 1e-14);
    require_close("u[4]", u[4], 1.0, 1e-14);
    assert(u.size() == x_uniform.size());
    assert(std::abs(u[0] - 10.0) < 1e-14);
    assert(std::abs(u[1] - 7.5) < 1e-14);
    assert(std::abs(u[2] - 5.0) < 1e-14);
    assert(std::abs(u[3] - 2.5) < 1e-14);
    assert(std::abs(u[4] - 1.0) < 1e-14);

    const std::vector<double> density(x_uniform.size(), 10.0);
    const auto div = dcr::solver::ion_flux_divergence_cm3_s(x_uniform, density, u);
    assert(div.size() == x_uniform.size());
    const std::vector<double> div_expected = {-100.0, -100.0, -100.0, -80.0, -60.0};
    for (size_t i = 0; i < div.size(); ++i) {
        require_close(("div_ion[" + std::to_string(i) + "]").c_str(), div[i], div_expected[i]);
    }

    const std::vector<double> falling_density = {10.0, 8.0, 6.0, 4.0, 2.0};
    const auto falling_div = dcr::solver::ion_flux_divergence_cm3_s(
        x_uniform, falling_density, u
    );
    const std::vector<double> falling_div_expected = {-160.0, -140.0, -100.0, -56.0, -32.0};
    for (size_t i = 0; i < falling_div.size(); ++i) {
        require_close(
            ("div_ion_falling_density[" + std::to_string(i) + "]").c_str(),
            falling_div[i],
            falling_div_expected[i]
        );
    }

    const std::vector<double> y_quadratic = {0.0, 1.0, 4.0, 9.0};
    const std::vector<double> x_quadratic = {0.0, 1.0, 2.0, 3.0};
    const auto dy = dcr::solver::finite_difference_derivative(x_quadratic, y_quadratic);
    require_close("dy[0]", dy[0], 1.0, 1e-14);
    require_close("dy[1]", dy[1], 2.0, 1e-14);
    require_close("dy[2]", dy[2], 4.0, 1e-14);
    require_close("dy[3]", dy[3], 5.0, 1e-14);
    assert(dy.size() == y_quadratic.size());
    assert(std::abs(dy[0] - 1.0) < 1e-14);
    assert(std::abs(dy[1] - 2.0) < 1e-14);
    assert(std::abs(dy[2] - 4.0) < 1e-14);
    assert(std::abs(dy[3] - 5.0) < 1e-14);

    std::vector<dcr::atomic::EnergyLevel> levels(3);
    levels[0].atomicity = 1;
    levels[2].atomicity = 2;
    const std::vector<dcr::base::Vector> background = {
        full_state(10.0, 3.0),
        full_state(10.0, 3.0),
        full_state(10.0, 3.0),
        full_state(10.0, 3.0),
        full_state(10.0, 3.0),
    };
    const auto nuclei_div = dcr::solver::nuclei_weighted_ion_flux_divergence_cm3_s(
        x_uniform,
        background,
        levels,
        {0, 2},
        {10.0, 5.0},
        1.0,
        0.1
    );
    assert(nuclei_div.size() == x_uniform.size());
    const std::vector<double> nuclei_div_expected = {-130.0, -130.0, -130.0, -104.0, -78.0};
    for (size_t i = 0; i < nuclei_div.size(); ++i) {
        require_close(("nuclei_div[" + std::to_string(i) + "]").c_str(), nuclei_div[i], nuclei_div_expected[i]);
    }

    const auto truncated_div = dcr::solver::nuclei_weighted_ion_flux_divergence_cm3_s(
        x_uniform,
        background,
        levels,
        {0},
        {10.0},
        0.5
    );
    assert(std::abs(truncated_div[0] + 200.0) < 1e-12);
    assert(std::abs(truncated_div[1] + 180.0) < 1e-12);
    assert(std::abs(truncated_div[2] + 100.0) < 1e-12);
    assert(truncated_div[3] == 0.0);
    assert(truncated_div[4] == 0.0);

    levels[0].mass_amu = 1.0;
    levels[2].mass_amu = 4.0;
    const auto bohm_speeds = dcr::solver::ion_bohm_speeds_cm_s(levels, {0, 2, 99}, 1.0);
    const double expected_bohm_h = std::sqrt(
        dcr::base::constants::eV_to_erg / dcr::base::constants::amu_g
    );
    require_relative("u_B_H_1eV", bohm_speeds[0], expected_bohm_h);
    require_relative("u_B_mass4_1eV", bohm_speeds[1], 0.5 * expected_bohm_h);
    require_close("u_B_invalid_index", bohm_speeds[2], 0.0, 1e-14);
    const auto zero_te_speeds = dcr::solver::ion_bohm_speeds_cm_s(levels, {0, 2}, 0.0);
    require_close("u_B_zero_Te[0]", zero_te_speeds[0], 0.0, 1e-14);
    require_close("u_B_zero_Te[1]", zero_te_speeds[1], 0.0, 1e-14);

    std::vector<dcr::atomic::EnergyLevel> neutral_levels(4);
    neutral_levels[1].atomicity = 1;
    neutral_levels[2].atomicity = 1;
    neutral_levels[3].atomicity = 2;
    const std::vector<dcr::base::Vector> neutral_background = {
        neutral_state(1.0, 3.0, 3.0),
        neutral_state(0.0, 0.0, 0.0),
    };
    const auto comp = dcr::solver::build_neutral_transport_compensation(
        neutral_background,
        neutral_levels,
        {1, 2},
        {3},
        {100.0, 50.0},
        {20.0, 10.0}
    );
    require_close("P_rem[0]", comp.remainder_nuclei_cm3_s[0], 80.0, 1e-14);
    require_close("atomic_nuc_div[0]", comp.atomic_group_nuclei_divergence_cm3_s[0], 32.0, 1e-14);
    require_close("molecular_nuc_div[0]", comp.molecular_group_nuclei_divergence_cm3_s[0], 48.0, 1e-14);
    require_close("atomic_state_div[0][0]", comp.atomic_state_divergence_cm3_s[0](0), 8.0, 1e-14);
    require_close("atomic_state_div[0][1]", comp.atomic_state_divergence_cm3_s[0](1), 24.0, 1e-14);
    require_close("molecular_state_div[0][0]", comp.molecular_state_divergence_cm3_s[0](0), 24.0, 1e-14);
    assert(comp.remainder_nuclei_cm3_s.size() == 2);
    assert(std::abs(comp.remainder_nuclei_cm3_s[0] - 80.0) < 1e-14);
    assert(std::abs(comp.atomic_group_nuclei_divergence_cm3_s[0] - 32.0) < 1e-14);
    assert(std::abs(comp.molecular_group_nuclei_divergence_cm3_s[0] - 48.0) < 1e-14);
    assert(comp.atomic_state_divergence_cm3_s[0].size() == 2);
    assert(comp.molecular_state_divergence_cm3_s[0].size() == 1);
    assert(std::abs(comp.atomic_state_divergence_cm3_s[0](0) - 8.0) < 1e-14);
    assert(std::abs(comp.atomic_state_divergence_cm3_s[0](1) - 24.0) < 1e-14);
    assert(std::abs(comp.molecular_state_divergence_cm3_s[0](0) - 24.0) < 1e-14);
    assert(std::abs(comp.remainder_nuclei_cm3_s[1] - 40.0) < 1e-14);
    assert(comp.atomic_group_nuclei_divergence_cm3_s[1] == 0.0);
    assert(comp.molecular_group_nuclei_divergence_cm3_s[1] == 0.0);
    assert(comp.atomic_state_divergence_cm3_s[1](0) == 0.0);
    assert(comp.atomic_state_divergence_cm3_s[1](1) == 0.0);
    assert(comp.molecular_state_divergence_cm3_s[1](0) == 0.0);

    const auto pump = dcr::solver::neutral_pump_exhaust_nuclei_rate_cm3_s(
        neutral_background,
        neutral_levels,
        {1, 2},
        {3},
        10.0,
        4.0,
        2.0
    );
    require_close("pump[0]", pump[0], 32.0, 1e-14);
    require_close("pump[1]", pump[1], 0.0, 1e-14);
    assert(pump.size() == neutral_background.size());
    assert(std::abs(pump[0] - 32.0) < 1e-14);
    assert(pump[1] == 0.0);

    const auto bad_width_pump = dcr::solver::neutral_pump_exhaust_nuclei_rate_cm3_s(
        neutral_background,
        neutral_levels,
        {1, 2},
        {3},
        10.0,
        4.0,
        0.0
    );
    assert(bad_width_pump.size() == neutral_background.size());
    assert(bad_width_pump[0] == 0.0);
    assert(bad_width_pump[1] == 0.0);

    std::cout << "[PASS] AdaptiveRecycling checks.\n";
    return 0;
}
