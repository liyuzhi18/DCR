#include <algorithm>
#include <cassert>
#include <cmath>
#include <iostream>

#include "../src/atomic/AtomicData.hpp"
#include "../src/physics/WallBoundary.hpp"
#include "../src/solver/core/RateAnalysis.hpp"
#include "../src/solver/marching/LocalSystemAssembler.hpp"
#include "../src/solver/boundary/BoundaryPhase.hpp"
#include "TestDCRSetup.hpp"

namespace {

double max_abs_diff(const dcr::base::Vector& a, const dcr::base::Vector& b) {
    assert(a.size() == b.size());
    double out = 0.0;
    for (int i = 0; i < a.size(); ++i) {
        out = std::max(out, std::abs(a(i) - b(i)));
    }
    return out;
}

} // namespace

int main() {
    std::cout << "--- Testing EEDF Boundary Regression ---\n";

    auto cfg_default = test_dcr::load_reference_config(/*num_cells=*/4, /*verbose_logging=*/false);
    auto cfg_explicit = cfg_default;
    cfg_explicit.plasma.eedf.type = "maxwellian";
    cfg_explicit.plasma.eedf.power_p = 1.0;

    dcr::atomic::AtomicData atomic_data(cfg_default);
    const auto boundary_temperatures = test_dcr::plasma_temperatures_at(cfg_default, 0.0);
    const double ion_mass_amu = test_dcr::estimate_ion_mass_amu(cfg_default);
    const auto wall = dcr::physics::compute_wall_recycling(
        cfg_default.wall.material,
        boundary_temperatures.electron_eV,
        boundary_temperatures.ion_eV,
        ion_mass_amu,
        cfg_default.wall.sheath_potential_drop
    );

    auto plasma_default = test_dcr::make_plasma_state(cfg_default, atomic_data.get_total_states());
    auto plasma_explicit = test_dcr::make_plasma_state(cfg_explicit, atomic_data.get_total_states());

    test_dcr::EEDFContext eedf_default(boundary_temperatures.electron_eV, test_dcr::make_eedf_config(cfg_default));
    test_dcr::EEDFContext eedf_explicit(boundary_temperatures.electron_eV, test_dcr::make_eedf_config(cfg_explicit));

    const auto boundary_default = dcr::solver::run_boundary_phase(
        cfg_default, atomic_data, plasma_default, eedf_default.grid, ion_mass_amu, wall
    );
    const auto boundary_explicit = dcr::solver::run_boundary_phase(
        cfg_explicit, atomic_data, plasma_explicit, eedf_explicit.grid, ion_mass_amu, wall
    );

    assert(boundary_default.population.size() == boundary_explicit.population.size());
    assert(max_abs_diff(boundary_default.population, boundary_explicit.population) < 1e-9);
    assert(boundary_default.flowA_last.size() == boundary_explicit.flowA_last.size());
    assert(boundary_default.flowM_last.size() == boundary_explicit.flowM_last.size());
    assert(max_abs_diff(boundary_default.flowA_last, boundary_explicit.flowA_last) < 1e-9);
    assert(max_abs_diff(boundary_default.flowM_last, boundary_explicit.flowM_last) < 1e-9);

    const dcr::base::Vector nP_default = test_dcr::compact_background_from_boundary(boundary_default);
    const dcr::base::Vector nP_explicit = test_dcr::compact_background_from_boundary(boundary_explicit);
    const dcr::base::Vector bg_full_default = test_dcr::full_background_from_compact(
        nP_default, boundary_default, atomic_data.get_total_states()
    );
    const dcr::base::Vector bg_full_explicit = test_dcr::full_background_from_compact(
        nP_explicit, boundary_explicit, atomic_data.get_total_states()
    );

    const auto local_default = dcr::solver::assemble_local_system(
        cfg_default,
        atomic_data,
        plasma_default,
        eedf_default.grid,
        boundary_default,
        bg_full_default,
        test_dcr::flowA_from_boundary(boundary_default),
        test_dcr::flowM_from_boundary(boundary_default),
        0.0
    );
    const auto local_explicit = dcr::solver::assemble_local_system(
        cfg_explicit,
        atomic_data,
        plasma_explicit,
        eedf_explicit.grid,
        boundary_explicit,
        bg_full_explicit,
        test_dcr::flowA_from_boundary(boundary_explicit),
        test_dcr::flowM_from_boundary(boundary_explicit),
        0.0
    );

    dcr::solver::AtomicRateCalculator calculator(atomic_data);
    const auto rates_default = calculator.evaluate(
        cfg_default, boundary_default, local_default, bg_full_default, 0.0
    );
    const auto rates_explicit = calculator.evaluate(
        cfg_explicit, boundary_explicit, local_explicit, bg_full_explicit, 0.0
    );

    assert(rates_default.atomic_effective.valid);
    assert(rates_explicit.atomic_effective.valid);
    assert(std::abs(rates_default.atomic_effective.scd_cm3_s - rates_explicit.atomic_effective.scd_cm3_s) < 1e-18);
    assert(std::abs(rates_default.atomic_effective.acd_cm3_s - rates_explicit.atomic_effective.acd_cm3_s) < 1e-18);

    std::cout << "[PASS] EEDF boundary regression checks.\n";
    return 0;
}
