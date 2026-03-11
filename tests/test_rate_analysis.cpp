#include <cassert>
#include <cmath>
#include <iostream>

#include "../src/atomic/AtomicData.hpp"
#include "../src/physics/WallBoundary.hpp"
#include "../src/solver/core/RateAnalysis.hpp"
#include "../src/solver/marching/CellSolve.hpp"
#include "../src/solver/marching/LocalSystemAssembler.hpp"
#include "../src/solver/marching/MarchingDriver.hpp"
#include "TestDCRSetup.hpp"

int main() {
    std::cout << "--- Testing RateAnalysis ---\n";

    auto cfg = test_dcr::load_reference_config(/*num_cells=*/6, /*verbose_logging=*/false);
    cfg.grid.length_cm = 1.0;
    cfg.grid.type = "linear";
    cfg.grid.first_cell_cm = 0.0;

    dcr::atomic::AtomicData atomic_data(cfg);
    auto plasma = test_dcr::make_plasma_state(cfg, atomic_data.get_total_states());
    const auto boundary_temperatures = test_dcr::plasma_temperatures_at(cfg, 0.0);
    test_dcr::EEDFContext eedf(boundary_temperatures.electron_eV);

    const double ion_mass_amu = test_dcr::estimate_ion_mass_amu(cfg);
    const auto wall = dcr::physics::compute_wall_recycling(
        cfg.wall.material,
        boundary_temperatures.electron_eV,
        boundary_temperatures.ion_eV,
        ion_mass_amu,
        cfg.wall.sheath_potential_drop
    );

    const auto boundary = dcr::solver::run_boundary_phase(
        cfg, atomic_data, plasma, eedf.grid, ion_mass_amu, wall
    );

    const dcr::base::Vector nP_boundary = test_dcr::compact_background_from_boundary(boundary);
    const dcr::base::Vector bg_full_boundary = test_dcr::full_background_from_compact(
        nP_boundary, boundary, atomic_data.get_total_states()
    );
    const dcr::base::Vector flowA_boundary = test_dcr::flowA_from_boundary(boundary);
    const dcr::base::Vector flowM_boundary = test_dcr::flowM_from_boundary(boundary);

    const auto local_boundary = dcr::solver::assemble_local_system(
        cfg,
        atomic_data,
        plasma,
        eedf.grid,
        boundary,
        bg_full_boundary,
        flowA_boundary,
        flowM_boundary,
        0.0
    );

    dcr::solver::AtomicRateCalculator calculator(atomic_data);
    const auto boundary_rates = calculator.evaluate(
        cfg, boundary, local_boundary, bg_full_boundary, 0.0
    );

    assert(boundary_rates.atomic_effective.valid);
    assert(boundary_rates.atomic_effective.atom_ground_index >= 0);
    assert(boundary_rates.atomic_effective.ion_ground_index >= 0);
    assert(std::isfinite(boundary_rates.atomic_effective.scd_cm3_s));
    assert(std::isfinite(boundary_rates.atomic_effective.acd_cm3_s));
    assert(boundary_rates.atomic_effective.scd_cm3_s >= 0.0);
    assert(boundary_rates.atomic_effective.acd_cm3_s >= 0.0);
    assert(boundary_rates.atomic_qss.valid);
    assert(!boundary_rates.atomic_qss.excited_indices.empty());
    assert(boundary_rates.atomic_qss.transport_rate_cm3_s.size() ==
           static_cast<int>(boundary_rates.atomic_qss.excited_indices.size()));
    assert(boundary_rates.atomic_qss.local_source_rate_cm3_s.size() ==
           boundary_rates.atomic_qss.transport_rate_cm3_s.size());
    assert(boundary_rates.atomic_qss.local_loss_rate_cm3_s.size() ==
           boundary_rates.atomic_qss.transport_rate_cm3_s.size());
    assert(boundary_rates.atomic_qss.transport_to_local_ratio.size() ==
           boundary_rates.atomic_qss.transport_rate_cm3_s.size());
    assert(boundary_rates.atomic_qss.transport_to_loss_frequency_ratio.size() ==
           boundary_rates.atomic_qss.transport_rate_cm3_s.size());

    for (int i = 0; i < boundary_rates.atomic_qss.transport_rate_cm3_s.size(); ++i) {
        assert(std::isfinite(boundary_rates.atomic_qss.transport_rate_cm3_s(i)));
        assert(std::isfinite(boundary_rates.atomic_qss.local_source_rate_cm3_s(i)));
        assert(std::isfinite(boundary_rates.atomic_qss.local_loss_rate_cm3_s(i)));
        assert(std::isfinite(boundary_rates.atomic_qss.local_loss_frequency_s(i)));
        assert(std::isfinite(boundary_rates.atomic_qss.transport_to_local_ratio(i)));
        assert(std::isfinite(boundary_rates.atomic_qss.transport_to_loss_frequency_ratio(i)));
        assert(boundary_rates.atomic_qss.transport_rate_cm3_s(i) >= 0.0);
        assert(boundary_rates.atomic_qss.local_source_rate_cm3_s(i) >= 0.0);
        assert(boundary_rates.atomic_qss.local_loss_rate_cm3_s(i) >= 0.0);
        assert(boundary_rates.atomic_qss.local_loss_frequency_s(i) >= 0.0);
        assert(boundary_rates.atomic_qss.transport_to_local_ratio(i) >= 0.0);
        assert(boundary_rates.atomic_qss.transport_to_loss_frequency_ratio(i) >= 0.0);
    }

    const auto history = dcr::solver::run_full_marching(cfg, atomic_data, plasma, eedf.grid, boundary);
    assert(history.rate_diagnostics.size() == static_cast<size_t>(cfg.grid.num_cells));
    for (const auto& snapshot : history.rate_diagnostics) {
        assert(std::isfinite(snapshot.electron_temperature_eV));
        assert(std::isfinite(snapshot.ion_temperature_eV));
        assert(std::isfinite(snapshot.electron_density_cm3));
        assert(snapshot.electron_density_cm3 >= 0.0);
        assert(snapshot.atomic_effective.valid);
        assert(std::isfinite(snapshot.atomic_effective.scd_cm3_s));
        assert(std::isfinite(snapshot.atomic_effective.acd_cm3_s));
        assert(snapshot.atomic_effective.scd_cm3_s >= 0.0);
        assert(snapshot.atomic_effective.acd_cm3_s >= 0.0);
        assert(snapshot.atomic_qss.valid);
        assert(snapshot.atomic_qss.excited_indices.size() ==
               boundary_rates.atomic_qss.excited_indices.size());
    }

    std::cout << "[PASS] RateAnalysis checks.\n";
    return 0;
}
