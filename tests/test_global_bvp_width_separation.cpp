#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>

#include "../src/atomic/AtomicData.hpp"
#include "../src/physics/WallBoundary.hpp"
#include "../src/solver/boundary/BoundaryPhase.hpp"
#include "TestDCRSetup.hpp"

namespace {

void require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

double max_relative_difference(const dcr::base::Vector& left,
                               const dcr::base::Vector& right) {
    require(left.size() == right.size(), "Compared boundary vectors have different sizes.");
    double difference = 0.0;
    for (int i = 0; i < left.size(); ++i) {
        require(std::isfinite(left(i)) && std::isfinite(right(i)),
                "Compared boundary vector contains a non-finite value.");
        difference = std::max(
            difference,
            std::abs(left(i) - right(i)) /
                std::max({1.0, std::abs(left(i)), std::abs(right(i))}));
    }
    return difference;
}

} // namespace

int main() {
    auto narrow_spatial = test_dcr::load_reference_config(2, false);
    require(narrow_spatial.grid.boundary_poloidal_width_cm > 0.0,
            "Legacy width did not initialize the boundary width.");
    require(narrow_spatial.grid.spatial_exhaust_width_cm ==
                narrow_spatial.grid.boundary_poloidal_width_cm,
            "Legacy width did not initialize both separated widths equally.");

    narrow_spatial.numerics.marching_solver = "global_sparse_newton";
    narrow_spatial.numerics.boundary_tolerance = 1.0e-5;
    narrow_spatial.numerics.boundary_max_iterations = 200;
    narrow_spatial.numerics.relaxation = 0.9;
    narrow_spatial.grid.spatial_exhaust_width_cm = 1.0e-3;
    auto wide_spatial = narrow_spatial;
    wide_spatial.grid.spatial_exhaust_width_cm = 1.0e6;
    require(wide_spatial.grid.boundary_poloidal_width_cm ==
                narrow_spatial.grid.boundary_poloidal_width_cm,
            "Boundary width changed with spatial width.");

    dcr::atomic::AtomicData atomic_data(narrow_spatial);
    auto narrow_plasma = test_dcr::make_plasma_state(
        narrow_spatial, atomic_data.get_total_states());
    auto wide_plasma = test_dcr::make_plasma_state(
        wide_spatial, atomic_data.get_total_states());
    const auto temperatures = test_dcr::plasma_temperatures_at(narrow_spatial, 0.0);
    test_dcr::EEDFContext eedf(temperatures.electron_eV);
    const double ion_mass_amu = test_dcr::estimate_ion_mass_amu(narrow_spatial);
    const auto wall = dcr::physics::compute_wall_recycling(
        narrow_spatial.wall.material,
        temperatures.electron_eV,
        temperatures.ion_eV,
        ion_mass_amu,
        narrow_spatial.wall.sheath_potential_drop);

    const auto narrow_boundary = dcr::solver::run_boundary_phase(
        narrow_spatial, atomic_data, narrow_plasma, eedf.grid, ion_mass_amu, wall);
    const auto wide_boundary = dcr::solver::run_boundary_phase(
        wide_spatial, atomic_data, wide_plasma, eedf.grid, ion_mass_amu, wall);

    require(narrow_boundary.converged,
            "Target boundary did not converge with narrow spatial exhaust width.");
    require(wide_boundary.converged,
            "Target boundary did not converge with wide spatial exhaust width.");
    require(narrow_boundary.iterations == wide_boundary.iterations,
            "Spatial exhaust width changed target-boundary convergence iterations.");

    constexpr double tolerance = 1.0e-13;
    require(max_relative_difference(narrow_boundary.population, wide_boundary.population) < tolerance,
            "Spatial exhaust width changed the converged target-boundary population.");
    require(max_relative_difference(narrow_boundary.flowA_last, wide_boundary.flowA_last) < tolerance,
            "Spatial exhaust width changed the converged target atomic recycling flow.");
    require(max_relative_difference(narrow_boundary.flowM_last, wide_boundary.flowM_last) < tolerance,
            "Spatial exhaust width changed the converged target molecular recycling flow.");

    std::cout << "[PASS] Spatial exhaust width is isolated from the pre-BVP target boundary.\n";
    return 0;
}
