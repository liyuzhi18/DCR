#include "QSSSolver.hpp"

#include "QSSBoundary.hpp"
#include "QSSMarching.hpp"
#include "QSSModel.hpp"

namespace dcr::solver {

QSSSolveResult run_qss_dcr(
    const dcr::io::Config& config,
    const dcr::atomic::AtomicData& atomic_data,
    dcr::state::PlasmaState& plasma,
    const EEDFGridView& grid,
    double ion_mass_amu,
    const dcr::physics::WallRecycling& wall) {

    const auto qss_boundary = solve_qss_boundary(
        config, atomic_data, plasma, grid, ion_mass_amu, wall
    );
    const auto marching = run_qss_marching(
        config, atomic_data, plasma, grid, qss_boundary
    );

    QSSSolveResult out;
    out.boundary = qss_boundary.boundary;
    out.history = marching.history;
    out.boundary.population = lift_qss_background_to_full(
        atomic_data.get_total_states(),
        out.boundary,
        qss_boundary.layout,
        marching.retained_background,
        marching.transient_atomic
    );
    out.boundary.flowA_last = marching.flowA;
    out.boundary.flowM_last = marching.flowM;
    out.boundary.have_flow_last = true;

    return out;
}

} // namespace dcr::solver
