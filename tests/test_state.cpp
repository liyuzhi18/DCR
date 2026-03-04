#include <cassert>
#include <iostream>

#include "../src/state/PlasmaState.hpp"

int main() {
    using dcr::base::Index;
    using dcr::base::Real;
    using dcr::state::PlasmaState;

    std::cout << "--- Testing PlasmaState ---\n";

    const Index num_cells = 3;
    const Index total_states = 4;
    PlasmaState state(num_cells, total_states);

    // Initial values should be zero
    for (Index k = 0; k < num_cells; ++k) {
        for (Index i = 0; i < total_states; ++i) {
            assert(state.n_bg(i, k) == Real(0));
            assert(state.n_rcl(i, k) == Real(0));
        }
    }
    assert(state.Te().size() == num_cells);
    assert(state.Ti().size() == num_cells);
    assert(state.vi().size() == num_cells);
    assert(state.phi().size() == num_cells);
    assert(state.ne().size() == num_cells);

    // Scalar access
    state.n_bg(2, 1) = 1.23;
    assert(state.n_bg(2, 1) == Real(1.23));

    // Vector view access (column map)
    auto col = state.n_bg_vec(1);
    assert(col.size() == total_states);
    assert(col(2) == Real(1.23));
    col(0) = 4.56;
    assert(state.n_bg(0, 1) == Real(4.56));

    // Recycling subset helpers
    state.n_rcl(1, 2) = 7.0;
    state.n_rcl(3, 2) = 8.0;

    dcr::base::IndexVec indices = {1, 3};
    auto subset = state.get_rcl_subset(2, indices);
    assert(subset.size() == 2);
    assert(subset(0) == Real(7.0));
    assert(subset(1) == Real(8.0));

    dcr::base::Vector new_vals(2);
    new_vals << 9.0, 10.0;
    state.set_rcl_subset(2, indices, new_vals);

    assert(state.n_rcl(1, 2) == Real(9.0));
    assert(state.n_rcl(3, 2) == Real(10.0));

    // Field initialization accessors
    state.init_Te().setConstant(5.0);
    state.init_Ti().setConstant(1.0);
    state.init_vi().setConstant(2.0);
    state.init_phi().setConstant(3.0);
    state.init_ne().setConstant(4.0);

    assert(state.Te()(0) == Real(5.0));
    assert(state.Ti()(1) == Real(1.0));
    assert(state.vi()(2) == Real(2.0));
    assert(state.phi()(0) == Real(3.0));
    assert(state.ne()(0) == Real(4.0));
    assert(state.electron_density_cm3() == Real(4.0));
    assert(state.ion_temperature_ev() == Real(1.0));

    std::cout << "[PASS] PlasmaState checks.\n";
    return 0;
}
