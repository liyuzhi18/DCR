#include "PlasmaState.hpp"
#include <cassert>

namespace {
    inline void assert_in_range(dcr::base::Index idx, dcr::base::Index upper) {
        assert(idx >= 0 && idx < upper);
    }
}

namespace dcr::state {
    PlasmaState::PlasmaState(base::Index num_cells, base::Index total_states):
        num_cells_(num_cells), total_states_(total_states) 
    {
        density_bg_ = base::Matrix::Zero(total_states_, num_cells_);
        density_rcl_ = base::Matrix::Zero(total_states_, num_cells_);

        Te_ = base::Vector::Zero(num_cells_);
        Ti_ = base::Vector::Zero(num_cells_);
        vi_ = base::Vector::Zero(num_cells_);
        phi_ = base::Vector::Zero(num_cells_);
        ne_ = base::Vector::Zero(num_cells_);
    }

    base::Real& PlasmaState::n_bg(base::Index i, base::Index k) {
        assert_in_range(i, total_states_);
        assert_in_range(k, num_cells_);
        return density_bg_(i, k);
    }
    base::Real PlasmaState::n_bg(base::Index i, base::Index k) const {
        assert_in_range(i, total_states_);
        assert_in_range(k, num_cells_);
        return density_bg_(i, k);
    }

    Eigen::Map<base::Vector> PlasmaState::n_bg_vec(base::Index k) {
        assert_in_range(k, num_cells_);
        return Eigen::Map<base::Vector>(&density_bg_(0, k), total_states_);
    }
    Eigen::Map<const base::Vector> PlasmaState::n_bg_vec(base::Index k) const {
        assert_in_range(k, num_cells_);
        return Eigen::Map<const base::Vector>(&density_bg_(0, k), total_states_);
    }

    base::Real& PlasmaState::n_rcl(base::Index i, base::Index k) {
        assert_in_range(i, total_states_);
        assert_in_range(k, num_cells_);
        return density_rcl_(i, k);
    }
    base::Real PlasmaState::n_rcl(base::Index i, base::Index k) const {
        assert_in_range(i, total_states_);
        assert_in_range(k, num_cells_);
        return density_rcl_(i, k);
    }

    base::Vector PlasmaState::get_rcl_subset(base::Index k, const base::IndexVec& indices) const {
        assert_in_range(k, num_cells_);
        base::Index subset_size = static_cast<base::Index>(indices.size());
        base::Vector vec(subset_size);
        
        for (base::Index j = 0; j < subset_size; ++j) {
            assert_in_range(indices[j], total_states_);
            // Read from Global Index j at Cell k
            vec(j) = density_rcl_(indices[j], k);
        }
        return vec;
    }

    void PlasmaState::set_rcl_subset(base::Index k, const base::IndexVec& indices, const base::Vector& values) {
        assert_in_range(k, num_cells_);
        assert(indices.size() == (size_t)values.size());
        
        for (size_t j = 0; j < indices.size(); ++j) {
            assert_in_range(indices[j], total_states_);
            // Write to Global Index (indices[j]) at Cell k
            density_rcl_(indices[j], k) = values(j);
        }
    }

    const base::Vector& PlasmaState::Te() const { return Te_; }
    const base::Vector& PlasmaState::Ti() const { return Ti_; }
    const base::Vector& PlasmaState::vi() const { return vi_; }
    const base::Vector& PlasmaState::phi() const { return phi_; }
    const base::Vector& PlasmaState::ne() const { return ne_; }

    base::Real PlasmaState::electron_density_cm3() const {
        if (ne_.size() == 0) return 0.0;
        return ne_(0);
    }

    base::Real PlasmaState::electron_temperature_ev() const {
        if (Te_.size() == 0) return 0.0;
        return Te_(0);
    }

    base::Real PlasmaState::ion_temperature_ev() const {
        if (Ti_.size() == 0) return 0.0;
        return Ti_(0);
    }
}
