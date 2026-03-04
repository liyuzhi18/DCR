#pragma once
#include "../base/Types.hpp"
#include <map>
#include <vector>

namespace dcr::state {
    class PlasmaState { 
    public: 
        // Constructs plasma state storage at each cell 
        PlasmaState(base::Index num_cells, base::Index total_states);

        // Background population access helpers
        // Scalar Access
        base::Real& n_bg(base::Index i, base::Index k);
        base::Real n_bg(base::Index i, base::Index k) const;

        // Vector Access 
        Eigen::Map<base::Vector> n_bg_vec(base::Index k);
        Eigen::Map<const base::Vector> n_bg_vec(base::Index k) const;

        // Recycling population access helpers
        // Scalar Access 
        base::Real& n_rcl(base::Index i, base::Index k);
        base::Real n_rcl(base::Index i, base::Index k) const;

        // Extracts specific species into a recycling population vector
        base::Vector get_rcl_subset(base::Index k, const base::IndexVec& indices) const;
        // Writes the subpopulation vector back to global storage
        void set_rcl_subset(base::Index k, const base::IndexVec& indices, const base::Vector& values);

        // Fields
        const base::Vector& Te() const;
        const base::Vector& Ti() const;
        const base::Vector& vi() const;
        const base::Vector& phi() const;
        const base::Vector& ne() const;
        // Convenience: return a representative electron density for single-cell calls.
        base::Real electron_density_cm3() const;
        // Convenience: return a representative electron temperature (eV) for single-cell calls.
        base::Real electron_temperature_ev() const;
        // Convenience: return a representative ion temperature (eV) for single-cell calls.
        base::Real ion_temperature_ev() const;

        // Initialization 
        base::Vector& init_Te() { return Te_; }
        base::Vector& init_Ti() { return Ti_; }
        base::Vector& init_vi() { return vi_; }
        base::Vector& init_phi() { return phi_; }
        base::Vector& init_ne() { return ne_; }
    
    private:
        base::Index num_cells_;
        base::Index total_states_;

        base::Matrix density_bg_; 
        base::Matrix density_rcl_;

        base::Vector Te_, Ti_, vi_, phi_, ne_;
    };
}
