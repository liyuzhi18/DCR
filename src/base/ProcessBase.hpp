#pragma once
#include "Types.hpp"
#include <vector>

// Forward declaration 
namespace dcr::state { class PlasmaState; }
class EEDFGridView;
struct ReactionAccumulator;

struct EnergyLossAccumulator {
    double atomic_ionization_W_cm3 = 0.0;
    double molecular_ionization_W_cm3 = 0.0;
    double molecular_dissociation_W_cm3 = 0.0;
    double molecular_dissociation_rate_cm3_s = 0.0;
};

namespace dcr {
    // Abstract base class for atomic and molecular processes
    class ProcessBase { 
    public:
        virtual ~ProcessBase() = default;
        /* Calculate the reaction rate to put in rate matrix. 
         * @param plasma      The full state of the plasma (Te, ne, densities).
         * @param grid        The electron energy distribution function grid.
         * @param population  The current population vector (Size = Total_States).
         * @param R           The Rate Matrix to update (Size = Total_States x Total_States).
         * @param accumulator Optional pointer to record specific reaction events.
        */
        virtual void apply(
             const dcr::state::PlasmaState& plasma,
             const EEDFGridView& grid,
             const Eigen::VectorXd& population,
             Eigen::MatrixXd& R,
             ReactionAccumulator* accumulator
         ) const = 0;

        virtual void accumulate_energy_loss(
             const dcr::state::PlasmaState& plasma,
             const EEDFGridView& grid,
             const Eigen::VectorXd& population,
             EnergyLossAccumulator& accumulator
        ) const {
             (void)plasma;
             (void)grid;
             (void)population;
             (void)accumulator;
        }

        virtual std::vector<dcr::base::Index> population_dependencies() const {
             return {};
        }
     };
}
