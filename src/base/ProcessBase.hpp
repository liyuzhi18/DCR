#pragma once
#include "Types.hpp"

// Forward declaration 
namespace dcr::state { class PlasmaState; }
class EEDFGridView;
struct ReactionAccumulator;

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
    };
}