#pragma once

#include "../../atomic/AtomicData.hpp"
#include "../../base/Types.hpp"
#include "../../io/ConfigStructs.hpp"
#include "../boundary/BoundaryPhase.hpp"
#include "LocalSystemAssembler.hpp"

namespace dcr::solver {

struct FlowAdvanceResult {
    // Recycling-flow state at the next spatial location (x+dx).
    dcr::base::Vector flowA_next;
    dcr::base::Vector flowM_next;
    // RHS evaluated at the converged implicit state, useful for diagnostics.
    dcr::base::Vector rhsA;
    dcr::base::Vector rhsM;
    // Effective exhaust speeds used in Gamma_ex / w terms.
    double c_s_A = 0.0;
    double c_s_M = 0.0;
};

// Implicit backward-Euler update for recycling flows over one step:
// solves n_R^{k+1} from (I - dx/u * R_RR + dx/u * ex*I) n_R^{k+1} = n_R^k.
FlowAdvanceResult advance_recycling_flow_one_step(
    const dcr::io::Config& config,
    const dcr::atomic::AtomicData& atomic_data,
    const BoundaryPhaseResult& boundary,
    const LocalSystem& local_system,
    const dcr::base::Vector& flowA,
    const dcr::base::Vector& flowM,
    double dx_cm);

} // namespace dcr::solver
