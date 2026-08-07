# Activity Trace Historical Replay

This is a read-only evidence audit. No solver iteration was rerun, and final-state
activity was not substituted for event-time activity.

## Saved evidence

- The A2 stage summary records 500 accepted and 15 rejected P pseudo-time steps.
- It records 300 accepted and 0 rejected A pseudo-time steps.
- It records 300 accepted and 1 rejected M pseudo-time step.
- The detailed legacy log through macro 40 contains 14 ordinary rejected P step
  records, all marked `excessive_state_change`.
- The detailed legacy log also contains 2 rejected P calibration records marked
  `change_above_target`. Calibration retries are not counted identically to
  ordinary rejected steps in the stage summary.
- The detailed legacy log contains 2 M retry records: one
  `catastrophic_state_change` event at macro 35 and one
  `calibration_change_above_target` event at macro 38. The stage summary counts
  only the former as a rejected M step.
- Two initial A calibration probes are logged with `accepted_A=0` and
  `calibration_change_below_target`, but the stage summary correctly counts no A
  rejected steps because those probes increase the calibration timestep.
- The detailed legacy log contains 144 rejected Anderson trials and 144 rejected
  damped-map trials, all rejected by the nonmonotone merit window.

## Missing evidence

- Event-time activity and pathway throughput were not saved for P, A, or M
  limiting rows.
- Full PTC and Anderson candidate vectors were not saved.
- Negative candidate components and the component that bounded each positivity
  blend were not saved.
- The saved last-100-macro CSV contains block-level norms and timesteps, not
  event-time row activity or candidate components.

## Replay verdict

Every historical limiter and Anderson event is activity-unclassifiable at all
four candidate thresholds. Suppression fractions, trace/non-trace event counts,
and counterfactual accepted steps are unavailable. Inferring them from the A2
final state would be temporally invalid. This does not change the saved-state
threshold result, and it does not validate any threshold for production use.
