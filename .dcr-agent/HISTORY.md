# Experiment History

## Historical A2 activity scan

- Hypothesis: a global activity floor may safely identify rows that can be excluded from numerical controller norms.
- Experiment: replayed the saved A2 state at `epsilon_abs` values `1e-8`, `1e-10`, `1e-12`, and `1e-14` with pathway protection.
- Observation: all 455 protected rows survived at every threshold; the scan was complete, but the saved history did not contain event-time states needed for counterfactual controller decisions.
- Interpretation: historical replay alone cannot establish a safe production activity floor.
- Status: unresolved.
- Artifacts: `output/nested_dcr_fixed_gamma_T3p120_A2_from_T3p110/A2_activity_trace_deweighting_scan.log` and `output/nested_dcr_fixed_gamma_T3p120_A2_from_T3p110/A2_activity_trace_historical_replay.md`.

## Fixed-Gamma activity controller at 1e-10

- Hypothesis: activity de-weighting at `epsilon_abs=1e-10` prevents trace rows from controlling P/A/M timesteps or Anderson positivity without changing physical equations.
- Experiment: ran controller ON and a same-source controller-OFF/global-PTC control from the immutable 3.110 eV checkpoint at fixed Gamma and 3.120 eV.
- Observation: both trajectories had identical numerical metrics and stopped at macro 79 on `be_steady_operator_mismatch`; no trace row changed a controller decision.
- Interpretation: `epsilon_abs=1e-10` had no effect on this trajectory. Differences from historical A2 begin at an unrelated macro-21 watchdog-path divergence.
- Status: rejected for this trajectory.
- Artifacts: `output/nested_dcr_fixed_gamma_T3p120_A_activity_eps1e10_v2_optimized_from_T3p110/activity_controller_result.md`.

## Macro-79 BE/steady mismatch diagnostic

- Hypothesis: the macro-79 stop is a mismatch-detector state or scaling defect rather than a genuine difference between BE and steady physical operators.
- Experiment: reproduce the first stop once with read-only row, temporal-increment, conservation-projection, and source-term diagnostics for macros 75 through 79.
- Observation: no physical row exceeded tolerance. The stop was triggered by the nuclei-weighted physical-row projection at cell 164, where row-level floating-point defects project to `-6.057282645477e6` and are divided by a cancellation-sensitive net scale, producing `1.119800571231e-6`. The status line instead reports the unrelated maximum-relative row at cell 183. The jump appears immediately after the accepted macro-78 Anderson blend.
- Interpretation: BE copies the same steady operator and adds only the intended temporal diagonal/RHS. States, frozen partners, row scaling, source terms, and boundary treatment agree. The mismatch is a detector scaling and attribution bug, not a physical operator mismatch.
- Status: hypothesis accepted; classification `mismatch_detector_state_or_scaling_bug`.
- Artifact: `output/nested_dcr_fixed_gamma_T3p120_be_mismatch_macro79_diagnostic/run.log`.
- Report: `output/nested_dcr_fixed_gamma_T3p120_be_mismatch_macro79_diagnostic/diagnostic_report.md`.
