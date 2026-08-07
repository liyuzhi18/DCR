# Current Status

- Branch: `adaptive-recycling-closure`
- Numerical-work commit: `d9d1578`
- Baseline case: fixed-Gamma 3.120 eV A2 trajectory loaded from the accepted 3.110 eV temperature-continuation checkpoint.
- Current experiment: completed read-only diagnosis of the first `be_steady_operator_mismatch` reported at macro 79 in `nested_dcr_fixed_gamma_T3p120_A_activity_eps1e10_v2_optimized_from_T3p110`.
- Changed relative to baseline: diagnostic logging only for the existing BE/steady identity detector; no trace de-weighting, timestep, Anderson, physical equation, merit, or convergence behavior change. The one-run forensic logging is disabled after capture.
- Build status: `DCR_NestedDiagnostic`, `DCR_AlternatingSweep`, and `check_alternating_controller` built successfully before diagnostic instrumentation; the instrumented diagnostic also built and completed.
- Test status: `check_alternating_controller` passed before diagnostic instrumentation.
- Numerical outcome: `mismatch_detector_state_or_scaling_bug`. No physical row failed. The actual trigger was the conservation projection at cell 164 (`x=1.318558581937e1 cm`), dominated by roundoff in physical row 0. The detector attributed the stop to an unrelated maximum-relative row at cell 183.
- Important residuals: cell-183/row-31 `F_BE=3.683339634375e7`, transient `-1.517978740746e10`, `F_steady=1.521662080376e10`, `D_BE=4.984283447266e-2`, scaled `D_BE=8.811959921026e-14`. Cell-164 projected `D_BE=-6.057282645477e6`, scaled `-1.119800571231e-6`; detector tolerance `3e-7`.
- Controller events: only initial P calibration and two initial A calibration retries; no trace row changed a timestep or Anderson decision.
- Primary run log: `output/nested_dcr_fixed_gamma_T3p120_A_activity_eps1e10_v2_optimized_from_T3p110/run.log`
- Diagnostic artifact path: `output/nested_dcr_fixed_gamma_T3p120_be_mismatch_macro79_diagnostic/`
- Diagnostic report: `output/nested_dcr_fixed_gamma_T3p120_be_mismatch_macro79_diagnostic/diagnostic_report.md`
- Activity-controller result: `output/nested_dcr_fixed_gamma_T3p120_A_activity_eps1e10_v2_optimized_from_T3p110/activity_controller_result.md`
- Accepted checkpoints changed: no.
- Production checkpoint written: no.
- Open question for ChatGPT: what single next experiment, if any, should verify a corrected conservation-projection mismatch scale without changing the solver trajectory?
