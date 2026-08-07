# Fixed-Gamma Activity Controller Result

## Run definition

- Temperature: 3.120 eV from the accepted 3.110 eV checkpoint
- Fixed Gamma: `3.526510083586e19`
- Activity threshold: `epsilon_abs=1e-10`
- Pathway protection fraction: `1e-3`
- Plasma pseudo-time mode: global
- Physical equations, merit, and convergence gates: unchanged

## Isolation result

The controller-ON run and the fresh current-source controller-OFF/global-PTC
control produced identical solver metrics and termination:

- Termination: `be_steady_operator_mismatch` at macro 79
- P steady residual: `1.400390363299e-2`
- M map residual: `4.107537323609e-1`
- Profile change: `1.126111454330e-3`
- Final plasma pseudo-time step: `2.054521343638e-8`
- P steps: 390 accepted, 0 rejected
- Anderson trials: 11 accepted, 1160 rejected

The only controller events were the initial P calibration and two initial A
calibration retries. Their limiting rows were non-trace, so filtered and raw
changes were equal. No trace timestep limiter was rejected or rescued, and no
negative trace Anderson component was restored. Therefore `epsilon_abs=1e-10`
made no solver decision in this trajectory.

## Historical A2 comparison

A2 is not a same-source control after macro 20. At macro 21 A2 rolled back its
watchdog and returned to adaptive PTC; the current source remained in Anderson
mode. The later differences cannot be attributed to activity de-weighting.

| Metric | Current paired runs | Historical A2 |
|---|---:|---:|
| Termination macro | 79 | 101 |
| Termination status | `be_steady_operator_mismatch` | `inner_solve_stalled` |
| P steady residual | `1.400390363299e-2` | `1.026934491417e-1` |
| M map residual | `4.107537323609e-1` | `2.489931290929e-1` |
| Profile change | `1.126111454330e-3` | `1.811383590789e-2` |
| Final plasma pseudo-time step | `2.054521343638e-8` | `1.001883554736e-8` |
| P accepted/rejected steps | 390 / 0 | 500 / 15 |

At cell 138, the saved current trajectory has low-v H2+ populations 3.60 to
3.82 times the A2 populations for v=0 through v=5. Its directly postprocessed
hybrid residuals on those rows are about 117 to 271 times smaller. These are
trajectory differences, not demonstrated controller effects, because the
controller-ON and controller-OFF current-source trajectories are identical.

The read-only postprocess audit reported `state_unchanged=1`,
`checkpoint_written=0`, and `solver_iterations=0`.
