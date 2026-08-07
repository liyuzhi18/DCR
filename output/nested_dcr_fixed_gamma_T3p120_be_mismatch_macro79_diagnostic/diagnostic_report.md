# Macro-79 BE/Steady Mismatch Diagnostic

## Classification

`mismatch_detector_state_or_scaling_bug`

There is no genuine difference between the BE and steady physical operators.
The BE system is a copy of the same steady system with only the intended
temporal diagonal and old-state RHS increments added. The stop is caused by a
conservation-projection scaling and attribution defect in the detector.

## Exact first trigger

- Block: P
- Macro/inner: 79/1
- Actual failed detector condition: conservation projection at cell 164
- Position: `x=1.318558581937e1 cm`
- Projection: nuclei-weighted sum of all physical P rows
- Dominant roundoff contribution: physical row `pi=0`, `gi=0`
- Dominant state: `H_Atom barenucl[g=0,q=1,id=1]`
- Dominant row contribution to projected `D_BE`: `-6.058969609375e6`
- Conservation replacement row at cell 164: `pi=0`, `gi=0`

No individual physical row exceeded the `3e-7` row-relative tolerance. The
status line incorrectly attributed the stop to the independent maximum-relative
physical row:

- Reported cell: 183
- Position: `x=2.719200691010e1 cm`
- Physical row: `pi=31`, `gi=31`
- State: `H2_Molecule h2_v0[g=31,q=0,id=1]`
- Row-relative `D_BE`: `8.811959921026e-14`

The status also printed independent maximum-absolute row and conservation
defects from other cells, so its cell, row-relative value, raw `D_BE`,
conservation-relative value, and conservation raw value do not describe one
common failing equation.

## Detector-Reported Physical Row

All quantities use the macro-79/inner-1 trial state, old PTC state, frozen A/M
partners at cell 183, the same physical row scale, and the physical row before
any conservation replacement:

| Quantity | Raw dimensional | Row-scaled |
|---|---:|---:|
| `F_BE` | `3.683339634375e7` | `6.511957350950e-5` |
| `M Delta_z / Delta_tau` | `-1.517978740746e10` | `-2.683709296622e-2` |
| `F_steady` | `1.521662080376e10` | `2.690221253964e-2` |
| `D_BE` | `4.984283447266e-2` | `8.811959921026e-14` |

- Physical row scale: `5.656271126895e11`
- Detector identity scale: `5.656271126895e11`
- State: `3.071421753101e6`
- Old state: `3.071733625074e6`
- `Delta_tau`: `2.054521343638e-8`
- `1/Delta_tau`: `4.867313757030e7`
- Gross temporal LHS: `1.494957335251e14`
- Gross temporal RHS: `1.495109133125e14`
- Represented and intended net transient: `-1.517978740746e10`
- Long-double recomputation: `D_BE=-2.202785760164e-2`

The small difference between double and long-double `D_BE` confirms ordinary
floating-point summation/cancellation, many orders below the physical row
scale. The represented temporal diagonal and RHS increments equal the intended
increments.

## Physical Row Terms

The complete emitted term list is in `run.log`:

- Steady matrix contributions: lines containing `P_BE_STEADY_LHS_TERM`
- Steady RHS: `P_BE_RHS_TERM`
- Two BE-only temporal terms: `P_BE_TEMPORAL_TERM`
- All 36 process/donor chemistry terms: `P_BE_SOURCE_TERM`
- Conservation projection and all 46 row contributions:
  `P_BE_CONSERVATION_DETAIL` and `P_BE_CONSERVATION_ROW_TERM`

For row 31, the nonzero steady matrix contributions are from columns 1, 30,
and 31 through 45. Their values, coefficients, states, and products are printed
individually in the log. The aggregate physical reconstruction is:

- Local molecular exhaust: `6.721725376184e9`
- Assembled chemistry: `-8.494895427572e9`
- Channel production: `5.571316818029e11`
- Channel loss: `5.656265772305e11`
- Channel net: `-8.494895427572e9`
- Channel reconstruction error: `8.583068847656e-6`
- Reconstructed `F_steady`: `1.521662080376e10`

Term audit:

- Present only in BE: the intended temporal diagonal and old-state RHS terms.
- Present only in steady: none; BE copies the complete steady system.
- Different state: none.
- Stale partner values: none; both use the same frozen A/M vectors.
- Different physical scaling: none for the row identity.
- Conservation replacement: row 31 is not replaced at cell 183. At cell 164,
  row 0 is the solve replacement row, but the identity detector deliberately
  projects the original physical rows before replacement.
- Boundary treatment: row 31 is neutral and uses only local molecular exhaust;
  no upstream cell or boundary face enters that row.

## Actual Conservation Trigger

At cell 164:

| Projected quantity | Raw dimensional | Detector-scaled |
|---|---:|---:|
| `F_BE` | `-3.094569677886e10` | `-5.720883597201e-3` |
| transient | `5.378311615677e12` | `9.942802362034e-1` |
| `F_steady` | `-5.409251255173e12` | `-1.000000000000e0` |
| `D_BE` | `-6.057282645477e6` | `-1.119800571231e-6` |

- Projection scale: `5.409251255173e12`
- Projection of individually computed row defects: `-6.057282646995e6`
- Stored projected defect: `-6.057282645477e6`
- Dominant contribution: row 0, `-6.058969609375e6`

The conservation detector scales a cancellation residual by the net projected
BE/transient/steady values. It does not include physical row scales or gross
temporal operands. Consequently, harmless row-level arithmetic error that is
below `1.4e-13` on every physical row becomes `1.12e-6` after projection and
triggers the `3e-7` gate.

## Development History

For the reported cell-183/row-31 identity, macros 75 through 78 remain between
roughly `1e-15` and `1.4e-13`. The maximum conservation-projected values during
those macros remain between roughly `3e-10` and `9.34e-9`. Macro 79 inner 1
jumps suddenly to `1.119800571231e-6`; this is not a gradual row-operator drift.

The event immediately before the first mismatch is the accepted macro-78
Anderson candidate:

- `accepted=1`
- `beta_selected=0.95`
- `positivity_limited=0`
- `merit_limited=0`
- candidate merit `4.109427316146e4`

The accepted Anderson blend mutates the profiles before macro 79. There is no
intervening timestep calibration, rollback, history restart, boundary update,
or trace-controller intervention. The state change alters floating-point
cancellation in the cell-164 conservation projection; it does not introduce a
new BE or steady operator term.

## Conclusion

The exact cause of the macro-79 stop is a detector bug: conservation-projected
roundoff is normalized by a cancellation-sensitive net scale, and the failure
is then attributed to an unrelated maximum-relative physical row. The
underlying BE and steady assemblies are consistent.
