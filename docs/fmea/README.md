# FMEA Documentation — Two Artifacts, Two Purposes

This project carries **two FMEA-related artifacts** that intentionally use
different numbers. Read this before comparing them.

## 1. Design FMEA (this directory)

`AC_fmea.md`, `HX_fmea.md`, `CM_fmea.md` are machine-type design FMEAs in the
classic AIAG style: each failure mode carries Severity, Occurrence and
Detection ratings (1–10) and `RPN = S × O × D`. They document the physical
mechanism, current controls and standard references per machine type.

## 2. Runtime Signature Library (`src/decision/failure_modes.yaml`)

The decision service uses a **generic, component-level** signature library:
sensor-pattern rules mapped to failure modes, each with an `rpn` value. That
`rpn` is an **operational risk priority**, not a design-FMEA product:

- It feeds the diagnostic ranking (`confidence × rpn`) and the recommendation
  engine's option-count thresholds (>200 / 100–200 / <100).
- It escalates monotonically with progression stage
  (`bearing_stage_1..4`: 120 → 480) so that a later stage always outranks an
  earlier one. A textbook design FMEA behaves differently: late-stage damage
  is *easy* to detect, so its Detection rating — and often its RPN — drops.

## Mapping between the two

| Runtime modes (yaml) | Design FMEA mode(s) |
|---|---|
| `bearing_stage_1..4` | AC FM1 Bearing Fatigue · CM FM3 Bearing Failure |
| `oil_degradation`, `oil_contamination` | AC FM2 Oil Degradation · AC FM4 Oil Filter Blockage |
| `motor_overheating`, `motor_imbalance` | CM FM2 Motor Overload |
| `pump_cavitation`, `pump_seal_leak` | HX FM4 Flow Maldistribution · HX FM5 Gasket/Joint Leakage (flow-loop analogues) |
| `shaft_misalignment`, `coupling_wear` | AC FM6 Coupling Misalignment · CM FM4 Gearbox Wear |
| — (covered by the ML fault classifier, `src/ml/fault_classifier.py`: FOULING, BELT_SLIP signatures) | HX FM1/FM6 Fouling · CM FM1 Belt Slip · CM FM6 Cross-Line Load Spike |

One design-level mode can fan out to several staged runtime modes and vice
versa, which is why the numbers do not (and should not) match one-to-one.
