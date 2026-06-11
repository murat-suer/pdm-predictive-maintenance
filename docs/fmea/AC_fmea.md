# FMEA — Rotary Air Compressor (AC-201 / AC-301)

**Standard References:** ISO 10816, API 619, IEEE 493 Table 7-2
**Machine Type:** Rotary screw air compressor
**Sensor Count:** 5 (vibration_rms, bearing_temp, oil_pressure, motor_current, outlet_pressure)

---

## Failure Mode 1: Bearing Fatigue (Drive End)

| Field | Value |
|---|---|
| Failure Mode | Rolling element bearing surface fatigue (spalling) |
| Physical Mechanism | Cyclic Hertzian contact stress causes subsurface crack initiation and propagation. Material flakes from raceway, increasing vibration and temperature. ISO 281 L10 model governs rated life. |
| Severity | 8 |
| Occurrence | 6 |
| Detectability | 3 |
| **RPN** | **144** |
| Current Controls | vibration_rms (BPFO harmonic at 0.4x coeff), bearing_temp correlation (r=0.75), ISO 10816 Zone C/D thresholds at 4.5/7.1 mm/s |
| References | ISO 281:2007, CWRU Bearing Data Center, IEEE 493 Table 7-2 |

---

## Failure Mode 2: Oil Degradation (Oxidation)

| Field | Value |
|---|---|
| Failure Mode | Lubricant oxidation and viscosity loss |
| Physical Mechanism | Arrhenius-driven oxidation at operating temperature (80 C nominal). Oil film thickness decreases, increasing metal-to-metal contact. Rate doubles per 10 C rise above reference (100 C). Mobil SHC 624 datasheet reference. |
| Severity | 7 |
| Occurrence | 5 |
| Detectability | 4 |
| **RPN** | **140** |
| Current Controls | oil_pressure sensor (degradation_direction=-1, warning at 2.8 bar), Arrhenius model with Ea=85 kJ/mol, bearing_temp correlation |
| References | Schewe ASME J. Tribol. 2009, Mobil SHC 624 datasheet |

---

## Failure Mode 3: Rotor Timing Loss

| Field | Value |
|---|---|
| Failure Mode | Male/female rotor profile wear causing internal leakage |
| Physical Mechanism | Rotor tip clearance increases with wear, reducing volumetric efficiency. Outlet pressure drops as compressed gas recirculates internally. Motor current rises to compensate. |
| Severity | 7 |
| Occurrence | 4 |
| Detectability | 5 |
| **RPN** | **140** |
| Current Controls | outlet_pressure (warning 6.5 bar, critical 5.0 bar), motor_current correlation (r=0.70), downstream UPSTREAM_EFFECT on HX inlet_flow_proxy |
| References | API 619 Section 4, TEMA RGP-T-2.4 |

---

## Failure Mode 4: Oil Filter Blockage

| Field | Value |
|---|---|
| Failure Mode | Oil filter element clogging |
| Physical Mechanism | Particulate accumulation reduces oil flow rate. Bearing oil starvation accelerates wear. Oil pressure drops abruptly as differential pressure across filter increases. |
| Severity | 9 |
| Occurrence | 3 |
| Detectability | 4 |
| **RPN** | **108** |
| Current Controls | oil_pressure rapid-decline detection, bearing_temp secondary indicator, warning threshold at 2.8 bar |
| References | API 619 Section 5.3, ISO 10816 |

---

## Failure Mode 5: Inlet Valve Sticking

| Field | Value |
|---|---|
| Failure Mode | Inlet butterfly/valve fails to modulate |
| Physical Mechanism | Carbon deposit buildup or actuator spring fatigue causes inlet valve to stick in partial-open position. Outlet pressure oscillates, motor current shows cyclic loading pattern. |
| Severity | 6 |
| Occurrence | 4 |
| Detectability | 5 |
| **RPN** | **120** |
| Current Controls | outlet_pressure oscillation pattern, motor_current FFT analysis, vibration_rms broadband increase |
| References | API 619 Section 4.5, ISO 10816-3 |

---

## Failure Mode 6: Coupling Misalignment

| Field | Value |
|---|---|
| Failure Mode | Motor-compressor coupling angular/parallel misalignment |
| Physical Mechanism | Thermal growth or foundation settlement causes shaft misalignment. Generates 1x and 2x rotational frequency vibration, increases bearing load asymmetrically. |
| Severity | 7 |
| Occurrence | 3 |
| Detectability | 3 |
| **RPN** | **63** |
| Current Controls | vibration_rms FFT (1x/2x amplitude ratio), bearing_temp differential, BSF_coeff=0.2 in FFT config |
| References | ISO 10816-3 Table 2, API 619 |
