# FMEA — Conveyor Belt Drive System (CM-203 / CM-303)

**Standard References:** ISO 5048, CEMA 7th Ed., ISO 10816
**Machine Type:** Belt conveyor drive system with pulley ratio 1.65
**Sensor Count:** 5 (belt_tension, drive_temp, motor_load, speed_rpm, vibration_rms)

---

## Failure Mode 1: Belt Slip / Tension Loss

| Field | Value |
|---|---|
| Failure Mode | Drive belt slip due to tension decay |
| Physical Mechanism | Belt elongation and pulley lagging wear reduce friction grip. Tension follows exponential decay: T(t) = T0 * exp(-lambda * t). Slip ratio increases, causing speed_rpm drop and drive_temp rise from friction heat. CEMA 7th Ed. Section 6.4 reference. |
| Severity | 7 |
| Occurrence | 7 |
| Detectability | 3 |
| **RPN** | **147** |
| Current Controls | belt_tension (degradation_weight=0.30, BELT_SLIP model, warning 11.5 kN, critical 13.8 kN), speed_rpm (degradation_direction=-1), drive_temp square_law thermal model |
| References | CEMA 7th Ed. Belt Conveyors for Bulk Materials Section 6.4, ISO 5048 |

---

## Failure Mode 2: Motor Overload

| Field | Value |
|---|---|
| Failure Mode | Drive motor thermal overload from excessive mechanical load |
| Physical Mechanism | Belt friction, material buildup, or idler seizure increases torque demand. Motor_load rises above nominal (68%). Thermal protection trips at critical threshold (96%). Square-law thermal model: heat proportional to load^2. |
| Severity | 8 |
| Occurrence | 5 |
| Detectability | 3 |
| **RPN** | **120** |
| Current Controls | motor_load (degradation_weight=0.20, warning 88%, critical 96%), drive_temp (k_nominal=0.0015 square-law), cross_line_load_spike detection for CM-303 |
| References | CEMA 7th Ed. Section 4.2, IEC 60034-1 |

---

## Failure Mode 3: Bearing Failure (Drive Pulley)

| Field | Value |
|---|---|
| Failure Mode | Drive pulley bearing fatigue and seizure |
| Physical Mechanism | Radial load from belt tension (8.2 kN nominal) causes rolling contact fatigue. ISO 281 L10 life model applies. Vibration_rms increases with BPFO harmonics. Bearing temp rises as lubricant film breaks down. |
| Severity | 9 |
| Occurrence | 4 |
| Detectability | 3 |
| **RPN** | **108** |
| Current Controls | vibration_rms (degradation_weight=0.10, warning 3.5 mm/s, critical 5.0 mm/s), drive_temp correlation, pulley_ratio=1.65 speed/torque relationship |
| References | ISO 281:2007, CEMA 7th Ed. Section 3.5, ISO 10816 |

---

## Failure Mode 4: Gearbox Wear

| Field | Value |
|---|---|
| Failure Mode | Speed reducer gear tooth wear and pitting |
| Physical Mechanism | Gear mesh fatigue at pulley_ratio=1.65 reduction. Surface pitting increases backlash, causing speed_rpm fluctuation and vibration harmonics. Motor_load oscillates as load transfer becomes intermittent. |
| Severity | 8 |
| Occurrence | 3 |
| Detectability | 4 |
| **RPN** | **96** |
| Current Controls | speed_rpm fluctuation analysis (degradation_direction=-1, warning 1380 RPM), vibration_rms gear-mesh frequency, motor_load oscillation pattern |
| References | AGMA 2001-D04, ISO 6336, CEMA 7th Ed. Section 3.4 |

---

## Failure Mode 5: Belt Cover / Lagging Degradation

| Field | Value |
|---|---|
| Failure Mode | Pulley lagging wear and belt cover degradation |
| Physical Mechanism | Rubber lagging on drive pulley wears smooth, reducing friction coefficient. Belt cover cracks from thermal cycling (drive_temp). Slip events become more frequent, accelerating tension loss. |
| Severity | 6 |
| Occurrence | 6 |
| Detectability | 5 |
| **RPN** | **180** |
| Current Controls | belt_tension decay rate monitoring, drive_temp thermal model deviation, speed_rpm slip-ratio calculation |
| References | CEMA 7th Ed. Section 6.3, DIN 22101 |

---

## Failure Mode 6: Cross-Line Load Spike (CM-303 Specific)

| Field | Value |
|---|---|
| Failure Mode | CM-303 overload when Line A fails |
| Physical Mechanism | When AC-201/HX-202/CM-203 enter FAILED state, Line B absorbs redistributed material flow. CM-303 belt_tension *= 1.4, degradation_rate *= 1.8, motor_load += 15%. CASCADE_RULES trigger: LINE_A -> CM-303 LOAD_SPIKE. |
| Severity | 9 |
| Occurrence | 3 |
| Detectability | 2 |
| **RPN** | **54** |
| Current Controls | cross_line_load_spike config (belt_tension_mult=1.4, deg_rate_mult=1.8, motor_load_delta=+15%), CASCADE_RULES LINE_A->CM-303, L1 notification level |
| References | CEMA 7th Ed. Section 2.1, system-level cascade analysis |

---

## Failure Mode 7: Idler Seizure

| Field | Value |
|---|---|
| Failure Mode | Carry or return idler bearing seizure |
| Physical Mechanism | Idler bearing grease degradation or contaminant ingress causes seizure. Seized idler creates belt drag point, increasing motor_load locally. Belt slides over stationary idler, generating heat and wear debris. |
| Severity | 7 |
| Occurrence | 5 |
| Detectability | 4 |
| **RPN** | **140** |
| Current Controls | motor_load step-increase detection, vibration_rms broadband spike, drive_temp local hot-spot from friction |
| References | CEMA 7th Ed. Section 5.3, ISO 10816-3 |
