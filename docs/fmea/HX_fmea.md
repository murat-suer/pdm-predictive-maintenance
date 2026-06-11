# FMEA — Shell & Tube Heat Exchanger (HX-202 / HX-302)

**Standard References:** TEMA RGP-T-2.4, ISO 10816
**Machine Type:** Shell and tube heat exchanger (cooling water service)
**Sensor Count:** 5 (inlet_temp, outlet_temp, pressure_drop, flow_rate, fouling_index)

---

## Failure Mode 1: Tube Fouling (Scaling)

| Field | Value |
|---|---|
| Failure Mode | Mineral scale deposition on tube surfaces |
| Physical Mechanism | Dissolved calcium/magnesium salts precipitate on heated tube walls. Fouling resistance Rf(t) follows TEMA asymptotic model: Rf(t) = Rf_max * (1 - exp(-k_foul * t)). Thermal resistance increases, outlet temperature rises. |
| Severity | 7 |
| Occurrence | 8 |
| Detectability | 3 |
| **RPN** | **168** |
| Current Controls | pressure_drop (degradation_weight=0.35, exponential fouling_growth_model), fouling_index (warning 0.35, critical 0.55), TEMA RGP-T-2.4 Rf_max=0.00035 m2K/W |
| References | TEMA RGP-T-2.4, Muller-Steinhagen Heat Exchanger Fouling |

---

## Failure Mode 2: Tube Corrosion

| Field | Value |
|---|---|
| Failure Mode | Through-wall corrosion of tube bundle |
| Physical Mechanism | Chloride-induced pitting corrosion or erosion-corrosion at tube inlet. Wall thickness decreases until leak-through occurs. Shell-side fluid contaminates tube-side. |
| Severity | 9 |
| Occurrence | 4 |
| Detectability | 5 |
| **RPN** | **180** |
| Current Controls | pressure_drop sudden-change detection, outlet_temp deviation from inlet_temp correlation (r=0.65), flow_rate decrease |
| References | TEMA RGP-T-2.4 Section 5, NACE MR0175 |

---

## Failure Mode 3: Baffle Degradation

| Field | Value |
|---|---|
| Failure Mode | Baffle plate erosion or collapse |
| Physical Mechanism | High-velocity cross-flow at baffle window causes tube vibration and baffle edge erosion. Flow bypass increases, reducing heat transfer effectiveness. Outlet temperature rises without proportional pressure_drop increase. |
| Severity | 7 |
| Occurrence | 3 |
| Detectability | 6 |
| **RPN** | **126** |
| Current Controls | outlet_temp / inlet_temp differential analysis, flow_rate efficiency calculation, fouling_index decoupling |
| References | TEMA RGP-T-2.4 Table 3, HTRI Xchanger Suite |

---

## Failure Mode 4: Flow Maldistribution

| Field | Value |
|---|---|
| Failure Mode | Uneven flow distribution across tube bundle |
| Physical Mechanism | Partial blockage of inlet header or nozzle causes channeling. Some tubes see high velocity (erosion risk), others see stagnation (fouling risk). Flow_rate decreases while pressure_drop may not increase proportionally. |
| Severity | 6 |
| Occurrence | 5 |
| Detectability | 5 |
| **RPN** | **150** |
| Current Controls | flow_rate (degradation_direction=-1, warning 9.0 m3/h), UPSTREAM_EFFECT from AC outlet_pressure, pressure_drop cross-validation |
| References | TEMA RGP-T-2.4, Bell KJ flow distribution guidelines |

---

## Failure Mode 5: Gasket / Joint Leakage

| Field | Value |
|---|---|
| Failure Mode | Channel cover or floating head gasket failure |
| Physical Mechanism | Thermal cycling causes gasket creep and bolt relaxation. Shell-side and tube-side fluids mix. Outlet temperature shows step change, fouling_index may drop as fluids dilute. |
| Severity | 8 |
| Occurrence | 3 |
| Detectability | 4 |
| **RPN** | **96** |
| Current Controls | outlet_temp step detection, fouling_index anomaly (unexpected decrease), inlet_temp mass-balance check |
| References | TEMA RGP-T-2.4 Section 7, EJMA Standards |

---

## Failure Mode 6: Biological Fouling

| Field | Value |
|---|---|
| Failure Mode | Microbiologically influenced corrosion and biofilm growth |
| Physical Mechanism | Cooling water biofilm ( Legionella, sulfate-reducing bacteria) forms insulating layer on tubes. Fouling follows logistic growth pattern. pressure_drop increases as flow area decreases. |
| Severity | 6 |
| Occurrence | 5 |
| Detectability | 4 |
| **RPN** | **120** |
| Current Controls | fouling_index growth rate analysis, pressure_drop exponential model deviation, flow_rate gradual decline |
| References | TEMA RGP-T-2.4, NACE RP01-99 |
