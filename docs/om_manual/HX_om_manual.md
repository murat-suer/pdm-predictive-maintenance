# O&M Manual — Shell & Tube Heat Exchanger (HX)

## Applicable Machines: HX-202, HX-302

## Standards Reference
- TEMA RGP-T-2.4 — Standards of the Tubular Exchanger Manufacturers Association
- TEMA 9th Edition — Mechanical design and operating guidelines
- ASME Section VIII — Pressure Vessels
- API 660 — Shell-and-Tube Heat Exchangers

---

## Maintenance Procedures

### Procedure HX-01: Fouling Monitoring via Pressure Drop
- **Sensor**: pressure_drop
- **Warning Threshold**: 1.4 bar | **Critical Threshold**: 1.8 bar
- **Interval**: Continuous online monitoring; trending weekly
- **Standard**: TEMA RGP-T-2.4 — fouling resistance limits
- **Action**: Schedule chemical cleaning when pressure_drop exceeds 1.4 bar. Emergency cleaning at 1.8 bar.
- **OM Section Reference**: Fouling index — Section HX-05

### Procedure HX-02: Outlet Temperature Surveillance
- **Sensor**: outlet_temp
- **Warning Threshold**: 95 °C
- **Interval**: Continuous; log every 5 minutes
- **Standard**: TEMA — thermal performance criteria
- **Action**: Investigate fouling or flow maldistribution at 95 °C. Check inlet conditions.
- **OM Section Reference**: Inlet temperature — Section HX-03

### Procedure HX-03: Inlet Temperature Monitoring
- **Sensor**: inlet_temp
- **Interval**: Continuous monitoring
- **Standard**: TEMA — operating temperature envelope
- **Action**: Verify upstream process (AC-201/AC-301) is delivering within spec. High inlet_temp indicates upstream flow reduction.
- **OM Section Reference**: Outlet temperature — Section HX-02

### Procedure HX-04: Flow Rate Verification
- **Sensor**: flow_rate
- **Warning Threshold**: 9.0 m³/h | **Critical Threshold**: 7.0 m³/h
- **Interval**: Continuous; calibrate flowmeter every 12 months
- **Standard**: TEMA — minimum flow velocity for erosion prevention
- **Action**: Check for blockages, valve position, and pump performance at 9.0 m³/h. Shutdown at 7.0 m³/h.
- **OM Section Reference**: Pressure drop — Section HX-01

### Procedure HX-05: Fouling Index Calculation
- **Sensor**: fouling_index
- **Warning Threshold**: 0.35 | **Critical Threshold**: 0.55
- **Interval**: Calculated continuously from temperature and flow data
- **Standard**: TEMA RGP-T-2.4 — typical fouling resistance values
- **Action**: Plan cleaning when fouling_index reaches 0.35. Urgent cleaning at 0.55.
- **OM Section Reference**: Pressure drop — Section HX-01

### Procedure HX-06: Tube Bundle Inspection
- **Sensor**: pressure_drop, fouling_index
- **Interval**: Every 12 months or when fouling_index > 0.35
- **Standard**: TEMA 9th Ed. — inspection and testing
- **Action**: Remove channel head. Inspect tubes for fouling, corrosion, and erosion. Eddy current test for wall thickness.
- **OM Section Reference**: Fouling — Section HX-05

### Procedure HX-07: Chemical Cleaning (CIP)
- **Sensor**: pressure_drop, fouling_index
- **Interval**: As needed based on fouling_index trend (typically every 6-12 months)
- **Standard**: TEMA RGP-T-2.4 — cleaning methods
- **Action**: Circulate cleaning solution (acid or alkaline per deposit type). Monitor pH and temperature. Rinse thoroughly.
- **OM Section Reference**: Fouling index — Section HX-05, Pressure drop — Section HX-01

### Procedure HX-08: Tube Plug and Repair
- **Sensor**: pressure_drop
- **Interval**: During scheduled shutdowns or when leak detected
- **Standard**: TEMA — maximum tube plug percentage (10% of bundle)
- **Action**: Isolate leaking tube. Install tapered plug. Document plug location and count.
- **OM Section Reference**: Tube bundle — Section HX-06

### Procedure HX-09: Gasket and Joint Inspection
- **Sensor**: pressure_drop
- **Interval**: Every opening of the exchanger (during cleaning or inspection)
- **Standard**: ASME Section VIII Div. 1 — pressure boundary integrity
- **Action**: Replace all gaskets upon reassembly. Torque bolts per TEMA tables.
- **OM Section Reference**: Pressure boundary — Section HX-10

### Procedure HX-10: Pressure Test
- **Sensor**: pressure_drop
- **Interval**: After any maintenance that opens the pressure boundary
- **Standard**: ASME Section VIII — hydrostatic test at 1.3× MAWP
- **Action**: Hydrotest shell side and tube side independently. Hold for 30 minutes. Inspect for leaks.

### Procedure HX-11: Baffle and Support Inspection
- **Sensor**: pressure_drop, flow_rate
- **Interval**: During major overhaul (every 3-5 years)
- **Standard**: TEMA — baffle spacing and support requirements
- **Action**: Check baffle cut, spacing, and tie-rod integrity. Replace damaged baffles.
- **OM Section Reference**: Flow rate — Section HX-04

### Procedure HX-12: Nozzle and Piping Connection Check
- **Sensor**: pressure_drop, inlet_temp
- **Interval**: Annually — visual inspection; UT thickness every 3 years
- **Standard**: API 660 — nozzle loading criteria
- **Action**: Check for erosion/corrosion at nozzle entrances. Verify pipe support and alignment.
