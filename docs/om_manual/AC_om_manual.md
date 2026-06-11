# O&M Manual — Rotary Air Compressor (AC)

## Applicable Machines: AC-201, AC-301

## Standards Reference
- API 619 — Rotary Positive Displacement Compressors
- ISO 10816 — Mechanical vibration evaluation of machine vibration
- ISO 281 — Rolling bearings — dynamic load ratings and rating life
- IEEE 493 — Recommended Practice for Design of Reliable Industrial Power Systems

---

## Maintenance Procedures

### Procedure AC-01: Vibration Monitoring and Analysis
- **Sensor**: vibration_rms
- **Warning Threshold**: 4.5 mm/s | **Critical Threshold**: 7.1 mm/s
- **Interval**: Continuous online monitoring; manual verification every 2 weeks
- **Standard**: ISO 10816 Zone B/C boundary at 4.5 mm/s
- **Action**: When vibration_rms exceeds 4.5 mm/s, schedule bearing inspection within 48 hours. At 7.1 mm/s, initiate controlled shutdown.
- **OM Section Reference**: Bearing inspection — Section AC-03

### Procedure AC-02: Bearing Temperature Surveillance
- **Sensor**: bearing_temp
- **Warning Threshold**: 85 °C | **Critical Threshold**: 105 °C
- **Interval**: Continuous; log every 5 minutes
- **Standard**: API 619 §4.9 — bearing metal temperature limits
- **Action**: Investigate cooling system at 85 °C. Emergency shutdown at 105 °C.
- **OM Section Reference**: Lubrication system — Section AC-04

### Procedure AC-03: Bearing Inspection and Replacement
- **Sensor**: vibration_rms, bearing_temp
- **Interval**: Every 8,000 operating hours or upon vibration alarm
- **Standard**: ISO 281 L10 life calculation; API 619 §6.3
- **Action**: Replace bearings when L10 life consumed or vibration exceeds ISO 10816 Zone C.
- **OM Section Reference**: Vibration — Section AC-01, Temperature — Section AC-02

### Procedure AC-04: Lubricating Oil Analysis and Change
- **Sensor**: oil_pressure
- **Warning Threshold**: 2.8 bar | **Critical Threshold**: 2.0 bar
- **Interval**: Oil analysis every 1,000 hours; full change every 4,000 hours
- **Standard**: API 619 §5.2 — lubrication system requirements
- **Action**: Top up oil at 2.8 bar warning. Shutdown at 2.0 bar to prevent bearing damage.
- **OM Section Reference**: Oil pressure — Section AC-04, Bearing temp — Section AC-02

### Procedure AC-05: Oil Pressure System Inspection
- **Sensor**: oil_pressure
- **Warning Threshold**: 2.8 bar | **Critical Threshold**: 2.0 bar
- **Interval**: Monthly visual inspection; quarterly pump flow test
- **Standard**: API 614 — Lubrication, Shaft-sealing, and Control-oil Systems
- **Action**: Check oil pump, filter, and relief valve. Replace filter element if pressure drop exceeds 0.5 bar.

### Procedure AC-06: Motor Current Analysis
- **Sensor**: motor_current
- **Warning Threshold**: 28.0 A | **Critical Threshold**: 34.0 A
- **Interval**: Continuous monitoring; monthly power quality analysis
- **Standard**: NEMA MG-1 — Motors and Generators
- **Action**: Investigate mechanical binding at 28 A. Trip at 34 A (motor protection).
- **OM Section Reference**: Drive motor — Section AC-07

### Procedure AC-07: Drive Motor Maintenance
- **Sensor**: motor_current, vibration_rms
- **Interval**: Annually — insulation resistance test, bearing grease, alignment check
- **Standard**: NEMA MG-1; IEEE 43 — Insulation Resistance
- **Action**: Megger test (>100 MΩ). Re-grease motor bearings per OEM schedule.
- **OM Section Reference**: Motor current — Section AC-06

### Procedure AC-08: Outlet Pressure Verification
- **Sensor**: outlet_pressure
- **Warning Threshold**: 6.5 bar | **Critical Threshold**: 5.0 bar
- **Interval**: Continuous; calibrate transducer every 6 months
- **Standard**: API 619 — performance test code
- **Action**: Check inlet valve, discharge valve, and unloader at 6.5 bar. Shutdown at 5.0 bar.
- **OM Section Reference**: Performance — Section AC-09

### Procedure AC-09: Compressor Performance Testing
- **Sensor**: outlet_pressure, vibration_rms, bearing_temp
- **Interval**: Quarterly — capacity test, specific power measurement
- **Standard**: API 619 Annex D — performance test procedures
- **Action**: Compare actual vs. nameplate capacity. Investigate if >10% degradation.

### Procedure AC-10: Rotor and Timing Gear Inspection
- **Sensor**: vibration_rms
- **Interval**: Every 16,000 operating hours or major overhaul
- **Standard**: API 619 §6.4 — clearances and timing
- **Action**: Check rotor profile, timing gear backlash, and axial clearance. Replace if wear exceeds OEM limits.
- **OM Section Reference**: Vibration — Section AC-01

### Procedure AC-11: Inlet Filter and Air Path
- **Sensor**: outlet_pressure, motor_current
- **Interval**: Monthly filter inspection; replace when differential pressure exceeds 25 mbar
- **Standard**: ISO 8573-1 — Compressed air quality
- **Action**: Replace inlet filter element. Inspect inlet valve for carbon buildup.

### Procedure AC-12: Cooling System Maintenance
- **Sensor**: bearing_temp, oil_pressure
- **Interval**: Monthly — check coolant level, fan operation, radiator fins
- **Standard**: API 619 §5.3 — cooling requirements
- **Action**: Clean cooler fins, check thermostat, verify coolant flow rate.
- **OM Section Reference**: Bearing temperature — Section AC-02

### Procedure AC-13: Coupling and Alignment Check
- **Sensor**: vibration_rms
- **Interval**: Annually or after any major maintenance
- **Standard**: API 686 — Recommended Practice for Installation
- **Action**: Laser alignment check. Replace coupling element if wear visible.
