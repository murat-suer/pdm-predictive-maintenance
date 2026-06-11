# O&M Manual — Conveyor Belt Drive System (CM)

## Applicable Machines: CM-203, CM-303

## Standards Reference
- CEMA 7th Edition — Belt Conveyors for Bulk Materials
- ISO 5048 — Continuous mechanical handling equipment
- ISO 10816 — Mechanical vibration evaluation
- CEMA 550 — Belt Pulley Lagging
- ISO 281 — Rolling bearings life calculation

---

## Maintenance Procedures

### Procedure CM-01: Belt Tension Measurement and Adjustment
- **Sensor**: belt_tension
- **Warning Threshold**: 11.5 kN | **Critical Threshold**: 13.8 kN
- **Interval**: Weekly manual check; continuous via sensor
- **Standard**: CEMA 7th Ed. §6.4 — belt tension calculation
- **Action**: Re-tension belt at 11.5 kN warning. Inspect splice and idlers at 13.8 kN critical.
- **OM Section Reference**: Drive temperature — Section CM-03

### Procedure CM-02: Belt Condition Inspection
- **Sensor**: belt_tension, vibration_rms
- **Interval**: Monthly visual inspection; continuous vibration monitoring
- **Standard**: CEMA 575 — Belt Splicing and Maintenance
- **Action**: Check for edge damage, splice integrity, and cover wear. Schedule splice repair if damage >25% width.
- **OM Section Reference**: Belt tension — Section CM-01

### Procedure CM-03: Drive Temperature Monitoring
- **Sensor**: drive_temp
- **Warning Threshold**: 78 °C | **Critical Threshold**: 92 °C
- **Interval**: Continuous; log every 5 minutes
- **Standard**: CEMA 7th Ed. — drive component temperature limits
- **Action**: Check lubrication and alignment at 78 °C. Shutdown at 92 °C to prevent gearbox damage.
- **OM Section Reference**: Motor load — Section CM-04

### Procedure CM-04: Motor Load Analysis
- **Sensor**: motor_load
- **Warning Threshold**: 88% | **Critical Threshold**: 96%
- **Interval**: Continuous monitoring; monthly power analysis
- **Standard**: NEMA MG-1 — motor overload protection
- **Action**: Investigate mechanical binding, belt misalignment, or excessive material load at 88%. Trip at 96%.
- **OM Section Reference**: Drive temperature — Section CM-03

### Procedure CM-05: Speed RPM Verification
- **Sensor**: speed_rpm
- **Warning Threshold**: 1380 RPM
- **Interval**: Continuous; calibrate tachometer every 6 months
- **Standard**: CEMA 7th Ed. — belt speed design criteria
- **Action**: Check VFD settings, belt slip, and motor performance at 1380 RPM.
- **OM Section Reference**: Belt tension — Section CM-01

### Procedure CM-06: Vibration Monitoring (Drive Assembly)
- **Sensor**: vibration_rms
- **Warning Threshold**: 3.5 mm/s | **Critical Threshold**: 5.0 mm/s
- **Interval**: Continuous online; monthly route-based verification
- **Standard**: ISO 10816 — Zone B/C boundary
- **Action**: Inspect bearings, gearbox, and coupling at 3.5 mm/s. Shutdown at 5.0 mm/s.
- **OM Section Reference**: Drive temperature — Section CM-03

### Procedure CM-07: Gearbox Maintenance
- **Sensor**: drive_temp, vibration_rms
- **Interval**: Oil analysis every 2,000 hours; oil change every 8,000 hours
- **Standard**: AGMA 6177 — Gear Lubrication
- **Action**: Sample oil for particle count and viscosity. Replace oil and filter at scheduled interval.
- **OM Section Reference**: Drive temperature — Section CM-03, Vibration — Section CM-06

### Procedure CM-08: Bearing Inspection and Replacement
- **Sensor**: vibration_rms, drive_temp
- **Interval**: Grease every 2,000 hours; replace per ISO 281 L10 life
- **Standard**: ISO 281 — bearing life; CEMA 7th Ed. §8.2
- **Action**: Re-grease bearings per OEM schedule. Replace when vibration exceeds ISO 10816 Zone C.
- **OM Section Reference**: Vibration — Section CM-06

### Procedure CM-09: Pulley Lagging and Crown Inspection
- **Sensor**: belt_tension, speed_rpm
- **Interval**: Every 6 months or when slip detected
- **Standard**: CEMA 550 — pulley lagging criteria
- **Action**: Inspect lagging wear. Re-lag when thickness <50% original. Check pulley crown profile.
- **OM Section Reference**: Belt tension — Section CM-01, Speed — Section CM-05

### Procedure CM-10: Idler and Roller Inspection
- **Sensor**: motor_load, vibration_rms
- **Interval**: Monthly walk-down; infrared scan quarterly
- **Standard**: CEMA 7th Ed. §7.3 — idler selection and spacing
- **Action**: Replace seized idlers. Check idler alignment. Record failed idler positions.
- **OM Section Reference**: Motor load — Section CM-04

### Procedure CM-11: Belt Alignment Check
- **Sensor**: belt_tension, vibration_rms
- **Interval**: Monthly or after belt splice
- **Standard**: CEMA 7th Ed. §9.1 — belt training
- **Action**: Adjust training idlers. Check structure squareness. Verify loading chute centering.
- **OM Section Reference**: Belt tension — Section CM-01

### Procedure CM-12: Take-up System Maintenance
- **Sensor**: belt_tension
- **Interval**: Monthly — check take-up travel, counterweight, and auto-tensioner
- **Standard**: CEMA 7th Ed. §6.5 — take-up requirements
- **Action**: Verify take-up has full travel range. Lubricate screw take-up. Check hydraulic tensioner pressure.
- **OM Section Reference**: Belt tension — Section CM-01

### Procedure CM-13: Drive Motor Maintenance
- **Sensor**: motor_load, vibration_rms
- **Interval**: Annually — insulation test, bearing service, alignment
- **Standard**: NEMA MG-1; IEEE 43
- **Action**: Megger test motor windings. Re-grease motor bearings. Laser-align motor-to-gearbox.
- **OM Section Reference**: Motor load — Section CM-04, Vibration — Section CM-06
