"""
src/data_generator/machines.py
================================
Machine topology, sensor configurations, Weibull parameters,
cascade rules, and shift load profiles for all 6 industrial machines.

CRITICAL RULES:
- All sensor nominal values defined as mu + sigma pairs (never single numbers)
- degradation_weight per machine must sum to 1.0 (enforced at import)
- Weibull beta/eta drawn once per machine start from Gauss(beta, beta_std)
- random.uniform is FORBIDDEN inside the data_generator package (use seeded
  np.random.Generator for reproducible physics simulation). Operational
  jitter outside this package (e.g. scheduling delays in database/models.py)
  may use stdlib random where reproducibility is not required.

NOTE ON ETA VALUES:
  Original IEEE 493 / TEMA / CEMA reference values are 720-1020 hours
  (≈30-40 days). For the live demo these are scaled to 120-180 hours
  (≈5-7.5 days) so that visitors can observe a full degradation cycle
  within a reasonable session. The beta (shape) parameters are kept
  identical to the literature sources — only the scale (eta) is
  adjusted by a DEMO_TIME_SCALE factor of 0.2.

  UNIT CONVENTION: every `eta` value below is in HOURS, not seconds.
  The 5-7.5 day range corresponds to the 95% point of the Weibull CDF
  (the FAILED-phase transition, where `overall_degradation ≥ 0.95`).
  `create_machine_state` reads this field via `_lookup_weibull_prior`
  and converts to Weibull scale via:
      eta_seconds = t95_hours * 3600 / (-ln 0.05)^(1/beta)
  The resulting *median* is ~1.9-3.9 days (first signs of trouble
  appear early, escalating to FAILED by 5-7.5 days).
"""

MACHINE_CONFIGS = {
    # ─── LINE A ──────────────────────────────────────────────────────────
    "AC-201": {
        "name": "Rotary Air Compressor — Line A",
        "line": "A",
        "type": "AC",
        "has_bearings": True,
        "pulley_ratio": None,
        "standard": "ISO 10816, API 619",
        "failure_mode": "Bearing fatigue (BPFO harmonic excitation, race spalling)",
        "weibull": {
            # Source: IEEE 493 Table 7-2, rotary compressor category
            # Base eta = 720h, scaled 0.2x for demo visibility
            "beta": 2.1,
            "beta_std": 0.15,
            "eta": 144.0,       # 144 hours = 6 days (t95, FAILED-phase point)
            "eta_std": 18.0,    # ±~0.75 day variation
            "source": "IEEE 493 Table 7-2, rotary compressor category (demo-scaled 0.2x)",
        },
        "cwru_calibration": {
            "dataset": "Case Western Reserve University Bearing Data Center",
            "config": "Drive End, 1750 RPM, 12k samples/sec",
            "classes": ["normal", "0.007in", "0.014in", "0.021in"],
            "usage": "vibration_rms baseline mean/std + BPFO harmonic amplitudes",
            "url": "https://engineering.case.edu/bearingdatacenter",
        },
        "sensors": {
            # AC-201 target fault: BEARING_FAULT
            # Dominant degrading sensors are vibration_rms + bearing_temp, matching
            # the BEARING_FAULT classifier rule (weights 0.35/0.30 → now 0.45/0.35).
            # oil_pressure and outlet_pressure carry minimal weight so they do not
            # reach the VALVE_LEAK or OIL_DEGRADATION pattern thresholds first.
            "vibration_rms": {
                "unit": "mm/s",
                "nominal_mu": 2.5,
                "nominal_sigma": 0.15,
                "warning_threshold": 4.5,
                "critical_threshold": 7.1,
                "degradation_weight": 0.45,
                "degradation_direction": 1,
                "fft": {
                    "BPFO_coeff": 0.4,
                    "BPFI_coeff": 0.6,
                    "BSF_coeff": 0.2,
                    "N_balls": 9,
                    "rpm_nominal": 1750,
                },
            },
            "bearing_temp": {
                "unit": "°C",
                "nominal_mu": 62.0,
                "nominal_sigma": 2.0,
                "warning_threshold": 85.0,
                "critical_threshold": 105.0,
                "degradation_weight": 0.35,
                "degradation_direction": 1,
                "correlation_with": {"vibration_rms": {"mu": 0.75, "sigma": 0.05}},
            },
            "oil_pressure": {
                "unit": "bar",
                "nominal_mu": 4.5,
                "nominal_sigma": 0.15,
                "warning_threshold": 2.8,
                "critical_threshold": 2.0,
                "degradation_weight": 0.10,
                "degradation_direction": -1,
            },
            "motor_current": {
                "unit": "A",
                "nominal_mu": 21.0,
                "nominal_sigma": 0.6,
                "warning_threshold": 28.0,
                "critical_threshold": 34.0,
                "degradation_weight": 0.07,
                "degradation_direction": 1,
                "correlation_with": {"vibration_rms": {"mu": 0.70, "sigma": 0.05}},
            },
            "outlet_pressure": {
                "unit": "bar",
                "nominal_mu": 8.2,
                "nominal_sigma": 0.15,
                "warning_threshold": 6.5,
                "critical_threshold": 5.0,
                "degradation_weight": 0.03,
                "degradation_direction": -1,
                "downstream_effect": {
                    "target_machine": "HX-202",
                    "target_sensor": "inlet_flow_proxy",
                    "effect_type": "UPSTREAM_EFFECT",
                },
            },
        },
        "startup_penalty": {
            "rul_penalty_mu": -0.10,
            "rul_penalty_sigma": 0.02,
            "oil_pressure_low_s": 30,
            "warm_restart_s": 45,
            "cold_restart_s": 210,
        },
    },
    "HX-202": {
        "name": "Shell & Tube Heat Exchanger — Line A",
        "line": "A",
        "type": "HX",
        "has_bearings": False,
        "pulley_ratio": None,
        "standard": "TEMA RGP-T-2.4",
        "failure_mode": "Fouling — thermal scaling and tube-side deposit growth (TEMA RGP-T-2.4)",
        "weibull": {
            # Source: TEMA RGP-T-2.4, shell&tube fouling statistics
            # Base eta = 960h, scaled 0.2x for demo visibility
            "beta": 2.5,
            "beta_std": 0.20,
            "eta": 168.0,       # 168 hours = 7 days (t95, FAILED-phase point; higher beta = more reliable)
            "eta_std": 24.0,    # ±1 day variation
            "source": "TEMA RGP-T-2.4, shell&tube fouling statistics (demo-scaled 0.2x)",
        },
        "cwru_calibration": None,
        "sensors": {
            # HX-202 target fault: FOULING
            # Dominant sensors: fouling_index (unique to FOULING rule) + outlet_temp +
            # pressure_drop. Elevated fouling_index + outlet_temp weight drives the
            # FOULING signature. flow_rate is kept low so it does not dominate the
            # FLOW_RESTRICTION pattern (which keys on flow_rate + pressure_drop).
            "inlet_temp": {
                "unit": "°C",
                "nominal_mu": 145.0,
                "nominal_sigma": 3.0,
                "nominal_range": [130, 160],
                "warning_threshold": None,
                "critical_threshold": None,
                "degradation_weight": 0.05,
                "degradation_direction": 1,
                "upstream_sensitivity": {
                    "source_machine": "AC-201",
                    "source_sensor": "outlet_pressure",
                    "effect": "inlet_temp_rise_on_low_flow",
                    "flag": "UPSTREAM_EFFECT",
                },
            },
            "outlet_temp": {
                "unit": "°C",
                "nominal_mu": 78.0,
                "nominal_sigma": 2.0,
                "warning_threshold": 95.0,
                "critical_threshold": None,
                "degradation_weight": 0.30,
                "degradation_direction": 1,
                "correlation_with": {"inlet_temp": {"mu": 0.65, "sigma": 0.05}},
            },
            "pressure_drop": {
                "unit": "bar",
                "nominal_mu": 0.85,
                "nominal_sigma": 0.04,
                "warning_threshold": 1.4,
                "critical_threshold": 1.8,
                "degradation_weight": 0.30,
                "degradation_direction": 1,
                "fouling_growth_model": "exponential",
            },
            "flow_rate": {
                "unit": "m³/h",
                "nominal_mu": 12.5,
                "nominal_sigma": 0.4,
                "warning_threshold": 9.0,
                "critical_threshold": 7.0,
                "degradation_weight": 0.05,
                "degradation_direction": -1,
            },
            "fouling_index": {
                "unit": "-",
                "nominal_mu": 0.08,
                "nominal_sigma": 0.01,
                "warning_threshold": 0.35,
                "critical_threshold": 0.55,
                "degradation_weight": 0.30,
                "degradation_direction": 1,
            },
        },
        "startup_penalty": None,
    },
    "CM-203": {
        "name": "Conveyor Belt Drive System — Line A",
        "line": "A",
        "type": "CM",
        "has_bearings": True,
        "pulley_ratio": 1.65,
        "standard": "ISO 5048, CEMA 7th Ed.",
        "failure_mode": "Bearing fatigue — drive-end vibration and thermal rise (ISO 281 L10)",
        "weibull": {
            # Source: CEMA 7th Ed., belt conveyor drive reliability
            # Base eta = 540h, scaled 0.2x for demo visibility
            "beta": 2.2,
            "beta_std": 0.18,
            "eta": 132.0,       # 132 hours = 5.5 days (t95, FAILED-phase point; wear-out dominant, beta > 2)
            "eta_std": 16.0,
            "source": "CEMA 7th Ed., belt conveyor drive reliability (demo-scaled 0.2x)",
        },
        "cwru_calibration": None,
        "sensors": {
            # CM-203 target fault: BEARING_FAULT
            # Dominant sensors: vibration_rms + drive_temp, matching the CM BEARING_FAULT
            # rule. belt_tension is reduced so it does not dominate the BELT_SLIP pattern.
            "belt_tension": {
                "unit": "kN",
                "nominal_mu": 8.2,
                "nominal_sigma": 0.25,
                "warning_threshold": 11.5,
                "critical_threshold": 13.8,
                "degradation_weight": 0.10,
                "degradation_direction": 1,
            },
            "drive_temp": {
                "unit": "°C",
                "nominal_mu": 55.0,
                "nominal_sigma": 2.5,
                "warning_threshold": 78.0,
                "critical_threshold": 92.0,
                "degradation_weight": 0.35,
                "degradation_direction": 1,
                "thermal_model": "square_law",
                "k_nominal": 0.0015,
                "k_sigma_pct": 0.08,
            },
            "motor_load": {
                "unit": "%",
                "nominal_mu": 68.0,
                "nominal_sigma": 3.0,
                "warning_threshold": 88.0,
                "critical_threshold": 96.0,
                "degradation_weight": 0.10,
                "degradation_direction": 1,
            },
            "speed_rpm": {
                "unit": "RPM",
                "nominal_mu": 1450.0,
                "nominal_sigma": 7.5,
                "warning_threshold": 1380.0,
                "critical_threshold": None,
                "degradation_weight": 0.10,
                "degradation_direction": -1,
            },
            "vibration_rms": {
                "unit": "mm/s",
                "nominal_mu": 1.8,
                "nominal_sigma": 0.15,
                "warning_threshold": 3.5,
                "critical_threshold": 5.0,
                "degradation_weight": 0.35,
                "degradation_direction": 1,
            },
        },
        "startup_penalty": None,
    },
    # ─── LINE B ──────────────────────────────────────────────────────────
    "AC-301": {
        "name": "Rotary Air Compressor — Line B",
        "line": "B",
        "type": "AC",
        "has_bearings": True,
        "pulley_ratio": None,
        "standard": "ISO 10816, API 619",
        "failure_mode": "Valve seat leak — internal recirculation and outlet pressure loss (API 619)",
        "weibull": {
            # Source: IEEE 493, newer equipment adjustment +10%
            # Base eta = 680h, scaled 0.2x for demo visibility
            "beta": 2.3,
            "beta_std": 0.16,
            "eta": 156.0,       # 156 hours = 6.5 days (t95, FAILED-phase point)
            "eta_std": 20.0,
            "source": "IEEE 493, newer equipment adjustment (demo-scaled 0.2x)",
        },
        "cwru_calibration": {
            "dataset": "CWRU — same dataset, new equipment deviation factor",
            "config": "Drive End, 1750 RPM, +5% amplitude offset (new equipment)",
            "usage": "Same calibration base as AC-201, slightly different beta/eta",
        },
        "sensors": {
            # AC-301 target fault: VALVE_LEAK
            # Dominant sensors: outlet_pressure (drops) + motor_current (rises as
            # compressor compensates for recirculation), matching the VALVE_LEAK rule.
            # vibration_rms and bearing_temp are kept low to avoid BEARING_FAULT
            # or MOTOR_OVERLOAD cross-fire.
            "vibration_rms": {
                "unit": "mm/s",
                "nominal_mu": 2.5,
                "nominal_sigma": 0.15,
                "warning_threshold": 4.5,
                "critical_threshold": 7.1,
                "degradation_weight": 0.05,
                "degradation_direction": 1,
                "fft": {
                    "BPFO_coeff": 0.4,
                    "BPFI_coeff": 0.6,
                    "BSF_coeff": 0.2,
                    "N_balls": 9,
                    "rpm_nominal": 1750,
                },
            },
            "bearing_temp": {
                "unit": "°C",
                "nominal_mu": 62.0,
                "nominal_sigma": 2.0,
                "warning_threshold": 85.0,
                "critical_threshold": 105.0,
                "degradation_weight": 0.10,
                "degradation_direction": 1,
                "correlation_with": {"vibration_rms": {"mu": 0.75, "sigma": 0.05}},
            },
            "oil_pressure": {
                "unit": "bar",
                "nominal_mu": 4.5,
                "nominal_sigma": 0.15,
                "warning_threshold": 2.8,
                "critical_threshold": 2.0,
                "degradation_weight": 0.10,
                "degradation_direction": -1,
            },
            "motor_current": {
                "unit": "A",
                "nominal_mu": 21.0,
                "nominal_sigma": 0.6,
                "warning_threshold": 28.0,
                "critical_threshold": 34.0,
                "degradation_weight": 0.35,
                "degradation_direction": 1,
                "correlation_with": {"vibration_rms": {"mu": 0.70, "sigma": 0.05}},
            },
            "outlet_pressure": {
                "unit": "bar",
                "nominal_mu": 8.2,
                "nominal_sigma": 0.15,
                "warning_threshold": 6.5,
                "critical_threshold": 5.0,
                "degradation_weight": 0.40,
                "degradation_direction": -1,
                "downstream_effect": {
                    "target_machine": "HX-302",
                    "target_sensor": "inlet_flow_proxy",
                    "effect_type": "UPSTREAM_EFFECT",
                },
            },
        },
        "startup_penalty": {
            "rul_penalty_mu": -0.10,
            "rul_penalty_sigma": 0.02,
            "oil_pressure_low_s": 30,
            "warm_restart_s": 45,
            "cold_restart_s": 210,
        },
    },
    "HX-302": {
        "name": "Shell & Tube Heat Exchanger — Line B",
        "line": "B",
        "type": "HX",
        "has_bearings": False,
        "pulley_ratio": None,
        "standard": "TEMA RGP-T-2.4",
        "failure_mode": "Flow restriction — tube-side blockage and progressive flow area loss (TEMA RGP-T-2.4)",
        "weibull": {
            # Source: TEMA, newer unit extended service factor
            # Base eta = 1020h, scaled 0.2x for demo visibility
            "beta": 2.5,
            "beta_std": 0.18,
            "eta": 180.0,       # 180 hours = 7.5 days (t95, FAILED-phase point; most reliable)
            "eta_std": 26.0,
            "source": "TEMA, newer unit extended service factor (demo-scaled 0.2x)",
        },
        "cwru_calibration": None,
        "sensors": {
            # HX-302 target fault: FLOW_RESTRICTION
            # Dominant sensors: flow_rate (drops) + pressure_drop (rises), matching
            # the FLOW_RESTRICTION rule. fouling_index is kept minimal to avoid
            # cross-firing with the FOULING rule.
            "inlet_temp": {
                "unit": "°C",
                "nominal_mu": 145.0,
                "nominal_sigma": 3.0,
                "nominal_range": [130, 160],
                "warning_threshold": None,
                "critical_threshold": None,
                "degradation_weight": 0.05,
                "degradation_direction": 1,
                "upstream_sensitivity": {
                    "source_machine": "AC-301",
                    "source_sensor": "outlet_pressure",
                    "effect": "inlet_temp_rise_on_low_flow",
                    "flag": "UPSTREAM_EFFECT",
                },
            },
            "outlet_temp": {
                "unit": "°C",
                "nominal_mu": 78.0,
                "nominal_sigma": 2.0,
                "warning_threshold": 95.0,
                "critical_threshold": None,
                "degradation_weight": 0.10,
                "degradation_direction": 1,
                "correlation_with": {"inlet_temp": {"mu": 0.65, "sigma": 0.05}},
            },
            "pressure_drop": {
                "unit": "bar",
                "nominal_mu": 0.85,
                "nominal_sigma": 0.04,
                "warning_threshold": 1.4,
                "critical_threshold": 1.8,
                "degradation_weight": 0.40,
                "degradation_direction": 1,
                "fouling_growth_model": "exponential",
            },
            "flow_rate": {
                "unit": "m³/h",
                "nominal_mu": 12.5,
                "nominal_sigma": 0.4,
                "warning_threshold": 9.0,
                "critical_threshold": 7.0,
                "degradation_weight": 0.40,
                "degradation_direction": -1,
            },
            "fouling_index": {
                "unit": "-",
                "nominal_mu": 0.08,
                "nominal_sigma": 0.01,
                "warning_threshold": 0.35,
                "critical_threshold": 0.55,
                "degradation_weight": 0.05,
                "degradation_direction": 1,
            },
        },
        "startup_penalty": None,
    },
    "CM-303": {
        "name": "Conveyor Belt Drive System — Line B",
        "line": "B",
        "type": "CM",
        "has_bearings": True,
        "pulley_ratio": 1.65,
        "standard": "ISO 5048, CEMA 7th Ed.",
        "failure_mode": "Belt slip — progressive tension loss and speed deficit under high-duty load (CEMA 7th Ed.)",
        "weibull": {
            # Source: CEMA, higher duty cycle adjustment
            # Base eta = 600h, scaled 0.2x for demo visibility
            "beta": 2.1,
            "beta_std": 0.17,
            "eta": 120.0,       # 120 hours = 5 days (t95, FAILED-phase point; wear-out dominant, beta > 2)
            "eta_std": 15.0,
            "source": "CEMA, higher duty cycle adjustment (demo-scaled 0.2x)",
        },
        "cwru_calibration": None,
        # CM-303 target fault: BELT_SLIP
        # When Line A fully stops, CM-303 takes extra load
        "cross_line_load_spike": {
            "trigger": "LINE_A_FULL_STOP",
            "belt_tension_mult": {"mu": 1.4, "sigma": 0.05},
            "deg_rate_mult": {"mu": 1.8, "sigma": 0.15},
            "motor_load_delta": {"mu": +15.0, "sigma": 1.5},
            "notification_level": "L1",
            "message": "Line B operating at ~140% conveyor load due to Line A failure",
        },
        "sensors": {
            # CM-303 target fault: BELT_SLIP
            # Dominant sensors: belt_tension (rises as tension becomes uneven) +
            # speed_rpm (drops as slip increases), matching the BELT_SLIP rule.
            # drive_temp and motor_load are reduced to avoid MOTOR_OVERLOAD cross-fire.
            "belt_tension": {
                "unit": "kN",
                "nominal_mu": 8.2,
                "nominal_sigma": 0.25,
                "warning_threshold": 11.5,
                "critical_threshold": 13.8,
                "degradation_weight": 0.45,
                "degradation_direction": 1,
            },
            "drive_temp": {
                "unit": "°C",
                "nominal_mu": 55.0,
                "nominal_sigma": 2.5,
                "warning_threshold": 78.0,
                "critical_threshold": 92.0,
                "degradation_weight": 0.10,
                "degradation_direction": 1,
                "thermal_model": "square_law",
                "k_nominal": 0.0015,
                "k_sigma_pct": 0.08,
            },
            "motor_load": {
                "unit": "%",
                "nominal_mu": 68.0,
                "nominal_sigma": 3.0,
                "warning_threshold": 88.0,
                "critical_threshold": 96.0,
                "degradation_weight": 0.10,
                "degradation_direction": 1,
            },
            "speed_rpm": {
                "unit": "RPM",
                "nominal_mu": 1450.0,
                "nominal_sigma": 7.5,
                "warning_threshold": 1380.0,
                "critical_threshold": None,
                "degradation_weight": 0.30,
                "degradation_direction": -1,
            },
            "vibration_rms": {
                "unit": "mm/s",
                "nominal_mu": 1.8,
                "nominal_sigma": 0.15,
                "warning_threshold": 3.5,
                "critical_threshold": 5.0,
                "degradation_weight": 0.05,
                "degradation_direction": 1,
            },
        },
        "startup_penalty": None,
    },
}

# ─── Shift Load Profiles ──────────────────────────────────────────────────
SHIFT_LOAD_PROFILES = {
    "morning": {
        "hours": (6, 14),
        "load_factor": {"mu": 1.00, "sigma": 0.05},
        "tolerance_mult": {"mu": 1.00, "sigma": 0.03},
    },
    "afternoon": {
        "hours": (14, 22),
        "load_factor": {"mu": 1.15, "sigma": 0.06},
        "tolerance_mult": {"mu": 0.95, "sigma": 0.03},
    },
    "night": {
        "hours": (22, 6),
        "load_factor": {"mu": 0.75, "sigma": 0.04},
        "tolerance_mult": {"mu": 1.10, "sigma": 0.04},
    },
}

# ─── Cascade Rules ────────────────────────────────────────────────────────
CASCADE_RULES = {
    ("AC-201", "HX-202"): {
        "effect_type": "UPSTREAM_EFFECT",
        "physical_mechanism": "outlet_pressure → HX-202 inlet_flow",
        "repair_cost": False,
    },
    ("AC-201", "CM-203"): {
        "effect_type": "IDLE",
        "physical_mechanism": "Line A fully stopped",
        "repair_cost": False,
    },
    ("HX-202", "CM-203"): {
        "effect_type": "IDLE",
        "physical_mechanism": "Line A output stopped",
        "repair_cost": False,
    },
    ("LINE_A", "CM-303"): {
        "effect_type": "LOAD_SPIKE",
        "physical_mechanism": "Line B ~40% extra load",
        "repair_cost": False,
    },
    ("AC-301", "HX-302"): {
        "effect_type": "UPSTREAM_EFFECT",
        "physical_mechanism": "outlet_pressure → HX-302 inlet_flow",
        "repair_cost": False,
    },
    ("AC-301", "CM-303"): {
        "effect_type": "IDLE",
        "physical_mechanism": "Line B fully stopped",
        "repair_cost": False,
    },
}

REQUIRED_SENSOR_FIELDS = ("nominal_mu", "nominal_sigma", "degradation_weight", "degradation_direction")


def _validate_configs():
    """
    Validate all machine configs at import time.
    Raises ValueError if any machine has sensors whose degradation_weight != 1.0.
    """
    for machine_id, config in MACHINE_CONFIGS.items():
        sensors = config["sensors"]
        # Check required fields
        for sensor_name, s_cfg in sensors.items():
            for field in REQUIRED_SENSOR_FIELDS:
                if field not in s_cfg:
                    raise ValueError(f"{machine_id}.{sensor_name} missing required field: {field}")
        # Check weight sum
        total_weight = sum(s["degradation_weight"] for s in sensors.values())
        if abs(total_weight - 1.0) > 1e-9:
            raise ValueError(f"{machine_id} degradation_weight sum = {total_weight:.10f}, expected 1.0")

    # Validate cascade rules reference valid machine IDs
    valid_ids = set(MACHINE_CONFIGS.keys()) | {"LINE_A", "LINE_B"}
    for src, tgt in CASCADE_RULES:
        if src not in valid_ids:
            raise ValueError(f"CASCADE_RULES source '{src}' not in MACHINE_CONFIGS")
        if tgt not in valid_ids and tgt not in MACHINE_CONFIGS:
            raise ValueError(f"CASCADE_RULES target '{tgt}' not in MACHINE_CONFIGS")


# Validate on import — prevents misconfigured deployments
_validate_configs()


# ═══════════════════════════════════════════════════════════════════════════════
# 6-Machine Factory: MACHINE_CONFIGS → machine_specs for SimulationEngine
# ═══════════════════════════════════════════════════════════════════════════════


def build_machine_specs() -> dict[str, dict[str, dict]]:
    """
    Convert MACHINE_CONFIGS into the `machine_specs` shape expected by
    `SimulationEngine` (v3):

        {
            machine_id: {
                sensor_name: {
                    "nominal_mu": ...,
                    "nominal_sigma": ...,
                    "degradation_weight": ...,
                    "degradation_direction": ...,
                    "warning_threshold": ...,
                    "critical_threshold": ...,
                },
                ...
            },
            ...
        }

    Returns:
        dict: 6-machine spec dict, deterministic from MACHINE_CONFIGS.
    """
    specs: dict[str, dict[str, dict]] = {}
    for machine_id, config in MACHINE_CONFIGS.items():
        sensors: dict[str, dict] = {}
        for sensor_name, sensor_cfg in config["sensors"].items():
            sensors[sensor_name] = {
                "nominal_mu": sensor_cfg["nominal_mu"],
                "nominal_sigma": sensor_cfg["nominal_sigma"],
                "degradation_weight": sensor_cfg["degradation_weight"],
                "degradation_direction": sensor_cfg["degradation_direction"],
                "warning_threshold": sensor_cfg.get("warning_threshold"),
                "critical_threshold": sensor_cfg.get("critical_threshold"),
            }
        specs[machine_id] = sensors
    return specs


# ═══════════════════════════════════════════════════════════════════════════════
# Anomaly Scenarios (chaos router)
# ═══════════════════════════════════════════════════════════════════════════════
#
# Per-sensor positive strength (scalar multiplier) for each fault scenario.
# Injected via SimulationEngine.inject_anomaly(machine_id, scenario, ramp_seconds).
#
# The effect on each sensor is multiplicative on its degradation level:
#     d_effective = d_base * (1 + strength * smoothstep(ramp_progress))
# The strength is always a POSITIVE magnitude. The physical sign of the
# effect (e.g. oil_pressure decreases) is handled by the sensor's existing
# `degradation_direction` field; this dict specifies how much *extra*
# degradation pressure the scenario applies.
#
# Special key "_all_" applies the same strength to every sensor on the
# machine (used for "full_cascade").

ANOMALY_SCENARIOS: dict[str, dict[str, float]] = {
    # Per-sensor positive strength (multiplicative on degradation level).
    # Source: demo-grade engineering judgment, calibrated to produce a
    # visible anomaly signal in sensor readings without saturating them.
    # Strengths are POSITIVE magnitudes; the physical sign is the sensor's
    # `degradation_direction` field.
    "oil_leak": {
        "bearing_temp": 0.3,   # +30% deg acceleration (heat from friction)
        "oil_pressure": 0.4,   # +40% deg acceleration (loss of pressure)
    },
    "fouling_spike": {
        "pressure_drop": 0.5,  # +50% deg acceleration (tube scaling)
        "flow_rate": 0.3,      # +30% deg acceleration (flow restriction)
    },
    "belt_slip": {
        "vibration_rms": 0.4,  # +40% deg acceleration (slip-induced vibration)
        "motor_current": 0.2,  # +20% deg acceleration (compensating load)
        "belt_tension": 0.4,   # +40% deg acceleration (tension loss, CM native)
        "motor_load": 0.2,     # +20% deg acceleration (drive compensating)
    },
    "full_cascade": {
        "_all_": 0.5,          # +50% deg acceleration on every sensor
    },
}


def get_anomaly_scenarios() -> dict[str, dict[str, float]]:
    """
    Public accessor — returns the canonical ANOMALY_SCENARIOS dict.

    The factory function exists so callers don't need to know the module-level
    name. `scheduler.py` passes this to `SimulationEngine.anomaly_scenarios`.
    """
    return ANOMALY_SCENARIOS
