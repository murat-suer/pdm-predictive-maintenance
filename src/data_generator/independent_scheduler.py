"""
src/data_generator/independent_scheduler.py
=============================================
Async per-machine, per-sensor degradation scheduler for PdM v3.

KEY PROPERTIES:
- Each machine has its own asyncio-based scheduler (independent task).
- Each sensor within a machine has independent random degradation timing.
- Random interval between degradation checks: uniform(180, 300) simulated seconds.
- 75% probability of degradation occurring each check.
- NO synchronization between machines — completely independent scheduling.
- Seed per machine (and per sensor) for full reproducibility.

Design:
    SimulationEngine
        └── asyncio.gather(*[machine_scheduler(machine) for machine in machines])
            └── Each machine_scheduler() runs an infinite loop:
                1. Advance simulated time by randomly-sampled interval
                2. For each sensor, check if its timer has elapsed
                3. With 75% probability, apply degradation to that sensor
                4. Generate new sensor readings
                5. Sleep (wall time) = interval / speed_multiplier
"""

from __future__ import annotations

import asyncio
import logging
import math
import time as wall_time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from src.config import get_settings

from .fft_engine import generate_fft_data
from .machines import MACHINE_CONFIGS, SHIFT_LOAD_PROFILES
from .weibull_engine import (
    MachineDegradationState,
    check_plausibility,
    create_degradation_model,
    create_machine_state,
    sample_degradation_probability,
    sample_per_machine_interval,
)

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# Sensor dropout configuration (Phase 4 Item 3)
# ═══════════════════════════════════════════════════════════════════════════════

# Phase 4 Item 3 — sensor dropout (missingness as signal).
#
# Real factories have intermittent sensor connectivity. The
# `_PRESENT_PROBABILITY` knob controls the per-reading probability
# that a reading is marked `present=True` (sensor online). The
# default 0.99 (1% dropout) is calibrated to produce a visible-but-rare
# "sensor health" pattern on the dashboard without overwhelming the ML
# pipeline.
#
# Configured via Settings.SENSOR_DROPOUT_PROBABILITY (env var of the
# same name), making config.py the single source of truth. Examples:
#   SENSOR_DROPOUT_PROBABILITY=0.999  — pristine lab fixture
#   SENSOR_DROPOUT_PROBABILITY=0.95   — noisy industrial deployment
_PRESENT_PROBABILITY = get_settings().SENSOR_DROPOUT_PROBABILITY


# ═══════════════════════════════════════════════════════════════════════════════
# Fault injection pacing
# ═══════════════════════════════════════════════════════════════════════════════

# Minimum degradation base for an injected fault's smoothstep boost. The
# boost is proportional to degradation; on a young machine at low
# SIMULATION_SPEED that proportionality alone makes injection invisible.
INJECTION_FLOOR_D = 0.30
# After the ramp completes the injected fault keeps developing on a
# real-time clock (per unit anomaly_strength per real second). At 0.004,
# a strength-0.4 sensor adds ~0.10 degradation per real minute — the
# machine reaches the ANOMALY zone within ~5 minutes of injection
# regardless of simulation speed.
INJECTION_PROGRESSION_PER_REAL_S = 0.004


# ═══════════════════════════════════════════════════════════════════════════════
# Failure mode correlation (Phase 4 Item 8)
# ═══════════════════════════════════════════════════════════════════════════════

# Phase 4 Item 8 — when a fault scenario is injected, related sensors
# should degrade TOGETHER (not just the primary sensor). The physics
# rationale: bearing_fault produces vibration AND temperature rise
# (friction); oil_leak causes pressure drop AND temperature (less
# cooling); fouling_spike correlates pressure_drop and flow_rate.
#
# The correlation matrix is keyed by scenario name. Each value is
# `{source_sensor: correlation_strength}`. When the source sensor is
# in its anomaly ramp window, the *target* sensor (one in the
# matrix but not the source) gets a correlated boost. Correlation
# values are in [0, 1] representing a fraction of the primary
# sensor's anomaly strength applied to the target.
#
# These correlations are derived from physical principles (bearing
# friction → heat, oil leak → pressure + temperature, fouling →
# pressure + flow restriction) and are the kind of "obvious in
# hindsight" correlations that a real PdM model should capture.
_FAILURE_MODE_CORRELATIONS: dict[str, dict[str, float]] = {
    "oil_leak": {
        "oil_pressure": 0.85,    # source: oil pressure drop → bearing_temp rise
    },
    "fouling_spike": {
        "pressure_drop": 0.80,   # source: pressure drop → flow_rate drop
    },
    "belt_slip": {
        "vibration_rms": 0.75,   # source: vibration → motor_current rise
    },
    "bearing_fault": {
        "vibration_rms": 0.90,   # source: vibration → bearing_temp rise
    },
    "full_cascade": {
        # For full_cascade we just use any primary sensor (the
        # scenario applies to ALL sensors via _all_, so all are
        # "primary" and there's no secondary — the correlation
        # matrix is unused).
        "vibration_rms": 0.50,
    },
}


# ═══════════════════════════════════════════════════════════════════════════════
# Reading record type
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class SensorReading:
    """A single sensor reading produced by the simulation.

    Phase 4 Item 3: `present` is a per-reading boolean flag that
    indicates whether the sensor was online at the time of the reading.
    `present=False` means the sensor went offline / dropped the packet
    for this step. The ML subscriber and alarm_decision consumer
    consult this flag and skip dropped readings (they are NOT passed
    to the anomaly detector). The dashboard can display dropped
    readings as a separate "sensor health" indicator.
    """
    machine_id: str
    sensor_name: str
    value: float
    degradation_level: float
    phase: str
    simulated_time: float  # seconds since simulation start
    wall_time: float       # Unix timestamp of generation
    operational_seconds: float  # accumulated operational time for this sensor
    present: bool = True   # Phase 4 Item 3: sensor availability flag
    fft_data: dict | None = None


# ═══════════════════════════════════════════════════════════════════════════════
# Degradation step application
# ═══════════════════════════════════════════════════════════════════════════════

def apply_degradation_step(
    state: MachineDegradationState,
    sensor_name: str,
    dt_seconds: float,
    load_factor: float = 1.0,
) -> float:
    """
    Apply one degradation step to a single sensor.

    Uses the sensor's physics-based degradation model to compute the new
    degradation level, then blends with the Weibull-predicted degradation.
    If a fault scenario is in its ramp window, the blended degradation is
    scaled by (1 + strength * smoothstep_progress) — a smooth, monotonic
    acceleration (NOT a jump) used by the chaos router.

    Returns the new degradation level for this sensor.
    """
    sensor = state.sensors[sensor_name]
    model = create_degradation_model(sensor.config)

    # The physics model is evaluated at the MACHINE's operational time, not
    # the per-call dt. The scheduler advances state.operational_seconds by
    # ~180-300s per due sensor; if we used dt_seconds here, the sensor clock
    # would lag ~180x behind the machine and physics_d would never mature.
    sensor.operational_seconds = state.operational_seconds

    if (
        hasattr(sensor, "gamma_process")
        and sensor.gamma_process is not None
    ):
        gamma_val = sensor.gamma_process.step(dt=dt_seconds)
        physics_d = float(min(1.0, gamma_val))
    else:
        physics_d = model.compute_degradation(sensor.operational_seconds, sensor.rng)

    # Blend with Weibull-predicted degradation for consistency
    weibull_d = state.weibull.degradation_level_from_weibull(
        state.operational_seconds
    )

    # Weighted blend: Weibull sets the time scale, physics informs the shape.
    # Blend ratio: 50% physics model, 50% Weibull CDF — balanced so the
    # physics models (ISO 281 L10, Arrhenius, TEMA fouling) contribute
    # meaningfully to the per-sensor trajectory while the Weibull still
    # anchors the phase transition timing.
    blended = (0.50 * physics_d + 0.50 * weibull_d) * load_factor

    # Anomaly ramp blending: if this sensor is in a fault scenario ramp,
    # add a smooth boost on top of the Weibull curve so the effect is
    # visible in the sensor reading. The boost grows with smoothstep
    # (3p² - 2p³) so the trajectory is monotonic, never a step jump.
    #
    # The boost base is floored: proportional-to-age alone makes injection
    # a silent no-op on a young machine at low simulation speeds (weibull_d
    # ≈ 0 ⇒ boost ≈ 0). After the ramp the fault keeps progressing on a
    # REAL-time clock, so an injected fault walks degradation into the
    # ANOMALY zone within minutes at any SIMULATION_SPEED — at high speeds
    # the natural Weibull growth dominates and behavior is unchanged.
    t_now = state.operational_seconds
    anomaly_boost = 0.0
    if (
        sensor.anomaly_scenario is not None
        and sensor.anomaly_strength > 0.0
        and sensor.anomaly_ramp_end_t > sensor.anomaly_ramp_start_t
        and t_now >= sensor.anomaly_ramp_start_t
    ):
        if t_now >= sensor.anomaly_ramp_end_t:
            ramp_progress = 1.0
        else:
            span = sensor.anomaly_ramp_end_t - sensor.anomaly_ramp_start_t
            ramp_progress = (t_now - sensor.anomaly_ramp_start_t) / span
        smooth = ramp_progress * ramp_progress * (3.0 - 2.0 * ramp_progress)
        base_d = max(weibull_d, INJECTION_FLOOR_D)
        anomaly_boost = base_d * sensor.anomaly_strength * smooth
        post_ramp_sim_s = max(0.0, t_now - sensor.anomaly_ramp_end_t)
        if post_ramp_sim_s > 0.0:
            from src.config import settings as _settings

            speed = max(float(_settings.SIMULATION_SPEED), 1.0)
            progression = (
                sensor.anomaly_strength
                * INJECTION_PROGRESSION_PER_REAL_S
                * (post_ramp_sim_s / speed)
            )
            anomaly_boost += progression
        # Injection alone never forces FAILED (0.95); it parks the machine
        # deep in the ANOMALY zone and lets the decision loop act.
        anomaly_boost = min(anomaly_boost, max(0.0, 0.94 - weibull_d))

    # Phase 4 Item 8 — failure mode correlation. When the chaos
    # router injects a fault scenario, related sensors should
    # degrade TOGETHER (not just the primary sensor). The mechanism:
    # we look up the scenario's correlation matrix (defined below)
    # and add a correlated boost to each "secondary" sensor. The
    # correlation is a fraction of the primary sensor's strength
    # times a smoothstep factor, so the secondary sensor accelerates
    # IN SYNC with the primary.
    correlated_boost = 0.0
    correlation_matrix = _FAILURE_MODE_CORRELATIONS.get(sensor.anomaly_scenario or "")
    if correlation_matrix and anomaly_boost > 0.0:
        # correlation_matrix: { secondary_sensor: correlation_with_primary }
        # The primary sensor is the one in the scenario spec; we look
        # it up to find the "source" of the correlation.
        for src_scenario_sensor in correlation_matrix:
            if src_scenario_sensor == sensor.name:
                continue
            src_sensor = state.sensors.get(src_scenario_sensor)
            if src_sensor is None:
                continue
            if (
                src_sensor.anomaly_scenario == sensor.anomaly_scenario
                and src_sensor.anomaly_ramp_end_t > src_sensor.anomaly_ramp_start_t
            ):
                # Compute the source sensor's ramp progress
                if t_now >= src_sensor.anomaly_ramp_end_t:
                    src_progress = 1.0
                else:
                    span = src_sensor.anomaly_ramp_end_t - src_sensor.anomaly_ramp_start_t
                    src_progress = max(0.0, (t_now - src_sensor.anomaly_ramp_start_t) / span)
                src_smooth = src_progress * src_progress * (3.0 - 2.0 * src_progress)
                corr = correlation_matrix[src_scenario_sensor]
                correlated_boost += weibull_d * float(sensor.anomaly_strength) * corr * src_smooth
        # Cap so the boost is never larger than the primary's
        correlated_boost = min(correlated_boost, anomaly_boost * 0.95)
    anomaly_boost = anomaly_boost + correlated_boost

    # Per-sensor level = max(prev, weibull + anomaly_boost, natural blend).
    # The Weibull CDF is the time-scale signal: every sensor tracks it so
    # slow-degrading physics models (e.g. Arrhenius oil at 80 °C) still
    # reach FAILED in the documented 5-7 day window. The 0.50/0.50 blend
    # is a small physics perturbation — it can boost a sensor above the
    # Weibull curve but never pull it below. The anomaly boost is added
    # on top of the Weibull floor so the chaos-router signal is visible.
    candidate = max(
        sensor.degradation_level,
        blended,
        weibull_d + anomaly_boost,
    )

    if sensor.rng is not None:
        jitter = float(np.clip(sensor.rng.normal(1.0, 0.03), 0.91, 1.09))
        increment = candidate - sensor.degradation_level
        if increment > 0:
            candidate = sensor.degradation_level + increment * jitter

    # Phase 4 Item 7 — maintenance history effect. The degradation
    # rate is scaled by `degradation_rate_modifier`, which is < 1.0
    # for a freshly maintained machine and ~1.0 for one that has
    # been running for months. The candidate level is therefore
    # INTERPOLATED between the previous level (preserved) and the
    # new candidate, scaled by the modifier. The previous level
    # still anchors the monotone-non-decreasing property.
    try:
        modifier = state.degradation_rate_modifier()
    except Exception:
        modifier = 1.0
    if modifier < 1.0:
        # Interpolate: new_level = prev + (candidate - prev) * modifier
        interpolated = sensor.degradation_level + (
            candidate - sensor.degradation_level
        ) * modifier
        sensor.degradation_level = float(interpolated)
    else:
        sensor.degradation_level = float(candidate)

    return sensor.degradation_level


def generate_sensor_reading(
    state: MachineDegradationState,
    sensor_name: str,
) -> float:
    """Generate a single sensor reading based on current degradation level.

    Phase 4 Item 6: the per-sensor degradation model's
    `compute_sensor_value` returns a base value. We add an AR(1)
    noise increment on top using the sensor's `phi` and rolling
    `prev_noise` so the series is continuous across scheduler
    iterations. The AR(1) noise variance is set to
    `sigma = 0.5 * nominal_sigma` so it stays in the same order
    of magnitude as the model's own Gaussian noise (which uses
    `nominal_sigma` directly inside the model). This makes the
    dashboard charts look like real data (smooth drift with
    white noise on top) rather than pure white noise.
    """
    sensor = state.sensors[sensor_name]
    model = create_degradation_model(sensor.config)
    base = model.compute_sensor_value(sensor.degradation_level, sensor.rng)
    if sensor.phi > 0.0 and sensor.rng is not None:
        # AR(1) increment — use the sensor's own RNG (not the model's,
        # which doesn't carry an rng). The noise sigma is half the
        # nominal sigma so it stays in the same order of magnitude
        # as the model's own Gaussian noise.
        sigma = 0.5 * float(sensor.config.nominal_sigma)
        try:
            eps = float(sensor.rng.normal(0.0, sigma))
            scale = math.sqrt(max(0.0, 1.0 - sensor.phi * sensor.phi))
            ar_increment = sensor.phi * sensor.prev_noise + scale * eps
            sensor.prev_noise = float(ar_increment)
            base = base + float(ar_increment)
        except (ValueError, OverflowError):
            pass
    return float(base)


# ═══════════════════════════════════════════════════════════════════════════════
# Shift load profiles & cross-line load spike
# ═══════════════════════════════════════════════════════════════════════════════


def _apply_shift_load(hour: int) -> float:
    """
    Return the degradation-rate multiplier for the shift that contains `hour`.

    Looks up the current shift from SHIFT_LOAD_PROFILES:
      - morning   (06-14): load_factor mu=1.00
      - afternoon (14-22): load_factor mu=1.15
      - night     (22-06): load_factor mu=0.75

    The returned value is the `mu` of the shift's load_factor distribution.
    Callers multiply their per-step degradation increment by this value.
    """
    hour = int(hour) % 24
    for _shift_name, profile in SHIFT_LOAD_PROFILES.items():
        start, end = profile["hours"]
        if start < end:
            if start <= hour < end:
                return float(profile["load_factor"]["mu"])
        else:
            if hour >= start or hour < end:
                return float(profile["load_factor"]["mu"])
    return 1.0


def apply_cross_line_load_spike(state: MachineDegradationState) -> None:
    """
    Apply the cross-line load spike to CM-303 when Line A fully stops.

    Reads CM-303's `cross_line_load_spike` config and applies:
      - belt_tension model: lambda_per_second *= belt_tension_mult
      - all CM sensors: degradation rate *= deg_rate_mult
      - motor_load: nominal_mu += motor_load_delta

    Sets `cross_line_active=True` in affected sensor model_params so
    downstream code can detect the spike state.
    """
    cm_config = MACHINE_CONFIGS.get(state.machine_id, {})
    spike = cm_config.get("cross_line_load_spike")
    if spike is None:
        return

    tension_mult = float(spike["belt_tension_mult"]["mu"])
    deg_rate_mult = float(spike["deg_rate_mult"]["mu"])
    motor_delta = float(spike["motor_load_delta"]["mu"])

    for sensor_name, sensor in state.sensors.items():
        sensor.config.model_params["cross_line_active"] = True
        if sensor_name == "belt_tension":
            cur_lambda = sensor.config.model_params.get("lambda_per_second", 0.0)
            sensor.config.model_params["lambda_per_second"] = cur_lambda * tension_mult
        if sensor_name == "motor_load":
            sensor.config.nominal_mu = sensor.config.nominal_mu + motor_delta
        sensor.config.model_params["deg_rate_mult"] = deg_rate_mult

    logger.info(
        f"[{state.machine_id}] Cross-line load spike applied: "
        f"tension_x{tension_mult}, deg_rate_x{deg_rate_mult}, motor+{motor_delta}%"
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Per-machine async scheduler
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class MachineSchedulerConfig:
    """
    Configuration for one machine's independent scheduler.

    Attributes:
        state: The machine's degradation state.
        speed_multiplier: Initial simulation speed (500x for calibration, 1x for demo).
        speed_controller: Optional SpeedController — if set, `get_speed()` reads its
            current value (thread-safe), enabling live transitions from 500x to 1x.
        reading_callback: Optional async callback invoked for each reading batch.
        alarm_callback: Optional async callback invoked when machine enters ANOMALY/FAILED.
        maintenance_probability: Per-cycle probability of maintenance recovery when FAILED.
            Default 0.0 — FAILED is terminal. Auto-recovery is opt-in only.
        is_paused: Optional Callable[[], bool] that returns True when the machine should
            hold (no readings, no time advance). Used by the Phase 3A pre-staged dataset
            path: when a pre-staged dataset is playing, the live loop for that machine
            sleeps until the dataset is exhausted. Callable form so pause/resume takes
            effect without task restart.
    """
    state: MachineDegradationState
    speed_multiplier: float = 1.0
    speed_controller: Any | None = None
    reading_callback: Any | None = None  # async callable
    alarm_callback: Any | None = None    # async callable
    maintenance_probability: float = 0.0
    is_paused: Callable[[], bool] | None = None  # async-callable-friendly
    engine_states: dict[str, MachineDegradationState] | None = None

    # Internal: get speed multiplier. If a speed_controller is wired, prefer
    # its live value so transitions (calibration → demo) take effect within
    # the next event loop iteration without restarting tasks.
    def get_speed(self) -> float:
        if self.speed_controller is not None:
            return float(self.speed_controller.current_speed)
        return self.speed_multiplier


async def machine_scheduler(config: MachineSchedulerConfig) -> None:
    """
    Independent async scheduler for ONE machine.

    Runs an infinite loop:
    1. Determine next degradation check time (per-sensor timers).
    2. Sleep wall time = simulated_interval / speed_multiplier.
    3. For each sensor whose timer has elapsed:
       a. With 75% probability, apply degradation.
       b. Generate new reading.
    4. Update machine phase.
    5. Optionally invoke callbacks.

    This runs INDEPENDENTLY for each machine — no global tick, no synchronization.
    """
    state = config.state
    machine_id = state.machine_id

    logger.info(
        f"[{machine_id}] Scheduler started | beta={state.weibull.beta:.3f} "
        f"eta={state.weibull.eta:.0f}s | speed={config.get_speed():.1f}x"
    )

    while True:
        # ── Phase 3A: pause gate (pre-staged dataset takes over) ──────
        # When `is_paused()` returns True, the live loop sleeps without
        # advancing simulated time or emitting readings. The pre-staged
        # dataset is being driven by an external coroutine during this
        # window — see `ProductionSimulationEngine.play_pre_staged`.
        if config.is_paused is not None and config.is_paused():
            await asyncio.sleep(0.05)
            continue

        speed = config.get_speed()
        if speed <= 0.0:
            await asyncio.sleep(0.1)
            continue

        # ── Step 1: Find the minimum next-check time across all sensors ──
        # Each sensor has its own independent timer
        min_interval = float("inf")
        sensors_due: list[str] = []

        for sensor_name, sensor in state.sensors.items():
            # How much simulated time until this sensor's next check?
            remaining = sensor.next_step_at - state.operational_seconds
            if remaining <= 0.0:
                sensors_due.append(sensor_name)
                remaining = 0.0
            min_interval = min(min_interval, max(remaining, 0.0))

        # If no sensor is due soon, sleep until the next one. The sleep is
        # broken into 1s wall chunks so the scheduler advances simulated time
        # continuously and remains responsive to cancellation/stop.
        if not sensors_due and min_interval < float("inf") and min_interval > 0.0:
            wall_sleep = min_interval / speed
            chunk_wall = 1.0
            elapsed_wall = 0.0
            while elapsed_wall < wall_sleep:
                chunk = min(chunk_wall, wall_sleep - elapsed_wall)
                # Advance simulated time BEFORE sleeping so that the state
                # is consistent even if the engine is cancelled mid-sleep.
                state.operational_seconds += chunk * speed
                elapsed_wall += chunk
                await asyncio.sleep(chunk)
            continue

        # ── Step 2: Advance simulated time to next event ──
        if sensors_due:
            # All due sensors are at the same simulated time (or past it)
            # We advance time to the next step for all of them
            dt = 0.0
            for sn in sensors_due:
                sensor = state.sensors[sn]
                behind = state.operational_seconds - sensor.next_step_at
                if behind > 0:
                    dt = max(dt, behind)
            state.operational_seconds += dt if dt > 0 else 1.0
        else:
            # No sensors due, advance by a small amount
            state.operational_seconds += min_interval

        # ── Step 3: Process each due sensor independently ──
        readings: list[SensorReading] = []
        for sensor_name in sensors_due:
            sensor = state.sensors[sensor_name]
            sensor_rng = sensor.rng

            # 75% probability of degradation occurring this step
            if sensor_rng is not None and sample_degradation_probability(sensor_rng):
                current_hour = int((state.operational_seconds / 3600) % 24)
                shift_load = _apply_shift_load(current_hour)
                apply_degradation_step(
                    state, sensor_name, dt_seconds=dt,
                    load_factor=shift_load * getattr(state, "load_override", 1.0),
                )

            # Phase 4 Item 3: sensor dropout. The per-sensor RNG draws
            # a uniform sample; if it falls below the dropout
            # probability (1 - present_probability), the reading is
            # marked present=False. The ML subscriber consults this
            # flag and skips the row.
            present = True
            if sensor_rng is not None:
                dropout_threshold = 1.0 - _PRESENT_PROBABILITY
                if sensor_rng.random() < dropout_threshold:
                    present = False

            # Generate sensor reading (always — the dashboard can
            # still display the last-known value for a dropped
            # reading, but the consumer treats it as missing).
            value = generate_sensor_reading(state, sensor_name)
            sensor.last_value = value

            # Schedule next degradation step for this sensor
            if sensor_rng is not None:
                next_interval = sample_per_machine_interval(sensor_rng, state.interval_mu, state.interval_sigma)
                sensor.next_step_at = state.operational_seconds + max(next_interval, 1.0)

            fft_data = None
            machine_cfg = MACHINE_CONFIGS.get(machine_id, {})
            sensor_cfg = machine_cfg.get("sensors", {}).get(sensor_name, {})
            fft_cfg = sensor_cfg.get("fft")
            if (
                fft_cfg is not None
                and sensor.degradation_level > 0.3
                and machine_cfg.get("has_bearings", False)
            ):
                fft_data = generate_fft_data(
                    N_balls=fft_cfg.get("N_balls", 9),
                    rpm=fft_cfg.get("rpm_nominal", 1750),
                    ball_dia=0.4,
                    pitch_dia=1.5,
                    contact_angle=0.0,
                    degradation_level=sensor.degradation_level,
                    BPFO_coeff=fft_cfg.get("BPFO_coeff", 0.4),
                    BPFI_coeff=fft_cfg.get("BPFI_coeff", 0.6),
                    BSF_coeff=fft_cfg.get("BSF_coeff", 0.2),
                    rng=sensor_rng,
                )

            readings.append(SensorReading(
                machine_id=machine_id,
                sensor_name=sensor_name,
                value=value,
                degradation_level=sensor.degradation_level,
                phase=state.phase,
                simulated_time=state.operational_seconds,
                wall_time=wall_time.time(),
                operational_seconds=sensor.operational_seconds,
                present=present,
                fft_data=fft_data,
            ))

        # ── Step 4: Update machine phase ──
        old_phase = state.phase
        # Phase 4 Item 1: stochastic Weibull failure check. Fires the
        # exact conditional probability hazard against the dedicated
        # failure RNG. If it fires, it transitions the machine to
        # FAILED and records failure_operational_seconds for the
        # conformal predictor's baseline target (Item 2).
        if state.phase != "FAILED":
            step_dt = (dt if sensors_due else 0.0) if "dt" in locals() else 0.0
            if step_dt > 0.0:
                t_before = max(0.0, state.operational_seconds - step_dt)
                state.stochastic_step_failure(
                    t_current=t_before,
                    dt_seconds=step_dt,
                )
        state.update_phase()

        plausibility_state = {
            "degradation_history": [
                s.degradation_level for s in state.sensors.values()
            ],
            "current_degradation": state.overall_degradation,
            "sensor_values": {
                sn: s.last_value for sn, s in state.sensors.items()
                if s.last_value is not None
            },
        }
        plausibility_result = check_plausibility(plausibility_state)
        if plausibility_result["violations"]:
            for v in plausibility_result["violations"]:
                logger.warning(f"[{machine_id}] Plausibility violation: {v}")

        if (
            machine_id in ("AC-201", "HX-202", "CM-203")
            and state.phase == "FAILED"
            and config.engine_states is not None
        ):
            cm303_state = config.engine_states.get("CM-303")
            if cm303_state is not None:
                already_active = any(
                    s.config.model_params.get("cross_line_active", False)
                    for s in cm303_state.sensors.values()
                )
                if not already_active:
                    apply_cross_line_load_spike(cm303_state)

        # ── Step 5: Handle phase transitions ──
        if state.phase != old_phase:
            logger.info(
                f"[{machine_id}] Phase transition: {old_phase} → {state.phase} "
                f"(d={state.overall_degradation:.4f}, t={state.operational_seconds:.0f}s)"
            )

        # ── Step 6: Invoke callbacks ──
        if readings and config.reading_callback is not None:
            try:
                await config.reading_callback(readings)
            except Exception as exc:
                logger.error(f"[{machine_id}] Reading callback error: {exc}")

        if state.phase in ("ANOMALY", "FAILED") and config.alarm_callback is not None:
            try:
                await config.alarm_callback(machine_id, state.phase, state.overall_degradation)
            except Exception as exc:
                logger.error(f"[{machine_id}] Alarm callback error: {exc}")

        # ── Step 7: Small sleep to yield control ──
        await asyncio.sleep(0.001)


# ═══════════════════════════════════════════════════════════════════════════════
# Simulation Engine: orchestrates all independent machine schedulers
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class SimulationEngine:
    """
    Top-level simulation orchestrator for PdM v3.

    Creates N independent MachineScheduler tasks, each running its own
    asyncio event loop with independent random degradation timing.

    Usage:
        engine = SimulationEngine(machine_configs, speed_multiplier=500.0)
        await engine.run()   # runs until cancelled
        await engine.stop()  # graceful shutdown

    Anomaly injection (chaos router API):
        engine.inject_anomaly("AC-201", "oil_leak", ramp_seconds=10.0)
    """

    machine_specs: dict[str, dict[str, dict]]
    speed_multiplier: float = 1.0
    global_seed: int = 42
    reading_callback: Any | None = None
    alarm_callback: Any | None = None
    # Dynamic alarm-suppression source. Accepts either a bool (legacy
    # construction) or a Callable[[], bool] that is evaluated on every alarm
    # invocation. The callable form is required so calibration state changes
    # (SUPPRESSED → ENABLED) take effect without recreating the engine.
    is_alarm_suppressed: Any | None = False
    # Optional live speed source — if provided, machine tasks read its
    # current value each iteration, enabling 500x → 1x transitions at runtime.
    speed_controller: Any | None = None
    # Per-scenario per-sensor strength map for anomaly injection. Keys are
    # scenario names ("oil_leak", ...); values are {sensor_name: strength} or
    # {"_all_": strength} to apply the same multiplier to every sensor.
    # If None, inject_anomaly() will raise ValueError with a clear message.
    anomaly_scenarios: dict | None = None

    # Internal state
    _states: dict[str, MachineDegradationState] = field(default_factory=dict)
    _tasks: list[asyncio.Task] = field(default_factory=list)
    _running: bool = False
    # Phase 3A: per-machine pause list. When a machine_id is in this set,
    # its scheduler loop sleeps without emitting readings or advancing
    # simulated time. Used by `play_pre_staged` to let a pre-recorded
    # dataset drive the pipeline for the duration of a demo, without the
    # live generator contaminating the downstream consumer with conflicting
    # rows for the same machine.
    _paused_machines: set[str] = field(default_factory=set)

    def __post_init__(self):
        """Create MachineDegradationState for each machine spec."""
        # Normalize alarm-suppression input: bool → constant callable.
        # This means `alarm_suppressed=True` (legacy) and
        # `is_alarm_suppressed=lambda: True` (new) behave identically from
        # the scheduler's perspective, but the callable form supports live
        # state changes without recreating the engine.
        if not callable(self.is_alarm_suppressed):
            suppressed_value = bool(self.is_alarm_suppressed)
            self.is_alarm_suppressed = lambda val=suppressed_value: val

        for i, (machine_id, sensors) in enumerate(self.machine_specs.items()):
            state = create_machine_state(
                machine_id=machine_id,
                sensor_specs=sensors,
                global_seed=self.global_seed,
                machine_index=i,
            )
            self._states[machine_id] = state
            logger.info(
                f"[{machine_id}] Created | beta={state.weibull.beta:.3f} "
                f"eta={state.weibull.eta:.0f}s "
                f"(median TTF={state.weibull.median_time_to_failure() / 86400:.1f} days)"
            )

    @property
    def states(self) -> dict[str, MachineDegradationState]:
        return self._states

    def get_readings_since(self, since_wall_time: float) -> list[SensorReading]:
        """Return all readings generated since a given wall time. (Placeholder for DB)"""
        return []

    def inject_anomaly(
        self,
        machine_id: str,
        scenario: str,
        ramp_seconds: float = 10.0,
    ) -> None:
        """
        Inject a fault scenario that smoothly accelerates the per-sensor
        degradation over `ramp_seconds` of simulated time. NOT a jump.

        Behaviour per scenario (per-sensor strengths live in `machines.py`):
          - oil_leak:      bearing_temp +0.3, oil_pressure +0.4
          - fouling_spike: pressure_drop +0.5, flow_rate +0.3
          - belt_slip:     vibration_rms +0.4, motor_current +0.2
          - full_cascade:  all sensors +0.5

        The ramp function is smoothstep from 0→1 over the window; the
        effective degradation is multiplied by (1 + strength * ramp_progress)
        — so a machine that was at d=0.2 with oil_leak strength=0.3 will,
        over 10 simulated seconds, see the effective d rise smoothly to
        0.2 * (1 + 0.3) = 0.26 (NOT a jump to 0.5). The downstream ML
        service reads this and computes the matching RUL drop.

        This method mutates per-sensor `SensorState` fields. Since the
        engine is single-threaded asyncio, no lock is needed; the next
        `apply_degradation_step()` call (within the ramp window) will see
        the new ramp bounds.
        """
        state = self._states.get(machine_id)
        if state is None:
            raise ValueError(f"Unknown machine_id: {machine_id!r}")

        if self.anomaly_scenarios is None:
            raise ValueError(
                "SimulationEngine.anomaly_scenarios is not configured; "
                "pass anomaly_scenarios=machines.ANOMALY_SCENARIOS at construction."
            )

        scenario_spec = self.anomaly_scenarios.get(scenario)
        if scenario_spec is None:
            raise ValueError(
                f"Unknown scenario: {scenario!r}. "
                f"Available: {sorted(self.anomaly_scenarios.keys())}"
            )

        # If a "_all_" key is present, apply that strength to every sensor.
        # Otherwise, apply per-sensor strengths from the dict, skipping sensors
        # that are not in the spec.
        all_strength = scenario_spec.get("_all_") if isinstance(scenario_spec, dict) else None

        start_t = float(state.operational_seconds)
        end_t = start_t + float(ramp_seconds)

        applied: list[str] = []
        for sensor_name, sensor in state.sensors.items():
            if all_strength is not None:
                strength = float(all_strength)
            elif isinstance(scenario_spec, dict) and sensor_name in scenario_spec:
                strength = float(scenario_spec[sensor_name])
            else:
                continue
            sensor.anomaly_scenario = scenario
            sensor.anomaly_ramp_start_t = start_t
            sensor.anomaly_ramp_end_t = end_t
            sensor.anomaly_strength = strength
            applied.append(sensor_name)

        logger.info(
            f"[{machine_id}] Injected scenario={scenario!r} over {ramp_seconds:.1f}s "
            f"(sensors: {', '.join(applied) or 'none'})"
        )

    def inject_chaos_goal(
        self,
        machine_id: str,
        scenario: str,
        target_hours: float = 48.0,
        n_iterations: int = 20,
    ) -> float:
        """
        Phase 4 Item 4 — goal-directed chaos injection (binary search
        for the exact degradation level that yields a target RUL).

        The chaos router previously took a "strength" multiplier and
        applied it to the per-sensor degradation. This worked, but it
        didn't *guarantee* any particular RUL — the operator couldn't
        say "I want this machine to fail in 48h" and have the system
        actually land there.

        This method solves that: given a target RUL in hours, it
        binary-searches for the degradation level d* such that the
        closed-form Weibull MRL at t=0 equals target_hours:

            d* = 1 - exp(-((target_seconds / eta)^beta - 1)^0 / beta)
                 (rearranging the MRL formula) — but we binary-search
                 for numerical robustness, especially near the MRL
                 numerical-integration fallback boundary.

        The found d* is then APPLIED to the machine's sensors (clamped
        to [0, 0.95] so the FAILED threshold backstop still holds),
        and the chaos scenario is also injected so the smoothstep
        ramp is visible in the chart.

        This is the world-class "demo button" experience: the operator
        picks a target, the system finds the exact d* to hit it.

        Parameters
        ----------
        machine_id
            Target machine.
        scenario
            Chaos scenario (one of ANOMALY_SCENARIOS).
        target_hours
            Desired RUL in hours. The binary search converges on the
            d* that produces this MRL.
        n_iterations
            Number of binary-search iterations. 20 gives ~1e-6
            precision on d.

        Returns
        -------
        float
            The d* that was applied to the machine (for diagnostics).
        """
        state = self._states.get(machine_id)
        if state is None:
            raise ValueError(f"Unknown machine_id: {machine_id!r}")

        if target_hours <= 0.0:
            raise ValueError(f"target_hours must be positive, got {target_hours}")

        # Convert the target to seconds for the MRL formula
        target_seconds = float(target_hours) * 3600.0
        beta = state.weibull.beta
        eta = state.weibull.eta

        # Initial guess from the Weibull inverse CDF:
        # If we want MRL(0) = target_seconds and the MRL is
        # approximately eta * Gamma(1 + 1/beta) - t_to_d for t_to_d
        # representing the "elapsed" portion, the inversion is
        #   MRL(0) for a fresh machine = MTTF = eta * Gamma(1 + 1/beta)
        # so target_seconds = MTTF is the natural upper bound.
        # For shorter targets, we need the d=0 MRL to be smaller,
        # which corresponds to a higher d at t=0. The d* we apply
        # to the sensors SETS the simulated "elapsed time" forward
        # by t_to_d hours where t_to_d is solved from
        # MRL(t_to_d) = target_seconds.
        mttf = float(state.weibull.mean_time_to_failure())
        if target_seconds > mttf:
            raise ValueError(
                f"target_hours={target_hours} exceeds the Weibull MTTF "
                f"({mttf/3600:.1f}h) for {machine_id}"
            )

        # Binary search for t* such that MRL(t*) = target_seconds.
        # t* is the "effective age" we want the machine to be at
        # after the chaos ramp. d* is then 1 - exp(-(t*/eta)^beta).
        lo_t, hi_t = 0.0, eta * 100.0  # generous upper bound (>> t95)
        t_star = hi_t
        for _ in range(n_iterations):
            mid = (lo_t + hi_t) / 2.0
            mrl = state.weibull.mean_remaining_life(mid)
            if mrl > target_seconds:
                # Need MORE elapsed age to reach this small MRL
                lo_t = mid
            else:
                # This much elapsed age produces an acceptable MRL
                hi_t = mid
                t_star = mid
        # Solve d* from t_star: d = 1 - exp(-(t_star/eta)^beta)
        try:
            d_star = 1.0 - math.exp(-((t_star / eta) ** beta))
        except (ValueError, OverflowError):
            d_star = 0.95
        # Clamp to [overall_degradation, 0.95] so we don't "rewind"
        # a partially degraded machine, and we keep the 0.95
        # threshold backstop intact.
        d_star = max(state.overall_degradation, min(0.95, d_star))

        # Apply d* to every sensor proportionally to its current
        # degradation level (preserves relative weighting) — the
        # machine is now "as if it had been running for t_star
        # simulated seconds".
        for sensor in state.sensors.values():
            sensor.degradation_level = max(
                sensor.degradation_level,
                d_star * (0.95 + 0.05 * float(state.rng.random()))
                if state.rng is not None
                else d_star,
            )
            sensor.operational_seconds = max(
                sensor.operational_seconds,
                t_star,
            )
        # Force a phase re-evaluation — the machine is now in
        # DEGRADING/ANOMALY depending on d*. Note: machine-level
        # degradation is computed on-the-fly from sensor levels via
        # `overall_degradation`; we don't store it as a field.
        state.operational_seconds = max(state.operational_seconds, t_star)
        state.update_phase()

        # Also inject the scenario so the smoothstep ramp is visible
        self.inject_anomaly(machine_id, scenario, ramp_seconds=10.0)

        logger.info(
            f"[{machine_id}] Goal-directed chaos: scenario={scenario!r} "
            f"target={target_hours:.1f}h → d*={d_star:.4f} "
            f"(t*={t_star/3600:.2f}h, mrl_at_t*={state.weibull.mean_remaining_life(t_star)/3600:.2f}h)"
        )
        return float(d_star)

    async def run(self) -> None:
        """
        Start all independent machine schedulers.
        Runs until cancelled (Ctrl+C or stop()).
        """
        if self._running:
            logger.warning("SimulationEngine already running")
            return

        self._running = True
        self._tasks = []

        for machine_id, state in self._states.items():
            # Phase 3A: wire the pause callback so `play_pre_staged` can
            # hold a single machine's live loop while the pre-staged
            # dataset drives the pipeline.
            config = MachineSchedulerConfig(
                state=state,
                speed_multiplier=self.speed_multiplier,
                speed_controller=self.speed_controller,
                reading_callback=self._wrap_reading_callback,
                alarm_callback=self._wrap_alarm_callback,
                is_paused=(lambda mid=machine_id: self.is_paused(mid)),
                engine_states=self._states,
            )
            task = asyncio.create_task(
                machine_scheduler(config),
                name=f"scheduler-{machine_id}",
            )
            self._tasks.append(task)

        logger.info(
            f"SimulationEngine started | {len(self._tasks)} machines | "
            f"speed={self.speed_multiplier:.0f}x"
        )

        try:
            await asyncio.gather(*self._tasks)
        except asyncio.CancelledError:
            logger.info("SimulationEngine cancelled")
            await self.stop()

    async def stop(self) -> None:
        """Gracefully stop all machine schedulers."""
        self._running = False
        for task in self._tasks:
            if not task.done():
                task.cancel()
        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks.clear()
        logger.info("SimulationEngine stopped")

    def pause_machine(self, machine_id: str) -> None:
        """
        Phase 3A: mark a machine as paused. Its scheduler loop will sleep
        without emitting readings or advancing simulated time until
        `resume_machine` is called. The pause is a soft sleep, NOT a task
        cancellation — when the dataset is exhausted, `resume_machine`
        brings the loop back without restart.
        """
        if machine_id not in self._states:
            raise ValueError(f"Unknown machine_id: {machine_id!r}")
        self._paused_machines.add(machine_id)
        logger.info(f"[{machine_id}] Scheduler paused (pre-staged or external control)")

    def resume_machine(self, machine_id: str) -> None:
        """
        Phase 3A: counterpart to `pause_machine`. Removes the machine from
        the paused set; the scheduler loop resumes on its next iteration.
        """
        self._paused_machines.discard(machine_id)
        logger.info(f"[{machine_id}] Scheduler resumed")

    def is_paused(self, machine_id: str) -> bool:
        """Phase 3A: return True if the machine is currently in the pause set."""
        return machine_id in self._paused_machines

    def reset_machine(self, machine_id: str, maintenance_quality: float | None = None) -> None:
        """
        Maintenance completed: return the machine to full health.

        Resets per-sensor degradation via `reset_after_maintenance`, zeroes
        the machine-level operational clock (the Weibull hazard restarts at
        t=0 for the fresh part) and clears any operator load override.
        """
        state = self._states.get(machine_id)
        if state is None:
            raise ValueError(f"Unknown machine_id: {machine_id!r}")
        state.reset_after_maintenance(maintenance_quality=maintenance_quality)
        state.operational_seconds = 0.0
        state.failure_operational_seconds = None
        state.load_override = 1.0
        logger.info(f"[{machine_id}] Maintenance reset: machine restored to full health")

    def set_load_factor(self, machine_id: str, factor: float) -> None:
        """
        Operator-controlled load throttle (REDUCE_LOAD scenario).

        Multiplies the shift load in the degradation step; 0.8 means the
        machine wears ~20% slower at the cost of reduced throughput.
        """
        state = self._states.get(machine_id)
        if state is None:
            raise ValueError(f"Unknown machine_id: {machine_id!r}")
        state.load_override = float(max(0.1, min(factor, 1.5)))
        logger.info(f"[{machine_id}] Load override set to {state.load_override:.2f}")

    async def _wrap_reading_callback(self, readings: list[SensorReading]) -> None:
        """Internal: wrap user reading callback."""
        if self.reading_callback is not None and self._running is not False:
            await self.reading_callback(readings)

    async def _wrap_alarm_callback(self, machine_id: str, phase: str, degradation: float) -> None:
        """Internal: wrap user alarm callback, respecting dynamic alarm suppression.

        `is_alarm_suppressed` is evaluated on every call (it's a `Callable[[], bool]`
        after `__post_init__` normalization), so calibration → ENABLED transitions
        take effect on the next alarm without restarting the engine.
        """
        if callable(self.is_alarm_suppressed) and self.is_alarm_suppressed():
            return
        if self.alarm_callback is not None:
            await self.alarm_callback(machine_id, phase, degradation)


# ═══════════════════════════════════════════════════════════════════════════════
# Demo / smoke test entry point
# ═══════════════════════════════════════════════════════════════════════════════

async def _demo_run():
    """Minimal demo showing the independent scheduler in action."""
    # Minimal machine specs for testing (5 sensors each)
    test_specs = {
        f"MACHINE-{i}": {
            f"sensor_{j}": {
                "nominal_mu": 50.0 + j * 10,
                "nominal_sigma": 2.0,
                "degradation_weight": 1.0 / 6.0,
                "degradation_direction": 1 if j % 2 == 0 else -1,
            }
            for j in range(6)
        }
        for i in range(3)
    }

    engine = SimulationEngine(
        machine_specs=test_specs,
        speed_multiplier=100.0,
        global_seed=12345,
    )

    # Run for 10 seconds then stop
    asyncio.create_task(engine.run())
    await asyncio.sleep(10.0)
    await engine.stop()

    for mid, state in engine.states.items():
        logging.getLogger(__name__).info(
            "[%s] phase=%s d=%.4f t=%.0fs",
            mid, state.phase, state.overall_degradation, state.operational_seconds,
        )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    asyncio.run(_demo_run())
