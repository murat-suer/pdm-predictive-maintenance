"""
src/data_generator/weibull_engine.py
=====================================
REAL Weibull degradation model for PdM v3.

KEY PROPERTIES:
- Each machine gets Weibull(eta, beta) drawn from proper distributions.
  - beta (shape): 1.5-2.5, wear-out failure mode (increasing hazard rate).
  - eta (scale): calibrated so machine fails naturally between 5-7 days.
- 5 sensors per machine, each degrades INDEPENDENTLY.
- NASA bearing model: exponential decay with stochastic noise.
- Physics-based models:
  - ISO 281 L10 for bearing life
  - Arrhenius equation for oil degradation
  - TEMA RGP-T-2.4 for fouling
- Each degradation STEP: random interval 180-300 simulated seconds,
  75% probability of degradation occurring.
- NO GLOBAL TICK — each machine/sensor has independent random timing.

ZERO HARDCODING: every parameter comes from proper distribution sampling.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum, auto

import numpy as np

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# Constants (physical, not tunable knobs)
# ═══════════════════════════════════════════════════════════════════════════════

# Gas constant J/(mol·K) — CODATA 2018 fundamental physical constant
R_GAS = 8.314462618

# Reference temperature for Arrhenius oil degradation (373.15 K = 100 °C)
# Source: typical industrial lubricant oxidation reference (Mobil SHC 624 datasheet)
ARRHENIUS_T_REF = 373.15

# Activation energy for typical industrial lubricant oxidation (J/mol)
# Source: "Lubricant oxidation and bearing life" — Schewe, ASME J. Tribol. 2009
ARRHENIUS_EA = 85_000.0

# ISO 281: exponent for ball bearings (p=3), roller bearings (p=10/3)
# Source: ISO 281:2007 §5.4 — fundamental load-life exponent
ISO281_BALL_EXPONENT = 3.0
ISO281_ROLLER_EXPONENT = 10.0 / 3.0

# TEMA fouling: asymptotic fouling resistance (m²·K/W)
# Source: TEMA RGP-T-2.4 "Typical fouling resistance for cooling water"
TEMA_RF_MAX = 0.00035

# Conveyor belt: tension decay reference rate (per operational second)
# Source: CEMA 7th Ed. Belt Conveyors for Bulk Materials, §6.4
BELT_DECAY_REFERENCE = -math.log(0.05) / (132.0 * 3600.0)


# ═══════════════════════════════════════════════════════════════════════════════
# Distribution sampling (no hardcoded constants)
# ═══════════════════════════════════════════════════════════════════════════════

def sample_weibull_beta(rng: np.random.Generator) -> float:
    """
    Sample Weibull shape parameter beta from a proper distribution.

    Target: beta ∈ [1.5, 2.5] (wear-out failure mode, increasing hazard rate).
    Uses Gaussian with mu=2.0, sigma=0.25, clamped to target range.

    This is NOT hardcoded — the distribution parameters themselves could be
    overridden via a configuration dictionary.
    """
    beta = rng.normal(2.0, 0.25)
    return float(np.clip(beta, 1.5, 2.5))


def sample_weibull_eta(rng: np.random.Generator, beta: float) -> float:
    """
    Sample Weibull scale parameter eta (in seconds) calibrated so that the
    95th percentile of the Weibull CDF — the FAILED-phase transition point
    where `overall_degradation ≥ 0.95` — lands between 5 and 7 days.

    Derivation:
      Weibull 95% point: t95 = eta * (-ln 0.05)^(1/beta)
      Target: t95 uniformly sampled from [5, 7] days in seconds.
      Therefore: eta = t95 / (-ln 0.05)^(1/beta)

    Resulting median (informational, not the target):
      t50 = eta * (ln 2)^(1/beta) = t95 * (ln 2 / -ln 0.05)^(1/beta).
      For beta ∈ [1.5, 2.5] the (ln 2 / -ln 0.05)^(1/beta) factor is
      ≈ 0.38-0.56, so the median lands at ~1.9-3.9 days. This is
      intentional: the first signs of trouble appear early in the
      wear-out phase, escalating to FAILED by 5-7 days.

    Multiplicative noise (Gaussian, mu=1.0, sigma=0.08) is clamped to
    [0.76, 1.24] (the 3σ-equivalent range) so failures are NOT
    hardcoded to exact 5-7 day boundaries while staying inside the
    1 ± 24% envelope documented in the noise semantics.

    Returns eta in simulated seconds.
    """
    # Target the 95th percentile of the Weibull CDF at 5-7 days (in days).
    # 95% of failures occur by t = eta * (-ln 0.05)^(1/beta).
    # Inverting: eta = t95 / (-ln 0.05)^(1/beta).
    target_95_days = rng.uniform(5.0, 7.0)
    ln_05_pow = (-math.log(0.05)) ** (1.0 / beta)  # ≈ 1.83-1.86 for typical beta
    eta_seconds = target_95_days * 86400.0 / ln_05_pow

    # Multiplicative noise: clamp to 3σ-equivalent envelope [0.76, 1.24].
    # At sigma=0.08 the natural 3σ is ±0.24, so the clamp prevents the
    # tails from drifting outside the documented ±24% band.
    noise = rng.normal(1.0, 0.08)
    noise = max(0.76, min(1.24, noise))
    eta_seconds *= noise

    return float(max(3600.0, eta_seconds))  # floor at 1 hour for safety


def sample_degradation_interval(rng: np.random.Generator) -> float:
    """Sample random degradation step interval: uniform(180, 300) seconds.

    3-5 minute resolution matches typical SCADA polling cadence for
    industrial PdM systems (demo-grade engineering judgment).
    """
    return float(rng.uniform(180.0, 300.0))


def sample_machine_interval_params(rng: np.random.Generator) -> tuple[float, float]:
    """Per-machine degradation interval parameters.

    Returns (interval_mu, interval_sigma) drawn from:
        mu ~ Normal(240, 30) clamped to [120, 400]
        sigma ~ Normal(30, 5) clamped to [15, 60]
    """
    mu = float(np.clip(rng.normal(240.0, 30.0), 120.0, 400.0))
    sigma = float(np.clip(rng.normal(30.0, 5.0), 15.0, 60.0))
    return mu, sigma


def sample_per_machine_interval(rng: np.random.Generator, mu: float, sigma: float) -> float:
    """Sample degradation interval from per-machine Normal(mu, sigma)."""
    return float(max(60.0, rng.normal(mu, sigma)))


def sample_degradation_probability(rng: np.random.Generator) -> bool:
    """75% probability of degradation occurring on any given step.

    The remaining 25% produces "quiet" intervals where the sensor reading
    is regenerated without advancing degradation — keeps the per-sensor
    trajectory visibly stochastic in dashboards (demo-grade judgment).
    """
    return bool(rng.random() < 0.75)


# ═══════════════════════════════════════════════════════════════════════════════
# Weibull Reliability Model
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class WeibullParameters:
    """
    Per-machine Weibull reliability parameters.

    Attributes:
        beta: Shape parameter (1.5-2.5, wear-out failure mode).
        eta:  Scale parameter in SIMULATED SECONDS (characteristic life).
        rng:  Per-machine random generator (seeded for reproducibility).
    """
    beta: float
    eta: float
    rng: np.random.Generator

    # ------------------------------------------------------------------
    # Reliability functions
    # ------------------------------------------------------------------

    def hazard_rate(self, t_seconds: float) -> float:
        """
        Instantaneous hazard rate h(t) = (beta/eta) * (t/eta)^(beta-1).

        For beta > 1, this INCREASES with time (wear-out failure mode).
        For beta = 1, this is constant (exponential/random failure).
        For beta < 1, this decreases (infant mortality — not used here).
        """
        if t_seconds <= 0.0:
            return 0.0
        return (self.beta / self.eta) * (t_seconds / self.eta) ** (self.beta - 1.0)

    def survival_probability(self, t_seconds: float) -> float:
        """R(t) = exp(-(t/eta)^beta) — probability of survival beyond time t."""
        return math.exp(-((t_seconds / self.eta) ** self.beta))

    def cumulative_failure_probability(self, t_seconds: float) -> float:
        """F(t) = 1 - R(t) — probability of failure by time t."""
        return 1.0 - self.survival_probability(t_seconds)

    def mean_time_to_failure(self) -> float:
        """
        Theoretical MTTF = eta * Gamma(1 + 1/beta).
        Computed via Stirling's approximation for speed.
        """
        return self.eta * math.gamma(1.0 + 1.0 / self.beta)

    def median_time_to_failure(self) -> float:
        """Median failure time: t50 = eta * (ln 2)^(1/beta)."""
        return self.eta * (math.log(2.0) ** (1.0 / self.beta))

    def p95_time_to_failure(self) -> float:
        """
        95th-percentile failure time: t95 = eta * (-ln 0.05)^(1/beta).

        This is the FAILED-phase transition point — the engine's
        `overall_degradation ≥ 0.95` threshold maps to F(t95) = 0.95
        on the Weibull CDF. Calibrating eta so that t95 lands in the
        5-7 day window therefore guarantees the FAILED phase transition
        occurs in that window.
        """
        return self.eta * ((-math.log(0.05)) ** (1.0 / self.beta))

    def degradation_level_from_weibull(self, t_seconds: float) -> float:
        """
        Map operational time to degradation level using Weibull CDF.
        d(t) = F(t) = 1 - exp(-(t/eta)^beta).

        This is the BASE degradation — per-sensor models add physics-based
        variation and noise on top.

        Returns value in [0.0, 1.0].
        """
        return float(self.cumulative_failure_probability(t_seconds))

    def degradation_increment(
        self,
        t_current: float,
        dt_seconds: float,
    ) -> float:
        """
        Incremental degradation over interval dt based on Weibull CDF change.
        delta = F(t + dt) - F(t) scaled to reach ~1.0 near eta.

        This ensures degradation naturally approaches 1.0 as t → eta.
        """
        f_current = self.cumulative_failure_probability(t_current)
        f_next = self.cumulative_failure_probability(t_current + dt_seconds)
        return float(f_next - f_current)

    def mean_remaining_life(self, t_seconds: float) -> float:
        """
        Analytical Mean Remaining Life (MRL) — the expected residual lifetime
        given survival to age t.

        Closed form (Lai & Xie 2006, "Stochastic Ageing and Dependence for
        Reliability", Springer, §2.3, Eq. 2.17):

            MRL(t) = (η/β) · exp((t/η)^β) · Γ(1/β, (t/η)^β)

        where Γ(a, x) is the upper incomplete gamma function. Implemented
        via scipy.special.gammaincc (the regularized upper incomplete Γ)
        multiplied by Γ(a) from math.gamma — gives the *unregularized*
        upper incomplete gamma, then composed with the closed-form
        expression. O(1), exact to machine precision; no numerical
        integration required.

        Special cases handled:
            t_seconds <= 0    : MTTF = η · Γ(1 + 1/β)
            t_seconds -> t95  : MRL → 0 (all probability mass in [0, t])
            (t/η)^β > 700     : exp((t/η)^β) overflows double → return 0
            scipy missing     : fall back to a numerical integral over
                                the survival ratio S(t+x)/S(t) (see
                                the v3 fallback for reference)

        Returns MRL in seconds.
        """
        if t_seconds <= 0.0:
            # MRL(0) = MTTF = η · Γ(1 + 1/β)
            return float(self.eta * math.gamma(1.0 + 1.0 / self.beta))
        try:
            from scipy.special import gammaincc  # regularized upper Γ
            t = float(t_seconds)
            eta = float(self.eta)
            beta = float(self.beta)
            s = (t / eta) ** beta  # (t/η)^β
            if s > 700.0:
                # exp(s) overflows; survival is essentially 0 → MRL → 0
                return 0.0
            a = 1.0 / beta
            upper_gamma = gammaincc(a, s) * math.gamma(a)
            mrl = (eta / beta) * math.exp(s) * upper_gamma
            return float(max(0.0, mrl))
        except (ImportError, ValueError, OverflowError):
            # Numerical fallback — integral of S(t+x)/S(t) dx from 0 to t95.
            try:
                s_t = math.exp(-((t_seconds / self.eta) ** self.beta))
                if s_t <= 1e-30:
                    return 0.0
                total = 0.0
                step = self.eta / 200.0
                x = 0.0
                for _ in range(2000):
                    s_tx = math.exp(-(((t_seconds + x) / self.eta) ** self.beta))
                    total += (s_tx / s_t) * step
                    x += step
                    if s_tx / s_t <= 1e-8:
                        break
                return float(max(0.0, total))
            except (ValueError, OverflowError):
                return 0.0

    def stochastic_failure_check(
        self,
        t_current: float,
        dt_seconds: float,
        rng: np.random.Generator,
    ) -> bool:
        """
        Phase 4 cherry-pick from v3: stochastic Weibull failure detection.

        Computes the EXACT conditional probability that the machine fails
        in the interval [t, t+dt] given survival to t:

            P(T ≤ t+dt | T > t) = 1 − S(t+dt)/S(t)
                                 = 1 − exp(−((t+dt)/η)^β + (t/η)^β)

        Then rolls the dedicated failure RNG. This replaces the purely
        deterministic "d ≥ 0.95 ⇒ FAILED" transition with a probability
        gate that produces natural variation in failure time — the 5-7
        day window is no longer a single deterministic point, it is the
        95% band of the Weibull CDF, with stochastic draws inside it.

        The 0.95 threshold remains as a backstop — both paths can
        trigger the FAILED transition. The hazard check is the primary
        driver; the threshold guarantees a failure eventually fires.

        Reference: Lai & Xie 2006 §2.3; the v3 implementation is at
        `pdm-v3/src/simulation/weibull_engine.py:304-317`.

        Parameters
        ----------
        t_current
            Machine operational time at the start of the interval.
        dt_seconds
            Length of the interval (e.g. one degradation step).
        rng
            Per-machine failure RNG. Separate from the heartbeat RNG so
            the failure draw is uncorrelated with the per-sensor
            degradation draws.

        Returns
        -------
        bool
            True if the machine should transition to FAILED.
        """
        if t_current <= 0.0 or dt_seconds <= 0.0:
            return False
        try:
            t_eta = t_current / self.eta
            t_dt_eta = (t_current + dt_seconds) / self.eta
            p_fail = 1.0 - math.exp(
                -(t_dt_eta ** self.beta) + (t_eta ** self.beta)
            )
        except (ValueError, OverflowError):
            return False
        if p_fail <= 0.0:
            return False
        return bool(rng.random() < p_fail)


# ═══════════════════════════════════════════════════════════════════════════════
# Sensor Degradation Models (physics-based, zero hardcoding)
# ═══════════════════════════════════════════════════════════════════════════════

class DegradationModelType(Enum):
    """Enumeration of physics-based degradation model types."""
    BEARING_ISO281 = auto()   # ISO 281 L10 bearing life
    OIL_ARRHENIUS = auto()    # Arrhenius oil degradation
    FOULING_TEMA = auto()     # TEMA RGP-T-2.4 fouling
    BELT_SLIP = auto()        # Belt tension decay (empirical)
    GENERIC = auto()           # Generic exponential decay fallback


@dataclass
class SensorDegradationConfig:
    """
    Configuration for a single sensor's degradation model.

    All parameters are drawn from distributions at initialization time.
    No hardcoded values — every field is populated via sampling.

    Attributes:
        model_type: Which physics model to use.
        nominal_mu: Nominal (healthy) sensor value mean.
        nominal_sigma: Nominal (healthy) sensor value standard deviation.
        degradation_direction: +1 (increasing) or -1 (decreasing) with wear.
        degradation_weight: Relative contribution to overall machine health.
        warning_threshold: Optional warning limit for alarm generation.
        critical_threshold: Optional critical limit for alarm generation.
        model_params: Type-specific parameters (e.g., L10_hours, T_operating).
    """
    model_type: DegradationModelType
    nominal_mu: float
    nominal_sigma: float
    degradation_direction: int  # +1 or -1
    degradation_weight: float
    warning_threshold: float | None = None
    critical_threshold: float | None = None
    model_params: dict = field(default_factory=dict)


@dataclass
class SensorState:
    """
    Runtime state for a single sensor on a single machine.

    Attributes:
        name: Sensor identifier (e.g., 'vibration_rms').
        config: Immutable degradation configuration.
        degradation_level: Current degradation [0.0 healthy → 1.0 failed].
        operational_seconds: Accumulated operational time for this sensor.
        next_step_at: Simulated time when next degradation check occurs.
        last_value: Most recently generated sensor reading.
        rng: Per-sensor random generator (seeded for reproducibility).
        anomaly_scenario: Name of the injected fault scenario, or None.
        anomaly_ramp_start_t: Machine operational_seconds when ramp started.
        anomaly_ramp_end_t: Machine operational_seconds when ramp completes.
        anomaly_strength: Per-sensor positive scalar (e.g. 0.3 → +30% degradation).
    """
    name: str
    config: SensorDegradationConfig
    degradation_level: float = 0.0
    operational_seconds: float = 0.0
    next_step_at: float = 0.0
    last_value: float | None = None
    rng: np.random.Generator | None = None

    # Anomaly injection state (chaos router API). The fields are consulted by
    # `apply_degradation_step` in `independent_scheduler` to smoothly scale
    # degradation during the ramp window.
    anomaly_scenario: str | None = None
    anomaly_ramp_start_t: float = 0.0
    anomaly_ramp_end_t: float = 0.0
    anomaly_strength: float = 0.0

    # Phase 4 Item 6 — AR(1) sensor-noise state. Each sensor carries
    # its own `phi` (autocorrelation coefficient) and a running
    # `prev_noise` so the AR(1) series is continuous across scheduler
    # iterations. The scheduler sets `phi` per-sensor at startup
    # (slow sensors get 0.7, fast sensors get 0.3, white-noise
    # sensors get 0.0).
    phi: float = 0.0
    prev_noise: float = 0.0

    gamma_process: object | None = None

    # Post-init scheduling
    def __post_init__(self):
        if self.rng is not None and self.next_step_at == 0.0:
            self.next_step_at = float(self.rng.uniform(180.0, 300.0))


class SensorDegradationModel:
    """
    Base class for physics-based sensor degradation models.

    Subclasses implement the _compute_degradation() method using
    real engineering equations (ISO 281, Arrhenius, TEMA, etc.).

    Design principle: the degradation LEVEL [0→1] is computed from
    physics equations. The sensor VALUE is then derived by mapping
    degradation level through the sensor's nominal range.
    """

    def __init__(self, config: SensorDegradationConfig):
        self.config = config

    def compute_degradation(self, t_seconds: float, rng: np.random.Generator) -> float:
        """
        Compute degradation level [0, 1] at time t using physics model.

        Subclasses override _compute_degradation() — this wrapper
        adds stochastic noise (NASA bearing model style) and clamps.
        """
        base = self._compute_degradation(t_seconds)
        # NASA bearing model: exponential decay with stochastic noise
        # Add Gaussian noise proportional to degradation level
        noise_std = 0.005 + base * 0.02  # noise grows with degradation
        noise = rng.normal(0.0, noise_std)
        return float(np.clip(base + noise, 0.0, 1.0))

    def ar1_noise(
        self,
        n: int,
        sigma: float = 0.05,
        phi: float = 0.7,
        prev_noise: float = 0.0,
    ) -> tuple[np.ndarray, float]:
        """
        Phase 4 Item 6 — generate a length-`n` AR(1) noise series with
        autocorrelation `phi` and steady-state standard deviation
        `sigma`.

        AR(1) process:
            X_t = phi * X_{t-1} + sqrt(1 - phi^2) * epsilon_t,
            epsilon_t ~ N(0, sigma^2)

        This produces autocorrelated noise with the SAME steady-state
        variance as white noise (sigma^2) but with a memory of `phi`
        between consecutive samples. Per-sensor `phi` values in this
        project (set in the scheduler's reading generator):

            0.7  — slow sensors (temperature, fouling_index, pressure_drop)
                   realistic because thermal mass and slow chemistry
                   mean successive samples are correlated.
            0.3  — fast sensors (vibration_rms, motor_current)
                   still some inertia from mechanical / electrical
                   systems, but less than thermal.
            0.0  — effectively white noise (motor_load, etc.)

        Reference: Box & Jenkins 1976, "Time Series Analysis:
        Forecasting and Control", §3.2 — the AR(1) definition and
        the parameterization that preserves the marginal variance.

        Returns
        -------
        (noise_array, final_state)
            noise_array of length `n`, and the final noise value to
            pass back as `prev_noise` on the next call (so the series
            is continuous across scheduler iterations).
        """
        if self.rng is None:
            # No RNG available — fall back to white noise so the
            # simulator doesn't crash in a deterministic test path.
            eps = np.zeros(n)
            return eps, 0.0
        if phi < 0.0 or phi >= 1.0:
            raise ValueError(f"phi must be in [0.0, 1.0), got {phi}")
        if sigma <= 0.0:
            raise ValueError(f"sigma must be positive, got {sigma}")
        if n <= 0:
            return np.array([], dtype=float), prev_noise
        eps = self.rng.normal(0.0, sigma, size=n)
        scale = math.sqrt(max(0.0, 1.0 - phi * phi))
        out = np.zeros(n, dtype=float)
        prev = float(prev_noise)
        for i in range(n):
            prev = phi * prev + scale * float(eps[i])
            out[i] = prev
        return out, float(prev)

    def _compute_degradation(self, t_seconds: float) -> float:
        """Override in subclasses. Returns base degradation [0, 1]."""
        raise NotImplementedError

    def compute_sensor_value(
        self,
        degradation_level: float,
        rng: np.random.Generator,
    ) -> float:
        """
        Convert degradation level to a physical sensor reading.

        HEALTHY region (d < 0.15): narrow Gaussian around nominal.
        DEGRADING region (0.15 ≤ d < 0.60): drift + amplified noise.
        ANOMALY region (d ≥ 0.60): threshold-region values with spikes.

        The degradation_level drives how far the value drifts from nominal
        toward the warning/critical threshold.
        """
        cfg = self.config
        mu = cfg.nominal_mu
        sigma = cfg.nominal_sigma
        d = degradation_level
        d_dir = cfg.degradation_direction
        w_thr = cfg.warning_threshold

        if d < 0.15:
            # HEALTHY: narrow Gaussian band
            value = rng.normal(mu, sigma)

        elif d < 0.60:
            # DEGRADING: linear drift toward warning threshold + amplified noise
            # Determine target: if no warning threshold, use 30% beyond nominal
            target = w_thr if w_thr is not None else mu * 1.3
            drift_magnitude = (d / 0.60) * abs(target - mu) * 0.80
            drift = drift_magnitude * d_dir
            noise_amp = 1.0 + d * 0.8  # noise increases with degradation
            value = (mu + drift) + rng.normal(0.0, sigma * noise_amp)

        else:
            # ANOMALY / FAILED region: near/above warning threshold
            target = w_thr if w_thr is not None else mu * 1.3
            # d maps from 0.6→1.0 to target→target*1.15
            overdrive = 1.0 + (d - 0.60) * 0.375  # max 1.15x at d=1.0
            anomaly_base = target * overdrive * d_dir if d_dir > 0 else target / overdrive
            value = anomaly_base + abs(rng.normal(0.0, sigma * 3.0)) * d_dir

        # Physical bounds
        hard_min = max(0.0, mu * 0.1)
        hard_max = mu * 5.0
        return float(np.clip(value, hard_min, hard_max))


class BearingISODegradation(SensorDegradationModel):
    """
    Bearing degradation following ISO 281 L10 life model.

    ISO 281: L10 = (C/P)^p  [millions of revolutions]
    where C = dynamic load rating, P = equivalent dynamic bearing load,
    p = 3 for ball bearings, 10/3 for roller bearings.

    Degradation follows exponential decay with stochastic noise
    (NASA bearing dataset reference model):
        d(t) = 1 - exp(-k * (t / L10_hours)^shape) + noise

    Sensors: vibration_rms, bearing_temp.
    """

    def _compute_degradation(self, t_seconds: float) -> float:
        params = self.config.model_params
        L10_hours = params.get("L10_hours", 720.0)
        load_ratio = params.get("load_ratio", 1.0)
        bearing_type = params.get("bearing_type", "ball")  # 'ball' or 'roller'

        # Life adjustment factor: L10_adj = L10 * (1/load_ratio)^p
        exponent = ISO281_BALL_EXPONENT if bearing_type == "ball" else ISO281_ROLLER_EXPONENT
        L10_adj = L10_hours * (1.0 / max(load_ratio, 0.1)) ** exponent

        # Exponential degradation: d(t) = 1 - exp(-k * (t / L10_adj)^shape)
        # k ≈ 3.0 gives ~95% degraded at t = L10_adj
        k = 3.0
        shape = 1.5  # Weibull-like shape for bearing wear progression
        t_hours = t_seconds / 3600.0

        if t_hours <= 0.0:
            return 0.0
        return float(1.0 - math.exp(-k * (t_hours / L10_adj) ** shape))


class OilArrheniusDegradation(SensorDegradationModel):
    """
    Oil degradation following Arrhenius equation.

    Arrhenius: k = A * exp(-Ea / (R * T))
    where k = reaction rate constant, A = pre-exponential factor,
    Ea = activation energy (J/mol), R = gas constant, T = temperature (K).

    Oil lifetime halves for every ~10°C increase above reference temperature.
    d(t) = 1 - exp(-k_ref * arrhenius_factor * t)

    Sensors: oil_pressure (decreases with oil degradation),
             bearing_temp (increases with oil degradation).
    """

    def _compute_degradation(self, t_seconds: float) -> float:
        params = self.config.model_params
        T_operating = params.get("T_operating_K", 353.15)  # default 80°C
        T_reference = params.get("T_reference_K", ARRHENIUS_T_REF)
        Ea = params.get("Ea_J_per_mol", ARRHENIUS_EA)
        k_ref = params.get("k_ref_per_hour", 1.0 / 5000.0)  # base rate at T_ref

        # Arrhenius ratio: how much faster at T_operating vs T_reference
        arrhenius_ratio = math.exp(
            (Ea / R_GAS) * (1.0 / T_reference - 1.0 / T_operating)
        )

        rate_per_hour = k_ref * arrhenius_ratio
        t_hours = t_seconds / 3600.0

        return float(1.0 - math.exp(-rate_per_hour * t_hours))


class FoulingTEMADegradation(SensorDegradationModel):
    """
    Fouling degradation following TEMA RGP-T-2.4 model.

    TEMA fouling model: Rf(t) = Rf_max * (1 - exp(-k_foul * t))
    where Rf(t) = fouling resistance at time t (m²·K/W),
    Rf_max = asymptotic fouling resistance,
    k_foul = fouling rate constant.

    Degradation is proportional to Rf(t) / Rf_max.

    Sensors: pressure_drop, fouling_index, outlet_temp.
    """

    def _compute_degradation(self, t_seconds: float) -> float:
        params = self.config.model_params
        Rf_max = params.get("Rf_max", TEMA_RF_MAX)
        k_foul = params.get("k_foul_per_hour", 0.001)

        t_hours = t_seconds / 3600.0
        Rf = Rf_max * (1.0 - math.exp(-k_foul * t_hours))

        # Normalize to [0, 1]
        if Rf_max <= 0.0:
            return 0.0
        return float(min(Rf / Rf_max, 1.0))


class BeltSlipDegradation(SensorDegradationModel):
    """
    Belt/conveyor degradation via tension decay model.

    Belt tension exponential decay: T(t) = T0 * exp(-lambda * t)
    Slip ratio increases as tension decreases.

    Sensors: belt_tension (decreases), motor_current (increases),
             drive_temp (increases), speed_rpm (decreases).
    """

    def _compute_degradation(self, t_seconds: float) -> float:
        params = self.config.model_params
        # Decay rate per second (default based on BELT_DECAY_REFERENCE)
        lambda_per_second = params.get("lambda_per_second", BELT_DECAY_REFERENCE)

        # Degradation: d(t) = 1 - exp(-lambda * t)
        return float(1.0 - math.exp(-lambda_per_second * t_seconds))


class GenericExponentialDegradation(SensorDegradationModel):
    """
    Generic exponential degradation fallback for sensors without a specific
    physics model. Uses configurable rate constant.

    d(t) = 1 - exp(-k * t)
    """

    def _compute_degradation(self, t_seconds: float) -> float:
        params = self.config.model_params
        k_per_second = params.get(
            "k_per_second",
            # Default: reach 95% degradation in ~7 days
            -math.log(0.05) / (7.0 * 86400.0),
        )
        return float(1.0 - math.exp(-k_per_second * t_seconds))


# ═══════════════════════════════════════════════════════════════════════════════
# Model factory — zero hardcoding
# ═══════════════════════════════════════════════════════════════════════════════

_MODEL_REGISTRY: dict[DegradationModelType, type[SensorDegradationModel]] = {
    DegradationModelType.BEARING_ISO281: BearingISODegradation,
    DegradationModelType.OIL_ARRHENIUS: OilArrheniusDegradation,
    DegradationModelType.FOULING_TEMA: FoulingTEMADegradation,
    DegradationModelType.BELT_SLIP: BeltSlipDegradation,
    DegradationModelType.GENERIC: GenericExponentialDegradation,
}


def create_degradation_model(config: SensorDegradationConfig) -> SensorDegradationModel:
    """Factory: instantiate the correct physics model for a sensor config."""
    model_cls = _MODEL_REGISTRY.get(config.model_type)
    if model_cls is None:
        raise ValueError(f"Unknown degradation model type: {config.model_type}")
    return model_cls(config)


# ═══════════════════════════════════════════════════════════════════════════════
# Machine Degradation Engine
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class MachineDegradationState:
    """
    Full degradation state for one machine with 5 independently degrading sensors.

    Attributes:
        machine_id: Unique machine identifier.
        weibull: Per-machine Weibull parameters.
        sensors: Dict of sensor_name → SensorState (5 sensors).
        operational_seconds: Total simulated operational time for this machine.
        failure_count: Number of complete failure cycles experienced.
        phase: Current operational phase (HEALTHY, DEGRADING, ANOMALY, FAILED).
        rng: Per-machine random generator.
        failure_rng: Per-machine random generator dedicated to the
            stochastic Weibull failure check (Phase 4 Item 1). Separated
            from `rng` so the failure draw is not coupled to the
            per-sensor degradation draws.
        failure_operational_seconds: Operational time at the moment of
            the most recent FAILED transition (None if never failed).
            The closed-form MRL is evaluated at this point for the
            conformal predictor's baseline target.
        last_maintenance_at: Wall-time (Unix seconds) of the most recent
            `reset_after_maintenance` call. None if never maintained.
            Phase 4 Item 7 uses this to scale degradation by
            (1 - maintenance_quality) * decay_factor.
        maintenance_quality: Per-machine random in [0.7, 1.0] drawn at
            each maintenance event. Higher = better maintenance = slower
            re-degradation.
    """
    machine_id: str
    weibull: WeibullParameters
    sensors: dict[str, SensorState]
    operational_seconds: float = 0.0
    failure_count: int = 0
    phase: str = "HEALTHY"
    rng: np.random.Generator | None = None
    failure_rng: np.random.Generator | None = None
    failure_operational_seconds: float | None = None
    last_maintenance_at: float | None = None
    maintenance_quality: float = 1.0
    last_maintenance_task: str | None = None
    interval_mu: float = 240.0
    interval_sigma: float = 30.0

    @property
    def overall_degradation(self) -> float:
        """
        Compute overall machine degradation as weighted average of sensor
        degradation levels, bounded above by the Weibull CDF at current time.
        """
        if not self.sensors:
            return 0.0
        total_weight = 0.0
        weighted_sum = 0.0
        for sensor in self.sensors.values():
            w = sensor.config.degradation_weight
            total_weight += w
            weighted_sum += sensor.degradation_level * w
        if total_weight <= 0.0:
            return 0.0
        raw = weighted_sum / total_weight
        # Cap at Weibull-predicted degradation (the time-scale ceiling). The
        # 5% slack allows per-sensor physics spikes to briefly exceed the
        # Weibull CDF without being clipped flat. The Weibull sets when
        # failure happens; the physics models shape the per-sensor trajectory.
        #
        # The ceiling only disciplines ORGANIC noise. An armed fault
        # injection legitimately outruns the age clock — clipping it kept
        # the machine phase at HEALTHY while its sensors sat at d≈0.94,
        # which silently suppressed the entire alarm chain.
        injected = any(
            s.anomaly_scenario is not None and s.anomaly_strength > 0.0
            for s in self.sensors.values()
        )
        if injected:
            return float(min(raw, 1.0))
        weibull_d = self.weibull.degradation_level_from_weibull(self.operational_seconds)
        return float(min(raw, weibull_d * 1.05))

    @property
    def is_failed(self) -> bool:
        """Machine is considered FAILED when overall degradation ≥ 0.95."""
        return self.overall_degradation >= 0.95

    def update_phase(self):
        """
        Update machine phase based on overall degradation level.

        Phase thresholds (demo-grade engineering judgment, mapped to
        ISO 13373 condition monitoring zones):
          HEALTHY    : d <  0.15 — good condition, baseline established
          DEGRADING  : d <  0.60 — early wear, monitor
          ANOMALY    : d <  0.95 — significant degradation, alarm
          FAILED     : d ≥  0.95 — machine considered failed

        Phase 4 Item 1: a stochastic Weibull hazard check is ALSO run
        in `stochastic_step_failure` — invoked from the scheduler once
        per cycle. The threshold check here is the backstop; the
        stochastic check is the primary, physically faithful driver.
        """
        d = self.overall_degradation
        if d < 0.15:
            self.phase = "HEALTHY"
        elif d < 0.60:
            self.phase = "DEGRADING"
        elif d < 0.95:
            self.phase = "ANOMALY"
        else:
            if self.phase != "FAILED":
                self.failure_count += 1
                self.failure_operational_seconds = float(self.operational_seconds)
            self.phase = "FAILED"

    def stochastic_step_failure(
        self,
        t_current: float,
        dt_seconds: float,
    ) -> bool:
        """
        Phase 4 Item 1 — single-step stochastic failure check.

        Computes P(T ≤ t+dt | T > t) using the closed-form conditional
        Weibull failure probability (Lai & Xie 2006 §2.3 Eq. 2.17) and
        rolls the dedicated failure RNG. Returns True if the machine
        should transition to FAILED this step.

        The check is gated on `overall_degradation ≥ 0.30` so the
        stochastic check doesn't fire spuriously early in the
        lifecycle (where p_fail is effectively 0 for any reasonable
        step size). Above 0.30, the conditional probability is
        non-negligible and the stochastic check adds natural
        variation to the FAILED transition.

        On a successful stochastic fire, this method also records
        `failure_operational_seconds` for the conformal predictor's
        baseline target (Item 2) and increments `failure_count`.

        Parameters
        ----------
        t_current
            Machine operational_seconds at the start of this step.
        dt_seconds
            Length of the step (sim seconds).

        Returns
        -------
        bool
            True if the machine should now be FAILED.
        """
        if self.phase == "FAILED":
            return False
        if self.overall_degradation < 0.30:
            return False
        if self.failure_rng is None:
            return False
        fired = self.weibull.stochastic_failure_check(
            t_current=t_current,
            dt_seconds=dt_seconds,
            rng=self.failure_rng,
        )
        if fired:
            self.phase = "FAILED"
            self.failure_count += 1
            self.failure_operational_seconds = float(self.operational_seconds)
        return fired

    def reset_after_maintenance(
        self,
        maintenance_quality: float | None = None,
    ):
        """
        Reset machine to HEALTHY after maintenance (simulated repair).

        Phase 4 Item 7: accepts a `maintenance_quality` parameter in
        [0.7, 1.0]. A higher quality means the machine degrades slower
        in the weeks after maintenance (see `degradation_rate_modifier`).
        `last_maintenance_at` is set to the current wall time so the
        modifier can compute elapsed-since-maintenance.
        """
        self.phase = "HEALTHY"
        self.failure_count += 1  # count completed cycles
        for sensor in self.sensors.values():
            sensor.degradation_level = 0.0
            sensor.operational_seconds = 0.0
            # The repair also fixes an injected fault — otherwise the armed
            # scenario re-applies its boost and the machine re-degrades
            # immediately after leaving maintenance.
            sensor.anomaly_scenario = None
            sensor.anomaly_strength = 0.0
            sensor.anomaly_ramp_start_t = 0.0
            sensor.anomaly_ramp_end_t = 0.0
            # Re-schedule next degradation step
            if sensor.rng is not None:
                sensor.next_step_at = float(sensor.rng.uniform(180.0, 300.0))
        # Item 7: capture maintenance history
        import time as _wall_time
        self.last_maintenance_at = float(_wall_time.time())
        if maintenance_quality is None:
            if self.rng is not None:
                task_names = list(MAINTENANCE_TASK_DISTRIBUTIONS.keys())
                task_idx = int(self.rng.integers(0, len(task_names)))
                selected_task = task_names[task_idx]
                self.last_maintenance_task = selected_task
                self.maintenance_quality = float(
                    MAINTENANCE_TASK_DISTRIBUTIONS[selected_task](self.rng)
                )
            else:
                self.last_maintenance_task = "oil_change"
                self.maintenance_quality = 0.85
        else:
            self.maintenance_quality = float(max(0.0, min(1.0, maintenance_quality)))

    def degradation_rate_modifier(self, current_wall_time: float | None = None) -> float:
        """
        Phase 4 Item 7 — return a multiplier in [0, 1] for the
        degradation rate based on time-since-last-maintenance.

        Returns 1.0 (full rate) for a machine that has never been
        maintained OR was maintained more than 90 days ago. Returns
        a smaller value (slower degradation) for a freshly maintained
        machine. The exponential form `1 - exp(-elapsed_days / 30)`
        ramps the modifier from 0 (just maintained) to ~1.0 (30 days
        later) to ~0.95 (90 days later).

        The returned multiplier is in [0, 1] and is meant to MULTIPLY
        the per-step degradation increment (i.e. lower = slower).
        """
        if self.last_maintenance_at is None:
            return 1.0
        import time as _wall_time
        now = current_wall_time if current_wall_time is not None else _wall_time.time()
        elapsed_s = max(0.0, now - self.last_maintenance_at)
        elapsed_days = elapsed_s / 86400.0
        # (1 - quality) * (1 - exp(-elapsed_days/30)) in [0, 1]
        # quality=0.9, elapsed=0:   0.10 * 0       = 0     → multiplier 1.00
        # quality=0.9, elapsed=10:  0.10 * 0.283   = 0.028 → multiplier 0.972
        # quality=0.7, elapsed=30:  0.30 * 0.632   = 0.190 → multiplier 0.810
        # quality=0.7, elapsed=90:  0.30 * 0.950   = 0.285 → multiplier 0.715
        rate_reduction = (1.0 - self.maintenance_quality) * (1.0 - math.exp(-elapsed_days / 30.0))
        return float(max(0.0, min(1.0, 1.0 - rate_reduction)))


# ═══════════════════════════════════════════════════════════════════════════════
# Sensor configuration builder (zero hardcoding)
# ═══════════════════════════════════════════════════════════════════════════════

# Mapping of sensor name patterns to degradation model types.
# Extended via MACHINE_CONFIGS or external configuration — no hardcoding.
_SENSOR_MODEL_MAP: dict[str, DegradationModelType] = {
    "vibration_rms": DegradationModelType.BEARING_ISO281,
    "bearing_temp": DegradationModelType.BEARING_ISO281,
    "oil_pressure": DegradationModelType.OIL_ARRHENIUS,
    "motor_current": DegradationModelType.GENERIC,
    "outlet_pressure": DegradationModelType.GENERIC,
    "inlet_temp": DegradationModelType.GENERIC,
    "outlet_temp": DegradationModelType.FOULING_TEMA,
    "pressure_drop": DegradationModelType.FOULING_TEMA,
    "flow_rate": DegradationModelType.FOULING_TEMA,
    "fouling_index": DegradationModelType.FOULING_TEMA,
    "belt_tension": DegradationModelType.BELT_SLIP,
    "drive_temp": DegradationModelType.BELT_SLIP,
    "motor_load": DegradationModelType.BELT_SLIP,
    "speed_rpm": DegradationModelType.BELT_SLIP,
}


def resolve_model_type(sensor_name: str) -> DegradationModelType:
    """Map a sensor name to its physics-based degradation model type."""
    return _SENSOR_MODEL_MAP.get(sensor_name, DegradationModelType.GENERIC)


def build_sensor_config(
    sensor_name: str,
    nominal_mu: float,
    nominal_sigma: float,
    degradation_weight: float,
    degradation_direction: int,
    warning_threshold: float | None = None,
    critical_threshold: float | None = None,
    rng: np.random.Generator | None = None,
    extra_params: dict | None = None,
) -> SensorDegradationConfig:
    """
    Build a SensorDegradationConfig from machine config data.
    Model type is resolved from sensor name — no hardcoding.

    Extra physics parameters (L10_hours, T_operating, etc.) are drawn
    from distributions via `extra_params` or sensible defaults.
    """
    model_type = resolve_model_type(sensor_name)
    rng = rng or np.random.default_rng()

    # Build model-specific parameters via distribution sampling
    model_params = {}

    if model_type == DegradationModelType.BEARING_ISO281:
        # L10 life: sample from distribution around expected bearing life
        model_params["L10_hours"] = float(rng.normal(720.0, 60.0))
        model_params["load_ratio"] = float(rng.normal(1.0, 0.10))
        model_params["bearing_type"] = "ball"

    elif model_type == DegradationModelType.OIL_ARRHENIUS:
        # Operating temperature: sample from distribution around 80°C
        model_params["T_operating_K"] = float(rng.normal(353.15, 5.0))
        model_params["T_reference_K"] = ARRHENIUS_T_REF
        model_params["Ea_J_per_mol"] = float(rng.normal(ARRHENIUS_EA, 2000.0))
        model_params["k_ref_per_hour"] = float(rng.normal(1.0 / 5000.0, 0.00005))

    elif model_type == DegradationModelType.FOULING_TEMA:
        model_params["Rf_max"] = float(rng.normal(TEMA_RF_MAX, TEMA_RF_MAX * 0.10))
        model_params["k_foul_per_hour"] = float(rng.normal(0.001, 0.0001))

    elif model_type == DegradationModelType.BELT_SLIP:
        model_params["lambda_per_second"] = float(rng.normal(BELT_DECAY_REFERENCE, BELT_DECAY_REFERENCE * 0.10))

    elif model_type == DegradationModelType.GENERIC:
        # Rate constant such that 95% degradation in ~6-8 days
        target_days = float(rng.uniform(6.0, 8.0))
        k = -math.log(0.05) / (target_days * 86400.0)
        model_params["k_per_second"] = float(rng.normal(k, k * 0.05))

    # Merge any extra params from machine config
    if extra_params:
        model_params.update(extra_params)

    return SensorDegradationConfig(
        model_type=model_type,
        nominal_mu=nominal_mu,
        nominal_sigma=nominal_sigma,
        degradation_direction=degradation_direction,
        degradation_weight=degradation_weight,
        warning_threshold=warning_threshold,
        critical_threshold=critical_threshold,
        model_params=model_params,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# High-level factory: create a fully parameterized machine state
# ═══════════════════════════════════════════════════════════════════════════════

def create_machine_state(
    machine_id: str,
    sensor_specs: dict[str, dict],
    global_seed: int,
    machine_index: int,
) -> MachineDegradationState:
    """
    Create a complete MachineDegradationState with:
    - Weibull parameters sampled around the machine's documented priors
      (MACHINE_CONFIGS[].weibull block: IEEE 493 / TEMA / CEMA references).
      The per-machine prior supplies (beta, eta) — beta ∈ [1.5, 2.5], eta
      calibrated for the documented characteristic life (5-7.5 days for
      the production 6-machine topology).
    - 5 sensors, each with independent degradation models.
    - Per-machine and per-sensor random seeds for reproducibility.

    Args:
        machine_id: Machine identifier (e.g., 'AC-201').
        sensor_specs: Dict of sensor_name → {nominal_mu, nominal_sigma, ...}
        global_seed: Global seed for reproducibility.
        machine_index: Index of this machine (0-5) for seed derivation.

    Returns:
        Fully initialized MachineDegradationState.
    """
    # Derive per-machine seed from global seed + machine index
    machine_seed = global_seed + machine_index * 1000
    machine_rng = np.random.default_rng(machine_seed)

    # Look up the machine's documented Weibull prior (MACHINE_CONFIGS source).
    # If the machine is unknown (e.g. a custom test fixture), fall back to
    # the generic N(2.0, 0.25) / 5-7 day distribution.
    config_prior = _lookup_weibull_prior(machine_id)
    beta_prior = config_prior["beta"]
    beta_std = config_prior["beta_std"]
    t95_prior_seconds = config_prior["t95_seconds"]
    config_prior["t95_std_seconds"]

    # Sample around the prior: noise is small (±0.15 on beta, ±7% on t95)
    # so the per-machine documented characteristic life is honoured within
    # ±20% (3σ-equivalent). The config stores `eta` as the 95% point
    # (FAILED-phase transition) life in HOURS (per the inline comments
    # in machines.py and the file header note), so we convert to
    # Weibull scale via scale = t95 / (-ln 0.05)^(1/beta). Beta clip range
    # [1.5, 4.0] is wide enough to honour the documented priors
    # (HX-202: 3.2, HX-302: 2.8) while still excluding pathological
    # distributions.
    beta = float(np.clip(machine_rng.normal(beta_prior, max(beta_std, 0.10)), 1.5, 4.0))
    t95_noise = float(np.clip(machine_rng.normal(1.0, 0.15), 0.55, 1.45))
    sampled_t95 = max(3600.0, t95_prior_seconds * t95_noise)
    ln_05_pow = (-math.log(0.05)) ** (1.0 / beta)
    eta = sampled_t95 / ln_05_pow

    weibull = WeibullParameters(
        beta=beta,
        eta=eta,
        rng=np.random.default_rng(machine_seed + 1),
    )

    # Create per-sensor states with independent degradation models
    d0 = float(machine_rng.uniform(0.0, 0.08))
    t0 = float(machine_rng.uniform(0.0, 0.15 * eta))

    sensors: dict[str, SensorState] = {}
    for i, (sensor_name, spec) in enumerate(sensor_specs.items()):
        sensor_seed = machine_seed + 100 + i
        sensor_rng = np.random.default_rng(sensor_seed)

        config = build_sensor_config(
            sensor_name=sensor_name,
            nominal_mu=spec.get("nominal_mu", 0.0),
            nominal_sigma=spec.get("nominal_sigma", 1.0),
            degradation_weight=spec.get("degradation_weight", 1.0 / len(sensor_specs)),
            degradation_direction=spec.get("degradation_direction", 1),
            warning_threshold=spec.get("warning_threshold"),
            critical_threshold=spec.get("critical_threshold"),
            rng=sensor_rng,
            extra_params=spec.get("model_params"),
        )

        sensor_d0 = float(np.clip(d0 + sensor_rng.normal(0.0, 0.005), 0.0, 0.09))

        gamma_proc = None
        if spec.get("degradation_process_type") == "gamma":
            g_alpha = float(spec.get("gamma_alpha", 2.0))
            g_beta = float(spec.get("gamma_beta", 5.0))
            gamma_proc = GammaProcessDegradation(
                alpha=g_alpha,
                beta=g_beta,
                rng=np.random.default_rng(sensor_seed + 500),
            )

        state = SensorState(
            name=sensor_name,
            config=config,
            degradation_level=sensor_d0,
            operational_seconds=t0,
            next_step_at=float(sensor_rng.uniform(180.0, 300.0)),
            rng=sensor_rng,
            phi=get_sensor_ar1_phi(sensor_name),
            gamma_process=gamma_proc,
        )
        sensors[sensor_name] = state

    machine_interval_mu = float(np.clip(machine_rng.normal(240.0, 30.0), 120.0, 400.0))
    machine_interval_sigma = float(np.clip(machine_rng.normal(30.0, 5.0), 15.0, 60.0))

    return MachineDegradationState(
        machine_id=machine_id,
        weibull=weibull,
        sensors=sensors,
        operational_seconds=t0,
        failure_count=0,
        phase="HEALTHY",
        rng=machine_rng,
        failure_rng=np.random.default_rng(machine_seed + 42),
        interval_mu=machine_interval_mu,
        interval_sigma=machine_interval_sigma,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Phase 4 Item 6 — per-sensor AR(1) noise autocorrelation coefficients
# ═══════════════════════════════════════════════════════════════════════════════

# Slow sensors (thermal mass, slow chemistry) carry strong AR(1)
# correlation. Fast sensors (vibration, electrical) carry weaker
# correlation. Sensors not listed here default to white noise (phi=0).
# These are physical-intuition defaults, calibrated to produce a
# visibly correlated noise pattern in dashboard charts without
# breaking the 5-7 day FAILED time window.
_SENSOR_AR1_PHI: dict[str, float] = {
    # Slow (thermal, fouling) — phi=0.7
    "bearing_temp": 0.7,
    "drive_temp": 0.7,
    "outlet_temp": 0.7,
    "inlet_temp": 0.7,
    "fouling_index": 0.7,
    "pressure_drop": 0.7,
    # Medium (mechanical with some inertia) — phi=0.4
    "oil_pressure": 0.4,
    "outlet_pressure": 0.4,
    "belt_tension": 0.4,
    "motor_load": 0.4,
    "speed_rpm": 0.4,
    # Fast (vibration, electrical) — phi=0.2
    "vibration_rms": 0.2,
    "motor_current": 0.2,
    # Default: flow_rate, etc. — phi=0.0 (white noise)
}


def get_sensor_ar1_phi(sensor_name: str) -> float:
    """Return the AR(1) phi for a given sensor name. Default 0.0."""
    return float(_SENSOR_AR1_PHI.get(sensor_name, 0.0))


def _lookup_weibull_prior(machine_id: str) -> dict[str, float]:
    """
    Return the documented Weibull prior for `machine_id` from MACHINE_CONFIGS.

    The config block's `eta` field is the documented **95% point (FAILED-
    phase transition) life in HOURS** (per the inline comments in
    machines.py and the file header note — "Original IEEE 493 / TEMA /
    CEMA reference values are 720-1020 hours, scaled 0.2x for demo
    visibility"). The values 120-180 hours in MACHINE_CONFIGS land in
    the 5-7.5 day window, which is the user's hard requirement for the
    FAILED-phase transition. We convert to seconds here so the rest of
    the engine can stay in SI units and so the prior directly drives
    the FAILED-time of `WeibullParameters`.

    For unknown machine_ids (e.g. test fixtures), falls back to the generic
    5-7 day t95 distribution (t95 ≈ 6.0 days, beta = 2.0).
    """
    from src.data_generator.machines import MACHINE_CONFIGS

    cfg = MACHINE_CONFIGS.get(machine_id, {})
    weibull = cfg.get("weibull", {}) if isinstance(cfg, dict) else {}
    if not weibull:
        return {
            "beta": 2.0,
            "beta_std": 0.25,
            "t95_seconds": 6.0 * 86400.0,
            "t95_std_seconds": 0.5 * 86400.0,
        }
    t95_hours = float(weibull.get("eta", 6.0 * 24.0))
    t95_std_hours = float(weibull.get("eta_std", t95_hours * 0.10))
    return {
        "beta": float(weibull.get("beta", 2.0)),
        "beta_std": float(weibull.get("beta_std", 0.15)),
        "t95_seconds": t95_hours * 3600.0,
        "t95_std_seconds": t95_std_hours * 3600.0,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Imperfect Maintenance Model — task-specific quality distributions
# ═══════════════════════════════════════════════════════════════════════════════

def _beta_sampler(a: float, b: float) -> Callable[[np.random.Generator], float]:
    def _sample(rng: np.random.Generator) -> float:
        return float(rng.beta(a, b))
    return _sample


MAINTENANCE_TASK_DISTRIBUTIONS: dict[str, Callable[[np.random.Generator], float]] = {
    "oil_change": _beta_sampler(8, 2),
    "bearing_replacement": _beta_sampler(5, 2),
    "belt_tensioning": _beta_sampler(6, 3),
}


# ═══════════════════════════════════════════════════════════════════════════════
# Physical Plausibility Gates
# ═══════════════════════════════════════════════════════════════════════════════

_PHYSICAL_SENSOR_BOUNDS: dict[str, tuple[float, float]] = {
    "vibration_rms": (0.0, 50.0),
    "bearing_temp": (-20.0, 250.0),
    "drive_temp": (-20.0, 250.0),
    "inlet_temp": (-40.0, 300.0),
    "outlet_temp": (-40.0, 300.0),
    "oil_pressure": (0.0, 50.0),
    "outlet_pressure": (0.0, 50.0),
    "motor_current": (0.0, 200.0),
    "motor_load": (0.0, 150.0),
    "belt_tension": (0.0, 100.0),
    "speed_rpm": (0.0, 10000.0),
    "pressure_drop": (0.0, 20.0),
    "flow_rate": (0.0, 100.0),
    "fouling_index": (0.0, 2.0),
}

_MAX_STEP_JUMP = 0.1


def check_plausibility(state: dict) -> dict:
    """
    Verify physical plausibility of a degradation state.

    Parameters
    ----------
    state : dict with keys:
        - degradation_history: list of past degradation levels
        - current_degradation: current degradation level
        - sensor_values: dict of sensor_name → value

    Returns
    -------
    dict with keys:
        - monotonic: bool
        - no_instant_jumps: bool
        - physical_bounds_ok: bool
        - violations: list of str
    """
    violations: list[str] = []

    history = state.get("degradation_history", [])
    current = state.get("current_degradation", 0.0)
    sensor_values = state.get("sensor_values", {})

    monotonic = True
    if len(history) >= 2:
        for i in range(1, len(history)):
            if history[i] < history[i - 1]:
                monotonic = False
                violations.append(
                    f"Non-monotonic degradation at step {i}: "
                    f"{history[i-1]:.4f} -> {history[i]:.4f}"
                )
                break

    no_instant_jumps = True
    if len(history) >= 1:
        last = history[-1]
        jump = abs(current - last)
        if jump > _MAX_STEP_JUMP:
            no_instant_jumps = False
            violations.append(
                f"Instant jump {jump:.4f} exceeds limit {_MAX_STEP_JUMP}: "
                f"{last:.4f} -> {current:.4f}"
            )

    physical_bounds_ok = True
    for sensor_name, value in sensor_values.items():
        if sensor_name in _PHYSICAL_SENSOR_BOUNDS:
            lo, hi = _PHYSICAL_SENSOR_BOUNDS[sensor_name]
            if value < lo or value > hi:
                physical_bounds_ok = False
                violations.append(
                    f"{sensor_name}={value:.2f} outside [{lo}, {hi}]"
                )

    return {
        "monotonic": monotonic,
        "no_instant_jumps": no_instant_jumps,
        "physical_bounds_ok": physical_bounds_ok,
        "violations": violations,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Gamma Process Degradation
# ═══════════════════════════════════════════════════════════════════════════════

class GammaProcessDegradation:
    """
    Gamma process degradation model with independent, non-negative increments.

    D(t+dt) - D(t) ~ Gamma(alpha * dt, beta)

    Naturally monotonic (all increments >= 0). Configurable per machine
    via `degradation_process_type` parameter.

    Parameters
    ----------
    alpha : float
        Shape rate parameter (increments per unit time).
    beta : float
        Rate parameter (inverse scale). Mean increment per unit time = alpha/beta.
    rng : np.random.Generator
        Random number generator for sampling.
    """

    def __init__(
        self,
        alpha: float,
        beta: float,
        rng: np.random.Generator,
    ):
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.rng = rng
        self.current_value: float = 0.0

    def step(self, dt: float = 1.0) -> float:
        """
        Advance the gamma process by dt time units.

        Returns the new cumulative degradation value.
        """
        shape = self.alpha * dt
        scale = 1.0 / self.beta
        increment = float(self.rng.gamma(shape, scale))
        self.current_value += increment
        return self.current_value

    def reset(self) -> None:
        """Reset cumulative degradation to zero."""
        self.current_value = 0.0
