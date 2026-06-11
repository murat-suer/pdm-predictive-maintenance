"""Physics-informed diagnostic features for simulated industrial machines.

References:
- ISO 13373-2 vibration condition monitoring guidance
- Randall and Antoni (2011), bearing envelope analysis survey
- TEMA heat-exchanger fouling resistance conventions
"""

from __future__ import annotations

import numpy as np
import polars as pl
from numpy.typing import NDArray
from scipy.signal import butter, hilbert, sosfiltfilt


def rolling_kurtosis(series: pl.Series, window: int = 30) -> pl.Series:
    """Return rolling excess kurtosis for bearing impact trend detection."""
    if hasattr(series, "rolling_kurtosis"):
        return series.rolling_kurtosis(window_size=window)
    return _manual_rolling_kurtosis(series, window)


def _manual_rolling_kurtosis(series: pl.Series, window: int) -> pl.Series:
    values = series.to_numpy()
    out = np.full(len(values), np.nan, dtype=float)
    for index in range(window - 1, len(values)):
        sample = values[index - window + 1 : index + 1]
        std = sample.std()
        out[index] = 0.0 if std == 0 else ((sample - sample.mean()) ** 4).mean() / (std**4) - 3
    return pl.Series(out)


def crest_factor(series: pl.Series, window: int = 30) -> pl.Series:
    """Return rolling peak-to-RMS ratio for impulsive vibration signatures."""
    values = series.to_numpy()
    out = np.full(len(values), np.nan, dtype=float)
    for index in range(window - 1, len(values)):
        sample = values[index - window + 1 : index + 1]
        rms = np.sqrt(np.mean(sample**2))
        out[index] = 0.0 if rms == 0 else np.max(np.abs(sample)) / rms
    return pl.Series(out)


def envelope_amplitude(
    vibration: NDArray[np.float64],
    sample_rate_hz: float = 10.0,
    band: tuple[float, float] = (1.0, 4.0),
) -> float:
    """Return mean Hilbert envelope amplitude inside a diagnostic trend band.

    The simulator's slow trend stream cannot recover real bearing harmonics;
    live deployments need high-rate accelerometers for BPFO/BPFI analysis.
    """
    if len(vibration) < 8:
        return 0.0
    sos = butter(4, band, btype="bandpass", fs=sample_rate_hz, output="sos")
    filtered = sosfiltfilt(sos, vibration)
    return float(np.abs(hilbert(filtered)).mean())


def delta_p_per_flow(pressure_drop: pl.Series, flow_rate: pl.Series) -> pl.Series:
    """Return pressure-drop over flow-squared fouling resistance indicator."""
    flow_safe = flow_rate.fill_null(1e-6).clip(lower_bound=1e-6)
    return pressure_drop / (flow_safe**2)


def belt_slip_ratio(motor_rpm: pl.Series, output_rpm: pl.Series, pulley_ratio: float) -> pl.Series:
    """Return expected-versus-actual output RPM slip fraction."""
    expected = motor_rpm * pulley_ratio
    return (expected - output_rpm) / expected.clip(lower_bound=1e-6)


def oil_consumption_rate(level_series: pl.Series, window: int = 60) -> pl.Series:
    """Return rolling oil-level slope; negative values indicate consumption."""
    values = level_series.to_numpy()
    out = np.full(len(values), np.nan, dtype=float)
    for index in range(window - 1, len(values)):
        sample = values[index - window + 1 : index + 1]
        out[index] = np.polyfit(np.arange(len(sample)), sample, 1)[0]
    return pl.Series(out)
