"""
src/data_generator/fft_engine.py
=================================
FFT harmonic frequency generation for bearing fault detection.

Implements bearing characteristic frequencies per ISO 15243 / CWRU methodology:
  BPFO — Ball Pass Frequency, Outer race
  BPFI — Ball Pass Frequency, Inner race
  BSF  — Ball Spin Frequency

Amplitudes scale with degradation_level * coefficient, matching the
CWRU bearing dataset pattern where fault harmonics grow with defect severity.
"""

from __future__ import annotations

import math

import numpy as np


def compute_bearing_frequencies(
    N_balls: int,
    rpm: float,
    ball_dia: float,
    pitch_dia: float,
    contact_angle: float = 0.0,
) -> dict[str, float]:
    """
    Compute bearing characteristic frequencies (Hz).

    Parameters
    ----------
    N_balls
        Number of rolling elements.
    rpm
        Shaft rotational speed (revolutions per minute).
    ball_dia
        Ball/rolling element diameter (any consistent unit).
    pitch_dia
        Bearing pitch diameter (same unit as ball_dia).
    contact_angle
        Contact angle in radians (0 for deep groove ball bearing).

    Returns
    -------
    dict with keys: BPFO, BPFI, BSF (all in Hz)

    References
    ----------
    - ISO 15243-2:2017 — Rolling bearings — damage and failures
    - CWRU Bearing Data Center — fault frequency definitions
    """
    freq_shaft = rpm / 60.0
    bd_pd = ball_dia / pitch_dia
    cos_alpha = math.cos(contact_angle)

    bpfo = (N_balls / 2.0) * freq_shaft * (1.0 - bd_pd * cos_alpha)
    bpfi = (N_balls / 2.0) * freq_shaft * (1.0 + bd_pd * cos_alpha)
    bsf = (pitch_dia / ball_dia) * freq_shaft * (1.0 - (bd_pd * cos_alpha) ** 2)

    return {
        "BPFO": float(bpfo),
        "BPFI": float(bpfi),
        "BSF": float(bsf),
    }


def generate_fft_data(
    N_balls: int,
    rpm: float,
    ball_dia: float,
    pitch_dia: float,
    contact_angle: float,
    degradation_level: float,
    BPFO_coeff: float,
    BPFI_coeff: float,
    BSF_coeff: float,
    rng: np.random.Generator | None = None,
) -> dict[str, float]:
    """
    Generate FFT harmonic data with amplitudes scaled by degradation.

    Parameters
    ----------
    N_balls, rpm, ball_dia, pitch_dia, contact_angle
        Bearing geometry and operating speed.
    degradation_level
        Current bearing degradation [0.0, 1.0].
    BPFO_coeff, BPFI_coeff, BSF_coeff
        Amplitude scaling coefficients per harmonic.
    rng
        Optional RNG for stochastic noise on amplitudes.

    Returns
    -------
    dict with keys:
        BPFO, BPFI, BSF — frequencies (Hz)
        BPFO_amplitude, BPFI_amplitude, BSF_amplitude — scaled amplitudes
    """
    freqs = compute_bearing_frequencies(
        N_balls=N_balls,
        rpm=rpm,
        ball_dia=ball_dia,
        pitch_dia=pitch_dia,
        contact_angle=contact_angle,
    )

    d = max(0.0, min(1.0, degradation_level))

    noise = 0.0
    if rng is not None:
        noise = float(rng.normal(0.0, 0.02))

    bpfo_amp = BPFO_coeff * d * (1.0 + noise)
    bpfi_amp = BPFI_coeff * d * (1.0 + noise)
    bsf_amp = BSF_coeff * d * (1.0 + noise)

    return {
        "BPFO": freqs["BPFO"],
        "BPFI": freqs["BPFI"],
        "BSF": freqs["BSF"],
        "BPFO_amplitude": max(0.0, bpfo_amp),
        "BPFI_amplitude": max(0.0, bpfi_amp),
        "BSF_amplitude": max(0.0, bsf_amp),
    }
