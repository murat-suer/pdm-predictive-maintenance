"""Deterministic calibration buffer with isolated numpy RNG per sensor."""
from __future__ import annotations

import numpy as np


class CalibrationBuffer:
    """Deterministic calibration buffer with isolated RNG per sensor.

    Each sensor gets an independent but reproducible random number generator
    seeded from: global_seed + machine_index * 1000 + sensor_offset.
    """

    def __init__(self, machine_index: int, sensor_offset: int, global_seed: int = 42):
        seed = global_seed + (machine_index * 1000) + sensor_offset
        self.rng = np.random.default_rng(seed)

    def generate_synthetic_data(self, mu: float, sigma: float, samples: int = 10800) -> np.ndarray:
        """Generate deterministic synthetic calibration data."""
        return self.rng.normal(loc=mu, scale=sigma, size=samples)
