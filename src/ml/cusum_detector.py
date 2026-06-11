"""CUSUM change-point detection for univariate sensor streams."""


class CUSUMDetector:
    """Detect small sustained mean shifts in a single sensor signal."""

    def __init__(self, k=0.5, h=5.0):
        self.k = k
        self.h = h
        self.cusum_pos = 0.0
        self.cusum_neg = 0.0

    @classmethod
    def from_baseline_sigma(cls, sigma: float) -> "CUSUMDetector":
        """Derive raw thresholds from one nominal sensor standard deviation."""
        if sigma <= 0:
            raise ValueError("baseline sigma must be positive")
        return cls(k=0.5 * sigma, h=5.0 * sigma)

    def update(self, value, target, sigma):
        """Feed one observation and report whether its drift is sustained."""
        if sigma <= 0:
            return {"alarm": False, "cusum_pos": self.cusum_pos, "cusum_neg": self.cusum_neg, "score": 0.0}
        residual = value - target
        self.cusum_pos = max(0.0, self.cusum_pos + residual - self.k)
        self.cusum_neg = max(0.0, self.cusum_neg - residual - self.k)
        alarm = self.cusum_pos > self.h or self.cusum_neg > self.h
        score = max(self.cusum_pos, self.cusum_neg) / max(sigma, 1e-9)
        if alarm:
            self.reset()
        return {
            "alarm": alarm,
            "cusum_pos": round(self.cusum_pos, 4),
            "cusum_neg": round(self.cusum_neg, 4),
            "score": round(score, 4),
        }

    def reset(self):
        """Reset cumulative sums to zero."""
        self.cusum_pos = 0.0
        self.cusum_neg = 0.0
