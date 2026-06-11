import pytest

from src.ml.cusum_detector import CUSUMDetector


class TestCUSUMDetectorInit:
    def test_default_params(self):
        det = CUSUMDetector()
        assert det.k == 0.5
        assert det.h == 5.0
        assert det.cusum_pos == 0.0
        assert det.cusum_neg == 0.0

    def test_custom_params(self):
        det = CUSUMDetector(k=1.0, h=10.0)
        assert det.k == 1.0
        assert det.h == 10.0


class TestFromBaselineSigma:
    def test_valid_sigma(self):
        det = CUSUMDetector.from_baseline_sigma(2.0)
        assert det.k == 1.0
        assert det.h == 10.0

    def test_sigma_zero_raises(self):
        with pytest.raises(ValueError, match="baseline sigma must be positive"):
            CUSUMDetector.from_baseline_sigma(0)

    def test_sigma_negative_raises(self):
        with pytest.raises(ValueError, match="baseline sigma must be positive"):
            CUSUMDetector.from_baseline_sigma(-1.0)


class TestUpdate:
    def test_no_alarm_on_small_drift(self):
        det = CUSUMDetector(k=0.5, h=5.0)
        result = det.update(value=10.1, target=10.0, sigma=1.0)
        assert result["alarm"] is False

    def test_alarm_triggers_on_sustained_shift(self):
        det = CUSUMDetector(k=0.5, h=5.0)
        result = None
        for _ in range(50):
            result = det.update(value=15.0, target=10.0, sigma=1.0)
        assert result["alarm"] is True

    def test_reset_after_alarm(self):
        det = CUSUMDetector(k=0.5, h=5.0)
        for _ in range(50):
            det.update(value=15.0, target=10.0, sigma=1.0)
        assert det.cusum_pos == 0.0
        assert det.cusum_neg == 0.0

    def test_sigma_zero_returns_no_alarm(self):
        det = CUSUMDetector()
        result = det.update(value=100.0, target=10.0, sigma=0)
        assert result["alarm"] is False
        assert result["score"] == 0.0

    def test_sigma_negative_returns_no_alarm(self):
        det = CUSUMDetector()
        result = det.update(value=100.0, target=10.0, sigma=-1.0)
        assert result["alarm"] is False


class TestReset:
    def test_reset_clears_accumulators(self):
        det = CUSUMDetector(k=0.5, h=5.0)
        det.update(value=11.0, target=10.0, sigma=1.0)
        assert det.cusum_pos > 0
        det.reset()
        assert det.cusum_pos == 0.0
        assert det.cusum_neg == 0.0

    def test_reset_on_fresh_detector(self):
        det = CUSUMDetector()
        det.reset()
        assert det.cusum_pos == 0.0
        assert det.cusum_neg == 0.0
