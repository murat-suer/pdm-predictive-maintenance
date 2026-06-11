
from src.ml.mhi_calculator import MHICalculator, reliability_score


class TestReliabilityScore:
    def test_at_baseline_is_one(self):
        assert reliability_score(0.05) == 1.0

    def test_below_baseline_is_one(self):
        assert reliability_score(0.02) == 1.0

    def test_at_baseline_plus_tolerance_is_zero(self):
        assert reliability_score(0.05 + 0.15) == 0.0

    def test_above_tolerance_is_zero(self):
        assert reliability_score(0.50) == 0.0

    def test_midpoint(self):
        score = reliability_score(0.125)
        assert 0.0 < score < 1.0


class TestMHICalculator:
    def test_healthy_machine_high_mhi(self):
        calc = MHICalculator()
        result = calc.compute(
            machine_id="AC-201",
            phase="RUNNING",
            degradation_level=0.05,
            recent_anomaly_count=1,
            recent_readings_count=100,
            recent_downtime_minutes=0.0,
        )
        assert result["health_score"] > 0.70
        assert result["classification"] in ("Excellent", "Good")

    def test_failed_phase_zero_mhi(self):
        calc = MHICalculator()
        result = calc.compute(
            machine_id="AC-201",
            phase="FAILED",
            degradation_level=0.9,
            recent_anomaly_count=50,
            recent_readings_count=100,
        )
        assert result["health_score"] == 0.0

    def test_anomaly_phase_capped_at_0_60(self):
        calc = MHICalculator()
        result = calc.compute(
            machine_id="AC-201",
            phase="ANOMALY",
            degradation_level=0.1,
            recent_anomaly_count=2,
            recent_readings_count=100,
        )
        assert result["health_score"] <= 0.60

    def test_idle_phase_penalizes_availability(self):
        calc = MHICalculator()
        running = calc.compute(
            machine_id="A",
            phase="RUNNING",
            degradation_level=0.1,
            recent_anomaly_count=0,
            recent_readings_count=100,
        )
        calc2 = MHICalculator()
        idle = calc2.compute(
            machine_id="B",
            phase="IDLE",
            degradation_level=0.1,
            recent_anomaly_count=0,
            recent_readings_count=100,
        )
        assert idle["availability_score"] < running["availability_score"]

    def test_ema_smoothing(self):
        calc = MHICalculator(ema_alpha=0.3)
        first = calc.compute(
            machine_id="AC-201",
            phase="RUNNING",
            degradation_level=0.05,
            recent_anomaly_count=0,
            recent_readings_count=100,
        )
        second = calc.compute(
            machine_id="AC-201",
            phase="RUNNING",
            degradation_level=0.8,
            recent_anomaly_count=50,
            recent_readings_count=100,
        )
        assert second["health_score"] > 0.0

    def test_mhi_formula_components(self):
        calc = MHICalculator()
        result = calc.compute(
            machine_id="AC-201",
            phase="RUNNING",
            degradation_level=0.2,
            recent_anomaly_count=5,
            recent_readings_count=100,
        )
        expected = round(
            result["availability_score"]
            * result["reliability_score"]
            * result["condition_score"],
            4,
        )
        assert abs(result["health_score"] - expected) < 0.01

    def test_zero_readings_no_crash(self):
        calc = MHICalculator()
        result = calc.compute(
            machine_id="AC-201",
            phase="RUNNING",
            degradation_level=0.1,
            recent_anomaly_count=0,
            recent_readings_count=0,
        )
        assert 0.0 <= result["health_score"] <= 1.0

    def test_classification_labels(self):
        calc = MHICalculator()
        assert calc._classify(0.90) == "Excellent"
        assert calc._classify(0.75) == "Good"
        assert calc._classify(0.60) == "Degrading"
        assert calc._classify(0.30) == "Critical — Action Required"


class TestReliabilityMinSamples:
    """Small post-maintenance buffers must not report a fake 0% reliability."""

    def test_below_min_samples_reports_none(self):
        calc = MHICalculator()
        result = calc.compute(
            machine_id="AC-201",
            phase="RUNNING",
            degradation_level=0.05,
            recent_anomaly_count=2,
            recent_readings_count=9,  # 22% anomaly rate on 9 readings
        )
        assert result["reliability_score"] is None
        # reliability is neutral in the product, not zero
        assert result["health_score"] > 0.5

    def test_at_min_samples_reports_value(self):
        calc = MHICalculator()
        result = calc.compute(
            machine_id="AC-202",
            phase="RUNNING",
            degradation_level=0.05,
            recent_anomaly_count=0,
            recent_readings_count=MHICalculator.MIN_RELIABILITY_SAMPLES,
        )
        assert result["reliability_score"] == 1.0


class TestEventBasedReliability:
    """Reliability follows confirmed anomaly events, not raw detector flags."""

    def test_quiet_machine_reads_full_reliability(self):
        calc = MHICalculator()
        result = calc.compute(
            machine_id="AC-201",
            phase="RUNNING",
            degradation_level=0.05,
            # noisy raw detector: would read 0% under the rate model
            recent_anomaly_count=25,
            recent_readings_count=30,
            confirmed_events=0,
        )
        assert result["reliability_score"] == 1.0

    def test_each_confirmed_event_docks_ten_percent(self):
        calc = MHICalculator()
        result = calc.compute(
            machine_id="CM-303",
            phase="ANOMALY",
            degradation_level=0.5,
            recent_anomaly_count=0,
            recent_readings_count=30,
            confirmed_events=3,
        )
        assert result["reliability_score"] == 0.7

    def test_reliability_floors_at_zero(self):
        calc = MHICalculator()
        result = calc.compute(
            machine_id="CM-303",
            phase="ANOMALY",
            degradation_level=0.5,
            recent_anomaly_count=0,
            recent_readings_count=30,
            confirmed_events=15,
        )
        assert result["reliability_score"] == 0.0
