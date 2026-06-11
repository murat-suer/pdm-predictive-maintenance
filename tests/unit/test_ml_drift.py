from datetime import UTC, datetime
from unittest.mock import MagicMock


def test_anomaly_rate_below_threshold_no_drift():
    from src.ml.model_health import DriftDetector
    dd = DriftDetector()
    rate = 5 / 100
    drift = rate > dd.ANOMALY_RATE_THRESHOLD
    assert drift is False
    assert rate == 0.05


def test_anomaly_rate_above_threshold_drift():
    from src.ml.model_health import DriftDetector
    dd = DriftDetector()
    rate = 20 / 100
    drift = rate > dd.ANOMALY_RATE_THRESHOLD
    assert drift is True


def test_operator_alignment_no_overrides():
    from src.ml.model_health import DriftDetector
    dd = DriftDetector()
    decisions = [MagicMock(overridden=False) for _ in range(10)]
    db = MagicMock()
    db.query.return_value.filter.return_value.all.return_value = decisions
    result = dd.check_operator_alignment("AC-201", db)
    assert result["drift_detected"] is False
    assert result["operator_alignment_pct"] == 100.0


def test_operator_alignment_high_override_rate():
    from src.ml.model_health import DriftDetector
    dd = DriftDetector()
    decisions = [MagicMock(overridden=True) for _ in range(4)] + [MagicMock(overridden=False) for _ in range(6)]
    db = MagicMock()
    db.query.return_value.filter.return_value.all.return_value = decisions
    result = dd.check_operator_alignment("AC-201", db)
    assert result["drift_detected"] is True
    assert result["override_count"] == 4


def test_maintenance_validation_high_fp_rate():
    from src.ml.model_health import DriftDetector
    dd = DriftDetector()
    logs = [MagicMock(fault_found=False, performed_at=datetime.now(UTC)) for _ in range(3)] + [
        MagicMock(fault_found=True, performed_at=datetime.now(UTC)) for _ in range(2)
    ]
    db = MagicMock()
    db.query.return_value.filter.return_value.all.return_value = logs
    result = dd.check_maintenance_validation("AC-201", db)
    assert result["drift_detected"] is True
    assert result["fp_count"] == 3


def test_no_decisions_returns_no_drift():
    from src.ml.model_health import DriftDetector
    dd = DriftDetector()
    db = MagicMock()
    db.query.return_value.filter.return_value.all.return_value = []
    result = dd.check_operator_alignment("AC-201", db)
    assert result["drift_detected"] is False
    assert result["total_decisions"] == 0
