"""
Unit tests for src.ml.ai_act_compliance — AI Act compliance logger.

Tests cover:
- Initialization
- Logging compliance events (DB success and fallback)
- Audit trail retrieval
- Edge cases: missing data, invalid events, empty results
"""

from datetime import UTC, datetime
from unittest.mock import MagicMock, patch

from src.ml.ai_act_compliance import AIActComplianceLogger, ComplianceLogger


class TestAIActComplianceLoggerInit:
    """Test initialization of AIActComplianceLogger."""

    def test_init_creates_empty_fallback(self):
        logger = AIActComplianceLogger()
        assert logger._fallback_entries == []

    def test_compliance_logger_alias(self):
        """ComplianceLogger should be an alias for AIActComplianceLogger."""
        assert ComplianceLogger is AIActComplianceLogger


class TestLogDecision:
    """Test log_decision method."""

    def test_log_decision_db_success(self):
        """When DB is available, entry should be written to DB, not fallback."""
        logger = AIActComplianceLogger()
        mock_db = MagicMock()
        mock_session = MagicMock()
        mock_session.__enter__ = MagicMock(return_value=mock_db)
        mock_session.__exit__ = MagicMock(return_value=False)

        with patch("src.ml.ai_act_compliance.get_db_context", return_value=mock_session, create=True):
            # Patch the lazy imports inside log_decision
            with patch.dict("sys.modules", {
                "src.database.connection": MagicMock(get_db_context=lambda: mock_session),
                "src.database.models": MagicMock(AIActLog=MagicMock()),
            }):
                logger.log_decision(
                    machine_id="AC-201",
                    model_version="isolation_forest_v1",
                    features_snapshot={"vibration": 4.5, "temperature": 72.1},
                    output={"anomaly_score": 0.85, "is_anomaly": True},
                    shap_values={"vibration": 0.6, "temperature": 0.25},
                    action="ANOMALY_RAISED",
                    human_action=None,
                    decision_type="anomaly_detection",
                )

        # Should NOT be in fallback since DB succeeded
        assert len(logger._fallback_entries) == 0

    def test_log_decision_db_failure_uses_fallback(self):
        """When DB write fails, entry should be stored in fallback list."""
        logger = AIActComplianceLogger()

        # Force the import to fail
        with patch.dict("sys.modules", {
            "src.database.connection": None,
            "src.database.models": None,
        }):
            logger.log_decision(
                machine_id="AC-201",
                model_version="isolation_forest_v1",
                features_snapshot={"vibration": 4.5},
                output={"anomaly_score": 0.85},
                shap_values={"vibration": 0.6},
                action="ANOMALY_RAISED",
            )

        # Should be in fallback
        assert len(logger._fallback_entries) == 1
        entry = logger._fallback_entries[0]
        assert entry["machine_id"] == "AC-201"
        assert entry["model_version"] == "isolation_forest_v1"
        assert entry["action_taken"] == "ANOMALY_RAISED"
        assert entry["decision_type"] == "anomaly_detection"
        assert "timestamp" in entry

    def test_log_decision_with_human_action(self):
        """Human action override should be recorded."""
        logger = AIActComplianceLogger()

        with patch.dict("sys.modules", {
            "src.database.connection": None,
            "src.database.models": None,
        }):
            logger.log_decision(
                machine_id="AC-302",
                model_version="rul_voting_v2",
                features_snapshot={"vibration": 3.2},
                output={"rul_days": 45},
                shap_values={"vibration": 0.4},
                action="MAINTENANCE_SCHEDULED",
                human_action="OVERRIDE_IGNORED",
                decision_type="rul_prediction",
            )

        entry = logger._fallback_entries[0]
        assert entry["human_action"] == "OVERRIDE_IGNORED"
        assert entry["decision_type"] == "rul_prediction"

    def test_log_decision_default_decision_type(self):
        """Default decision_type should be 'anomaly_detection'."""
        logger = AIActComplianceLogger()

        with patch.dict("sys.modules", {
            "src.database.connection": None,
            "src.database.models": None,
        }):
            logger.log_decision(
                machine_id="AC-101",
                model_version="v1",
                features_snapshot={},
                output={},
                shap_values={},
                action="NO_ACTION",
            )

        entry = logger._fallback_entries[0]
        assert entry["decision_type"] == "anomaly_detection"

    def test_log_decision_empty_features(self):
        """Should handle empty features/output/shap dicts."""
        logger = AIActComplianceLogger()

        with patch.dict("sys.modules", {
            "src.database.connection": None,
            "src.database.models": None,
        }):
            logger.log_decision(
                machine_id="AC-101",
                model_version="v1",
                features_snapshot={},
                output={},
                shap_values={},
                action="NO_ACTION",
            )

        assert len(logger._fallback_entries) == 1
        assert logger._fallback_entries[0]["features_snapshot"] == {}

    def test_log_decision_none_human_action(self):
        """human_action=None should be stored as None."""
        logger = AIActComplianceLogger()

        with patch.dict("sys.modules", {
            "src.database.connection": None,
            "src.database.models": None,
        }):
            logger.log_decision(
                machine_id="AC-101",
                model_version="v1",
                features_snapshot={"x": 1},
                output={"score": 0.5},
                shap_values={"x": 0.5},
                action="NO_ACTION",
                human_action=None,
            )

        assert logger._fallback_entries[0]["human_action"] is None

    def test_multiple_log_decisions(self):
        """Multiple log calls should accumulate in fallback."""
        logger = AIActComplianceLogger()

        with patch.dict("sys.modules", {
            "src.database.connection": None,
            "src.database.models": None,
        }):
            for i in range(5):
                logger.log_decision(
                    machine_id=f"AC-{i:03d}",
                    model_version="v1",
                    features_snapshot={"x": i},
                    output={"score": i * 0.1},
                    shap_values={"x": 0.5},
                    action="NO_ACTION",
                )

        assert len(logger._fallback_entries) == 5
        assert logger._fallback_entries[0]["machine_id"] == "AC-000"
        assert logger._fallback_entries[4]["machine_id"] == "AC-004"


class TestGetLast:
    """Test get_last method."""

    def test_get_last_from_fallback(self):
        """When DB is unavailable, get_last should return from fallback."""
        logger = AIActComplianceLogger()

        # Populate fallback
        with patch.dict("sys.modules", {
            "src.database.connection": None,
            "src.database.models": None,
        }):
            for i in range(5):
                logger.log_decision(
                    machine_id=f"AC-{i:03d}",
                    model_version="v1",
                    features_snapshot={"x": i},
                    output={"score": i * 0.1},
                    shap_values={"x": 0.5},
                    action="NO_ACTION",
                )

        # get_last with DB failing should return fallback entries
        results = logger.get_last(n=3)
        assert len(results) == 3
        # Should return last 3 entries
        assert results[0]["machine_id"] == "AC-002"
        assert results[2]["machine_id"] == "AC-004"

    def test_get_last_empty_fallback(self):
        """When fallback is empty and DB fails, return empty list."""
        logger = AIActComplianceLogger()
        results = logger.get_last(n=10)
        assert results == []

    def test_get_last_n_larger_than_fallback(self):
        """Requesting more entries than available returns all available."""
        logger = AIActComplianceLogger()

        with patch.dict("sys.modules", {
            "src.database.connection": None,
            "src.database.models": None,
        }):
            logger.log_decision(
                machine_id="AC-001",
                model_version="v1",
                features_snapshot={},
                output={},
                shap_values={},
                action="NO_ACTION",
            )

        results = logger.get_last(n=100)
        assert len(results) == 1

    def test_get_last_with_db_session(self):
        """When db_session is provided, should use it for querying."""
        logger = AIActComplianceLogger()
        mock_session = MagicMock()

        # Mock the AIActLog model class with timestamp column attribute
        mock_ai_act_log_class = MagicMock()
        mock_ai_act_log_class.timestamp = MagicMock()
        mock_ai_act_log_class.timestamp.desc.return_value = "timestamp_desc"

        # Mock a row result
        mock_row = MagicMock()
        mock_row.timestamp = datetime(2024, 1, 1, tzinfo=UTC)
        mock_row.machine_id = "AC-001"
        mock_row.model_version = "v1"
        mock_row.decision_type = "anomaly_detection"
        mock_row.features_snapshot = {"x": 1}
        mock_row.output = {"score": 0.5}
        mock_row.shap_values = {"x": 0.5}
        mock_row.action_taken = "NO_ACTION"
        mock_row.human_action = None

        mock_query = MagicMock()
        mock_query.order_by.return_value.limit.return_value.all.return_value = [mock_row]
        mock_session.query.return_value = mock_query

        with patch.dict("sys.modules", {
            "src.database.models": MagicMock(AIActLog=mock_ai_act_log_class),
        }):
            results = logger.get_last(n=5, db_session=mock_session)

        assert len(results) == 1
        assert results[0]["machine_id"] == "AC-001"
        assert results[0]["model_version"] == "v1"
        assert results[0]["action_taken"] == "NO_ACTION"

    def test_get_last_db_session_query_fails(self):
        """When db_session query fails, should fall back to fallback entries."""
        logger = AIActComplianceLogger()
        mock_session = MagicMock()
        mock_session.query.side_effect = Exception("DB error")

        # Add something to fallback first
        with patch.dict("sys.modules", {
            "src.database.connection": None,
            "src.database.models": None,
        }):
            logger.log_decision(
                machine_id="AC-001",
                model_version="v1",
                features_snapshot={},
                output={},
                shap_values={},
                action="NO_ACTION",
            )

        results = logger.get_last(n=5, db_session=mock_session)
        assert len(results) == 1
        assert results[0]["machine_id"] == "AC-001"


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_timestamp_is_iso_format(self):
        """Timestamp in fallback entries should be ISO format string."""
        logger = AIActComplianceLogger()

        with patch.dict("sys.modules", {
            "src.database.connection": None,
            "src.database.models": None,
        }):
            logger.log_decision(
                machine_id="AC-001",
                model_version="v1",
                features_snapshot={},
                output={},
                shap_values={},
                action="NO_ACTION",
            )

        ts = logger._fallback_entries[0]["timestamp"]
        # Should be parseable as ISO format
        parsed = datetime.fromisoformat(ts)
        assert parsed.tzinfo is not None

    def test_special_characters_in_machine_id(self):
        """Should handle special characters in machine_id."""
        logger = AIActComplianceLogger()

        with patch.dict("sys.modules", {
            "src.database.connection": None,
            "src.database.models": None,
        }):
            logger.log_decision(
                machine_id="AC-001/Special#Chars!",
                model_version="v1",
                features_snapshot={},
                output={},
                shap_values={},
                action="NO_ACTION",
            )

        assert logger._fallback_entries[0]["machine_id"] == "AC-001/Special#Chars!"

    def test_large_features_snapshot(self):
        """Should handle large feature snapshots."""
        logger = AIActComplianceLogger()
        large_features = {f"feature_{i}": float(i) for i in range(1000)}

        with patch.dict("sys.modules", {
            "src.database.connection": None,
            "src.database.models": None,
        }):
            logger.log_decision(
                machine_id="AC-001",
                model_version="v1",
                features_snapshot=large_features,
                output={"score": 0.5},
                shap_values={f"feature_{i}": 0.001 for i in range(1000)},
                action="NO_ACTION",
            )

        assert len(logger._fallback_entries[0]["features_snapshot"]) == 1000

    def test_all_required_fields_present_in_entry(self):
        """Every fallback entry should have all required fields."""
        logger = AIActComplianceLogger()

        with patch.dict("sys.modules", {
            "src.database.connection": None,
            "src.database.models": None,
        }):
            logger.log_decision(
                machine_id="AC-001",
                model_version="v1",
                features_snapshot={"x": 1},
                output={"score": 0.5},
                shap_values={"x": 0.5},
                action="NO_ACTION",
            )

        entry = logger._fallback_entries[0]
        required_keys = [
            "timestamp", "machine_id", "model_version", "decision_type",
            "features_snapshot", "output", "shap_values",
            "action_taken", "human_action"
        ]
        for key in required_keys:
            assert key in entry, f"Missing key: {key}"
