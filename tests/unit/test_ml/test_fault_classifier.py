"""
Unit tests for src.ml.fault_classifier
"""


from src.ml.fault_classifier import (
    AC_FAULT_RULES,
    BEARING_FAULT,
    BELT_SLIP,
    CM_FAULT_RULES,
    CONFIDENCE_THRESHOLD,
    FLOW_RESTRICTION,
    FOULING,
    HX_FAULT_RULES,
    MACHINE_RULES,
    MOTOR_OVERLOAD,
    OIL_DEGRADATION,
    UNCLASSIFIED_ANOMALY,
    VALVE_LEAK,
    FaultClassification,
    FaultClassifier,
)


class TestFaultClassifierInit:
    def test_default_init(self):
        clf = FaultClassifier()
        assert clf.ac_rules is AC_FAULT_RULES
        assert clf.hx_rules is HX_FAULT_RULES
        assert clf.cm_rules is CM_FAULT_RULES
        assert clf.machine_rules is MACHINE_RULES
        assert clf.confidence_threshold == CONFIDENCE_THRESHOLD
        assert clf.db_session_factory is None
        assert clf._baseline_cache == {}

    def test_init_with_db_factory(self):
        factory = lambda: None
        clf = FaultClassifier(db_session_factory=factory)
        assert clf.db_session_factory is factory


class TestClassifyAC:
    def setup_method(self):
        self.clf = FaultClassifier()

    def test_bearing_fault_ac(self):
        result = self.clf.classify(
            machine_id="AC-001",
            machine_type="AC",
            anomaly_score=0.8,
            sensor_readings={"vibration_rms": 5.0, "bearing_temp": 90.0},
            shap_values={"vibration_rms": 0.5, "bearing_temp": 0.3},
            top_contributing_sensor="vibration_rms",
        )
        assert result.fault_type == BEARING_FAULT
        assert result.fault_confidence >= CONFIDENCE_THRESHOLD
        assert BEARING_FAULT in result.matched_rules

    def test_oil_degradation_ac(self):
        result = self.clf.classify(
            machine_id="AC-001",
            machine_type="AC",
            anomaly_score=0.7,
            sensor_readings={"oil_pressure": 2.5, "bearing_temp": 80.0},
            shap_values={"oil_pressure": 0.4},
        )
        assert result.fault_type == OIL_DEGRADATION
        assert result.fault_confidence >= CONFIDENCE_THRESHOLD

    def test_valve_leak_ac(self):
        result = self.clf.classify(
            machine_id="AC-001",
            machine_type="AC",
            anomaly_score=0.6,
            sensor_readings={"outlet_pressure": 6.0, "motor_current": 25.0},
        )
        assert result.fault_type == VALVE_LEAK
        assert result.fault_confidence >= CONFIDENCE_THRESHOLD

    def test_motor_overload_ac(self):
        result = self.clf.classify(
            machine_id="AC-001",
            machine_type="AC",
            anomaly_score=0.7,
            sensor_readings={"motor_current": 30.0, "vibration_rms": 4.0},
        )
        assert result.fault_type == MOTOR_OVERLOAD
        assert result.fault_confidence >= CONFIDENCE_THRESHOLD

    def test_unclassified_ac_no_match(self):
        result = self.clf.classify(
            machine_id="AC-001",
            machine_type="AC",
            anomaly_score=0.5,
            sensor_readings={"vibration_rms": 1.0, "bearing_temp": 40.0},
        )
        assert result.fault_type == UNCLASSIFIED_ANOMALY


class TestClassifyHX:
    def setup_method(self):
        self.clf = FaultClassifier()

    def test_fouling_hx(self):
        result = self.clf.classify(
            machine_id="HX-001",
            machine_type="HX",
            anomaly_score=0.8,
            sensor_readings={
                "fouling_index": 0.4,
                "pressure_drop": 1.6,
                "outlet_temp": 100.0,
            },
            shap_values={"fouling_index": 0.5, "pressure_drop": 0.4},
        )
        assert result.fault_type == FOULING
        assert result.fault_confidence >= CONFIDENCE_THRESHOLD

    def test_flow_restriction_hx(self):
        # pressure_drop=1.3 is below fouling threshold (1.4) so only FLOW_RESTRICTION matches
        result = self.clf.classify(
            machine_id="HX-001",
            machine_type="HX",
            anomaly_score=0.7,
            sensor_readings={"flow_rate": 8.0, "pressure_drop": 1.3},
        )
        assert result.fault_type == FLOW_RESTRICTION
        assert result.fault_confidence >= CONFIDENCE_THRESHOLD


class TestClassifyCM:
    def setup_method(self):
        self.clf = FaultClassifier()

    def test_belt_slip_cm(self):
        result = self.clf.classify(
            machine_id="CM-001",
            machine_type="CM",
            anomaly_score=0.7,
            sensor_readings={"belt_tension": 12.0, "speed_rpm": 1350.0},
        )
        assert result.fault_type == BELT_SLIP
        assert result.fault_confidence >= CONFIDENCE_THRESHOLD

    def test_motor_overload_cm(self):
        result = self.clf.classify(
            machine_id="CM-001",
            machine_type="CM",
            anomaly_score=0.7,
            sensor_readings={"motor_load": 90.0, "drive_temp": 80.0},
        )
        assert result.fault_type == MOTOR_OVERLOAD
        assert result.fault_confidence >= CONFIDENCE_THRESHOLD

    def test_bearing_fault_cm(self):
        result = self.clf.classify(
            machine_id="CM-001",
            machine_type="CM",
            anomaly_score=0.7,
            sensor_readings={"vibration_rms": 4.0, "drive_temp": 75.0},
        )
        assert result.fault_type == BEARING_FAULT
        assert result.fault_confidence >= CONFIDENCE_THRESHOLD


class TestEdgeCases:
    def setup_method(self):
        self.clf = FaultClassifier()

    def test_unknown_machine_type(self):
        result = self.clf.classify(
            machine_id="XX-001",
            machine_type="UNKNOWN",
            anomaly_score=0.9,
            sensor_readings={"vibration_rms": 10.0},
        )
        assert result.fault_type == UNCLASSIFIED_ANOMALY
        assert result.fault_confidence == 0.0
        assert result.matched_rules == []

    def test_missing_sensor_readings(self):
        result = self.clf.classify(
            machine_id="AC-001",
            machine_type="AC",
            anomaly_score=0.5,
            sensor_readings={},
        )
        assert result.fault_type == UNCLASSIFIED_ANOMALY

    def test_none_sensor_readings(self):
        result = self.clf.classify(
            machine_id="AC-001",
            machine_type="AC",
            anomaly_score=0.5,
            sensor_readings=None,
            shap_values=None,
        )
        assert result.fault_type == UNCLASSIFIED_ANOMALY

    def test_partial_sensor_match(self):
        """Only one sensor matches - still classified if above threshold."""
        result = self.clf.classify(
            machine_id="AC-001",
            machine_type="AC",
            anomaly_score=0.6,
            sensor_readings={"vibration_rms": 5.0},
            shap_values={"vibration_rms": 0.6},
            top_contributing_sensor="vibration_rms",
        )
        # vibration_rms matches BEARING_FAULT pattern
        assert result.fault_type == BEARING_FAULT

    def test_shap_values_boost(self):
        """SHAP values should boost confidence score."""
        result_no_shap = self.clf.classify(
            machine_id="AC-001",
            machine_type="AC",
            anomaly_score=0.6,
            sensor_readings={"vibration_rms": 5.0},
        )
        result_with_shap = self.clf.classify(
            machine_id="AC-001",
            machine_type="AC",
            anomaly_score=0.6,
            sensor_readings={"vibration_rms": 5.0},
            shap_values={"vibration_rms": 1.0},
        )
        assert result_with_shap.fault_confidence > result_no_shap.fault_confidence

    def test_top_contributing_sensor_bonus(self):
        """Top contributing sensor matching rule sensors adds bonus."""
        result_no_top = self.clf.classify(
            machine_id="AC-001",
            machine_type="AC",
            anomaly_score=0.6,
            sensor_readings={"vibration_rms": 5.0},
            shap_values={"vibration_rms": 0.5},
        )
        result_with_top = self.clf.classify(
            machine_id="AC-001",
            machine_type="AC",
            anomaly_score=0.6,
            sensor_readings={"vibration_rms": 5.0},
            shap_values={"vibration_rms": 0.5},
            top_contributing_sensor="vibration_rms",
        )
        assert result_with_top.fault_confidence > result_no_top.fault_confidence

    def test_confidence_capped_at_095(self):
        """Confidence should never exceed 0.95."""
        result = self.clf.classify(
            machine_id="AC-001",
            machine_type="AC",
            anomaly_score=0.9,
            sensor_readings={"vibration_rms": 10.0, "bearing_temp": 100.0},
            shap_values={"vibration_rms": 2.0, "bearing_temp": 2.0},
            top_contributing_sensor="vibration_rms",
        )
        assert result.fault_confidence <= 0.95


class TestGetSupportedFaultTypes:
    def setup_method(self):
        self.clf = FaultClassifier()

    def test_ac_fault_types(self):
        types = self.clf.get_supported_fault_types("AC")
        assert BEARING_FAULT in types
        assert OIL_DEGRADATION in types
        assert VALVE_LEAK in types
        assert MOTOR_OVERLOAD in types
        assert len(types) == 4

    def test_hx_fault_types(self):
        types = self.clf.get_supported_fault_types("HX")
        assert FOULING in types
        assert FLOW_RESTRICTION in types
        assert len(types) == 2

    def test_cm_fault_types(self):
        types = self.clf.get_supported_fault_types("CM")
        assert BELT_SLIP in types
        assert MOTOR_OVERLOAD in types
        assert BEARING_FAULT in types
        assert len(types) == 3

    def test_unknown_machine_type(self):
        types = self.clf.get_supported_fault_types("UNKNOWN")
        assert types == []


class TestGetFaultDescription:
    def setup_method(self):
        self.clf = FaultClassifier()

    def test_known_fault_types(self):
        desc = self.clf.get_fault_description(BEARING_FAULT)
        assert desc == {"key": "fault_bearing_short"}

        desc = self.clf.get_fault_description(OIL_DEGRADATION)
        assert desc == {"key": "fault_oil_short"}

        desc = self.clf.get_fault_description(FOULING)
        assert desc == {"key": "fault_fouling_short"}

    def test_unclassified(self):
        desc = self.clf.get_fault_description(UNCLASSIFIED_ANOMALY)
        assert desc == {"key": "fault_unclassified_short"}

    def test_unknown_fault_type(self):
        desc = self.clf.get_fault_description("NONEXISTENT")
        assert desc == {"key": "fault_unknown"}


class TestTopShapSensor:
    def setup_method(self):
        self.clf = FaultClassifier()

    def test_returns_top_sensor(self):
        result = self.clf._top_shap_sensor({"a": 0.1, "b": 0.9, "c": -0.5})
        assert result == "b"

    def test_negative_values(self):
        result = self.clf._top_shap_sensor({"a": -0.9, "b": 0.1})
        assert result == "a"

    def test_empty_dict(self):
        result = self.clf._top_shap_sensor({})
        assert result is None


class TestMatchesSensorPattern:
    def setup_method(self):
        self.clf = FaultClassifier()

    def test_min_condition_met(self):
        result = self.clf._matches_sensor_pattern("M1", "sensor", 5.0, {"min": 4.5})
        assert result is True

    def test_min_condition_not_met(self):
        result = self.clf._matches_sensor_pattern("M1", "sensor", 3.0, {"min": 4.5})
        assert result is False

    def test_max_condition_met(self):
        result = self.clf._matches_sensor_pattern("M1", "sensor", 2.0, {"max": 2.8})
        assert result is True

    def test_max_condition_not_met(self):
        result = self.clf._matches_sensor_pattern("M1", "sensor", 3.5, {"max": 2.8})
        assert result is False

    def test_boundary_min(self):
        result = self.clf._matches_sensor_pattern("M1", "sensor", 4.5, {"min": 4.5})
        assert result is True

    def test_boundary_max(self):
        result = self.clf._matches_sensor_pattern("M1", "sensor", 2.8, {"max": 2.8})
        assert result is True


class TestGetBaseline:
    def test_no_db_session(self):
        clf = FaultClassifier()
        result = clf._get_baseline("M1", "sensor")
        assert result is None

    def test_with_mock_db_session(self):
        """Test baseline retrieval with a mocked DB session."""
        from contextlib import contextmanager
        from unittest.mock import MagicMock

        mock_row = MagicMock()
        mock_row.sensor = "vibration_rms"
        mock_row.mean_value = 2.0
        mock_row.std_value = 0.5

        mock_session = MagicMock()
        mock_session.query.return_value.filter.return_value.all.return_value = [mock_row]

        @contextmanager
        def mock_factory():
            yield mock_session

        clf = FaultClassifier(db_session_factory=mock_factory)
        result = clf._get_baseline("M1", "vibration_rms")
        assert result == (2.0, 0.5)

    def test_baseline_caching(self):
        """Second call should use cache, not query DB again."""
        from unittest.mock import MagicMock

        mock_row = MagicMock()
        mock_row.sensor = "vibration_rms"
        mock_row.mean_value = 2.0
        mock_row.std_value = 0.5

        mock_session = MagicMock()
        mock_session.query.return_value.filter.return_value.all.return_value = [mock_row]

        mock_factory = MagicMock()
        mock_factory.return_value.__enter__ = MagicMock(return_value=mock_session)
        mock_factory.return_value.__exit__ = MagicMock(return_value=False)

        clf = FaultClassifier(db_session_factory=mock_factory)
        clf._get_baseline("M1", "vibration_rms")
        clf._get_baseline("M1", "vibration_rms")
        # Factory should only be called once (first call populates cache)
        assert mock_factory.call_count == 1

    def test_sensor_not_in_baseline(self):
        """If sensor not in baseline rows, return None."""
        from unittest.mock import MagicMock

        mock_row = MagicMock()
        mock_row.sensor = "vibration_rms"
        mock_row.mean_value = 2.0
        mock_row.std_value = 0.5

        mock_session = MagicMock()
        mock_session.query.return_value.filter.return_value.all.return_value = [mock_row]

        mock_factory = MagicMock()
        mock_factory.return_value.__enter__ = MagicMock(return_value=mock_session)
        mock_factory.return_value.__exit__ = MagicMock(return_value=False)

        clf = FaultClassifier(db_session_factory=mock_factory)
        result = clf._get_baseline("M1", "nonexistent_sensor")
        assert result is None


class TestMatchesSensorPatternWithBaseline:
    """Test _matches_sensor_pattern when baseline is available (z-score path)."""

    def test_z_score_min_condition(self):
        from unittest.mock import MagicMock

        mock_row = MagicMock()
        mock_row.sensor = "vibration_rms"
        mock_row.mean_value = 2.0
        mock_row.std_value = 0.5

        mock_session = MagicMock()
        mock_session.query.return_value.filter.return_value.all.return_value = [mock_row]

        mock_factory = MagicMock()
        mock_factory.return_value.__enter__ = MagicMock(return_value=mock_session)
        mock_factory.return_value.__exit__ = MagicMock(return_value=False)

        clf = FaultClassifier(db_session_factory=mock_factory)
        # z_score = (6.0 - 2.0) / 0.5 = 8.0 >= 4.0 (default k_sigma)
        result = clf._matches_sensor_pattern("M1", "vibration_rms", 6.0, {"min": 4.5})
        assert result is True

    def test_z_score_max_condition(self):
        from unittest.mock import MagicMock

        mock_row = MagicMock()
        mock_row.sensor = "oil_pressure"
        mock_row.mean_value = 4.0
        mock_row.std_value = 0.5

        mock_session = MagicMock()
        mock_session.query.return_value.filter.return_value.all.return_value = [mock_row]

        mock_factory = MagicMock()
        mock_factory.return_value.__enter__ = MagicMock(return_value=mock_session)
        mock_factory.return_value.__exit__ = MagicMock(return_value=False)

        clf = FaultClassifier(db_session_factory=mock_factory)
        # z_score = (1.0 - 4.0) / 0.5 = -6.0 <= -4.0
        result = clf._matches_sensor_pattern("M1", "oil_pressure", 1.0, {"max": 2.8})
        assert result is True

    def test_z_score_not_exceeded(self):
        from unittest.mock import MagicMock

        mock_row = MagicMock()
        mock_row.sensor = "vibration_rms"
        mock_row.mean_value = 2.0
        mock_row.std_value = 1.0

        mock_session = MagicMock()
        mock_session.query.return_value.filter.return_value.all.return_value = [mock_row]

        mock_factory = MagicMock()
        mock_factory.return_value.__enter__ = MagicMock(return_value=mock_session)
        mock_factory.return_value.__exit__ = MagicMock(return_value=False)

        clf = FaultClassifier(db_session_factory=mock_factory)
        # z_score = (3.0 - 2.0) / 1.0 = 1.0 < 4.0
        result = clf._matches_sensor_pattern("M1", "vibration_rms", 3.0, {"min": 4.5})
        assert result is False


class TestFaultClassificationDataclass:
    def test_dataclass_creation(self):
        fc = FaultClassification(
            fault_type=BEARING_FAULT,
            fault_description={"key": "test"},
            fault_confidence=0.8,
            matched_rules=[BEARING_FAULT],
            shap_contributors={"sensor": 0.5},
        )
        assert fc.fault_type == BEARING_FAULT
        assert fc.fault_confidence == 0.8
        assert fc.matched_rules == [BEARING_FAULT]
        assert fc.shap_contributors == {"sensor": 0.5}
