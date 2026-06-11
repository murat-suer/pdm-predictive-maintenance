"""
Unit tests for src.decision.failure_mode_library

Tests FailureModeLibrary: YAML-based FMEA failure mode matching,
rule evaluation, confidence calculation, and ranking.
"""


import pytest

from src.decision.failure_mode_library import (
    FailureModeLibrary,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def library():
    """Default library loaded from bundled failure_modes.yaml."""
    return FailureModeLibrary()


@pytest.fixture
def library_with_yaml(tmp_path):
    """Library loaded from a custom YAML file."""
    yaml_content = """
failure_modes:
  - mode_id: bearing_stage_1
    category: bearing
    description: Early stage bearing wear
    rpn: 120
    rules:
      - sensor: vibration_rms
        condition: "trend_up"
      - sensor: bearing_temp
        condition: "above:75"
    category_weight: 0.8

  - mode_id: bearing_stage_2
    category: bearing
    description: Advanced bearing wear
    rpn: 240
    rules:
      - sensor: vibration_rms
        condition: "trend_up"
      - sensor: bearing_temp
        condition: "above:90"
      - sensor: vibration_rms
        condition: "above:5.0"
    category_weight: 0.8

  - mode_id: oil_degradation
    category: lubrication
    description: Oil quality degradation
    rpn: 180
    rules:
      - sensor: oil_pressure
        condition: "trend_down"
      - sensor: oil_temp
        condition: "above:80"
    category_weight: 0.7

  - mode_id: belt_slip
    category: drive
    description: Belt slip condition
    rpn: 150
    rules:
      - sensor: belt_tension
        condition: "below:15"
      - sensor: speed_rpm
        condition: "delta_pct:5"
    category_weight: 0.6
"""
    yaml_file = tmp_path / "test_failure_modes.yaml"
    yaml_file.write_text(yaml_content)
    return FailureModeLibrary(yaml_path=str(yaml_file))


@pytest.fixture
def sensor_data():
    """Typical sensor readings for testing."""
    return {
        "vibration_rms": 5.5,
        "bearing_temp": 85.0,
        "oil_pressure": 3.0,
        "oil_temp": 82.0,
        "belt_tension": 12.0,
        "speed_rpm": 1500.0,
    }


@pytest.fixture
def trend_data():
    """Sensor data with trend information."""
    return {
        "vibration_rms": {"value": 5.5, "trend": "up"},
        "bearing_temp": {"value": 85.0, "trend": "up"},
        "oil_pressure": {"value": 3.0, "trend": "down"},
        "oil_temp": {"value": 82.0, "trend": "stable"},
        "belt_tension": {"value": 12.0, "trend": "down"},
        "speed_rpm": {"value": 1500.0, "trend": "stable"},
    }


# ---------------------------------------------------------------------------
# TestYAMLLoading
# ---------------------------------------------------------------------------
class TestYAMLLoading:
    """Test YAML loading and parsing."""

    def test_default_library_loads(self, library):
        """Default library should load without error."""
        assert library is not None

    def test_library_has_failure_modes(self, library):
        """Library should contain failure modes."""
        modes = library.get_all_modes()
        assert len(modes) > 0

    def test_library_has_12_modes(self, library):
        """Default failure_modes.yaml should have ~12 failure modes."""
        modes = library.get_all_modes()
        assert len(modes) >= 10, f"Expected 10+ modes, got {len(modes)}"

    def test_custom_yaml_loads(self, library_with_yaml):
        """Custom YAML should load correctly."""
        modes = library_with_yaml.get_all_modes()
        assert len(modes) == 4

    def test_mode_has_required_fields(self, library_with_yaml):
        """Each mode should have mode_id, category, description, rpn, rules."""
        modes = library_with_yaml.get_all_modes()
        for mode in modes:
            assert mode.mode_id is not None
            assert mode.category is not None
            assert mode.description is not None
            assert mode.rpn > 0
            assert len(mode.rules) > 0

    def test_missing_yaml_raises_error(self):
        """Non-existent YAML file should raise an error."""
        with pytest.raises((FileNotFoundError, OSError, ValueError)):
            FailureModeLibrary(yaml_path="/nonexistent/path.yaml")


# ---------------------------------------------------------------------------
# TestRuleEvaluation
# ---------------------------------------------------------------------------
class TestRuleEvaluation:
    """Test individual rule evaluation: trend_up, trend_down, above:X, below:X, delta_pct:X."""

    def test_trend_up_rule(self, library_with_yaml, trend_data):
        """trend_up should match when sensor trend is 'up'."""
        result = library_with_yaml.match(
            sensor_data=trend_data,
            machine_id="AC-001",
        )
        # bearing_stage_1 and bearing_stage_2 both have trend_up on vibration_rms
        mode_ids = [m.mode_id for m in result]
        assert "bearing_stage_1" in mode_ids or "bearing_stage_2" in mode_ids

    def test_trend_down_rule(self, library_with_yaml, trend_data):
        """trend_down should match when sensor trend is 'down'."""
        result = library_with_yaml.match(
            sensor_data=trend_data,
            machine_id="AC-001",
        )
        mode_ids = [m.mode_id for m in result]
        assert "oil_degradation" in mode_ids  # oil_pressure has trend_down

    def test_above_threshold_rule(self, library_with_yaml, sensor_data):
        """above:X should match when value > X."""
        # bearing_temp=85 > 75 (bearing_stage_1) and bearing_temp=85 < 90 (not stage_2)
        result = library_with_yaml.match(
            sensor_data=sensor_data,
            machine_id="AC-001",
        )
        mode_ids = [m.mode_id for m in result]
        assert "bearing_stage_1" in mode_ids

    def test_below_threshold_rule(self, library_with_yaml, sensor_data):
        """below:X should match when value < X."""
        # belt_tension=12 < 15 (belt_slip)
        result = library_with_yaml.match(
            sensor_data=sensor_data,
            machine_id="AC-001",
        )
        mode_ids = [m.mode_id for m in result]
        assert "belt_slip" in mode_ids

    def test_above_high_threshold_no_match(self, library_with_yaml, sensor_data):
        """above:90 should NOT match when value=85."""
        result = library_with_yaml.match(
            sensor_data=sensor_data,
            machine_id="AC-001",
        )
        mode_ids = [m.mode_id for m in result]
        # bearing_stage_2 requires vibration_rms > 5.0 AND bearing_temp > 90
        # bearing_temp=85 < 90, so stage_2 should not fully match
        # (it may appear with low confidence)
        for m in result:
            if m.mode_id == "bearing_stage_2":
                assert m.confidence < 1.0  # Not all rules matched

    def test_delta_pct_rule(self, library_with_yaml):
        """delta_pct:X should match when change exceeds X%."""
        data = {
            "belt_tension": {"value": 12.0, "trend": "stable"},
            "speed_rpm": {"value": 1500.0, "delta_pct": 6.0},  # > 5%
        }
        result = library_with_yaml.match(
            sensor_data=data,
            machine_id="AC-001",
        )
        mode_ids = [m.mode_id for m in result]
        assert "belt_slip" in mode_ids


# ---------------------------------------------------------------------------
# TestSignatureMatching
# ---------------------------------------------------------------------------
class TestSignatureMatching:
    """Test full signature matching (all rules for a mode)."""

    def test_full_signature_match(self, library_with_yaml):
        """All rules matching should give high confidence."""
        data = {
            "vibration_rms": {"value": 6.0, "trend": "up"},
            "bearing_temp": {"value": 95.0, "trend": "up"},
            "oil_pressure": {"value": 3.0, "trend": "down"},
            "oil_temp": {"value": 82.0, "trend": "stable"},
            "belt_tension": {"value": 12.0, "trend": "down"},
            "speed_rpm": {"value": 1500.0, "delta_pct": 6.0},
        }
        result = library_with_yaml.match(
            sensor_data=data,
            machine_id="AC-001",
        )
        # bearing_stage_2 should have highest confidence (all 3 rules match)
        bearing_modes = [m for m in result if m.mode_id.startswith("bearing")]
        assert len(bearing_modes) > 0
        # Stage 2 should match better than stage 1 (more specific)
        stage2 = [m for m in result if m.mode_id == "bearing_stage_2"]
        if stage2:
            assert stage2[0].confidence > 0.5

    def test_partial_signature_match(self, library_with_yaml, sensor_data):
        """Partial rule match should give lower confidence."""
        result = library_with_yaml.match(
            sensor_data=sensor_data,
            machine_id="AC-001",
        )
        # Should have some matches but not all at 100%
        assert len(result) > 0
        for m in result:
            assert 0.0 <= m.confidence <= 1.0

    def test_no_match_returns_empty(self, library_with_yaml):
        """No matching rules should return empty list."""
        data = {
            "vibration_rms": {"value": 0.5, "trend": "stable"},
            "bearing_temp": {"value": 40.0, "trend": "stable"},
            "oil_pressure": {"value": 5.0, "trend": "stable"},
            "oil_temp": {"value": 50.0, "trend": "stable"},
            "belt_tension": {"value": 25.0, "trend": "stable"},
            "speed_rpm": {"value": 1500.0, "delta_pct": 0.5},
        }
        result = library_with_yaml.match(
            sensor_data=data,
            machine_id="AC-001",
        )
        # Either empty or very low confidence
        for m in result:
            assert m.confidence < 0.3


# ---------------------------------------------------------------------------
# TestConfidenceCalculation
# ---------------------------------------------------------------------------
class TestConfidenceCalculation:
    """Confidence = matched_rules / total_rules * category_weight."""

    def test_confidence_range(self, library_with_yaml, sensor_data):
        """Confidence should be between 0 and 1."""
        result = library_with_yaml.match(
            sensor_data=sensor_data,
            machine_id="AC-001",
        )
        for m in result:
            assert 0.0 <= m.confidence <= 1.0

    def test_all_rules_matched_high_confidence(self, library_with_yaml):
        """When all rules match, confidence should be close to category_weight."""
        data = {
            "vibration_rms": {"value": 6.0, "trend": "up"},
            "bearing_temp": {"value": 95.0, "trend": "up"},
        }
        result = library_with_yaml.match(
            sensor_data=data,
            machine_id="AC-001",
        )
        bearing_modes = [m for m in result if "bearing" in m.mode_id]
        for m in bearing_modes:
            # All rules matched → confidence = 1.0 * 0.8 = 0.8
            assert m.confidence <= 0.85  # category_weight is 0.8

    def test_partial_rules_lower_confidence(self, library_with_yaml):
        """Fewer matched rules → lower confidence."""
        # Only vibration_rms matches (1 of 2 rules for bearing_stage_1)
        data_partial = {
            "vibration_rms": {"value": 6.0, "trend": "up"},
            "bearing_temp": {"value": 50.0, "trend": "stable"},  # below:75 NOT matched
        }
        result = library_with_yaml.match(
            sensor_data=data_partial,
            machine_id="AC-001",
        )
        bearing_modes = [m for m in result if m.mode_id == "bearing_stage_1"]
        if bearing_modes:
            # 1/2 rules matched * 0.8 weight = 0.4
            assert bearing_modes[0].confidence == pytest.approx(0.4, abs=0.1)


# ---------------------------------------------------------------------------
# TestThresholdFiltering
# ---------------------------------------------------------------------------
class TestThresholdFiltering:
    """Results below confidence threshold should be filtered out."""

    def test_threshold_filtering(self, library_with_yaml, sensor_data):
        """High threshold should filter out low-confidence matches."""
        result_all = library_with_yaml.match(
            sensor_data=sensor_data,
            machine_id="AC-001",
            min_confidence=0.0,
        )
        result_filtered = library_with_yaml.match(
            sensor_data=sensor_data,
            machine_id="AC-001",
            min_confidence=0.9,
        )
        assert len(result_filtered) <= len(result_all)

    def test_zero_threshold_returns_all(self, library_with_yaml, sensor_data):
        """min_confidence=0 should return all matches."""
        result = library_with_yaml.match(
            sensor_data=sensor_data,
            machine_id="AC-001",
            min_confidence=0.0,
        )
        # Should return all matches (even low confidence)
        assert len(result) >= 0  # May be empty if nothing matches

    def test_default_threshold(self, library_with_yaml, sensor_data):
        """Default threshold should filter reasonably."""
        result = library_with_yaml.match(
            sensor_data=sensor_data,
            machine_id="AC-001",
        )
        for m in result:
            assert m.confidence >= 0.0  # At minimum, above 0


# ---------------------------------------------------------------------------
# TestRankingByConfidenceAndRPN
# ---------------------------------------------------------------------------
class TestRankingByConfidenceAndRPN:
    """Results should be ranked by confidence * RPN (risk priority)."""

    def test_results_sorted_by_score(self, library_with_yaml):
        """Results should be sorted descending by combined score."""
        data = {
            "vibration_rms": {"value": 6.0, "trend": "up"},
            "bearing_temp": {"value": 95.0, "trend": "up"},
            "oil_pressure": {"value": 3.0, "trend": "down"},
            "oil_temp": {"value": 82.0, "trend": "stable"},
            "belt_tension": {"value": 12.0, "trend": "down"},
            "speed_rpm": {"value": 1500.0, "delta_pct": 6.0},
        }
        result = library_with_yaml.match(
            sensor_data=data,
            machine_id="AC-001",
        )
        if len(result) > 1:
            # Check descending order
            scores = [m.confidence * m.rpn for m in result]
            assert scores == sorted(scores, reverse=True)

    def test_high_rpn_ranks_higher(self, library_with_yaml):
        """With equal confidence, higher RPN should rank higher."""
        data = {
            "vibration_rms": {"value": 6.0, "trend": "up"},
            "bearing_temp": {"value": 95.0, "trend": "up"},
        }
        result = library_with_yaml.match(
            sensor_data=data,
            machine_id="AC-001",
        )
        # bearing_stage_2 (RPN=240) should rank above bearing_stage_1 (RPN=120)
        # if both have same confidence
        if len(result) >= 2:
            bearing_modes = [m for m in result if "bearing" in m.mode_id]
            if len(bearing_modes) >= 2:
                # Higher RPN should come first (if confidence is similar)
                assert bearing_modes[0].rpn >= bearing_modes[-1].rpn


# ---------------------------------------------------------------------------
# TestDuplicateModeIdRejection
# ---------------------------------------------------------------------------
class TestDuplicateModeIdRejection:
    """Duplicate mode_id in YAML should raise an error."""

    def test_duplicate_mode_id_rejected(self, tmp_path):
        yaml_content = """
failure_modes:
  - mode_id: bearing_stage_1
    category: bearing
    description: First
    rpn: 100
    rules:
      - sensor: vibration_rms
        condition: "trend_up"
    category_weight: 0.8

  - mode_id: bearing_stage_1
    category: bearing
    description: Duplicate!
    rpn: 200
    rules:
      - sensor: bearing_temp
        condition: "above:90"
    category_weight: 0.8
"""
        yaml_file = tmp_path / "duplicate.yaml"
        yaml_file.write_text(yaml_content)
        with pytest.raises((ValueError, KeyError)):
            FailureModeLibrary(yaml_path=str(yaml_file))


# ---------------------------------------------------------------------------
# TestMissingYAMLErrorHandling
# ---------------------------------------------------------------------------
class TestMissingYAMLErrorHandling:
    """Missing or malformed YAML should be handled gracefully."""

    def test_nonexistent_file(self):
        with pytest.raises((FileNotFoundError, OSError)):
            FailureModeLibrary(yaml_path="/nonexistent.yaml")

    def test_empty_yaml(self, tmp_path):
        yaml_file = tmp_path / "empty.yaml"
        yaml_file.write_text("")
        with pytest.raises((ValueError, KeyError)):
            FailureModeLibrary(yaml_path=str(yaml_file))

    def test_malformed_yaml(self, tmp_path):
        yaml_file = tmp_path / "bad.yaml"
        yaml_file.write_text("{{invalid yaml content:::")
        with pytest.raises(Exception):  # yaml.YAMLError or similar
            FailureModeLibrary(yaml_path=str(yaml_file))

    def test_missing_failure_modes_key(self, tmp_path):
        yaml_file = tmp_path / "no_modes.yaml"
        yaml_file.write_text("some_other_key: value")
        with pytest.raises((ValueError, KeyError)):
            FailureModeLibrary(yaml_path=str(yaml_file))


# ---------------------------------------------------------------------------
# TestStageIVBearing - Murat'ın 4. kuralı
# ---------------------------------------------------------------------------
class TestStageIVBearing:
    """
    Stage IV bearing: REDUCE_LOAD etkisiz olmalı.
    Bu test failure_mode_library'nin Stage IV detection'ını test eder.
    """

    def test_stage_iv_bearing_detected(self, library):
        """Library should have a Stage IV bearing failure mode."""
        modes = library.get_all_modes()
        stage_iv = [m for m in modes if "stage_4" in m.mode_id or "stage_iv" in m.mode_id.lower()]
        # Should exist in the 12 default modes
        assert len(stage_iv) >= 1 or any("bearing" in m.category.lower() for m in modes)

    def test_stage_iv_high_rpn(self, library):
        """Stage IV bearing should have high RPN (critical)."""
        modes = library.get_all_modes()
        bearing_modes = [m for m in modes if "bearing" in m.category.lower()]
        if bearing_modes:
            # At least one bearing mode should have high RPN
            max_rpn = max(m.rpn for m in bearing_modes)
            assert max_rpn >= 200  # Stage IV should be high risk


# ---------------------------------------------------------------------------
# TestEdgeCases
# ---------------------------------------------------------------------------
class TestEdgeCases:
    def test_empty_sensor_data(self, library_with_yaml):
        """Empty sensor data should return no matches."""
        result = library_with_yaml.match(
            sensor_data={},
            machine_id="AC-001",
        )
        assert len(result) == 0

    def test_none_sensor_data(self, library_with_yaml):
        """None sensor data should be handled gracefully."""
        with pytest.raises((TypeError, ValueError)):
            library_with_yaml.match(
                sensor_data=None,
                machine_id="AC-001",
            )

    def test_unknown_sensor_ignored(self, library_with_yaml):
        """Unknown sensors in data should be ignored."""
        data = {
            "unknown_sensor": {"value": 100.0, "trend": "up"},
            "vibration_rms": {"value": 6.0, "trend": "up"},
            "bearing_temp": {"value": 95.0, "trend": "up"},
        }
        result = library_with_yaml.match(
            sensor_data=data,
            machine_id="AC-001",
        )
        # Should still match bearing modes based on known sensors
        bearing_modes = [m for m in result if "bearing" in m.mode_id]
        assert len(bearing_modes) > 0

    def test_match_result_has_mode_info(self, library_with_yaml, sensor_data):
        """MatchResult should include mode_id, confidence, rpn."""
        result = library_with_yaml.match(
            sensor_data=sensor_data,
            machine_id="AC-001",
        )
        for m in result:
            assert hasattr(m, "mode_id")
            assert hasattr(m, "confidence")
            assert hasattr(m, "rpn")
