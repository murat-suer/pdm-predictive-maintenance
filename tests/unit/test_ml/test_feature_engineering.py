"""Unit tests for feature_engineering and physical_features modules."""

import numpy as np
import polars as pl
import pytest

from src.ml.feature_engineering import (
    compute_features,
    filter_calibration_data,
    get_feature_names,
)
from src.ml.physical_features import (
    belt_slip_ratio,
    crest_factor,
    delta_p_per_flow,
    oil_consumption_rate,
    rolling_kurtosis,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def simple_sensor_df():
    """Create a simple sensor DataFrame with 35 rows (enough for WINDOW_SIZE=30)."""
    n = 35
    timestamps = [f"2024-01-01T00:{i:02d}:00" for i in range(n)]
    rows = []
    for ts in timestamps:
        rows.append({"timestamp": ts, "sensor_name": "vibration_rms", "value": np.random.randn()})
        rows.append({"timestamp": ts, "sensor_name": "bearing_temp", "value": 50.0 + np.random.randn() * 0.5})
    return pl.DataFrame(rows)


@pytest.fixture
def full_sensor_df():
    """DataFrame with all sensors needed for physical features."""
    n = 35
    timestamps = [f"2024-01-01T00:{i:02d}:00" for i in range(n)]
    rows = []
    for i, ts in enumerate(timestamps):
        rows.append({"timestamp": ts, "sensor_name": "vibration_rms", "value": 2.0 + 0.1 * i})
        rows.append({"timestamp": ts, "sensor_name": "bearing_temp", "value": 50.0 + 0.05 * i})
        rows.append({"timestamp": ts, "sensor_name": "pressure_drop", "value": 10.0 + 0.2 * i})
        rows.append({"timestamp": ts, "sensor_name": "flow_rate", "value": 100.0 - 0.5 * i})
        rows.append({"timestamp": ts, "sensor_name": "motor_rpm", "value": 1500.0})
        rows.append({"timestamp": ts, "sensor_name": "output_rpm", "value": 745.0})
        rows.append({"timestamp": ts, "sensor_name": "oil_level", "value": 100.0 - 0.01 * i})
    return pl.DataFrame(rows)


# ---------------------------------------------------------------------------
# Tests: physical_features.py
# ---------------------------------------------------------------------------

class TestRollingKurtosis:
    def test_basic(self):
        """Kurtosis of a constant series should be 0 or NaN (undefined when std=0)."""
        series = pl.Series([1.0] * 40)
        result = rolling_kurtosis(series, window=10)
        # All values after the window should be 0 or NaN (std=0 => kurtosis undefined)
        valid = result.to_numpy()[9:]
        assert all(v == 0.0 or np.isnan(v) for v in valid)

    def test_window_nan_prefix(self):
        """Values before window-1 should be NaN."""
        series = pl.Series(np.random.randn(20))
        result = rolling_kurtosis(series, window=10)
        arr = result.to_numpy()
        assert all(np.isnan(v) for v in arr[:9])

    def test_length_preserved(self):
        series = pl.Series(np.random.randn(50))
        result = rolling_kurtosis(series, window=15)
        assert len(result) == 50


class TestCrestFactor:
    def test_constant_series(self):
        """Crest factor of constant non-zero series should be 1.0."""
        series = pl.Series([5.0] * 40)
        result = crest_factor(series, window=10)
        valid = result.to_numpy()[9:]
        assert all(abs(v - 1.0) < 1e-10 for v in valid)

    def test_zero_series(self):
        """Crest factor of all-zero series should be 0.0."""
        series = pl.Series([0.0] * 40)
        result = crest_factor(series, window=10)
        valid = result.to_numpy()[9:]
        assert all(v == 0.0 for v in valid)

    def test_impulse(self):
        """A single impulse in a window should give high crest factor."""
        values = [1.0] * 30
        values[15] = 100.0  # impulse
        series = pl.Series(values)
        result = crest_factor(series, window=30)
        # The window containing the impulse should have high crest factor
        arr = result.to_numpy()
        assert arr[29] > 1.0  # crest factor > 1 due to impulse


class TestBeltSlipRatio:
    def test_no_slip(self):
        """When output RPM matches expected, slip should be 0."""
        motor_rpm = pl.Series([1500.0] * 10)
        output_rpm = pl.Series([750.0] * 10)  # pulley_ratio=0.5
        result = belt_slip_ratio(motor_rpm, output_rpm, pulley_ratio=0.5)
        np.testing.assert_allclose(result.to_numpy(), 0.0, atol=1e-10)

    def test_positive_slip(self):
        """When output RPM is less than expected, slip should be positive."""
        motor_rpm = pl.Series([1500.0] * 10)
        output_rpm = pl.Series([700.0] * 10)  # expected=750, actual=700
        result = belt_slip_ratio(motor_rpm, output_rpm, pulley_ratio=0.5)
        # slip = (750 - 700) / 750 = 0.0667
        np.testing.assert_allclose(result.to_numpy(), 50.0 / 750.0, atol=1e-6)


class TestDeltaPPerFlow:
    def test_basic(self):
        """Basic computation: dp / flow^2."""
        dp = pl.Series([10.0, 20.0, 30.0])
        flow = pl.Series([2.0, 4.0, 5.0])
        result = delta_p_per_flow(dp, flow)
        expected = np.array([10.0 / 4.0, 20.0 / 16.0, 30.0 / 25.0])
        np.testing.assert_allclose(result.to_numpy(), expected, atol=1e-10)

    def test_near_zero_flow_clipped(self):
        """Very small flow values should be clipped to avoid division explosion."""
        dp = pl.Series([10.0])
        flow = pl.Series([0.0])
        result = delta_p_per_flow(dp, flow)
        # flow clipped to 1e-6, so result = 10.0 / (1e-6)^2 = 1e13
        assert result.to_numpy()[0] > 0


class TestOilConsumptionRate:
    def test_declining_level(self):
        """Linearly declining oil level should give negative slope."""
        values = [100.0 - 0.1 * i for i in range(70)]
        series = pl.Series(values)
        result = oil_consumption_rate(series, window=60)
        arr = result.to_numpy()
        # After window fills, slope should be approximately -0.1
        assert arr[59] < 0
        np.testing.assert_allclose(arr[59], -0.1, atol=1e-6)


# ---------------------------------------------------------------------------
# Tests: feature_engineering.py
# ---------------------------------------------------------------------------

class TestComputeFeatures:
    def test_basic_output_columns(self, simple_sensor_df):
        """Output should contain expected feature columns for each sensor."""
        result = compute_features(simple_sensor_df)
        for sensor in ["vibration_rms", "bearing_temp"]:
            assert f"{sensor}_value" in result.columns
            assert f"{sensor}_rolling_mean_5m" in result.columns
            assert f"{sensor}_rolling_std_5m" in result.columns
            assert f"{sensor}_rate_of_change" in result.columns
            assert f"{sensor}_z_score" in result.columns
            assert f"{sensor}_shift_adj_z" in result.columns

    def test_row_count_preserved(self, simple_sensor_df):
        """Output should have same number of rows as unique timestamps."""
        result = compute_features(simple_sensor_df)
        n_timestamps = simple_sensor_df["timestamp"].n_unique()
        assert len(result) == n_timestamps

    def test_with_physical_features(self, full_sensor_df):
        """With machine_profile, physical features should be added."""
        profile = {"has_bearings": True, "pulley_ratio": 0.5}
        result = compute_features(full_sensor_df, machine_profile=profile)
        assert "vibration_kurtosis" in result.columns
        assert "vibration_crest" in result.columns
        assert "fouling_indicator" in result.columns
        assert "belt_slip" in result.columns
        assert "oil_consumption_rate" in result.columns

    def test_inter_sensor_ratio(self, full_sensor_df):
        """Inter-sensor ratios should be present when both sensors exist."""
        result = compute_features(full_sensor_df)
        assert "vibration_rms_over_bearing_temp_inter_sensor_ratio" in result.columns
        assert "pressure_drop_over_flow_rate_inter_sensor_ratio" in result.columns


class TestGetFeatureNames:
    def test_returns_list(self):
        names = get_feature_names(["vibration_rms", "bearing_temp"])
        assert isinstance(names, list)
        assert len(names) > 0

    def test_includes_all_suffixes(self):
        names = get_feature_names(["sensor_a"])
        assert "sensor_a_value" in names
        assert "sensor_a_rolling_mean_5m" in names
        assert "sensor_a_rolling_std_5m" in names
        assert "sensor_a_rate_of_change" in names
        assert "sensor_a_z_score" in names
        assert "sensor_a_shift_adj_z" in names

    def test_inter_sensor_ratio_included(self):
        names = get_feature_names(["vibration_rms", "bearing_temp"])
        assert "vibration_rms_over_bearing_temp_inter_sensor_ratio" in names


class TestFilterCalibrationData:
    def test_excludes_anomaly_phase(self):
        df = pl.DataFrame({
            "machine_phase": ["NORMAL", "ANOMALY", "NORMAL", "FAILED"],
            "value": [1.0, 2.0, 3.0, 4.0],
        })
        result = filter_calibration_data(df)
        assert len(result) == 2
        assert result["machine_phase"].to_list() == ["NORMAL", "NORMAL"]

    def test_excludes_upstream_effect(self):
        df = pl.DataFrame({
            "machine_phase": ["NORMAL", "NORMAL"],
            "upstream_effect": [False, True],
            "value": [1.0, 2.0],
        })
        result = filter_calibration_data(df)
        assert len(result) == 1

    def test_no_columns_passthrough(self):
        df = pl.DataFrame({"value": [1.0, 2.0, 3.0]})
        result = filter_calibration_data(df)
        assert len(result) == 3
