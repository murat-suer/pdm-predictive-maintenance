"""
src/ml/feature_engineering.py
=======================================
Feature computation from raw sensor data using Polars.

RULES:
- This file ONLY uses Polars. pandas/sklearn/numpy imports are FORBIDDEN here.
- Output is a pivot table of rolling features plus profile-aware physical features.
- WINDOW_SIZE = 30 rows (5 minutes at 10s interval)
"""

import polars as pl

from src.ml.physical_features import (
    belt_slip_ratio,
    crest_factor,
    delta_p_per_flow,
    oil_consumption_rate,
    rolling_kurtosis,
)

WINDOW_SIZE = 30  # 30 rows = 5 minutes at 10s interval
SHIFT_WINDOW = 5  # shorter window for shift-adjusted z-score


def compute_features(raw_df: pl.DataFrame, machine_profile: dict | None = None) -> pl.DataFrame:
    """
    Compute 5 features per sensor from raw sensor readings.

    Input columns required: timestamp, sensor_name, value
    Output: wide pivoted DataFrame, one row per timestamp, columns:
        {sensor}_value
        {sensor}_rolling_mean_5m
        {sensor}_rolling_std_5m
        {sensor}_rate_of_change
        {sensor}_z_score
        {sensor}_shift_adj_z  (z-score over shorter shift window)

    Args:
        raw_df: Polars DataFrame with columns [timestamp, sensor_name, value]
                for a SINGLE machine over a time window (at least WINDOW_SIZE rows)
        machine_profile: Optional MACHINE_CONFIGS entry for profile-aware physical
                         features.

    Returns:
        Polars DataFrame with feature columns (last row = current features)
    """
    df = raw_df.sort("timestamp")

    # Get unique sensor names
    sensors = df["sensor_name"].unique().to_list()

    # Pivot to wide format: rows=timestamp, columns=sensor values
    pivoted = df.pivot(
        values="value",
        index="timestamp",
        on="sensor_name",
        aggregate_function="first",
    ).sort("timestamp")

    result = pivoted.clone()

    for sensor in sensors:
        if sensor not in pivoted.columns:
            continue

        # 1. Raw value (already exists as column)
        result = result.rename({sensor: f"{sensor}_value"})
        val_col = pl.col(f"{sensor}_value")

        # 2. Rolling mean (5 minutes = WINDOW_SIZE rows)
        result = result.with_columns(
            [
                val_col.rolling_mean(window_size=WINDOW_SIZE, min_samples=1).alias(f"{sensor}_rolling_mean_5m"),
            ]
        )

        # 3. Rolling std
        result = result.with_columns(
            [
                val_col.rolling_std(window_size=WINDOW_SIZE, min_samples=2)
                .fill_null(0.0)
                .alias(f"{sensor}_rolling_std_5m"),
            ]
        )

        # 4. Rate of change (current - previous)
        result = result.with_columns(
            [
                (val_col - val_col.shift(1)).fill_null(0.0).alias(f"{sensor}_rate_of_change"),
            ]
        )

        # 5. Z-score over WINDOW_SIZE
        result = result.with_columns(
            [
                _safe_z_score(f"{sensor}_value", WINDOW_SIZE).alias(f"{sensor}_z_score"),
            ]
        )

        # 6. Shift-adjusted z-score (shorter window = more sensitive)
        result = result.with_columns(
            [
                _safe_z_score(f"{sensor}_value", SHIFT_WINDOW).alias(f"{sensor}_shift_adj_z"),
            ]
        )

    # 7. Inter-sensor ratios
    sensor_pairs = [
        ("vibration_rms", "bearing_temp"),
        ("pressure_drop", "flow_rate"),
    ]
    for s1, s2 in sensor_pairs:
        if f"{s1}_value" in result.columns and f"{s2}_value" in result.columns:
            ratio_name = f"{s1}_over_{s2}_inter_sensor_ratio"
            result = result.with_columns(
                [
                    (pl.col(f"{s1}_value") / pl.col(f"{s2}_value").clip(0.001)).alias(ratio_name),
                ]
            )

    return _add_physical_features(result, machine_profile or {})


def _add_physical_features(result: pl.DataFrame, machine_profile: dict) -> pl.DataFrame:
    """Append sensor-specific diagnostic features when required inputs exist."""
    if machine_profile.get("has_bearings") and "vibration_rms_value" in result.columns:
        result = result.with_columns(
            [
                rolling_kurtosis(result["vibration_rms_value"], window=WINDOW_SIZE).alias("vibration_kurtosis"),
                crest_factor(result["vibration_rms_value"], window=WINDOW_SIZE).alias("vibration_crest"),
            ]
        )

    if {"pressure_drop_value", "flow_rate_value"}.issubset(result.columns):
        result = result.with_columns(
            delta_p_per_flow(result["pressure_drop_value"], result["flow_rate_value"]).alias("fouling_indicator")
        )

    pulley_ratio = machine_profile.get("pulley_ratio")
    if pulley_ratio and {"motor_rpm_value", "output_rpm_value"}.issubset(result.columns):
        result = result.with_columns(
            belt_slip_ratio(
                result["motor_rpm_value"],
                result["output_rpm_value"],
                pulley_ratio=pulley_ratio,
            ).alias("belt_slip")
        )

    if "oil_level_value" in result.columns:
        result = result.with_columns(oil_consumption_rate(result["oil_level_value"]).alias("oil_consumption_rate"))

    return result


def _safe_z_score(col_name: str, window: int) -> pl.Expr:
    """
    Compute z-score = (value - rolling_mean) / rolling_std.
    Returns 0.0 when rolling_std is 0 or null.
    """
    val = pl.col(col_name)
    mu = val.rolling_mean(window_size=window, min_samples=1).shift(1)
    std = val.rolling_std(window_size=window, min_samples=2).shift(1).fill_null(1e-10)
    safe_std = pl.when(std < 1e-10).then(pl.lit(1e-10)).otherwise(std)
    return (val - mu) / safe_std


def get_feature_names(sensor_names: list) -> list:
    """Return ordered list of feature column names for a given sensor list."""
    features = []
    suffixes = [
        "_value",
        "_rolling_mean_5m",
        "_rolling_std_5m",
        "_rate_of_change",
        "_z_score",
        "_shift_adj_z",
    ]
    for sensor in sorted(sensor_names):
        for suffix in suffixes:
            features.append(f"{sensor}{suffix}")

    sensor_pairs = [
        ("vibration_rms", "bearing_temp"),
        ("pressure_drop", "flow_rate"),
    ]
    for s1, s2 in sensor_pairs:
        if s1 in sensor_names and s2 in sensor_names:
            features.append(f"{s1}_over_{s2}_inter_sensor_ratio")

    return features


def filter_calibration_data(
    raw_df: pl.DataFrame,
    exclude_phases: list = None,
) -> pl.DataFrame:
    """
    Filter training data for ML model calibration.
    Excludes ANOMALY, FAILED, IDLE, CALIBRATING, and CANARY_EVENT phases.
    Excludes rows where upstream_effect=True.
    """
    if exclude_phases is None:
        exclude_phases = ["ANOMALY", "FAILED", "IDLE", "CALIBRATING", "CANARY_EVENT"]

    filtered = raw_df
    if "machine_phase" in raw_df.columns:
        filtered = filtered.filter(~pl.col("machine_phase").is_in(exclude_phases))
    if "upstream_effect" in raw_df.columns:
        filtered = filtered.filter(pl.col("upstream_effect") == False)
    return filtered
