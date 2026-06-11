"""
Tests for database CRUD operations using in-memory SQLite.

Tests insert and query operations for:
- SensorReading: insert and retrieve sensor data
- MachineBaseline: round-trip insert and query
"""
import sys
from datetime import UTC, datetime
from pathlib import Path

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from src.database.models import (
    MachineBaseline,
    SensorReading,
)


@pytest.fixture
def db_session():
    """Create an in-memory SQLite session for testing."""
    engine = create_engine("sqlite:///:memory:", echo=False)

    # SQLite doesn't support PostgreSQL ENUM types, so we need to
    # handle the schema creation carefully. We create tables that
    # don't use PG-specific features.
    # For this test, we only need SensorReading and MachineBaseline
    # which don't use PG_ENUM types directly.

    # Create only the tables we need (avoid PG-specific DDL)
    SensorReading.__table__.create(engine, checkfirst=True)
    MachineBaseline.__table__.create(engine, checkfirst=True)

    SessionLocal = sessionmaker(bind=engine)
    session = SessionLocal()
    yield session
    session.close()
    engine.dispose()


class TestSensorReadingCRUD:
    def test_insert_and_query_sensor_reading(self, db_session):
        """Insert a SensorReading and query it back."""
        now = datetime.now(UTC)
        reading = SensorReading(
            machine_id="CM-203",
            timestamp=now,
            sensor_name="vibration_rms",
            value=2.5,
            is_anomaly=False,
            anomaly_score=0.1,
            upstream_effect=False,
            machine_phase="HEALTHY",
            present=True,
        )
        db_session.add(reading)
        db_session.commit()

        # Query it back
        result = (
            db_session.query(SensorReading)
            .filter_by(machine_id="CM-203", sensor_name="vibration_rms")
            .first()
        )

        assert result is not None, "SensorReading should be found after insert"
        assert result.machine_id == "CM-203"
        assert result.sensor_name == "vibration_rms"
        assert result.value == 2.5
        assert result.is_anomaly is False
        assert result.machine_phase == "HEALTHY"
        assert result.present is True


class TestMachineBaselineCRUD:
    def test_machine_baseline_roundtrip(self, db_session):
        """Insert a MachineBaseline and query it back — full round-trip."""
        baseline = MachineBaseline(
            machine_id="CM-303",
            sensor="bearing_temp",
            mean_value=65.0,
            std_value=2.5,
            sample_count=1000,
            calibrated_at=datetime.now(UTC),
            is_active=True,
        )
        db_session.add(baseline)
        db_session.commit()

        # Query it back
        result = (
            db_session.query(MachineBaseline)
            .filter_by(machine_id="CM-303", sensor="bearing_temp")
            .first()
        )

        assert result is not None, "MachineBaseline should be found after insert"
        assert result.machine_id == "CM-303"
        assert result.sensor == "bearing_temp"
        assert result.mean_value == 65.0
        assert result.std_value == 2.5
        assert result.sample_count == 1000
        assert result.is_active is True

    def test_machine_baseline_update(self, db_session):
        """Update a MachineBaseline and verify the change persists."""
        baseline = MachineBaseline(
            machine_id="CM-203",
            sensor="vibration_rms",
            mean_value=1.0,
            std_value=0.1,
            sample_count=500,
            calibrated_at=datetime.now(UTC),
            is_active=True,
        )
        db_session.add(baseline)
        db_session.commit()

        # Update
        baseline.mean_value = 1.5
        baseline.sample_count = 1000
        db_session.commit()

        # Verify
        result = (
            db_session.query(MachineBaseline)
            .filter_by(machine_id="CM-203", sensor="vibration_rms")
            .first()
        )
        assert result.mean_value == 1.5
        assert result.sample_count == 1000
