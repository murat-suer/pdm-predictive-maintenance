import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))


class TestBaselineMigration:
    def test_single_baseline_exists(self):
        versions_dir = Path(__file__).resolve().parents[3] / "alembic" / "versions"
        migration_files = list(versions_dir.glob("*.py"))
        migration_files = [f for f in migration_files if f.name != "__pycache__" and not f.name.startswith("__")]
        assert len(migration_files) == 1, (
            f"Expected exactly 1 baseline migration, found {len(migration_files)}: "
            f"{[f.name for f in migration_files]}"
        )

    def test_baseline_filename(self):
        versions_dir = Path(__file__).resolve().parents[3] / "alembic" / "versions"
        baseline = versions_dir / "0001_baseline.py"
        assert baseline.exists(), "Migration file must be named 0001_baseline.py"

    def test_baseline_creates_all_16_tables(self):
        versions_dir = Path(__file__).resolve().parents[3] / "alembic" / "versions"
        baseline = versions_dir / "0001_baseline.py"
        content = baseline.read_text(encoding="utf-8")
        expected_tables = [
            "sensor_readings",
            "anomaly_log",
            "alarm_state",
            "alarm_state_transitions",
            "decision_log",
            "decision_audit_log",
            "machine_health_score",
            "canary_probe_log",
            "settings",
            "maintenance_log",
            "work_orders",
            "shift_reports",
            "ai_act_log",
            "machine_baselines",
            "decision_audit",
            "decision_work_orders",
        ]
        for table in expected_tables:
            assert f'"{table}"' in content or f"'{table}'" in content, (
                f"Baseline migration must create table '{table}'"
            )

    def test_baseline_includes_timescaledb(self):
        versions_dir = Path(__file__).resolve().parents[3] / "alembic" / "versions"
        baseline = versions_dir / "0001_baseline.py"
        content = baseline.read_text(encoding="utf-8")
        assert "timescaledb" in content.lower(), "Baseline must include TimescaleDB extension"
        assert "create_hypertable" in content, "Baseline must create hypertable on sensor_readings"

    def test_baseline_has_no_down_revision(self):
        versions_dir = Path(__file__).resolve().parents[3] / "alembic" / "versions"
        baseline = versions_dir / "0001_baseline.py"
        content = baseline.read_text(encoding="utf-8")
        assert 'down_revision' in content
        assert 'down_revision = None' in content or "down_revision: str = None" in content or "down_revision=None" in content.replace(" ", ""), (
            "Baseline migration must have down_revision = None (it's the first)"
        )
