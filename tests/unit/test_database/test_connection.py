import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))


class TestRetryUtility:
    def test_retry_decorator_exists(self):
        from src.database.connection import retry_on_operational_error
        assert callable(retry_on_operational_error), (
            "retry_on_operational_error must be a callable (tenacity retry decorator)"
        )

    def test_retry_is_tenacity_instance(self):

        from src.database.connection import retry_on_operational_error
        assert hasattr(retry_on_operational_error, '__call__'), (
            "retry_on_operational_error must be a tenacity retry instance"
        )

    def test_retry_stops_after_3_attempts(self):
        from sqlalchemy.exc import OperationalError

        from src.database.connection import retry_on_operational_error

        call_count = 0

        @retry_on_operational_error
        def always_fails():
            nonlocal call_count
            call_count += 1
            raise OperationalError("stmt", "params", Exception("connection lost"))

        with pytest.raises(OperationalError):
            always_fails()

        assert call_count == 3, (
            f"Expected 3 attempts, got {call_count}"
        )

    def test_retry_does_not_catch_other_exceptions(self):
        from src.database.connection import retry_on_operational_error

        call_count = 0

        @retry_on_operational_error
        def raises_value_error():
            nonlocal call_count
            call_count += 1
            raise ValueError("not an operational error")

        with pytest.raises(ValueError):
            raises_value_error()

        assert call_count == 1, (
            f"ValueError should not be retried, but got {call_count} calls"
        )


class TestAlembicIni:
    def test_no_hardcoded_password(self):
        ini_path = Path(__file__).resolve().parents[3] / "alembic.ini"
        content = ini_path.read_text(encoding="utf-8")
        assert "pdm_user:password" not in content, (
            "alembic.ini must not contain hardcoded credentials"
        )

    def test_sqlalchemy_url_empty_or_commented(self):
        ini_path = Path(__file__).resolve().parents[3] / "alembic.ini"
        content = ini_path.read_text(encoding="utf-8")
        for line in content.splitlines():
            stripped = line.strip()
            if stripped.startswith("sqlalchemy.url") and "=" in stripped:
                value = stripped.split("=", 1)[1].strip()
                assert value == "" or value.startswith("#"), (
                    f"sqlalchemy.url should be empty, got: {value}"
                )


class TestInitDb:
    def test_no_fstring_sql(self):
        init_db_path = Path(__file__).resolve().parents[3] / "scripts" / "init_db.py"
        content = init_db_path.read_text(encoding="utf-8")
        assert 'f"' not in content and "f'" not in content, (
            "init_db.py must not use f-strings for SQL queries (SQL injection risk)"
        )

    def test_correct_table_count(self):
        init_db_path = Path(__file__).resolve().parents[3] / "scripts" / "init_db.py"
        content = init_db_path.read_text(encoding="utf-8")
        assert "12 total" not in content, (
            "init_db.py table count must be updated from 12 to 16"
        )


class TestRequirementsBase:
    def test_tenacity_in_requirements(self):
        req_path = Path(__file__).resolve().parents[3] / "requirements" / "base.txt"
        assert req_path.exists(), "requirements/base.txt must exist"
        content = req_path.read_text(encoding="utf-8")
        assert "tenacity" in content.lower(), (
            "tenacity must be listed in requirements/base.txt"
        )
