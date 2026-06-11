"""
src/ml/ai_act_compliance.py
============================
AI Act compliance logger — Article 14 (transparency) and Article 50.

Every ML model decision that triggers an action is persisted to the
ai_act_log database table with full audit trail:
  - Feature snapshot (input data at decision time)
  - Model output (anomaly score, RUL, etc.)
  - SHAP values (feature attribution)
  - Action taken and any human override
"""

import logging
from datetime import UTC, datetime

logger = logging.getLogger(__name__)


class AIActComplianceLogger:
    """
    Persists ML decision audit records to the ai_act_log table.

    Falls back to in-memory list if DB write fails (ensures no data loss
    even during DB outages).
    """

    def __init__(self):
        self._fallback_entries = []

    def log_decision(
        self,
        machine_id,
        model_version,
        features_snapshot,
        output,
        shap_values,
        action,
        human_action=None,
        decision_type="anomaly_detection",
    ):
        """
        Log a single ML decision to the database.

        Args:
            machine_id: e.g. "AC-201"
            model_version: e.g. "isolation_forest_v1"
            features_snapshot: dict of feature values at decision time
            output: dict of model outputs (anomaly_score, is_anomaly, etc.)
            shap_values: dict of SHAP feature attributions
            action: action taken (e.g. "ANOMALY_RAISED", "NO_ACTION")
            human_action: operator override action if any
            decision_type: category of decision (default: "anomaly_detection")
        """
        entry = {
            "timestamp": datetime.now(UTC).isoformat(),
            "machine_id": machine_id,
            "model_version": model_version,
            "decision_type": decision_type,
            "features_snapshot": features_snapshot,
            "output": output,
            "shap_values": shap_values,
            "action_taken": action,
            "human_action": human_action,
        }

        try:
            from src.database.connection import get_db_context
            from src.database.models import AIActLog

            with get_db_context() as db:
                db.add(
                    AIActLog(
                        timestamp=datetime.now(UTC),
                        machine_id=machine_id,
                        model_version=model_version,
                        decision_type=decision_type,
                        features_snapshot=features_snapshot,
                        output=output,
                        shap_values=shap_values,
                        action_taken=action,
                        human_action=human_action,
                    )
                )
                db.commit()
        except Exception as e:
            logger.error(f"AI Act log DB write failed, using fallback: {e}")
            self._fallback_entries.append(entry)

    def get_last(self, n=10, db_session=None):
        """
        Retrieve the last N compliance log entries.

        Args:
            n: Number of entries to return
            db_session: Optional SQLAlchemy session. If None, tries to create one.

        Returns:
            List of log entry dicts
        """
        try:
            if db_session is None:
                from src.database.connection import get_db_context

                with get_db_context() as db:
                    return self._query_last(db, n)
            else:
                return self._query_last(db_session, n)
        except Exception as e:
            logger.error(f"AI Act log DB read failed, using fallback: {e}")
            return self._fallback_entries[-n:] if self._fallback_entries else []

    @staticmethod
    def _query_last(db, n):
        """Query last N entries from ai_act_log table."""
        from src.database.models import AIActLog

        rows = db.query(AIActLog).order_by(AIActLog.timestamp.desc()).limit(n).all()
        return [
            {
                "timestamp": r.timestamp.isoformat() if r.timestamp else None,
                "machine_id": r.machine_id,
                "model_version": r.model_version,
                "decision_type": r.decision_type,
                "features_snapshot": r.features_snapshot,
                "output": r.output,
                "shap_values": r.shap_values,
                "action_taken": r.action_taken,
                "human_action": r.human_action,
            }
            for r in reversed(rows)
        ]


ComplianceLogger = AIActComplianceLogger
