"""
Decision Subscriber (Phase 2D - Pipeline & Integration Layer).

Redis Streams consumer that:
- Reads anomaly events from anomaly_stream
- Creates AlarmState records
- Generates decision scenarios via DecisionEngine
- Implements ISA-18.2 alarm flood detection (R-010)
- Handles periodic tick (shelve expiry, L2 timeout)
- Handles demo operator tick (pending decisions)
- Logs EU AI Act compliance events
"""

import json
import logging
from collections import deque
from datetime import UTC, datetime, timedelta
from typing import Any

from src.database.models import (
    AIActLog,
    AlarmState,
    AlarmStateTransition,
    DecisionLog,
)
from src.decision.decision_engine import (
    DecisionEngine,
)


# ---------------------------------------------------------------------------
# ScenarioOption: wrapper that supports both attribute and dict-like access
# ---------------------------------------------------------------------------
class ScenarioOption:
    """
    Wrapper around RecommendedOption that supports both attribute access
    and dict-like .get() for test compatibility.
    """

    def __init__(
        self,
        scenario,
        cost: float,
        is_recommended: bool = False,
        is_valid: bool = True,
        expected_cost: float = 0.0,
        failure_probability: float = 0.0,
    ):
        self.scenario = scenario
        self.cost = cost
        self.is_recommended = is_recommended
        self.is_valid = is_valid
        self.expected_cost = expected_cost
        self.failure_probability = failure_probability

    def get(self, key, default=None):
        return getattr(self, key, default)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
ANOMALY_STREAM = "anomaly_stream"
ALARM_FLOOD_THRESHOLD = 10  # ISA-18.2 R-010: 10 alarms
ALARM_FLOOD_WINDOW_MINUTES = 10  # ISA-18.2 R-010: in 10 minutes
DEMO_FLOOD_THRESHOLD = 100  # DEMO_MODE threshold
FLOOD_RECOVERY_MINUTES = 5  # Recovery: 5 minutes below threshold
PERIODIC_TICK_INTERVAL_S = 300  # 5 minutes


# ---------------------------------------------------------------------------
# DecisionSubscriber
# ---------------------------------------------------------------------------
class DecisionSubscriber:
    """
    Redis Streams consumer for anomaly events.

    Reads from anomaly_stream, creates AlarmState records, generates
    decision scenarios, and handles alarm flood detection per ISA-18.2.
    """

    def __init__(
        self,
        redis_client,
        db_session,
        consumer_group: str = "decision_consumers",
        consumer_name: str = "decision-worker-1",
        demo_mode: bool = False,
    ):
        self._redis = redis_client
        self._db = db_session
        self._consumer_group = consumer_group
        self._consumer_name = consumer_name
        self._demo_mode = demo_mode

        # Decision engine for scenario generation
        self._decision_engine = DecisionEngine()

        # Alarm flood detection (ISA-18.2 R-010)
        self._alarm_timestamps: deque[datetime] = deque()
        self._flood_active: bool = False
        self._flood_last_active_at: datetime | None = None

        # Track processed anomaly IDs to avoid duplicates
        self._processed_anomaly_ids: set[int] = set()

    # -------------------------------------------------------------------
    # Consumer Group Management
    # -------------------------------------------------------------------
    def _ensure_consumer_group(self):
        """Create Redis consumer group, ignoring BUSYGROUP errors."""
        import redis as redis_lib

        try:
            self._redis.xgroup_create(
                ANOMALY_STREAM,
                self._consumer_group,
                id="0",
                mkstream=True,
            )
        except redis_lib.exceptions.ResponseError as e:
            if "BUSYGROUP" in str(e):
                # Group already exists, that's fine
                pass
            else:
                raise

    # -------------------------------------------------------------------
    # Event Parsing
    # -------------------------------------------------------------------
    def _parse_event_fields(self, event: dict) -> dict[str, str]:
        """
        Parse event fields from Redis Stream message.

        Handles both byte keys and string keys.
        """
        fields = {}
        for key, value in event.items():
            # Normalize key to string
            if isinstance(key, bytes):
                key = key.decode("utf-8")
            # Normalize value to string
            if isinstance(value, bytes):
                value = value.decode("utf-8")
            elif value is not None:
                value = str(value)
            fields[key] = value
        return fields

    # -------------------------------------------------------------------
    # Alarm Flood Detection (ISA-18.2 R-010)
    # -------------------------------------------------------------------
    def _record_alarm_timestamp(self, ts: datetime):
        """Record an alarm timestamp for flood detection."""
        self._alarm_timestamps.append(ts)

    def _is_flood_active(self, now: datetime | None = None) -> bool:
        """
        Check if alarm flood is active per ISA-18.2 R-010.

        Flood is active when >= threshold alarms within the window.
        Recovery after FLOOD_RECOVERY_MINUTES since last active detection.
        """
        if now is None:
            now = datetime.now(UTC)

        threshold = DEMO_FLOOD_THRESHOLD if self._demo_mode else ALARM_FLOOD_THRESHOLD

        # Check recovery FIRST: if enough time has passed since last detection,
        # flood is considered recovered regardless of current alarm count.
        if self._flood_active and self._flood_last_active_at:
            recovery_time = self._flood_last_active_at + timedelta(
                minutes=FLOOD_RECOVERY_MINUTES
            )
            if now >= recovery_time:
                self._flood_active = False
                return False

        # Count alarms within window
        window_start = now - timedelta(minutes=ALARM_FLOOD_WINDOW_MINUTES)
        count = sum(1 for ts in self._alarm_timestamps if ts >= window_start)

        if count >= threshold:
            self._flood_active = True
            self._flood_last_active_at = now
            return True

        self._flood_active = False
        return False

    # -------------------------------------------------------------------
    # Event Processing
    # -------------------------------------------------------------------
    def _process_anomaly_event(self, event: dict) -> Any | None:
        """
        Process a single anomaly event from the stream.

        Creates AlarmState record and generates decision scenarios.
        Handles edge cases: empty events, missing fields, duplicates.
        """
        # Handle empty events
        if not event:
            return None

        # Parse fields
        fields = self._parse_event_fields(event)

        # Validate required fields
        machine_id = fields.get("machine_id")
        if not machine_id:
            return None

        anomaly_id_str = fields.get("anomaly_id")
        if not anomaly_id_str:
            return None

        # Validate anomaly_id is numeric
        try:
            anomaly_id = int(anomaly_id_str)
        except (ValueError, TypeError):
            return None

        # Check for duplicate anomaly_id
        if anomaly_id in self._processed_anomaly_ids:
            return None

        # Extract other fields
        anomaly_score = float(fields.get("anomaly_score", "0.0"))
        severity = fields.get("severity", "WARNING")
        fields.get("event_type", "ANOMALY_DETECTED")
        timestamp_str = fields.get("timestamp")
        fields.get("top_contributing_sensor", "")
        phase = fields.get("phase", "HEALTHY")
        rul_hours = fields.get("rul_hours")

        # Parse timestamp
        if timestamp_str:
            try:
                timestamp = datetime.fromisoformat(timestamp_str)
            except (ValueError, TypeError):
                timestamp = datetime.now(UTC)
        else:
            timestamp = datetime.now(UTC)

        # Check alarm flood
        now = datetime.now(UTC)
        flood_active = self._is_flood_active(now)

        # During flood, suppress WARNING-level alarms
        if flood_active and severity == "WARNING":
            # Still mark as processed but return with suppressed flag
            self._processed_anomaly_ids.add(anomaly_id)
            self._record_alarm_timestamp(now)

            class SuppressedResult:
                pass

            result = SuppressedResult()
            result.suppressed = True
            result.machine_id = machine_id
            result.anomaly_id = anomaly_id
            return result

        # Create AlarmState record
        try:
            alarm = AlarmState(
                anomaly_id=anomaly_id,
                machine_id=machine_id,
                level=self._severity_to_level(severity),
                status="UNACKNOWLEDGED",
                created_at=timestamp,
                last_updated=now,
            )
            self._db.add(alarm)

            # Mark as processed
            self._processed_anomaly_ids.add(anomaly_id)
            self._record_alarm_timestamp(now)

            # Generate decision scenarios
            try:
                rul = float(rul_hours) if rul_hours else 24.0
                scenarios = self._generate_scenarios(
                    machine_id=machine_id, rul_hours=rul
                )
                scenarios_data = [
                    {
                        "scenario": s.scenario.value
                        if hasattr(s.scenario, "value")
                        else str(s.scenario),
                        "cost": s.cost,
                        "expected_cost": round(s.expected_cost, 2),
                        "failure_probability": round(s.failure_probability, 4),
                        "is_recommended": s.is_recommended,
                    }
                    for s in scenarios
                ]

                # Decision context for the UI and the savings accounting:
                # the run-to-failure counterfactual this decision avoids.
                decision_context = None
                engine = getattr(self, "_last_engine", None)
                profile = getattr(self, "_last_profile", None)
                if engine is not None and profile is not None:
                    decision_context = json.dumps(
                        {
                            "run_to_failure_cost": round(
                                engine.run_to_failure_cost(profile), 2
                            ),
                            "rul_hours": rul,
                        }
                    )

                # Find recommended scenario
                ai_recommendation = None
                for s in scenarios:
                    if s.is_recommended:
                        ai_recommendation = (
                            s.scenario.value
                            if hasattr(s.scenario, "value")
                            else str(s.scenario)
                        )
                        break

                # EEMUA 191 anti-pattern guard: repeated OBSERVE escalates to
                # a technician dispatch and eventually loses the OBSERVE option.
                from src.decision.observation_policy import (
                    apply_observe_escalation,
                    fault_is_identified,
                    observe_streak,
                )

                streak = observe_streak(self._db, machine_id)
                if streak > 0:
                    scenarios_data, ai_recommendation = apply_observe_escalation(
                        scenarios_data,
                        ai_recommendation,
                        streak,
                        fault_is_identified(self._db, anomaly_id),
                    )

                # The AlarmState after_insert listener already creates a
                # PENDING DecisionLog for this alarm — enrich it instead of
                # inserting a duplicate row.
                self._db.flush()
                decision = (
                    self._db.query(DecisionLog)
                    .filter(DecisionLog.alarm_id == alarm.id)
                    .first()
                )
                if decision is None:
                    decision = DecisionLog(
                        alarm_id=alarm.id,
                        machine_id=machine_id,
                        action="PENDING",
                        created_at=now,
                    )
                    self._db.add(decision)
                decision.ai_recommendation = ai_recommendation
                decision.scenarios_presented = scenarios_data
                decision.notes = decision_context

                # Log EU AI Act compliance
                self._log_ai_act_compliance(
                    machine_id=machine_id,
                    model_version="v2.1.0",
                    decision_type="ANOMALY_RESPONSE",
                    features_snapshot={
                        "anomaly_score": anomaly_score,
                        "severity": severity,
                        "phase": phase,
                    },
                    output={"recommended": ai_recommendation},
                    action_taken="SCENARIO_GENERATED",
                )

            except Exception as e:
                # Scenario generation failed - create fallback
                logger.error(f"Scenario generation failed: {e}")
                self._handle_scenario_failure(
                    machine_id=machine_id,
                    anomaly_id=anomaly_id,
                    alarm_id=alarm.id if hasattr(alarm, "id") else 0,
                    error=str(e),
                )

            self._db.commit()
            return alarm

        except Exception:
            self._db.rollback()
            return None

    def _severity_to_level(self, severity: str) -> int:
        """Convert severity string to alarm level."""
        if severity == "CRITICAL":
            return 2
        return 1

    # -------------------------------------------------------------------
    # Scenario Generation
    # -------------------------------------------------------------------
    def _generate_scenarios(
        self, machine_id: str, rul_hours: float = 24.0
    ) -> list[ScenarioOption]:
        """
        Generate decision scenarios for a machine.

        Uses DecisionEngine to evaluate all scenarios and return
        sorted recommendations wrapped in ScenarioOption.
        """
        from src.decision.machine_profiles import (
            build_engine,
            build_profile,
            load_financials,
        )

        financials = load_financials(self._db)
        profile = build_profile(machine_id, financials)
        engine = build_engine(machine_id, financials)
        self._last_engine = engine
        self._last_profile = profile

        recommendations = engine.recommend(
            machine_profile=profile,
            rul_hours=rul_hours,
        )

        return [
            ScenarioOption(
                scenario=r.scenario,
                cost=r.cost,
                is_recommended=r.is_recommended,
                is_valid=r.is_valid,
                expected_cost=r.expected_cost,
                failure_probability=r.failure_probability,
            )
            for r in recommendations
        ]

    # -------------------------------------------------------------------
    # Scenario Failure Handling (Reconciliation)
    # -------------------------------------------------------------------
    def _handle_scenario_failure(
        self,
        machine_id: str,
        anomaly_id: int,
        alarm_id: int,
        error: str,
    ) -> DecisionLog:
        """
        Handle scenario generation failure by creating a fallback DecisionLog.

        Even on failure, EU AI Act compliance log is created.
        """
        now = datetime.now(UTC)

        # Create fallback DecisionLog
        fallback = DecisionLog(
            machine_id=machine_id,
            alarm_id=alarm_id,
            action="PENDING",
            resolution_source="FALLBACK",
            scenarios_presented=None,
            notes=f"Scenario generation failed: {error}",
            created_at=now,
        )
        self._db.add(fallback)

        # Log EU AI Act compliance even on failure
        self._log_ai_act_compliance(
            machine_id=machine_id,
            model_version="v2.1.0",
            decision_type="ANOMALY_RESPONSE",
            features_snapshot={"error": error},
            output={"fallback": True},
            action_taken="SCENARIO_GENERATION_FAILED",
        )

        try:
            self._db.commit()
        except Exception:
            self._db.rollback()

        return fallback

    # -------------------------------------------------------------------
    # EU AI Act Compliance Logging
    # -------------------------------------------------------------------
    def _log_ai_act_compliance(
        self,
        machine_id: str,
        model_version: str,
        decision_type: str,
        features_snapshot: dict,
        output: dict,
        action_taken: str,
    ):
        """
        Log an EU AI Act compliance event.

        Required for transparency and traceability under EU AI Act.
        """
        now = datetime.now(UTC)

        log_entry = AIActLog(
            timestamp=now,
            machine_id=machine_id,
            model_version=model_version,
            decision_type=decision_type,
            features_snapshot=features_snapshot,
            output=output,
            action_taken=action_taken,
        )
        self._db.add(log_entry)

    # -------------------------------------------------------------------
    # Periodic Tick (5 min): shelve expiry, L2 timeout
    # -------------------------------------------------------------------
    def _periodic_tick(self):
        """
        Periodic maintenance tick (every 5 minutes).

        Checks:
        - Shelve expiry (return alarms to NORMAL)
        - L2 timeout (escalate to manager)
        """
        self._check_shelve_expiry()
        self._check_l2_timeouts()

    def _check_shelve_expiry(self):
        """Check for expired shelves and return alarms to NORMAL."""
        now = datetime.now(UTC)

        # Query for expired shelves
        expired_alarms = (
            self._db.query(AlarmState)
            .filter(
                AlarmState.status == "SHELVED",
                AlarmState.shelved_until <= now,
            )
            .all()
        )

        for alarm in expired_alarms:
            old_status = alarm.status
            alarm.status = "NORMAL"
            alarm.last_updated = now

            # Record transition
            transition = AlarmStateTransition(
                alarm_id=alarm.id,
                from_state=old_status,
                to_state="NORMAL",
                reason="Shelve expired",
                timestamp=now,
            )
            self._db.add(transition)

        if expired_alarms:
            try:
                self._db.commit()
            except Exception:
                self._db.rollback()

    def _check_l2_timeouts(self):
        """Check for L2 escalation timeouts and escalate to manager."""
        now = datetime.now(UTC)

        # Query for overdue decisions at L2
        overdue_decisions = (
            self._db.query(DecisionLog)
            .filter(
                DecisionLog.action == "PENDING",
                DecisionLog.due_at <= now,
            )
            .all()
        )

        for decision in overdue_decisions:
            decision.escalation_level = max(
                (decision.escalation_level or 1) + 1, 2
            )

        if overdue_decisions:
            try:
                self._db.commit()
            except Exception:
                self._db.rollback()

    # NOTE: demo-time auto-approval lives in operator_simulator.act_on_due_decisions
    # (invoked by runner.py) — it resolves through the full chain: alarm
    # transition + audit log + maintenance scheduling, under a bot identity.

    # -------------------------------------------------------------------
    # Poll Loop
    # -------------------------------------------------------------------
    def _poll_once(self):
        """
        Poll Redis Streams once for new messages.

        Handles connection errors gracefully.
        """
        try:
            messages = self._redis.xreadgroup(
                groupname=self._consumer_group,
                consumername=self._consumer_name,
                streams={ANOMALY_STREAM: ">"},
                count=10,
                block=1000,
            )
        except ConnectionError as e:
            logger.error(f"Redis connection error: {e}")
            return
        except Exception as e:
            logger.error(f"Redis error during poll: {e}")
            return

        if not messages:
            return

        for _stream_name, stream_messages in messages:
            for message_id, event in stream_messages:
                try:
                    self._process_anomaly_event(event)
                    # Acknowledge message
                    self._redis.xack(
                        ANOMALY_STREAM,
                        self._consumer_group,
                        message_id,
                    )
                except Exception as e:
                    logger.error(
                        f"Error processing message {message_id}: {e}"
                    )
