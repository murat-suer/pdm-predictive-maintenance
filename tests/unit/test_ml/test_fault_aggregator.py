"""Tests for the FaultAggregator sliding-window voting module."""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest

from src.ml.fault_aggregator import (
    MIN_AGREEMENT,
    WINDOW_SIZE,
    WINDOW_TTL_SECONDS,
    ConfirmedFault,
    FaultAggregator,
    _decode_entry,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_redis():
    """Return a mock Redis client with a working pipeline."""
    client = MagicMock()
    pipeline = MagicMock()
    client.pipeline.return_value = pipeline
    return client, pipeline


def _make_entry(fault_type: str, confidence: float, t: float) -> str:
    return json.dumps({"t": t, "ft": fault_type, "c": confidence})


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

class TestConstants:
    def test_window_size(self):
        assert WINDOW_SIZE == 5

    def test_min_agreement(self):
        assert MIN_AGREEMENT == 3

    def test_window_ttl(self):
        assert WINDOW_TTL_SECONDS == 60


# ---------------------------------------------------------------------------
# ConfirmedFault dataclass
# ---------------------------------------------------------------------------

class TestConfirmedFault:
    def test_creation(self):
        cf = ConfirmedFault(
            machine_id="m1",
            fault_type="bearing",
            confidence=0.8,
            votes=4,
            window_size=5,
            time_to_consensus_s=2.5,
        )
        assert cf.machine_id == "m1"
        assert cf.fault_type == "bearing"
        assert cf.confidence == 0.8
        assert cf.votes == 4
        assert cf.window_size == 5
        assert cf.time_to_consensus_s == 2.5

    def test_frozen(self):
        cf = ConfirmedFault("m1", "bearing", 0.8, 4, 5, 2.5)
        with pytest.raises(AttributeError):
            cf.machine_id = "m2"


# ---------------------------------------------------------------------------
# _decode_entry helper
# ---------------------------------------------------------------------------

class TestDecodeEntry:
    def test_decode_str(self):
        entry = '{"t": 1.0, "ft": "bearing", "c": 0.9}'
        result = _decode_entry(entry)
        assert result == {"t": 1.0, "ft": "bearing", "c": 0.9}

    def test_decode_bytes(self):
        entry = b'{"t": 2.0, "ft": "gear", "c": 0.7}'
        result = _decode_entry(entry)
        assert result == {"t": 2.0, "ft": "gear", "c": 0.7}


# ---------------------------------------------------------------------------
# FaultAggregator initialization
# ---------------------------------------------------------------------------

class TestFaultAggregatorInit:
    def test_init(self, mock_redis):
        client, _ = mock_redis
        agg = FaultAggregator(client)
        assert agg.r is client


# ---------------------------------------------------------------------------
# FaultAggregator._key
# ---------------------------------------------------------------------------

class TestFaultAggregatorKey:
    def test_key_format(self):
        assert FaultAggregator._key("m1") == "fault_window:m1"
        assert FaultAggregator._key("machine-42") == "fault_window:machine-42"


# ---------------------------------------------------------------------------
# FaultAggregator.submit — None fault_type
# ---------------------------------------------------------------------------

class TestSubmitNoneFault:
    def test_returns_none_for_none_fault(self, mock_redis):
        client, _ = mock_redis
        agg = FaultAggregator(client)
        result = agg.submit("m1", None, 0.9)
        assert result is None
        # Pipeline should NOT have been called
        client.pipeline.assert_not_called()


# ---------------------------------------------------------------------------
# FaultAggregator.submit — consensus reached
# ---------------------------------------------------------------------------

class TestSubmitConsensus:
    def test_consensus_reached(self, mock_redis):
        """3 out of 5 votes for 'bearing' → consensus."""
        client, pipeline = mock_redis
        t0 = 1000.0
        window = [
            _make_entry("bearing", 0.9, t0),
            _make_entry("bearing", 0.85, t0 + 1),
            _make_entry("gear", 0.6, t0 + 2),
            _make_entry("bearing", 0.95, t0 + 3),
            _make_entry("gear", 0.7, t0 + 4),
        ]
        pipeline.execute.return_value = [None, None, None, window]
        # The last submit was "bearing"
        agg = FaultAggregator(client)
        result = agg.submit("m1", "bearing", 0.95)

        assert result is not None
        assert isinstance(result, ConfirmedFault)
        assert result.machine_id == "m1"
        assert result.fault_type == "bearing"
        assert result.votes == 3
        assert result.window_size == WINDOW_SIZE
        # confidence = (votes/WINDOW_SIZE) * mean_confidence
        # mean_confidence = (0.9 + 0.85 + 0.95) / 3 = 0.9
        # confidence = (3/5) * 0.9 = 0.54
        assert abs(result.confidence - 0.54) < 1e-6
        # time_to_consensus = items[-1]["t"] - items[0]["t"] = (t0+4) - t0 = 4.0
        assert abs(result.time_to_consensus_s - 4.0) < 1e-6

    def test_pipeline_operations(self, mock_redis):
        """Verify correct Redis pipeline calls."""
        client, pipeline = mock_redis
        t0 = 1000.0
        window = [
            _make_entry("bearing", 0.9, t0),
            _make_entry("bearing", 0.85, t0 + 1),
            _make_entry("bearing", 0.95, t0 + 2),
        ]
        pipeline.execute.return_value = [None, None, None, window]

        agg = FaultAggregator(client)
        agg.submit("m1", "bearing", 0.95)

        # Verify pipeline calls
        pipeline.rpush.assert_called_once()
        pipeline.ltrim.assert_called_once_with("fault_window:m1", -WINDOW_SIZE, -1)
        pipeline.expire.assert_called_once_with("fault_window:m1", WINDOW_TTL_SECONDS)
        pipeline.lrange.assert_called_once_with("fault_window:m1", 0, -1)

    def test_all_votes_same_fault(self, mock_redis):
        """All 5 votes for same fault."""
        client, pipeline = mock_redis
        t0 = 500.0
        window = [
            _make_entry("gear", 0.8, t0 + i) for i in range(5)
        ]
        pipeline.execute.return_value = [None, None, None, window]

        agg = FaultAggregator(client)
        result = agg.submit("m1", "gear", 0.8)

        assert result is not None
        assert result.votes == 5
        # confidence = (5/5) * 0.8 = 0.8
        assert abs(result.confidence - 0.8) < 1e-6


# ---------------------------------------------------------------------------
# FaultAggregator.submit — no consensus
# ---------------------------------------------------------------------------

class TestSubmitNoConsensus:
    def test_below_min_agreement(self, mock_redis):
        """Only 2 votes for any fault type → no consensus."""
        client, pipeline = mock_redis
        t0 = 1000.0
        window = [
            _make_entry("bearing", 0.9, t0),
            _make_entry("gear", 0.8, t0 + 1),
            _make_entry("bearing", 0.85, t0 + 2),
        ]
        pipeline.execute.return_value = [None, None, None, window]

        agg = FaultAggregator(client)
        result = agg.submit("m1", "bearing", 0.85)

        # top_fault = "bearing" with 2 votes, but 2 < MIN_AGREEMENT (3)
        assert result is None

    def test_top_fault_differs_from_current(self, mock_redis):
        """Top fault is 'gear' but current submission is 'bearing' → no consensus."""
        client, pipeline = mock_redis
        t0 = 1000.0
        window = [
            _make_entry("gear", 0.9, t0),
            _make_entry("gear", 0.85, t0 + 1),
            _make_entry("gear", 0.8, t0 + 2),
            _make_entry("bearing", 0.7, t0 + 3),
            _make_entry("bearing", 0.75, t0 + 4),
        ]
        pipeline.execute.return_value = [None, None, None, window]

        agg = FaultAggregator(client)
        # Current submission is "bearing" but top_fault is "gear"
        result = agg.submit("m1", "bearing", 0.75)

        # top_fault="gear" (3 votes) but fault_type="bearing" → mismatch → None
        assert result is None


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_single_entry_in_window(self, mock_redis):
        """Only one entry in window — can't reach consensus."""
        client, pipeline = mock_redis
        t0 = 1000.0
        window = [_make_entry("bearing", 0.9, t0)]
        pipeline.execute.return_value = [None, None, None, window]

        agg = FaultAggregator(client)
        result = agg.submit("m1", "bearing", 0.9)

        # 1 vote < MIN_AGREEMENT
        assert result is None

    def test_exactly_min_agreement(self, mock_redis):
        """Exactly MIN_AGREEMENT votes → consensus."""
        client, pipeline = mock_redis
        t0 = 1000.0
        window = [
            _make_entry("bearing", 0.9, t0),
            _make_entry("bearing", 0.85, t0 + 1),
            _make_entry("bearing", 0.95, t0 + 2),
        ]
        pipeline.execute.return_value = [None, None, None, window]

        agg = FaultAggregator(client)
        result = agg.submit("m1", "bearing", 0.95)

        assert result is not None
        assert result.votes == 3

    def test_bytes_entries(self, mock_redis):
        """Redis may return bytes instead of strings."""
        client, pipeline = mock_redis
        t0 = 1000.0
        window = [
            _make_entry("bearing", 0.9, t0).encode(),
            _make_entry("bearing", 0.85, t0 + 1).encode(),
            _make_entry("bearing", 0.95, t0 + 2).encode(),
        ]
        pipeline.execute.return_value = [None, None, None, window]

        agg = FaultAggregator(client)
        result = agg.submit("m1", "bearing", 0.95)

        assert result is not None
        assert result.fault_type == "bearing"
        assert result.votes == 3

    def test_zero_confidence(self, mock_redis):
        """Consensus with zero confidence."""
        client, pipeline = mock_redis
        t0 = 1000.0
        window = [
            _make_entry("bearing", 0.0, t0),
            _make_entry("bearing", 0.0, t0 + 1),
            _make_entry("bearing", 0.0, t0 + 2),
        ]
        pipeline.execute.return_value = [None, None, None, window]

        agg = FaultAggregator(client)
        result = agg.submit("m1", "bearing", 0.0)

        assert result is not None
        assert result.confidence == 0.0

    def test_window_full_size(self, mock_redis):
        """Full window of WINDOW_SIZE entries."""
        client, pipeline = mock_redis
        t0 = 1000.0
        window = [
            _make_entry("bearing", 0.9, t0 + i) for i in range(WINDOW_SIZE)
        ]
        pipeline.execute.return_value = [None, None, None, window]

        agg = FaultAggregator(client)
        result = agg.submit("m1", "bearing", 0.9)

        assert result is not None
        assert result.votes == WINDOW_SIZE
        assert result.window_size == WINDOW_SIZE
