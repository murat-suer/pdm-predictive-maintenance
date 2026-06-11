"""Tests for CausalKnowledgeGraph."""

import pytest

from src.ml.causal_graph import (
    FAILURE_MODES,
    SENSORS,
    CausalKnowledgeGraph,
)


class TestCausalKnowledgeGraph:
    """Test suite for CausalKnowledgeGraph."""

    def test_graph_initialization(self):
        """Test that the graph initializes with correct nodes and edges."""
        graph = CausalKnowledgeGraph()

        # Check all sensor nodes exist
        for sensor in SENSORS:
            assert sensor in graph.graph.nodes()
            assert graph.graph.nodes[sensor]["node_type"] == "sensor"

        # Check all failure mode nodes exist
        for failure in FAILURE_MODES:
            assert failure in graph.graph.nodes()
            assert graph.graph.nodes[failure]["node_type"] == "failure_mode"

        # Check edges exist (sensor -> failure_mode)
        assert graph.graph.number_of_edges() > 0

        # All edges should go from sensor to failure_mode
        for u, v in graph.graph.edges():
            assert graph.graph.nodes[u]["node_type"] == "sensor"
            assert graph.graph.nodes[v]["node_type"] == "failure_mode"

        # All edges should have weight attribute
        for u, v, data in graph.graph.edges(data=True):
            assert "weight" in data
            assert 0.0 <= data["weight"] <= 1.0

    def test_find_root_causes(self):
        """Test finding root causes for a failure mode."""
        graph = CausalKnowledgeGraph()

        # Test bearing_failure - vibration_rms should be top cause
        causes = graph.find_root_causes("bearing_failure")
        assert len(causes) > 0
        assert causes[0]["sensor"] == "vibration_rms"
        assert causes[0]["weight"] == 0.95
        assert causes[0]["rank"] == 1

        # Results should be sorted by weight descending
        weights = [c["weight"] for c in causes]
        assert weights == sorted(weights, reverse=True)

        # Test top_k parameter
        causes_top2 = graph.find_root_causes("bearing_failure", top_k=2)
        assert len(causes_top2) == 2

        # Test invalid failure mode raises ValueError
        with pytest.raises(ValueError, match="Unknown failure mode"):
            graph.find_root_causes("nonexistent_failure")

        # Test motor_overload - motor_load should be top cause
        causes_motor = graph.find_root_causes("motor_overload")
        assert causes_motor[0]["sensor"] == "motor_load"
        assert causes_motor[0]["weight"] == 0.95

    def test_blacklist_edge(self):
        """Test blacklisting edges based on operator feedback."""
        graph = CausalKnowledgeGraph()

        # Verify edge exists before blacklisting
        assert graph.graph.has_edge("vibration_rms", "bearing_failure")

        # Blacklist the edge
        result = graph.blacklist_edge("vibration_rms", "bearing_failure")
        assert result is True

        # Edge should be removed from active graph
        assert not graph.graph.has_edge("vibration_rms", "bearing_failure")

        # Should be recorded as blacklisted
        assert graph.is_blacklisted("vibration_rms", "bearing_failure")
        assert ("vibration_rms", "bearing_failure") in graph.get_blacklisted_edges()

        # Blacklisting again should return False (already blacklisted)
        result2 = graph.blacklist_edge("vibration_rms", "bearing_failure")
        assert result2 is False

        # Blacklisted edge should not appear in root causes
        causes = graph.find_root_causes("bearing_failure")
        sensors = [c["sensor"] for c in causes]
        assert "vibration_rms" not in sensors

        # Test restore
        restored = graph.restore_edge("vibration_rms", "bearing_failure")
        assert restored is True
        assert graph.graph.has_edge("vibration_rms", "bearing_failure")
        assert not graph.is_blacklisted("vibration_rms", "bearing_failure")

        # Test invalid sensor raises ValueError
        with pytest.raises(ValueError, match="Unknown sensor"):
            graph.blacklist_edge("nonexistent_sensor", "bearing_failure")

        # Test invalid failure mode raises ValueError
        with pytest.raises(ValueError, match="Unknown failure mode"):
            graph.blacklist_edge("vibration_rms", "nonexistent_failure")

    def test_export_for_visualization(self):
        """Test exporting graph for visualization."""
        graph = CausalKnowledgeGraph()
        export = graph.export_for_visualization()

        # Check structure
        assert "nodes" in export
        assert "edges" in export
        assert "blacklisted" in export
        assert "metadata" in export

        # Check nodes
        assert len(export["nodes"]) == len(SENSORS) + len(FAILURE_MODES)
        node_ids = [n["id"] for n in export["nodes"]]
        for sensor in SENSORS:
            assert sensor in node_ids
        for failure in FAILURE_MODES:
            assert failure in node_ids

        # Check edges have source/target
        for edge in export["edges"]:
            assert "source" in edge
            assert "target" in edge
            assert "weight" in edge

        # Check metadata
        assert export["metadata"]["sensor_count"] == len(SENSORS)
        assert export["metadata"]["failure_mode_count"] == len(FAILURE_MODES)
        assert export["metadata"]["edge_count"] == graph.graph.number_of_edges()
        assert export["metadata"]["blacklisted_count"] == 0

        # Blacklist an edge and re-export
        graph.blacklist_edge("vibration_rms", "bearing_failure")
        export2 = graph.export_for_visualization()
        assert len(export2["blacklisted"]) == 1
        assert export2["blacklisted"][0]["source"] == "vibration_rms"
        assert export2["blacklisted"][0]["target"] == "bearing_failure"
        assert export2["metadata"]["blacklisted_count"] == 1

        # Verify JSON-serializable
        import json
        json_str = json.dumps(export2)
        assert len(json_str) > 0
