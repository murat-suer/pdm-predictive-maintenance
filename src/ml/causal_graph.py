"""Causal Knowledge Graph for Root Cause Analysis.

Models causal relationships between sensor readings and failure modes
using a directed graph. Supports operator feedback (blacklisting edges)
and visualization export.
"""

from __future__ import annotations

from typing import Any

import networkx as nx

# Sensor nodes
SENSORS = [
    "vibration_rms",
    "bearing_temp",
    "oil_pressure",
    "pressure_drop",
    "outlet_temp",
    "flow_rate",
    "belt_tension",
    "drive_temp",
    "motor_load",
    "speed_rpm",
]

# Failure mode nodes
FAILURE_MODES = [
    "bearing_failure",
    "oil_degradation",
    "fouling",
    "belt_slip",
    "motor_overload",
]


class CausalKnowledgeGraph:
    """Directed graph mapping sensor anomalies to failure modes with causal weights.

    The graph encodes domain knowledge about how sensor readings causally
    relate to specific failure modes. Weights represent the strength of
    each causal link (0.0 to 1.0).
    """

    def __init__(self) -> None:
        self.graph = nx.DiGraph()
        self._blacklisted_edges: set[tuple[str, str]] = set()
        self._build_default_topology()

    def _build_default_topology(self) -> None:
        """Build the default sensor → failure mode topology with causal weights.

        Each edge weight represents the strength of the causal relationship
        between a sensor anomaly and a failure mode, based on domain knowledge.
        """
        # Add all nodes
        for sensor in SENSORS:
            self.graph.add_node(sensor, node_type="sensor")
        for failure in FAILURE_MODES:
            self.graph.add_node(failure, node_type="failure_mode")

        # Define causal relationships: (sensor, failure_mode, weight)
        # Weights represent causal strength (0.0 - 1.0)
        edges = [
            # bearing_failure causes
            ("vibration_rms", "bearing_failure", 0.95),
            ("bearing_temp", "bearing_failure", 0.90),
            ("oil_pressure", "bearing_failure", 0.60),
            ("speed_rpm", "bearing_failure", 0.40),
            # oil_degradation causes
            ("oil_pressure", "oil_degradation", 0.85),
            ("bearing_temp", "oil_degradation", 0.55),
            ("pressure_drop", "oil_degradation", 0.70),
            ("outlet_temp", "oil_degradation", 0.45),
            # fouling causes
            ("pressure_drop", "fouling", 0.90),
            ("outlet_temp", "fouling", 0.80),
            ("flow_rate", "fouling", 0.75),
            ("vibration_rms", "fouling", 0.35),
            # belt_slip causes
            ("belt_tension", "belt_slip", 0.95),
            ("speed_rpm", "belt_slip", 0.70),
            ("drive_temp", "belt_slip", 0.65),
            ("vibration_rms", "belt_slip", 0.45),
            # motor_overload causes
            ("motor_load", "motor_overload", 0.95),
            ("drive_temp", "motor_overload", 0.85),
            ("speed_rpm", "motor_overload", 0.50),
            ("oil_pressure", "motor_overload", 0.30),
        ]

        for sensor, failure, weight in edges:
            self.graph.add_edge(sensor, failure, weight=weight, causal=True)

    def find_root_causes(self, failure_mode: str, top_k: int = 5) -> list[dict[str, Any]]:
        """Find root causes (sensors) for a given failure mode, ranked by causal weight.

        Args:
            failure_mode: The failure mode to find root causes for.
            top_k: Maximum number of root causes to return.

        Returns:
            List of dicts with 'sensor', 'weight', and 'rank' keys,
            sorted by weight descending.

        Raises:
            ValueError: If failure_mode is not in the graph.
        """
        if failure_mode not in self.graph:
            raise ValueError(
                f"Unknown failure mode: '{failure_mode}'. "
                f"Valid modes: {FAILURE_MODES}"
            )

        # Get all predecessors (sensors pointing to this failure mode)
        predecessors = list(self.graph.predecessors(failure_mode))

        causes = []
        for sensor in predecessors:
            edge_data = self.graph.edges[sensor, failure_mode]
            causes.append({
                "sensor": sensor,
                "weight": edge_data["weight"],
            })

        # Sort by weight descending
        causes.sort(key=lambda x: x["weight"], reverse=True)

        # Add rank
        for i, cause in enumerate(causes[:top_k], start=1):
            cause["rank"] = i

        return causes[:top_k]

    def blacklist_edge(self, sensor: str, failure_mode: str) -> bool:
        """Blacklist a causal edge based on operator feedback.

        Removes the edge from the active graph and records it as blacklisted.
        Blacklisted edges are excluded from root cause analysis.

        Args:
            sensor: The sensor node.
            failure_mode: The failure mode node.

        Returns:
            True if the edge was successfully blacklisted, False if it
            didn't exist or was already blacklisted.

        Raises:
            ValueError: If sensor or failure_mode is not in the graph.
        """
        if sensor not in self.graph:
            raise ValueError(f"Unknown sensor: '{sensor}'. Valid sensors: {SENSORS}")
        if failure_mode not in self.graph:
            raise ValueError(
                f"Unknown failure mode: '{failure_mode}'. Valid modes: {FAILURE_MODES}"
            )

        edge_key = (sensor, failure_mode)
        if edge_key in self._blacklisted_edges:
            return False

        if not self.graph.has_edge(sensor, failure_mode):
            return False

        # Remove edge from active graph
        self.graph.remove_edge(sensor, failure_mode)
        self._blacklisted_edges.add(edge_key)
        return True

    def is_blacklisted(self, sensor: str, failure_mode: str) -> bool:
        """Check if an edge is blacklisted."""
        return (sensor, failure_mode) in self._blacklisted_edges

    def get_blacklisted_edges(self) -> list[tuple[str, str]]:
        """Return all blacklisted edges."""
        return list(self._blacklisted_edges)

    def restore_edge(self, sensor: str, failure_mode: str) -> bool:
        """Restore a previously blacklisted edge.

        Args:
            sensor: The sensor node.
            failure_mode: The failure mode node.

        Returns:
            True if the edge was restored, False if it wasn't blacklisted.
        """
        edge_key = (sensor, failure_mode)
        if edge_key not in self._blacklisted_edges:
            return False

        # Re-add the edge with default weight
        default_weights = self._get_default_weight(sensor, failure_mode)
        if default_weights is not None:
            self.graph.add_edge(
                sensor, failure_mode, weight=default_weights, causal=True
            )
        self._blacklisted_edges.discard(edge_key)
        return True

    def _get_default_weight(self, sensor: str, failure_mode: str) -> float | None:
        """Get the default weight for a sensor-failure_mode edge."""
        defaults = {
            ("vibration_rms", "bearing_failure"): 0.95,
            ("bearing_temp", "bearing_failure"): 0.90,
            ("oil_pressure", "bearing_failure"): 0.60,
            ("speed_rpm", "bearing_failure"): 0.40,
            ("oil_pressure", "oil_degradation"): 0.85,
            ("bearing_temp", "oil_degradation"): 0.55,
            ("pressure_drop", "oil_degradation"): 0.70,
            ("outlet_temp", "oil_degradation"): 0.45,
            ("pressure_drop", "fouling"): 0.90,
            ("outlet_temp", "fouling"): 0.80,
            ("flow_rate", "fouling"): 0.75,
            ("vibration_rms", "fouling"): 0.35,
            ("belt_tension", "belt_slip"): 0.95,
            ("speed_rpm", "belt_slip"): 0.70,
            ("drive_temp", "belt_slip"): 0.65,
            ("vibration_rms", "belt_slip"): 0.45,
            ("motor_load", "motor_overload"): 0.95,
            ("drive_temp", "motor_overload"): 0.85,
            ("speed_rpm", "motor_overload"): 0.50,
            ("oil_pressure", "motor_overload"): 0.30,
        }
        return defaults.get((sensor, failure_mode))

    def export_for_visualization(self) -> dict[str, Any]:
        """Export the graph as a JSON-compatible dict for visualization.

        Returns:
            Dict with 'nodes' and 'edges' keys suitable for D3.js,
            Cytoscape, or similar graph visualization libraries.
        """
        nodes = []
        for node in self.graph.nodes():
            node_data = dict(self.graph.nodes[node])
            node_data["id"] = node
            nodes.append(node_data)

        edges = []
        for u, v, data in self.graph.edges(data=True):
            edge_data = dict(data)
            edge_data["source"] = u
            edge_data["target"] = v
            edges.append(edge_data)

        return {
            "nodes": nodes,
            "edges": edges,
            "blacklisted": [
                {"source": s, "target": t} for s, t in self._blacklisted_edges
            ],
            "metadata": {
                "sensor_count": len([n for n in self.graph.nodes() if self.graph.nodes[n].get("node_type") == "sensor"]),
                "failure_mode_count": len([n for n in self.graph.nodes() if self.graph.nodes[n].get("node_type") == "failure_mode"]),
                "edge_count": self.graph.number_of_edges(),
                "blacklisted_count": len(self._blacklisted_edges),
            },
        }
