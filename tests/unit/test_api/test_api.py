"""Unit tests for the FastAPI API layer."""
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client():
    """Create a test client for the FastAPI app."""
    from src.api.app import app
    return TestClient(app)


class TestHealthEndpoint:
    """Tests for the health check endpoint."""

    def test_health_check(self, client):
        """Test GET /health returns healthy status."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["version"] == "3.0.0"


class TestRootEndpoint:
    """Tests for the root endpoint."""

    def test_root(self, client):
        """Test GET / returns API info."""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "PDM Intelligence API"
        assert data["version"] == "3.0.0"


class TestAnomalyEndpoint:
    """Tests for the anomaly detection endpoint."""

    @patch("src.ml.anomaly_detector.AnomalyDetector")
    def test_predict_anomaly(self, mock_detector_cls, client):
        """Test POST /anomaly/predict returns anomaly prediction."""
        mock_detector = MagicMock()
        mock_detector.predict.return_value = {
            "is_anomaly": True,
            "anomaly_score": 0.85,
            "top_contributing_sensor": "vibration_rms",
            "shap_values": {"vibration_rms_value": 0.42},
        }
        mock_detector_cls.return_value = mock_detector

        response = client.post(
            "/anomaly/predict",
            json={
                "machine_id": "pump-001",
                "features": {"vibration_rms_value": 5.2, "bearing_temp_value": 78.3},
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["is_anomaly"] is True
        assert data["anomaly_score"] == 0.85
        assert data["top_contributing_sensor"] == "vibration_rms"

    @patch("src.ml.anomaly_detector.AnomalyDetector")
    def test_predict_anomaly_no_anomaly(self, mock_detector_cls, client):
        """Test POST /anomaly/predict returns normal prediction."""
        mock_detector = MagicMock()
        mock_detector.predict.return_value = {
            "is_anomaly": False,
            "anomaly_score": 0.12,
            "top_contributing_sensor": None,
            "shap_values": {},
        }
        mock_detector_cls.return_value = mock_detector

        response = client.post(
            "/anomaly/predict",
            json={
                "machine_id": "pump-001",
                "features": {"vibration_rms_value": 1.2},
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["is_anomaly"] is False
        assert data["anomaly_score"] == 0.12


class TestRULEndpoint:
    """Tests for the RUL prediction endpoint."""

    @patch("src.ml.rul_predictor.RULPredictor")
    def test_predict_rul(self, mock_predictor_cls, client):
        """Test POST /rul/predict returns RUL prediction."""
        mock_predictor = MagicMock()
        mock_predictor.predict.return_value = {
            "rul_hours": 245.5,
            "rul_low_ci": 180.0,
            "rul_high_ci": 310.0,
            "confidence": 0.85,
            "failure_prob_24h": 3.2,
            "survive_shift_pct": 92.5,
            "method": "xgboost+ema+conformal",
            "fallback": False,
            "model_trained": True,
        }
        mock_predictor_cls.return_value = mock_predictor

        response = client.post(
            "/rul/predict",
            json={
                "machine_id": "pump-001",
                "features": {"vibration_rms_value": 5.2},
                "phase": "DEGRADING",
                "emergency_stop_count": 0,
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["rul_hours"] == 245.5
        assert data["confidence"] == 0.85
        assert data["model_trained"] is True

    @patch("src.ml.rul_predictor.RULPredictor")
    def test_predict_rul_healthy_phase(self, mock_predictor_cls, client):
        """Test POST /rul/predict returns null for HEALTHY phase."""
        mock_predictor = MagicMock()
        mock_predictor.predict.return_value = None
        mock_predictor_cls.return_value = mock_predictor

        response = client.post(
            "/rul/predict",
            json={
                "machine_id": "pump-001",
                "features": {"vibration_rms_value": 1.0},
                "phase": "HEALTHY",
            },
        )
        assert response.status_code == 200
        assert response.json() is None


class TestKnowledgeGraphEndpoint:
    """Tests for the knowledge graph root cause analysis endpoint.

    These tests use the REAL CausalKnowledgeGraph implementation (no mocks).
    The router creates a fresh CausalKnowledgeGraph() per request, so we
    exercise the actual graph topology and find_root_causes() logic.

    NOTE: Until the router's tuple-unpacking bug is fixed (it expects
    list[tuple] but find_root_causes returns list[dict]), these tests
    will FAIL with 500 errors. This is intentional — they expose the bug.
    """

    # --- Valid failure modes: structure and content ---

    def test_bearing_failure_returns_ranked_sensors(self, client):
        """GET /knowledge-graph/root-causes/bearing_failure returns 4 sensors
        ranked by causal weight descending with correct rank values."""
        response = client.get("/knowledge-graph/root-causes/bearing_failure")
        assert response.status_code == 200

        data = response.json()
        assert len(data) == 4

        # Expected order from causal_graph.py topology:
        # vibration_rms=0.95, bearing_temp=0.90, oil_pressure=0.60, speed_rpm=0.40
        expected = [
            {"sensor": "vibration_rms", "weight": 0.95, "rank": 1},
            {"sensor": "bearing_temp", "weight": 0.90, "rank": 2},
            {"sensor": "oil_pressure", "weight": 0.60, "rank": 3},
            {"sensor": "speed_rpm", "weight": 0.40, "rank": 4},
        ]
        assert data == expected

    def test_oil_degradation_returns_correct_sensors(self, client):
        """GET /knowledge-graph/root-causes/oil_degradation returns sensors
        in correct weight-descending order."""
        response = client.get("/knowledge-graph/root-causes/oil_degradation")
        assert response.status_code == 200

        data = response.json()
        assert len(data) == 4

        # oil_pressure=0.85, pressure_drop=0.70, bearing_temp=0.55, outlet_temp=0.45
        assert data[0]["sensor"] == "oil_pressure"
        assert data[0]["weight"] == 0.85
        assert data[1]["sensor"] == "pressure_drop"
        assert data[1]["weight"] == 0.70
        assert data[2]["sensor"] == "bearing_temp"
        assert data[2]["weight"] == 0.55
        assert data[3]["sensor"] == "outlet_temp"
        assert data[3]["weight"] == 0.45

    def test_fouling_returns_correct_sensors(self, client):
        """GET /knowledge-graph/root-causes/fouling returns 4 sensors
        with pressure_drop as top root cause (weight=0.90)."""
        response = client.get("/knowledge-graph/root-causes/fouling")
        assert response.status_code == 200

        data = response.json()
        assert len(data) == 4
        assert data[0]["sensor"] == "pressure_drop"
        assert data[0]["weight"] == 0.90
        assert data[0]["rank"] == 1

    def test_belt_slip_returns_correct_sensors(self, client):
        """GET /knowledge-graph/root-causes/belt_slip returns belt_tension
        as top root cause (weight=0.95)."""
        response = client.get("/knowledge-graph/root-causes/belt_slip")
        assert response.status_code == 200

        data = response.json()
        assert len(data) == 4
        assert data[0]["sensor"] == "belt_tension"
        assert data[0]["weight"] == 0.95
        assert data[0]["rank"] == 1

    def test_motor_overload_returns_correct_sensors(self, client):
        """GET /knowledge-graph/root-causes/motor_overload returns motor_load
        as top root cause (weight=0.95)."""
        response = client.get("/knowledge-graph/root-causes/motor_overload")
        assert response.status_code == 200

        data = response.json()
        assert len(data) == 4
        assert data[0]["sensor"] == "motor_load"
        assert data[0]["weight"] == 0.95
        assert data[0]["rank"] == 1

    # --- Response structure validation ---

    def test_response_items_have_required_fields(self, client):
        """Each item in the response must have 'sensor' (str), 'weight' (float),
        and 'rank' (int) fields."""
        response = client.get("/knowledge-graph/root-causes/bearing_failure")
        assert response.status_code == 200

        data = response.json()
        assert len(data) > 0

        for item in data:
            assert "sensor" in item, "Missing 'sensor' field"
            assert "weight" in item, "Missing 'weight' field"
            assert "rank" in item, "Missing 'rank' field"
            assert isinstance(item["sensor"], str)
            assert isinstance(item["weight"], (int, float))
            assert isinstance(item["rank"], int)

    def test_weights_are_in_descending_order(self, client):
        """Weights must be sorted in descending order across all failure modes."""
        for failure_mode in [
            "bearing_failure", "oil_degradation", "fouling",
            "belt_slip", "motor_overload",
        ]:
            response = client.get(f"/knowledge-graph/root-causes/{failure_mode}")
            assert response.status_code == 200
            data = response.json()
            weights = [item["weight"] for item in data]
            assert weights == sorted(weights, reverse=True), (
                f"Weights not descending for {failure_mode}: {weights}"
            )

    def test_ranks_are_sequential_starting_from_one(self, client):
        """Rank values must be sequential integers starting from 1."""
        for failure_mode in [
            "bearing_failure", "oil_degradation", "fouling",
            "belt_slip", "motor_overload",
        ]:
            response = client.get(f"/knowledge-graph/root-causes/{failure_mode}")
            assert response.status_code == 200
            data = response.json()
            ranks = [item["rank"] for item in data]
            expected_ranks = list(range(1, len(data) + 1))
            assert ranks == expected_ranks, (
                f"Ranks not sequential for {failure_mode}: {ranks}"
            )

    # --- top_k parameter tests ---

    def test_top_k_limits_results_to_2(self, client):
        """top_k=2 returns only the top 2 root causes."""
        response = client.get(
            "/knowledge-graph/root-causes/bearing_failure?top_k=2"
        )
        assert response.status_code == 200

        data = response.json()
        assert len(data) == 2
        assert data[0]["sensor"] == "vibration_rms"
        assert data[0]["rank"] == 1
        assert data[1]["sensor"] == "bearing_temp"
        assert data[1]["rank"] == 2

    def test_top_k_1_returns_single_result(self, client):
        """top_k=1 returns exactly one root cause with rank=1."""
        response = client.get(
            "/knowledge-graph/root-causes/bearing_failure?top_k=1"
        )
        assert response.status_code == 200

        data = response.json()
        assert len(data) == 1
        assert data[0]["sensor"] == "vibration_rms"
        assert data[0]["weight"] == 0.95
        assert data[0]["rank"] == 1

    def test_top_k_larger_than_available_returns_all(self, client):
        """top_k=100 returns all available sensors (4 for bearing_failure),
        not 100 items."""
        response = client.get(
            "/knowledge-graph/root-causes/bearing_failure?top_k=100"
        )
        assert response.status_code == 200

        data = response.json()
        # bearing_failure has exactly 4 sensors in the topology
        assert len(data) == 4

    def test_top_k_default_is_5(self, client):
        """Default top_k (no param) returns up to 5 results.
        Since all failure modes have ≤4 sensors, we get all of them."""
        response = client.get("/knowledge-graph/root-causes/bearing_failure")
        assert response.status_code == 200

        data = response.json()
        # 4 sensors exist, default top_k=5, so all 4 returned
        assert len(data) == 4

    # --- Error / edge cases ---

    def test_unknown_failure_mode_returns_404(self, client):
        """An invalid failure_mode returns HTTP 404 with detail message."""
        response = client.get("/knowledge-graph/root-causes/nonexistent_mode")
        assert response.status_code == 404
        data = response.json()
        assert "detail" in data
        assert "nonexistent_mode" in data["detail"]

    def test_empty_string_failure_mode_returns_404(self, client):
        """Empty string as failure_mode returns 404 (not a valid mode)."""
        response = client.get("/knowledge-graph/root-causes/")
        # FastAPI won't match empty path param — returns 404 from router
        # or 404/422 depending on routing. Either way, not 200.
        assert response.status_code in (404, 422, 307)

    def test_all_valid_failure_modes_return_200(self, client):
        """All 5 defined failure modes return HTTP 200 with non-empty results."""
        valid_modes = [
            "bearing_failure", "oil_degradation", "fouling",
            "belt_slip", "motor_overload",
        ]
        for mode in valid_modes:
            response = client.get(f"/knowledge-graph/root-causes/{mode}")
            assert response.status_code == 200, f"Failed for {mode}"
            data = response.json()
            assert len(data) > 0, f"No results for {mode}"


class TestCostOptimizerEndpoint:
    """Tests for the cost optimizer endpoint."""

    @patch("src.ml.maintenance_optimizer.MaintenanceCostOptimizer")
    def test_optimize_cost(self, mock_optimizer_cls, client):
        """Test POST /cost-optimize returns optimal replacement time."""
        mock_optimizer = MagicMock()
        mock_optimizer.find_optimal_replacement_time.return_value = (156.3, 12.45)
        mock_optimizer.compare_strategies.return_value = {
            "preventive_cost_rate": 12.45,
            "corrective_cost_rate": 18.90,
            "optimal_tp": 156.3,
            "savings_per_hour": 6.45,
            "savings_percent": 34.1,
        }
        mock_optimizer_cls.return_value = mock_optimizer

        response = client.post(
            "/cost-optimize",
            json={
                "eta": 720.0,
                "beta": 2.1,
                "preventive_cost": 1000.0,
                "corrective_cost": 5000.0,
                "downtime_cost_per_hour": 500.0,
                "production_loss_per_hour": 2000.0,
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["optimal_tp"] == 156.3
        assert data["min_cost_rate"] == 12.45
        assert "strategy_comparison" in data
        assert data["strategy_comparison"]["savings_percent"] == 34.1

    @patch("src.ml.maintenance_optimizer.MaintenanceCostOptimizer")
    def test_optimize_cost_defaults(self, mock_optimizer_cls, client):
        """Test POST /cost-optimize with minimal parameters uses defaults."""
        mock_optimizer = MagicMock()
        mock_optimizer.find_optimal_replacement_time.return_value = (200.0, 10.0)
        mock_optimizer.compare_strategies.return_value = {
            "preventive_cost_rate": 10.0,
            "corrective_cost_rate": 15.0,
            "optimal_tp": 200.0,
            "savings_per_hour": 5.0,
            "savings_percent": 33.3,
        }
        mock_optimizer_cls.return_value = mock_optimizer

        response = client.post(
            "/cost-optimize",
            json={
                "eta": 500.0,
                "beta": 2.5,
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["optimal_tp"] == 200.0
