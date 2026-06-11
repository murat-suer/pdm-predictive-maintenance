
import pytest

from src.ml.model_card import (
    build_model_card,
    read_model_cards,
    training_window_from_timestamps,
    write_model_card,
)


@pytest.fixture()
def artifact(tmp_path):
    p = tmp_path / "model.joblib"
    p.write_bytes(b"fake-model-bytes")
    return p


class TestWriteReadRoundtrip:
    def test_write_creates_json_file(self, artifact):
        card_path = write_model_card(
            artifact,
            model_kind="anomaly_detector",
            machine_id="AC-201",
            feature_list=["vibration_rms", "bearing_temp"],
            hyperparameters={"n_estimators": 100},
            training_rows=5000,
        )
        assert card_path.exists()
        assert card_path.suffix == ".json"

    def test_read_returns_written_card(self, artifact):
        write_model_card(
            artifact,
            model_kind="anomaly_detector",
            machine_id="AC-201",
            feature_list=["vibration_rms"],
            hyperparameters={"lr": 0.01},
            training_rows=1000,
        )
        cards = read_model_cards(artifact.parent)
        assert len(cards) == 1
        assert cards[0]["name"] == "anomaly_detector_AC-201"

    def test_roundtrip_preserves_features(self, artifact):
        write_model_card(
            artifact,
            model_kind="rul_predictor",
            machine_id="AC-202",
            feature_list=["f1", "f2", "f3"],
            hyperparameters={"layers": 3},
            training_rows=2000,
        )
        cards = read_model_cards(artifact.parent)
        assert cards[0]["feature_list"] == ["f1", "f2", "f3"]


class TestSHA256:
    def test_artifact_sha_present(self, artifact):
        card = build_model_card(
            artifact,
            model_kind="anomaly_detector",
            machine_id="AC-201",
            feature_list=["vibration_rms"],
            hyperparameters={},
            training_rows=100,
        )
        assert card["artifact_sha256"].startswith("sha256:")

    def test_sha_changes_with_content(self, tmp_path):
        p1 = tmp_path / "a.bin"
        p1.write_bytes(b"aaa")
        p2 = tmp_path / "b.bin"
        p2.write_bytes(b"bbb")
        c1 = build_model_card(p1, model_kind="m", machine_id="x", feature_list=[], hyperparameters={}, training_rows=1)
        c2 = build_model_card(p2, model_kind="m", machine_id="x", feature_list=[], hyperparameters={}, training_rows=1)
        assert c1["artifact_sha256"] != c2["artifact_sha256"]


class TestTrainingWindow:
    def test_from_timestamps(self):
        window = training_window_from_timestamps(["2024-01-03", "2024-01-01", "2024-01-02"])
        assert window["from"] == "2024-01-01"
        assert window["to"] == "2024-01-03"

    def test_empty_timestamps(self):
        window = training_window_from_timestamps([])
        assert window["from"] is None
        assert window["to"] is None

    def test_window_in_card(self, artifact):
        window = {"from": "2024-01-01", "to": "2024-06-01"}
        card = build_model_card(
            artifact,
            model_kind="anomaly_detector",
            machine_id="AC-201",
            feature_list=[],
            hyperparameters={},
            training_rows=100,
            training_window=window,
        )
        assert card["training_window"] == window


class TestBuildModelCard:
    def test_name_format(self, artifact):
        card = build_model_card(
            artifact,
            model_kind="rul_predictor",
            machine_id="AC-201",
            feature_list=[],
            hyperparameters={},
            training_rows=100,
        )
        assert card["name"] == "rul_predictor_AC-201"

    def test_machines_covered_defaults_to_machine_id(self, artifact):
        card = build_model_card(
            artifact,
            model_kind="anomaly_detector",
            machine_id="AC-201",
            feature_list=[],
            hyperparameters={},
            training_rows=100,
        )
        assert card["machines_covered"] == ["AC-201"]

    def test_known_limitations_present(self, artifact):
        card = build_model_card(
            artifact,
            model_kind="anomaly_detector",
            machine_id="AC-201",
            feature_list=[],
            hyperparameters={},
            training_rows=100,
        )
        assert len(card["known_limitations"]) >= 1
