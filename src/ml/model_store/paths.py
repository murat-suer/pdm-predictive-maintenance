import os
from pathlib import Path

MODEL_ARTIFACT_VERSION = "if-feature-space-v3"


def get_model_store() -> Path:
    env = os.environ.get("MODEL_STORE_PATH")
    if env:
        return Path(env)
    docker = Path("/app/model_store")
    if docker.exists():
        return docker
    return Path(__file__).resolve().parent.parent / "model_store"
