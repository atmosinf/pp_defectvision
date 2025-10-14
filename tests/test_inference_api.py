"""Integration tests for the FastAPI inference service."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from inference.api import MODEL_NAME_ENV, MODEL_PATH_ENV, CLASS_NAMES_ENV, _load_session, app


# Resolve repo root so tests work regardless of CWD.
ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_PATH = ROOT / "models" / "model.pth"

model_path = Path(os.environ.get(MODEL_PATH_ENV, DEFAULT_MODEL_PATH))
class_names_path = Path(
    os.environ.get(CLASS_NAMES_ENV, model_path.with_name("class_names.json"))
)

# Skip the suite automatically if the checkpoint or class names are missing.
MODEL_AVAILABLE = model_path.exists() and class_names_path.exists()

pytestmark = pytest.mark.skipif(
    not MODEL_AVAILABLE,
    reason="Model artifacts not available; set DEFECTVISION_MODEL_PATH and DEFECTVISION_CLASS_NAMES_PATH.",
)


if MODEL_AVAILABLE:
    # Ensure the API resolves the model paths the same way it would in prod.
    os.environ.setdefault(MODEL_PATH_ENV, str(model_path))
    os.environ.setdefault(CLASS_NAMES_ENV, str(class_names_path))
    os.environ.setdefault(MODEL_NAME_ENV, os.environ.get(MODEL_NAME_ENV, "resnet18"))


def _client() -> TestClient:
    # Bust the cached InferenceSession so every test starts clean.
    _load_session.cache_clear()
    return TestClient(app)


def test_healthz_returns_ok() -> None:
    client = _client()
    response = client.get("/healthz")

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert payload["classes"] > 0


def test_predict_returns_expected_label() -> None:
    sample_image = ROOT / "src" / "inference" / "test_images" / (
        "Tomato_healthy__000bf685-b305-408b-91f4-37030f8e62db___GH_HL Leaf 308.1.JPG"
    )
    if not sample_image.exists():
        pytest.skip("Sample image not found.")

    client = _client()
    with sample_image.open("rb") as handle:
        response = client.post(
            "/predict",
            params={"topk": 3},
            files={"file": ("sample.jpg", handle, "image/jpeg")},
        )

    assert response.status_code == 200
    payload = response.json()

    assert payload["filename"] == "sample.jpg"
    # The model was trained on PlantVillage tomato classes; this ensures we get the
    # healthy tomato prediction and confidence stays near the expected baseline.
    assert payload["label"] == "Tomato_healthy"
    assert pytest.approx(payload["confidence"], rel=1e-3) == 0.999629

    assert payload["topk"][0]["label"] == "Tomato_healthy"
    assert [item["label"] for item in payload["topk"]] == [
        "Tomato_healthy",
        "Tomato__Target_Spot",
        "Tomato_Spider_mites_Two_spotted_spider_mite",
    ]
