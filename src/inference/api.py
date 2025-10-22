"""FastAPI application wrapping the DefectVision inference session."""

from __future__ import annotations

import io
import os
from functools import lru_cache
from pathlib import Path

from fastapi import Depends, FastAPI, File, HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel

from inference import InferenceSession, load_session

try:
    from mangum import Mangum
except ImportError:  # pragma: no cover - optional dependency for AWS Lambda
    Mangum = None  # type: ignore[misc]

DEFAULT_MODEL_PATH = Path("models/model.pth")
MODEL_PATH_ENV = "DEFECTVISION_MODEL_PATH"
CLASS_NAMES_ENV = "DEFECTVISION_CLASS_NAMES_PATH"
MODEL_NAME_ENV = "DEFECTVISION_MODEL_NAME"


class TopPrediction(BaseModel):
    label: str
    confidence: float


class PredictionResponse(BaseModel):
    filename: str
    label: str
    confidence: float
    topk: list[TopPrediction]


def _resolve_model_path() -> Path:
    """Resolve the path to the model checkpoint from the environment."""
    path = os.environ.get(MODEL_PATH_ENV)
    if path:
        return Path(path)
    return DEFAULT_MODEL_PATH


@lru_cache(maxsize=1)
def _load_session() -> InferenceSession:
    model_path = _resolve_model_path()
    class_names_path_env = os.environ.get(CLASS_NAMES_ENV)
    class_names_path = Path(class_names_path_env) if class_names_path_env else None
    model_name = os.environ.get(MODEL_NAME_ENV, "resnet18")

    if not model_path.exists():
        raise RuntimeError(
            f"Model checkpoint not found at {model_path}. "
            f"Set {MODEL_PATH_ENV} to the correct model.pth location."
        )

    return load_session(model_path, class_names_path=class_names_path, model_name=model_name)


def get_session() -> InferenceSession:
    """FastAPI dependency wrapper for obtaining the cached inference session."""
    return _load_session()


app = FastAPI(title="DefectVision Inference API", version="1.0.0")


if Mangum is not None:
    handler = Mangum(app)


@app.get("/healthz")
def health_check(session: InferenceSession = Depends(get_session)) -> dict[str, object]:
    """Simple readiness endpoint to confirm the model is loaded."""
    return {"status": "ok", "classes": len(session.class_names)}


@app.post("/predict", response_model=PredictionResponse)
async def predict(
    file: UploadFile = File(...),
    topk: int = 3,
    session: InferenceSession = Depends(get_session),
) -> PredictionResponse:
    if topk <= 0:
        raise HTTPException(status_code=400, detail="Parameter 'topk' must be greater than zero.")

    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    try:
        image = Image.open(io.BytesIO(data))
    except UnidentifiedImageError as exc:
        raise HTTPException(status_code=400, detail="Uploaded file is not a valid image.") from exc

    filename = file.filename or "uploaded-image"
    prediction = session.predict_image(image, topk=topk, image_path=Path(filename))
    top_predictions = [
        TopPrediction(label=label, confidence=confidence)
        for label, confidence in zip(prediction.topk_labels, prediction.topk_confidences)
    ]

    return PredictionResponse(
        filename=filename,
        label=prediction.label,
        confidence=prediction.confidence,
        topk=top_predictions,
    )
