"""Local inference utilities for PlantVillage-style classifiers."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import models, transforms


def _load_class_names(path: Path) -> List[str]:
    if path.exists():
        with path.open() as f:
            return json.load(f)
    raise FileNotFoundError(
        f"{path} not found. Provide --class-names when exporting models during training."
    )


def _build_transform() -> transforms.Compose:
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    return transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )


@dataclass
class Prediction:
    image_path: Path
    label: str
    confidence: float
    topk_labels: Sequence[str]
    topk_confidences: Sequence[float]


class InferenceSession:
    """Wrap a trained classifier for local predictions."""

    def __init__(
        self,
        model: torch.nn.Module,
        class_names: Sequence[str],
        device: torch.device | None = None,
    ) -> None:
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.model.eval()
        self.class_names = list(class_names)
        self.transform = _build_transform()

    def _predict_from_tensor(self, tensor: torch.Tensor, image_path: Path, topk: int) -> Prediction:
        logits = self.model(tensor)
        probs = F.softmax(logits, dim=1)
        confs, indices = torch.topk(probs, k=min(topk, len(self.class_names)))

        confs = confs.squeeze(0).cpu().tolist()
        indices = indices.squeeze(0).cpu().tolist()
        top_labels = [self.class_names[i] for i in indices]

        return Prediction(
            image_path=image_path,
            label=top_labels[0],
            confidence=confs[0],
            topk_labels=top_labels,
            topk_confidences=confs,
        )

    @torch.inference_mode()
    def predict_batch(self, images: Iterable[Path], topk: int = 3) -> List[Prediction]:
        predictions: List[Prediction] = []
        for image_path in images:
            img = Image.open(image_path).convert("RGB")
            tensor = self.transform(img).unsqueeze(0).to(self.device)
            predictions.append(self._predict_from_tensor(tensor, image_path, topk))
        return predictions

    @torch.inference_mode()
    def predict_image(
        self,
        image: Image.Image,
        *,
        topk: int = 3,
        image_path: Path | None = None,
    ) -> Prediction:
        """Score a single in-memory PIL image and return the top predictions."""
        tensor = self.transform(image.convert("RGB")).unsqueeze(0).to(self.device)
        resolved_path = image_path or Path(getattr(image, "filename", "in-memory-image"))
        return self._predict_from_tensor(tensor, resolved_path, topk)


def load_session(
    model_path: Path,
    class_names_path: Path | None = None,
    model_name: str = "resnet18",
) -> InferenceSession:
    """Create an inference session from a trained checkpoint."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if model_name == "resnet18":
        model = models.resnet18(weights=None)
    elif model_name == "resnet34":
        model = models.resnet34(weights=None)
    else:
        raise ValueError(f"Unsupported model {model_name}")

    class_names_file = class_names_path or model_path.with_name("class_names.json")
    class_names = _load_class_names(class_names_file)

    state_dict = torch.load(model_path, map_location=device)
    model.fc = torch.nn.Linear(model.fc.in_features, len(class_names))
    model.load_state_dict(state_dict)

    if len(class_names) != model.fc.out_features:
        raise ValueError(
            "Mismatch between number of class names and model output features. "
            "Ensure you saved class names during training."
        )

    return InferenceSession(model=model, class_names=class_names, device=device)
