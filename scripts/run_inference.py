"""Run local inference against a trained DefectVision model."""

from __future__ import annotations

import argparse
from pathlib import Path

from inference import load_session


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path, help="Path to the trained model checkpoint (model.pth).")
    parser.add_argument("images", type=Path, help="Image file or directory containing images to score.")
    parser.add_argument(
        "--classes",
        type=Path,
        default=None,
        help="Optional custom path to class_names.json (defaults to sibling of model).",
    )
    parser.add_argument("--model-name", default="resnet18", help="Backbone model name (default: resnet18).")
    parser.add_argument("--topk", type=int, default=3, help="Number of top predictions to display per image.")
    return parser.parse_args()


def gather_images(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    if not path.exists():
        raise FileNotFoundError(f"{path} not found")
    return [p for p in path.rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]


def main() -> None:
    args = parse_args()
    session = load_session(args.model, args.classes, model_name=args.model_name)

    images = gather_images(args.images)
    if not images:
        raise ValueError("No images found to score")

    predictions = session.predict_batch(images, topk=args.topk)
    for pred in predictions:
        print(f"Image: {pred.image_path}")
        for label, conf in zip(pred.topk_labels, pred.topk_confidences):
            print(f"  {label}: {conf:.4f}")


if __name__ == "__main__":
    main()
