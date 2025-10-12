"""SageMaker-compatible training script for PlantVillage-style datasets."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--model-name", type=str, default="resnet18")
    parser.add_argument(
        "--train-dir",
        type=Path,
        default=None,
        help="Override training data directory (defaults to SM_CHANNEL_TRAIN).",
    )
    parser.add_argument(
        "--val-dir",
        type=Path,
        default=None,
        help="Override validation data directory (defaults to SM_CHANNEL_VAL).",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=None,
        help="Model output directory (defaults to SM_MODEL_DIR).",
    )
    return parser.parse_args()


def resolve_data_dirs(args: argparse.Namespace) -> Tuple[Path, Path]:
    train_dir = args.train_dir or Path(os.environ.get("SM_CHANNEL_TRAIN", "data/train"))
    val_dir = args.val_dir or Path(os.environ.get("SM_CHANNEL_VAL", "data/val"))
    if not train_dir.exists():
        raise FileNotFoundError(f"Training directory {train_dir} does not exist")
    if not val_dir.exists():
        raise FileNotFoundError(f"Validation directory {val_dir} does not exist")
    return train_dir, val_dir


def resolve_model_dir(args: argparse.Namespace) -> Path:
    model_dir = args.model_dir or Path(os.environ.get("SM_MODEL_DIR", "model"))
    model_dir.mkdir(parents=True, exist_ok=True)
    return model_dir


def build_transforms() -> Dict[str, transforms.Compose]:
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    train_transform = transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )
    eval_transform = transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )
    return {"train": train_transform, "val": eval_transform}


def create_dataloaders(
    train_dir: Path,
    val_dir: Path,
    batch_size: int,
    num_workers: int,
) -> Tuple[Dict[str, DataLoader], Dict[str, int], list[str]]:
    transforms_map = build_transforms()
    image_datasets = {
        "train": datasets.ImageFolder(train_dir, transform=transforms_map["train"]),
        "val": datasets.ImageFolder(val_dir, transform=transforms_map["val"]),
    }
    dataloaders = {
        split: DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(split == "train"),
            num_workers=num_workers,
        )
        for split, dataset in image_datasets.items()
    }
    dataset_sizes = {split: len(dataset) for split, dataset in image_datasets.items()}
    class_names = image_datasets["train"].classes
    return dataloaders, dataset_sizes, class_names


def build_model(model_name: str, num_classes: int) -> nn.Module:
    if model_name.lower() == "resnet18":
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
    elif model_name.lower() == "resnet34":
        model = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
    else:
        raise ValueError(f"Unsupported model {model_name}")
    return model


def train_model(
    model: nn.Module,
    dataloaders: Dict[str, DataLoader],
    dataset_sizes: Dict[str, int],
    device: torch.device,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    epochs: int,
) -> Tuple[nn.Module, Dict[str, list]]:
    start_time = time.time()
    best_acc = 0.0
    best_weights = None
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        print("-" * 20)

        for phase in ["train", "val"]:
            if phase == "train":
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data).item()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects / dataset_sizes[phase]

            history[f"{phase}_loss"].append(epoch_loss)
            history[f"{phase}_acc"].append(epoch_acc)

            print(f"{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

            if phase == "val" and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_weights = model.state_dict()

    elapsed = time.time() - start_time
    print(f"Training complete in {elapsed // 60:.0f}m {elapsed % 60:.0f}s")
    print(f"Best val Acc: {best_acc:.4f}")

    if best_weights is not None:
        model.load_state_dict(best_weights)

    return model, history


def save_outputs(model: nn.Module, history: Dict[str, list], model_dir: Path) -> None:
    model_path = model_dir / "model.pth"
    torch.save(model.state_dict(), model_path)
    print(f"Saved model to {model_path}")

    output_dir = Path(os.environ.get("SM_OUTPUT_DATA_DIR", model_dir))
    output_dir.mkdir(parents=True, exist_ok=True)
    history_path = output_dir / "training_metrics.json"
    with history_path.open("w") as f:
        json.dump(history, f, indent=2)
    print(f"Saved metrics to {history_path}")


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_dir, val_dir = resolve_data_dirs(args)
    dataloaders, dataset_sizes, class_names = create_dataloaders(
        train_dir=train_dir,
        val_dir=val_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    print(f"Found classes: {class_names}")

    model = build_model(args.model_name, num_classes=len(class_names))
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    model, history = train_model(
        model=model,
        dataloaders=dataloaders,
        dataset_sizes=dataset_sizes,
        device=device,
        optimizer=optimizer,
        criterion=criterion,
        epochs=args.epochs,
    )

    model_dir = resolve_model_dir(args)
    save_outputs(model, history, model_dir)


if __name__ == "__main__":
    main()
