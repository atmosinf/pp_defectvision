"""Create train/val/test directory splits from a class-organised dataset."""

from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path
from typing import Iterable

from tqdm import tqdm
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--local-dataset",
        type=Path,
        required=True,
        help="Root directory containing class subfolders (e.g. PlantVillage).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/splits"),
        help="Directory where train/val/test splits will be created.",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.7,
        help="Fraction of samples to use for training.",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.15,
        help="Fraction of samples to use for validation.",
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.15,
        help="Fraction of samples to use for testing.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed so splits are reproducible.",
    )
    parser.add_argument(
        "--class-filter",
        nargs="+",
        default=None,
        help="Optional whitelist of class folder names.",
    )
    parser.add_argument(
        "--link",
        action="store_true",
        help="Create symlinks instead of copying files (saves space).",
    )
    parser.add_argument(
        "--clean-output",
        action="store_true",
        help="Remove existing output directory before creating splits.",
    )
    return parser.parse_args()


def iter_image_files(path: Path) -> Iterable[Path]:
    for entry in sorted(path.iterdir()):
        if entry.is_dir():
            continue
        if entry.suffix.lower() in IMAGE_EXTENSIONS:
            yield entry


def build_split_indices(total: int, train_ratio: float, val_ratio: float) -> tuple[int, int]:
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)
    if total < 3:
        return max(train_end, 1), min(total, max(train_end + 1, 2))
    train_end = max(1, min(train_end, total - 2))
    val_end = max(train_end + 1, min(val_end, total - 1))
    return train_end, val_end


def prepare_output_dirs(output_dir: Path, splits: Iterable[str], clean: bool) -> None:
    if clean and output_dir.exists():
        shutil.rmtree(output_dir)
    for split in splits:
        (output_dir / split).mkdir(parents=True, exist_ok=True)


def place_file(src: Path, dst: Path, use_symlink: bool) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if use_symlink:
        if dst.exists():
            dst.unlink()
        dst.symlink_to(src.resolve())
    else:
        shutil.copy2(src, dst)


def main() -> None:
    args = parse_args()
    total_ratio = args.train_ratio + args.val_ratio + args.test_ratio
    if abs(total_ratio - 1.0) > 1e-6:
        raise ValueError("Train/val/test ratios must sum to 1.0")

    dataset_root = args.local_dataset
    if not dataset_root.exists():
        raise FileNotFoundError(f"{dataset_root} not found.")

    rng = random.Random(args.seed)
    prepare_output_dirs(args.output_dir, ["train", "val", "test"], args.clean_output)

    split_counts = {"train": 0, "val": 0, "test": 0}
    class_dirs = [p for p in sorted(dataset_root.iterdir()) if p.is_dir()]
    for class_dir in tqdm(class_dirs, desc="Classes"):
        class_name = class_dir.name
        if args.class_filter and class_name not in args.class_filter:
            continue

        files = list(iter_image_files(class_dir))
        if not files:
            continue

        rng.shuffle(files)
        train_end, val_end = build_split_indices(len(files), args.train_ratio, args.val_ratio)
        splits = {
            "train": files[:train_end],
            "val": files[train_end:val_end],
            "test": files[val_end:],
        }

        for split_name, split_files in splits.items():
            for file_path in tqdm(
                split_files,
                desc=f"{class_name} -> {split_name}",
                leave=False,
                unit="img",
            ):
                target = args.output_dir / split_name / class_name / file_path.name
                place_file(file_path, target, args.link)
            split_counts[split_name] += len(split_files)

    for split_name, count in split_counts.items():
        print(f"{split_name}: {count} images")
    print(f"Splits created at {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
