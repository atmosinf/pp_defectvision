"""Generate train/val/test manifests for uploaded datasets."""

import argparse
import csv
import random
from collections import defaultdict
from pathlib import Path
from typing import Iterable

from defectvision.config import get_aws_config

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--local-dataset",
        type=Path,
        required=True,
        help="Root directory containing class subfolders.",
    )
    parser.add_argument(
        "--s3-prefix",
        type=str,
        required=True,
        help="Prefix inside the S3 bucket where images reside (e.g. datasets/plantvillage).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/manifests"),
        help="Where to write the CSV manifests.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for deterministic splits.",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.7,
        help="Fraction of samples reserved for training.",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.15,
        help="Fraction of samples used for validation.",
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.15,
        help="Fraction of samples used for testing.",
    )
    parser.add_argument(
        "--class-filter",
        nargs="+",
        default=None,
        help="Optional list of class folder names to keep.",
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
    # Ensure at least one sample per split when possible.
    train_end = max(min(train_end, total - 2), 1) if total >= 3 else max(train_end, 1)
    val_end = max(min(val_end, total - 1), train_end + 1) if total >= 3 else max(val_end, train_end + 1)
    val_end = min(val_end, total)
    return train_end, val_end


def create_manifests(args: argparse.Namespace) -> None:
    cfg = get_aws_config()
    dataset_root = args.local_dataset
    if not dataset_root.exists():
        raise FileNotFoundError(f"{dataset_root} not found.")

    rng = random.Random(args.seed)
    entries = defaultdict(list)

    for class_dir in sorted(p for p in dataset_root.iterdir() if p.is_dir()):
        class_name = class_dir.name
        if args.class_filter and class_name not in args.class_filter:
            continue
        files = list(iter_image_files(class_dir))
        if not files:
            continue
        rng.shuffle(files)
        train_end, val_end = build_split_indices(
            total=len(files),
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
        )
        splits = {
            "train": files[:train_end],
            "val": files[train_end:val_end],
            "test": files[val_end:],
        }
        for split_name, split_files in splits.items():
            for file_path in split_files:
                relative_key = f"{class_name}/{file_path.name}"
                s3_uri = f"s3://{cfg.bucket}/{args.s3_prefix.rstrip('/')}/{relative_key}"
                entries[split_name].append((s3_uri, class_name))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for split_name, rows in entries.items():
        output_file = args.output_dir / f"{dataset_root.name}_{split_name}.csv"
        with output_file.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["s3_uri", "label"])
            writer.writerows(rows)
        print(f"Wrote {len(rows)} entries to {output_file}")


def main() -> None:
    args = parse_args()
    total_ratio = args.train_ratio + args.val_ratio + args.test_ratio
    if not abs(total_ratio - 1.0) < 1e-6:
        raise ValueError("Train/val/test ratios must sum to 1.0")
    create_manifests(args)


if __name__ == "__main__":
    main()
