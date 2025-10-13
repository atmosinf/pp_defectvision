"""Sample images from the dataset into the inference test_images folder."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        required=True,
        help="Directory containing class subfolders (e.g. data/splits/test).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("src/inference/test_images"),
        help="Destination where sampled images will be copied.",
    )
    parser.add_argument(
        "--per-class",
        type=int,
        default=5,
        help="Number of images to copy for each class.",
    )
    parser.add_argument(
        "--clean-output",
        action="store_true",
        help="Remove existing files in the output directory before copying.",
    )
    parser.add_argument(
        "--link",
        action="store_true",
        help="Create symlinks instead of copying files (saves disk space).",
    )
    return parser.parse_args()


def prepare_output(output_dir: Path, clean: bool) -> None:
    if clean and output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)


def main() -> None:
    args = parse_args()
    if args.per_class <= 0:
        raise ValueError("--per-class must be positive")

    dataset_dir = args.dataset_dir
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset directory {dataset_dir} not found")

    prepare_output(args.output_dir, args.clean_output)

    class_dirs = [p for p in sorted(dataset_dir.iterdir()) if p.is_dir()]
    for class_dir in tqdm(class_dirs, desc="Classes"):
        class_name = class_dir.name
        images = sorted(
            [p for p in class_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
        )
        if not images:
            continue

        for image_path in tqdm(images[: args.per_class], desc=class_name, leave=False):
            target_name = f"{class_name}__{image_path.name}"
            target_path = args.output_dir / target_name
            if args.link:
                if target_path.exists():
                    target_path.unlink()
                target_path.symlink_to(image_path.resolve())
            else:
                shutil.copy2(image_path, target_path)


if __name__ == "__main__":
    main()
