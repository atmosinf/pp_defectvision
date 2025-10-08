"""Upload local dataset assets to S3."""

from __future__ import annotations

import argparse
from pathlib import Path

from defectvision.aws import get_s3_client
from defectvision.config import get_aws_config
from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--local-path",
        type=Path,
        required=True,
        help="Directory or file to upload.",
    )
    parser.add_argument(
        "--s3-prefix",
        type=str,
        required=True,
        help="Prefix inside the configured bucket (e.g. datasets/plantvillage).",
    )
    return parser.parse_args()


def upload_path(local_path: Path, s3_prefix: str) -> None:
    cfg = get_aws_config()
    client = get_s3_client()

    if local_path.is_file():
        files = [local_path]
        base = local_path.parent
    else:
        if not local_path.exists():
            raise FileNotFoundError(f"{local_path} does not exist")
        files = [p for p in local_path.rglob("*") if p.is_file()]
        base = local_path

    for file_path in tqdm(files, desc="Uploading", unit="file"):
        relative_key = file_path.relative_to(base).as_posix()
        key = f"{s3_prefix.rstrip('/')}/{relative_key}"
        client.upload_file(
            Filename=str(file_path),
            Bucket=cfg.bucket,
            Key=key,
        )


def main() -> None:
    args = parse_args()
    upload_path(args.local_path, args.s3_prefix)


if __name__ == "__main__":
    main()
