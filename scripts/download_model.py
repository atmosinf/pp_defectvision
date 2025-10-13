import os
from pathlib import Path

import boto3


def download_prefix(bucket: str, prefix: str, destination_dir: str) -> None:
    """Download every object under `prefix` into `destination_dir`."""
    s3 = boto3.client("s3")
    paginator = s3.get_paginator("list_objects_v2")

    # Ensure prefix ends with / so relative paths compute correctly.
    normalized_prefix = prefix.rstrip("/") + "/"
    destination_root = Path(destination_dir)

    for page in paginator.paginate(Bucket=bucket, Prefix=normalized_prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]

            if key.endswith("/"):
                continue

            relative_key = key[len(normalized_prefix) :]
            local_path = destination_root / relative_key
            local_path.parent.mkdir(parents=True, exist_ok=True)

            s3.download_file(bucket, key, str(local_path))


if __name__ == "__main__":
    download_prefix(
        bucket="defectvision-bucket",
        prefix="defectvision-train-20251012-195857/",
        destination_dir=os.path.join("models", "defectvision-train-20251012-195857"),
    )
