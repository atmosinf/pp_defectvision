"""Minimal configuration helpers for DefectVision."""

import os
from dataclasses import dataclass


@dataclass
class AWSConfig:
    """Pulled from environment variables."""

    region: str
    bucket: str
    access_key_id: str | None = None
    secret_access_key: str | None = None
    session_token: str | None = None
    sagemaker_role_arn: str | None = None


def get_aws_config() -> AWSConfig:
    """Load settings needed for S3 interactions."""
    bucket = os.environ.get("S3_BUCKET")
    if not bucket:
        raise RuntimeError(
            "Missing S3_BUCKET environment variable. "
            "Set it to the target bucket name before running the upload script."
        )

    region = os.environ.get("AWS_DEFAULT_REGION", "eu-north-1")

    return AWSConfig(
        region=region,
        bucket=bucket,
        access_key_id=os.environ.get("AWS_ACCESS_KEY_ID"),
        secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY"),
        session_token=os.environ.get("AWS_SESSION_TOKEN"),
        sagemaker_role_arn=os.environ.get("SAGEMAKER_EXEC_ROLE_ARN"),
    )
