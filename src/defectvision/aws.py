"""Lightweight AWS helper utilities."""

import boto3

from .config import get_aws_config


def get_boto3_session() -> boto3.session.Session:
    """Create a boto3 session using environment variables."""
    cfg = get_aws_config()
    return boto3.session.Session(
        aws_access_key_id=cfg.access_key_id,
        aws_secret_access_key=cfg.secret_access_key,
        aws_session_token=cfg.session_token,
        region_name=cfg.region,
    )


def get_s3_client():
    """Return an S3 client in the configured region."""
    return get_boto3_session().client("s3")
