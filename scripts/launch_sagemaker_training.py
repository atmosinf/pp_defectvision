"""Launch a SageMaker PyTorch training job using the prepared manifests."""

from __future__ import annotations

import argparse
import datetime as dt
from pathlib import Path

from defectvision.aws import get_boto3_session
from defectvision.config import get_aws_config
from sagemaker.inputs import TrainingInput
from sagemaker.pytorch import PyTorch
from sagemaker.session import Session


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--train-prefix",
        required=True,
        help="S3 prefix containing training data (e.g. s3://bucket/datasets/.../train).",
    )
    parser.add_argument(
        "--val-prefix",
        required=True,
        help="S3 prefix containing validation data.",
    )
    parser.add_argument(
        "--test-prefix",
        default=None,
        help="Optional S3 prefix with test data.",
    )
    parser.add_argument("--instance-type", default="ml.p3.2xlarge", help="SageMaker instance type.")
    parser.add_argument("--instance-count", type=int, default=1, help="Number of instances for the job.")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--model-name", default="resnet18")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--framework-version", default="2.2", help="PyTorch image version.")
    parser.add_argument(
        "--checkpoint-prefix",
        default="checkpoints/plantvillage",
        help="S3 prefix where SageMaker stores training checkpoints.",
    )
    parser.add_argument("--max-run", type=int, default=3 * 60 * 60, help="Training job timeout in seconds.")
    parser.add_argument(
        "--max-wait",
        type=int,
        default=5 * 60 * 60,
        help="Spot wait time (>= max-run). Ignored when --no-spot is set.",
    )
    parser.add_argument(
        "--no-spot",
        action="store_true",
        help="Disable spot training (run on on-demand instances).",
    )
    parser.add_argument(
        "--job-name",
        default=None,
        help="Optional custom job name. Defaults to timestamped value.",
    )
    parser.add_argument(
        "--role-arn",
        default=None,
        help="Override SageMaker execution role ARN (defaults to DEFECTVISION_SAGEMAKER_ROLE_ARN).",
    )
    return parser.parse_args()


def ensure_s3_prefix(uri: str) -> str:
    if not uri.startswith("s3://"):
        raise ValueError(f"Expected S3 URI, got: {uri}")
    return uri.rstrip("/")


def main() -> None:
    args = parse_args()
    cfg = get_aws_config()

    role_arn = args.role_arn or cfg.sagemaker_role_arn
    if not role_arn:
        raise RuntimeError(
            "SageMaker role ARN not provided. Set SAGEMAKER_EXEC_ROLE_ARN or pass --role-arn."
        )
    use_spot = not args.no_spot
    if use_spot and args.max_wait < args.max_run:
        raise ValueError("--max-wait must be greater than or equal to --max-run for spot training.")

    boto_session = get_boto3_session()
    sm_session = Session(boto_session=boto_session, default_bucket=cfg.bucket)

    checkpoint_s3_uri = f"s3://{cfg.bucket}/{args.checkpoint_prefix.strip('/')}"

    estimator = PyTorch(
        entry_point="train.py",
        source_dir=str(Path("src") / "training"),
        role=role_arn,
        framework_version=args.framework_version,
        py_version="py310",
        instance_count=args.instance_count,
        instance_type=args.instance_type,
        hyperparameters={
            "epochs": args.epochs,
            "batch-size": args.batch_size,
            "learning-rate": args.learning_rate,
            "weight-decay": args.weight_decay,
            "model-name": args.model_name,
            "num-workers": args.num_workers,
        },
        sagemaker_session=sm_session,
        use_spot_instances=use_spot,
        max_run=args.max_run,
        max_wait=args.max_wait if use_spot else None,
        checkpoint_s3_uri=checkpoint_s3_uri,
        enable_sagemaker_metrics=True,
        disable_profiler=True,
    )

    inputs = {
        "train": TrainingInput(
            s3_data=ensure_s3_prefix(args.train_prefix),
            distribution="FullyReplicated",
            input_mode="File",
            s3_data_type="S3Prefix",
        ),
        "val": TrainingInput(
            s3_data=ensure_s3_prefix(args.val_prefix),
            distribution="FullyReplicated",
            input_mode="File",
            s3_data_type="S3Prefix",
        ),
    }

    if args.test_prefix:
        inputs["test"] = TrainingInput(
            s3_data=ensure_s3_prefix(args.test_prefix),
            distribution="FullyReplicated",
            input_mode="File",
            s3_data_type="S3Prefix",
        )

    timestamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    job_name = args.job_name or f"defectvision-train-{timestamp}"
    estimator.fit(inputs=inputs, job_name=job_name, wait=True)
    spot_msg = "spot instances" if use_spot else "on-demand instances"
    print(f"Job {job_name} launched using {spot_msg}.")


if __name__ == "__main__":
    main()
