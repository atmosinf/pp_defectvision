# pp_defectvision

## Uploading datasets to S3

1. Set the target bucket (and AWS credentials) in your environment:
   ```bash
   export S3_BUCKET=<your-bucket-name>
   export AWS_ACCESS_KEY_ID=...
   export AWS_SECRET_ACCESS_KEY=...
   export AWS_DEFAULT_REGION=eu-north-1  # or your region
   ```
2. Install requirements (includes `boto3`):
   ```bash
   pip install -r requirements.txt
   ```
3. Upload any file or directory:
   ```bash
   python scripts/upload_to_s3.py --local-path data/plantdisease/plantvillage --s3-prefix datasets/plantvillage
   ```
   The script mirrors the folder structure under the provided prefix.

## Creating train/val/test manifests

Generate CSV manifests that SageMaker (or your training code) can consume:

```bash
export PYTHONPATH=src  # if not already set
python scripts/create_manifest.py \
  --local-dataset data/plantdisease/plantvillage/PlantVillage \
  --s3-prefix datasets/plantvillage \
  --output-dir data/manifests \
  --class-filter Tomato_healthy Tomato_Early_blight Tomato_Late_blight
```

- Adjust `--class-filter` to include the class folders you want (omit it to include every class).
- The script writes CSV files like `data/manifests/PlantVillage_train.csv` with two columns: `s3_uri` and `label`.

## Creating train/val/test directory splits

Use the splitting helper to materialise class-balanced folder structures (copying files by default):

```bash
python scripts/split_dataset.py \
  --local-dataset data/plantdisease/plantvillage/PlantVillage \
  --output-dir data/splits \
  --train-ratio 0.7 --val-ratio 0.15 --test-ratio 0.15
```

- Add `--link` to create symlinks instead of copying data (useful to save disk space locally).
- Add `--clean-output` if you want the script to remove any existing split directory before recreating it.

## Launch a SageMaker training job (spot instances)

1. Upload the prepared splits to S3:
   ```bash
   python scripts/upload_to_s3.py --local-path data/splits/train --s3-prefix datasets/plantvillage/train
   python scripts/upload_to_s3.py --local-path data/splits/val --s3-prefix datasets/plantvillage/val
   python scripts/upload_to_s3.py --local-path data/splits/test --s3-prefix datasets/plantvillage/test  # optional
   ```
2. Ensure both `DEFECTVISION_S3_BUCKET` and `DEFECTVISION_SAGEMAKER_ROLE_ARN` are set in your environment (the role needs SageMaker + S3 access).
3. Install dependencies locally (adds the `sagemaker` SDK):
   ```bash
   pip install -r requirements.txt
   ```
4. Launch training with spot instances:
   ```bash
   export PYTHONPATH=src
   python scripts/launch_sagemaker_training.py \
     --train-prefix s3://<bucket>/datasets/plantvillage/train \
     --val-prefix s3://<bucket>/datasets/plantvillage/val \
     --instance-type ml.p3.2xlarge \
     --epochs 5
   ```
   - The script enables spot instances by default; tune `--max-run` / `--max-wait` as required.
   - Checkpoints are written to `s3://<bucket>/checkpoints/plantvillage/`. Adjust with `--checkpoint-prefix` if needed.
   - Modify hyperparameters (`--batch-size`, `--learning-rate`, etc.) to experiment with different training configurations.
   - Add `--test-prefix s3://<bucket>/datasets/plantvillage/test` if you uploaded a held-out split.
   - Append `--no-spot` when you need on-demand instances (e.g. zero spot quota).
    - The training script saves `model.pth`, `class_names.json`, and `training_metrics.json` in the model output directory.

## Running local inference

1. Download the trained artifacts from SageMaker (e.g. `model.pth` and `class_names.json`) into the repo.
2. Score a directory or individual image:
   ```bash
   export PYTHONPATH=src
   python scripts/run_inference.py \
     models/model.pth \
     data/sample_images \
     --classes models/class_names.json \
     --topk 5
   ```
   Adjust the paths to wherever you placed the downloaded artifacts. The script prints the top predictions and confidences per image.

## Serving the FastAPI inference API

1. Make sure your trained artifacts exist locally and that the environment variables point to them (for example in a `.env` file):
   ```bash
   DEFECTVISION_MODEL_PATH="/workspaces/pp_defectvision/models/.../model.pth"
   DEFECTVISION_CLASS_NAMES_PATH="/workspaces/pp_defectvision/models/.../class_names.json"
   DEFECTVISION_MODEL_NAME="resnet18"
   ```
2. Launch the API with Uvicorn, loading the `.env` file so the model paths resolve:
   ```bash
   uvicorn inference.api:app \
     --host 0.0.0.0 \
     --port 8000 \
     --reload \
     --env-file .env \
     --app-dir src
   ```
   Using `--app-dir src` (or exporting `PYTHONPATH=src`) tells Uvicorn where to find the `inference` package. Drop `--reload` for production.
   - Alternatively, `make serve-api` runs the same command (override `ENV_FILE` or `APP_PORT` as needed).
3. Confirm the service is up:
   ```bash
   curl http://127.0.0.1:8000/healthz
   ```
   A 200 response with the class count indicates the model loaded correctly.
4. Score an image through the `/predict` endpoint (quote the path if it contains spaces):
   ```bash
   curl -X POST "http://127.0.0.1:8000/predict?topk=3" \
     -H "accept: application/json" \
     -F 'file=@src/inference/test_images/Tomato_healthy__000bf685-b305-408b-91f4-37030f8e62db___GH_HL Leaf 308.1.JPG'
   ```
   Replace the sample image path with any leaf photo you want to score. The response returns the predicted label, confidence, and top-k alternatives.

## Testing the inference API

Install dev dependencies and run the integration test against the live model artifacts:
```bash
pip install -r requirements-dev.txt
make test-integration
```
The test suite hits `/healthz` and `/predict` using a sample image to ensure the service loads the checkpoint and produces the expected prediction.

## Containerising the API

Build the image (requires internet access for Python wheels):
```bash
docker build -t defectvision-api .
```

Run the container, forwarding port 8000 and injecting your environment variables:
```bash
docker run --rm -p 8000:8000 \
  --env-file .env \
  defectvision-api
```

- `docker run --env-file .env` injects each variable into the container; the file itself stays on the host. To have the entrypoint source a file, bind-mount it and set `ENV_FILE=/app/.env`.
- The image does **not** bundle model artifacts. Provide them by either bind-mounting a local directory (e.g. `-v "$PWD/models/.../output:/app/models:ro"`) or exporting `DEFECTVISION_MODEL_S3_URI` / `DEFECTVISION_CLASS_NAMES_S3_URI` so the entrypoint can download from S3 at boot (requires valid AWS credentials in the container).
- By default the API expects `/app/models/model.pth` and `/app/models/class_names.json`. Override `DEFECTVISION_MODEL_PATH` / `DEFECTVISION_CLASS_NAMES_PATH` if you store them elsewhere.
- Override `PORT`, `APP_MODULE`, or `APP_DIR` if you need to customise the Uvicorn launch command.

### Lambda deployment

The FastAPI app exposes a Lambda-compatible handler via `inference.api.handler` (powered by `mangum`). To deploy the same container image to AWS Lambda:

1. Make sure the GitHub Actions workflow has pushed a fresh image to ECR (see “CI builds” below). Note the full image URI (`<account>.dkr.ecr.eu-north-1.amazonaws.com/defectvision-api:latest` or commit SHA).
2. In the AWS console (Lambda → Create function) choose **Container image** and provide that URI. Alternatively, use `aws lambda create-function --package-type Image ...`.
3. Set environment variables on the Lambda function:
   - `DEFECTVISION_MODEL_PATH` / `DEFECTVISION_CLASS_NAMES_PATH` (e.g. `/tmp/model.pth`, `/tmp/class_names.json`).
   - Either mount the artifacts via a Lambda layer/EFS, or supply `DEFECTVISION_MODEL_S3_URI` / `DEFECTVISION_CLASS_NAMES_S3_URI` so the entrypoint downloads them to those paths at cold start. Grant the function’s execution role `s3:GetObject` on the artifact bucket.
4. Update the Lambda execution role to include CloudWatch logging, S3 access, and (if needed) VPC permissions.
5. Optionally front the function with API Gateway or Lambda Function URLs for HTTPS access. Because the app uses `Mangum`, FastAPI routes work untouched.

For local testing before deploying, you can run `sam local start-api` or `sam local invoke --event events/sample.json --docker-network host` using the same image.

### Deploying with AWS SAM

The repository includes `sam/template.yaml`, letting you manage the Lambda + API Gateway stack with AWS SAM (Serverless Application Model).

1. [Install the SAM CLI](https://docs.aws.amazon.com/serverless-application-model/latest/developerguide/install-sam-cli.html) and ensure you have credentials with the necessary IAM permissions (`cloudformation:*`, `iam:PassRole` for execution roles, `lambda:*`, `apigateway:*`, plus S3 read on your model artifacts).
2. Populate the required parameters when you deploy. Example command:
   ```bash
   sam deploy --template-file sam/template.yaml \
     --stack-name defectvision-api \
     --capabilities CAPABILITY_IAM \
     --parameter-overrides \
       ImageUri=123456789012.dkr.ecr.eu-north-1.amazonaws.com/defectvision-api:latest \
       ModelPath=/tmp/model.pth \
       ClassNamesPath=/tmp/class_names.json \
       ModelS3Uri=s3://defectvision-bucket/checkpoints/model.pth \
       ClassNamesS3Uri=s3://defectvision-bucket/checkpoints/class_names.json
   ```
   Use `--guided` the first time to persist answers in `samconfig.toml`.
3. After deployment, SAM outputs the API Gateway URL (see `Outputs.ApiUrl`) so you can hit `/healthz` and `/predict`.
4. Update `sam/samconfig.toml` with your preferred defaults (stack name, artifact bucket, region). The template provides placeholders—ensure the bucket (`defectvision-sam-artifacts` in the file) exists in the target region or replace it with one you control.
4. Optional: `sam local start-api --parameter-overrides ImageUri=...` spins up the container locally through the SAM runtime for parity testing.

### CI builds

GitHub Actions workflow `.github/workflows/docker.yml` builds the image for every PR and push to `main`. PRs run a validation build only. On `main` pushes the workflow assumes an AWS IAM role (supplied via `AWS_ECR_ROLE_ARN`) and pushes the image to Amazon ECR as `<account>.dkr.ecr.<region>.amazonaws.com/defectvision-api` tagged with both the commit SHA and `latest`.

To enable the push:
- Create (or reuse) an ECR repository named `defectvision-api` in your target region (default `eu-north-1`—override via the workflow’s `AWS_REGION` env variable).
- Configure an IAM role that trusts GitHub’s OIDC provider and grants `ecr:GetAuthorizationToken`, `ecr:BatchGetImage`, `ecr:BatchCheckLayerAvailability`, `ecr:CompleteLayerUpload`, `ecr:CreateRepository` (optional), `ecr:InitiateLayerUpload`, `ecr:PutImage`, and `ecr:UploadLayerPart`. Store the role ARN in the repo secret `AWS_ECR_ROLE_ARN`.
- Ensure the workflow (or repo-level default) grants the `id-token: write` permission so GitHub can mint OIDC tokens for the role assumption.
- If you change the repository name or region, update `ECR_REPOSITORY` / `AWS_REGION` in `.github/workflows/docker.yml`.
.
