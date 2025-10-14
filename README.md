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
