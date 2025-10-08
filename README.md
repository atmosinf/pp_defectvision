# pp_defectvision

## Uploading datasets to S3

1. Set the target bucket (and AWS credentials) in your environment:
   ```bash
   export DEFECTVISION_S3_BUCKET=<your-bucket-name>
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
