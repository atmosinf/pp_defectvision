
import boto3
client = boto3.client("service-quotas")
quotas = client.list_service_quotas(ServiceCode="sagemaker")["Quotas"]
for q in quotas:
    if "training job usage" in q["QuotaName"].lower():
        print(f"{q['QuotaName']}: {q['Value']}")
PY
