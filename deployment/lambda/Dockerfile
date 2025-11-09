# syntax=docker/dockerfile:1.6

FROM public.ecr.aws/lambda/python:3.12

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /var/task

COPY requirements.txt .

RUN pip install --upgrade pip && \
    pip install -r requirements.txt

COPY src ./src
COPY models ./models

ENV PYTHONPATH=/var/task/src \
    DEFECTVISION_MODEL_PATH=/var/task/models/model.pth \
    DEFECTVISION_CLASS_NAMES_PATH=/var/task/models/class_names.json

CMD ["inference.api.handler"]
