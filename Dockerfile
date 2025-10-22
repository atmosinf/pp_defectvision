# syntax=docker/dockerfile:1.6

FROM python:3.12-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Install system packages needed for scientific/python imaging stack.
RUN apt-get update && \
    apt-get install --no-install-recommends -y \
        build-essential \
        libglib2.0-0 \
        libgl1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --upgrade pip && \
    pip install -r requirements.txt

COPY src ./src
COPY scripts ./scripts
COPY docker/entrypoint.sh /entrypoint.sh

# Copy models if they are included in the build context; otherwise users can mount or supply via env.
COPY models ./models

RUN chmod +x /entrypoint.sh

ENV PYTHONPATH=/app/src \
    PORT=8000

EXPOSE 8000

ENTRYPOINT ["/entrypoint.sh"]
