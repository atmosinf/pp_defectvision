SHELL := /bin/bash

PYTHON ?= python
UVICORN ?= uvicorn
ENV_FILE ?= .env
APP_MODULE ?= inference.api:app
APP_PORT ?= 8000

.PHONY: serve-api test-integration

serve-api:
	$(UVICORN) $(APP_MODULE) --host 0.0.0.0 --port $(APP_PORT) --reload --env-file $(ENV_FILE) --app-dir src

test-integration:
	@if [ -f $(ENV_FILE) ]; then \
		set -a; \
		. $(ENV_FILE); \
		set +a; \
	fi; \
	PYTHONPATH=src $(PYTHON) -m pytest tests/test_inference_api.py
