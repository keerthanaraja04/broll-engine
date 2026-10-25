.PHONY: test test-unit test-integration test-e2e test-e2e-mock lint format openapi install dev

PYTHON ?= python

install:
	$(PYTHON) -m pip install -e .

dev:
	$(PYTHON) -m pip install -e ".[dev]"

# Default target — runs unit + integration + mock e2e. No API keys, no docker needed.
test: test-unit test-integration test-e2e-mock
	@echo ""
	@echo "==================================="
	@echo " All tests passed (no keys needed)"
	@echo "==================================="

test-unit:
	$(PYTHON) -m pytest tests/unit -v --tb=short -x

test-integration:
	$(PYTHON) -m pytest tests/integration -v --tb=short -x

# E2E tests that only use in-process mocks — safe to run in CI with no keys
test-e2e-mock:
	$(PYTHON) -m pytest tests/e2e/test_mocked_e2e.py -v --tb=short -x -m e2e_mock

# Full E2E — requires `docker compose up -d` first
test-e2e:
	$(PYTHON) -m pytest tests/e2e -v --tb=short -m e2e

lint:
	$(PYTHON) -m ruff check app/ tests/ cli/

format:
	$(PYTHON) -m ruff format app/ tests/ cli/

# Generate openapi.yaml from the running FastAPI app
openapi:
	$(PYTHON) scripts/generate_openapi.py

# Generate a stub openapi.yaml (used when deps aren't installed)
openapi-stub:
	$(PYTHON) scripts/generate_openapi.py --output openapi.yaml
