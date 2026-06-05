# --- Configuration ---
PYTHON = python3
PIP = pip3
VENV = venv
ACTIVATE = . $(VENV)/bin/activate

# --- Setup ---
.PHONY: setup
setup:
	$(PYTHON) -m venv $(VENV)
	$(ACTIVATE) && $(PIP) install --upgrade pip
	$(ACTIVATE) && $(PIP) install -r requirements.txt
	@echo "🚀 Setup complete. Use 'source venv/bin/activate' to begin."

# --- Quality & Security ---
.PHONY: lint
lint:
	$(ACTIVATE) && ruff check src/
	$(ACTIVATE) && mypy --strict src/

.PHONY: security
security:
	$(ACTIVATE) && bandit -r src/ -ll

.PHONY: test
test:
	$(ACTIVATE) && pytest tests/

# --- Execution ---
.PHONY: run-api
run-api:
	$(ACTIVATE) && uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000

.PHONY: crawl
crawl:
	$(ACTIVATE) && python -m src.crawler

.PHONY: generate
generate:
	$(ACTIVATE) && python -m src.gemini_generator

.PHONY: consolidate
consolidate:
	$(ACTIVATE) && python -m src.consolidate_dataset

.PHONY: train
train:
	$(ACTIVATE) && python -m src.train_model

# --- Docker ---
.PHONY: docker-build
docker-build:
	docker build -t ai-slop-detector:latest -f deploy/Dockerfile .

.PHONY: docker-run
docker-run:
	docker run -p 8000:8000 --env-file .env ai-slop-detector:latest

# --- Cleanup ---
.PHONY: clean
clean:
	rm -rf $(VENV)
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
