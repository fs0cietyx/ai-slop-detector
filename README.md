# AI Slop Detector (SlopGuard-ML)

A high-fidelity machine learning suite engineered to detect AI-generated content ("slop") across multi-domain datasets. This project provides a full-stack solution from high-scale dataset training with LoRA adapters to real-time inference via a robust CLI and FastAPI microservice, achieving production-ready precision through hardware-accelerated computation.

## Key Features

- **Hardware-Agnostic Inference Engine**: Dynamically routes tensor computations to the most efficient hardware available (Apple Silicon MPS, NVIDIA CUDA, or CPU fallback).
- **LoRA-Optimized Fine-Tuning**: Trains robust detection models with a 99% reduction in trainable parameters, allowing for rapid adaptation without expensive compute clusters.
- **Microservice API**: A high-throughput, low-latency FastAPI gateway designed to integrate seamlessly into content moderation pipelines.
- **Enterprise-Grade Tooling**: Complete with rigorous static analysis (Ruff, Mypy, Bandit), 94% test coverage, and a fully automated GitHub Actions CI/CD pipeline.

## Tech Stack

- **Language**: Python 3.11+
- **Machine Learning**: PyTorch, Transformers (HuggingFace), PEFT, Evaluate, Scikit-Learn
- **API Framework**: FastAPI, Uvicorn, Pydantic V2
- **CLI Framework**: Typer
- **Testing & QA**: Pytest, Pytest-Cov, Pre-commit (Ruff, Mypy, Bandit)
- **Documentation**: MkDocs (Material theme)

## Prerequisites

- Python 3.11 or higher
- A compatible GPU/NPU for accelerated training and inference (optional, but recommended)
- Git

## Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/fs0cietyx/ai-slop-detector.git
cd ai-slop-detector
```

### 2. Install Dependencies

It is highly recommended to use a virtual environment. Install the package in editable mode along with its development dependencies:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
python -m pip install -e ".[dev]"
```

### 3. Initialize Pre-commit Hooks

Ensure code quality standards are maintained locally before pushing:

```bash
pre-commit install
```

### 4. Run Inference via CLI

You can immediately test the detection engine using the unified command-line interface:

```bash
slopguard predict "The quick brown fox jumps over the lazy dog. This is a baseline human text."
```

### 5. Start the API Server

Launch the REST service for integration testing:

```bash
python -m src.slopguard.api.main
```
The server will initialize the inference engine and listen on `http://127.0.0.1:8000`.

## Architecture

### Directory Structure

```text
├── src/
│   └── slopguard/
│       ├── api/               # FastAPI routes and middleware
│       │   └── main.py        # Microservice entrypoint
│       ├── cli/               # Typer CLI commands
│       │   ├── main.py        # CLI router
│       │   ├── predict.py     # Inference execution logic
│       │   └── train_model.py # LoRA training orchestration
│       ├── core/              # Core business logic
│       │   ├── config.py      # Pydantic environment configuration
│       │   └── engine.py      # Hardware-agnostic InferenceEngine singleton
│       └── data/              # Data ingestion pipelines
├── tests/                     # Pytest suite (Unit & Integration)
├── docs/                      # MkDocs markdown documentation files
├── .github/workflows/         # Enterprise CI/CD pipelines
├── pyproject.toml             # Project metadata, dependencies, and QA config
└── mkdocs.yml                 # Documentation site configuration
```

### Request Lifecycle (API)

1. Client sends an HTTP POST request to `/v1/predict` with a text payload.
2. FastAPI validates the request schema and API Key via Pydantic and middleware.
3. The request is routed to the `InferenceEngine` singleton.
4. The text is sanitized (stripping control characters and excess length).
5. The text is tokenized and processed through the sequence classification transformer on the optimal hardware (MPS/CUDA/CPU).
6. Logits are converted to probability distributions to extract the confidence score.
7. A JSON response with the label, confidence, and latency is returned to the client.

### Data Flow

```text
User Input → FastAPI Route/CLI → Input Sanitization → Tokenizer →
Hardware-Accelerated Transformer → Softmax Logits → Confidence Scoring → Output
```

## Environment Variables

Configuration is securely managed via `pydantic-settings`.

### Required (Production)

| Variable           | Description                                  | Default                                 |
| ------------------ | -------------------------------------------- | --------------------------------------- |
| `ENV`              | Execution environment (`development`, `production`) | `development`                           |
| `API_KEY_INTERNAL` | API key required to access REST endpoints.   | `sk_dev_super_secret_999` (development) |

### Optional Configuration

| Variable             | Description                                     | Default                                 |
| -------------------- | ----------------------------------------------- | --------------------------------------- |
| `MODEL_NAME`         | HuggingFace transformer model to utilize        | `distilbert-base-uncased`               |
| `MAX_INPUT_LENGTH`   | Maximum token sequence length                   | `512`                                   |
| `CONFIDENCE_THRESH`  | Threshold for classification flagging           | `0.85`                                  |

## Available Scripts

| Command                               | Description                                         |
| ------------------------------------- | --------------------------------------------------- |
| `slopguard predict "<text>"`          | Run local inference on the provided text payload.   |
| `slopguard train`                     | Initiate LoRA adapter training on the dataset.      |
| `python -m src.slopguard.api.main`    | Start the FastAPI microservice on port 8000.        |
| `pytest tests/`                       | Run the unit and integration test suite.            |
| `pre-commit run --all-files`          | Execute static analysis (Ruff, Mypy, Bandit).       |
| `mkdocs serve`                        | Serve the documentation site locally.               |
| `mkdocs build`                        | Build the static documentation site for deployment. |

## Testing

The project maintains a strict >80% test coverage requirement, verified automatically in CI.

### Running Tests

```bash
# Run all tests with coverage reporting
pytest tests/ --cov=slopguard --cov-report=term-missing

# Run a specific test module
pytest tests/test_core.py
```

### Test Architecture

- **`test_core.py`**: Validates the inference engine, model lazy-loading, text sanitization, and hardware fallback handling.
- **`test_api.py`**: Integration tests using `FastAPI.testclient` to ensure route logic, authentication, and error handling function correctly.
- **`test_cli.py` & `test_predict.py`**: Verifies CLI argument parsing and visual output formatting via mocked system arguments.
- **`test_train.py`**: Ensures the LoRA training configuration and dataset tokenization map correctly without requiring massive GPU cycles.

*Note: The test suite heavily utilizes `unittest.mock` to intercept large transformer model instantiation, allowing the suite to run completely in seconds on any machine.*

## Deployment

### API Deployment (Docker)

If you wish to deploy the API in a containerized production environment, you can use a standard Python Dockerfile:

```bash
# Build the image
docker build -t slopguard-api .

# Run the container
docker run -p 8000:8000 -e ENV="production" -e API_KEY_INTERNAL="your_secure_key" slopguard-api
```

### CI/CD Pipeline

The repository includes an enterprise-grade GitHub Actions pipeline (`.github/workflows/main.yml`) that runs on every Push and Pull Request. It includes:
1. Complete dependency caching.
2. Strict static analysis (Ruff, Mypy, Bandit).
3. Test suite execution ensuring >80% code coverage.
4. Container vulnerability scanning via Trivy.

## Troubleshooting

### HTTP 401 Unauthorized

**Error:** `{"detail": "Unauthorized. Invalid or missing API Key."}`

**Solution:**
When the `ENV` variable is set to `production`, you must provide the `X-API-KEY` header matching your `API_KEY_INTERNAL` configuration in every request.

### Missing Dataset

**Error:** `TRAINING_HALTED: Dataset missing at data/training_dataset.csv`

**Solution:**
The `slopguard train` command expects a CSV file containing `text` and `label` (0 or 1) columns located at `data/training_dataset.csv`. Ensure you have hydrated the data directory before initiating training.

### Hardware Acceleration Issues

**Error:** `UserWarning: MPS available but not used.` or sluggish inference.

**Solution:**
Ensure you are using the correct PyTorch binary for your system architecture.
- For Apple Silicon: Ensure you installed PyTorch natively via Conda or pip matching your ARM architecture.
- For NVIDIA: Ensure CUDA toolkits are installed and `torch.cuda.is_available()` returns `True`.
The engine will gracefully fallback to CPU if acceleration libraries are misconfigured, which will increase latency.
