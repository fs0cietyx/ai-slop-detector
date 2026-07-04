# Contributing to AI Slop Detector

We welcome contributions to the AI Slop Detector from internal engineers and the open-source community. This document outlines the enterprise standards required for submitting code.

## Code of Conduct

All contributors are expected to maintain professional communication. Harassment, unprofessional language, and dismissive remarks are strictly prohibited.

## Development Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-org/ai-slop-detector.git
   cd ai-slop-detector
   ```

2. **Install dependencies:**
   We use `hatchling` and standard `pip` for dependency management.
   ```bash
   python -m pip install -e ".[dev]"
   ```

3. **Install Pre-commit Hooks:**
   ```bash
   pre-commit install
   ```

## Workflow

1. Create a feature branch originating from `main`.
2. Implement your changes.
3. Ensure all static analysis passes:
   ```bash
   pre-commit run --all-files
   ```
4. Run the test suite and ensure coverage remains above 80%:
   ```bash
   pytest tests/
   ```
5. Submit a Pull Request.

## Pull Request Requirements

- **Descriptive Title**: Use conventional commits (e.g., `feat:`, `fix:`, `docs:`).
- **Comprehensive Description**: Detail the problem solved and the approach taken.
- **Test Coverage**: All new functionality must include corresponding unit or integration tests.
- **Documentation**: Update the MkDocs documentation in the `docs/` directory if your changes affect public APIs or CLI usage.

## Architecture Guidelines

- Avoid adding arbitrary logic to `api/main.py` or `cli/main.py`. Place core logic inside the `core/` module.
- Strict type hinting is enforced by `mypy`. Ensure all functions are fully typed.
- Handle exceptions gracefully; no raw stack traces should be exposed to end-users via the CLI or API.
