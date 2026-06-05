# SECURITY.md - DevSecOps & AppSec Architecture

## Overview
This project adheres to a **Zero-Trust Security Model**. Every component is designed to assume a hostile environment, enforcing strict boundaries between data, execution, and external APIs.

## Enforced Security Pillars

### 1. Secrets Management
- **Zero Exposure**: API keys and credentials are never hardcoded. 
- **Environment Isolation**: Strictly use `.env` files (excluded from VCE via `.gitignore`).
- **Placeholder Protection**: The repository contains only placeholder templates for secrets.

### 2. Input Validation & DoS Prevention
- **Size Boundaries**: Inference (`predict.py`) enforces a `MAX_INPUT_CHARS` limit (5000) to prevent memory-exhaustion attacks.
- **Type Checking**: All functions processing user-supplied or external data (Crawler, Generator) implement strict type and content verification.

### 3. Anti-Abuse & Bot Mitigation
- **Crawler Guardrails**: 
    - `Content-Type` verification to prevent processing of non-HTML blobs.
    - `MAX_SIZE` limits (1MB) per page to mitigate resource amplification attacks.
    - Explicit `User-Agent` and citizen-friendly rate limiting (0.5s - 1.0s delay).
- **AI Rate Limiting**: `gemini_generator.py` implements a semaphore-based concurrency limit (5) and a request-per-minute (15) delay.

### 4. Prompt Integrity
- **Sanitization**: AI prompts are sanitized to remove control characters and prevent prompt injection vectors during the two-pass generation phase.

### 5. Observability & Secure Deployment
- **Sanitized Logging**: Errors are logged without leaking PII or internal system state.
- **VENV Isolation**: The project is designed to run in a dedicated virtual environment to prevent dependency poisoning of the host system.

## Vulnerability Reporting
If you identify a security flaw, please do not open a public issue. Submit a detailed report via the established security contact (e.g., security@example.com).
