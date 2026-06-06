# ⚡ THE APEX PROTOCOL: AGENT MAINTENANCE GATEWAY

## [AI_AGENT_HANDSHAKE]
This codebase has been hardened under **The Apex Protocol**. It is a zero-trust, enterprise-grade ML suite. Any future agent modifying this repository MUST adhere to the following mandates to maintain architectural integrity and security.

### 1. SEC_MANDATE: ZERO-TRUST ENVIRONMENT
- **Secrets:** Never log or commit plain-text keys. All sensitive variables must be handled via `pydantic_settings.BaseSettings` and wrapped in `SecretStr`.
- **Inference Security:** Every input payload MUST pass through the `_sanitize_payload` engine in `src/core/engine.py`. Neutralize control characters and enforce strict byte-size truncation before tokenization.
- **SSRF Prevention:** The `AsyncCrawler` in `src/crawler.py` must maintain strict domain boundary enforcement.

### 2. ARCH_MANDATE: SINGLETON INFERENCE
- **Resource Discipline:** The `InferenceEngine` is a thread-safe Singleton. DO NOT instantiate multiple engines. Heavy ML weights must be pinned in memory exactly once to prevent OOM crashes on consumer-grade hardware.
- **Hardware Autonomy:** Maintain the `_select_device()` logic. The engine must remain silicon-agnostic, supporting MPS (Apple), CUDA (NVIDIA), and CPU fallback.

### 3. DEV_MANDATE: 100% STATIC TYPING
- **Strict Mode:** The project is configured for **Mypy Strict**. Zero `Any` types allowed. Every function must have precise type annotations for arguments and return values.
- **Validation:** All API request/response cycles must use Pydantic V2 schemas defined in `src/api/main.py`.

### 4. TEST_MANDATE: RIGOROUS VALIDATION
- **CI/CD Compliance:** Any new feature must pass the `APEX_PROTOCOL_CI` (Bandit, Ruff, Mypy, Pytest).
- **Adversarial Testing:** Unit tests must include weaponized payload neutralization checks (null bytes, malformed Unicode).

## [DEPLOYMENT_CONTEXT]
- **Container:** Multi-stage, rootless Docker build (`deploy/Dockerfile`).
- **Asset Management:** ML weights (models/) are mounted as volumes, never baked into the image.

---
**Certified by:** Gemini CLI (The Apex Protocol v1.0)
