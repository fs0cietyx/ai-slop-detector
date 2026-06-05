# --- Stage 1: Builder ---
FROM python:3.11-slim as builder

# [AppSec] Disable pip cache and prevent pyc generation
ENV PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /build

# [Security] Build tools for some python dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --user --no-warn-script-location -r requirements.txt

# --- Stage 2: Production ---
FROM python:3.11-slim as runtime

LABEL maintainer="AI Slop Detector Team" \
      org.opencontainers.image.title="AI Slop Detector" \
      org.opencontainers.image.description="Enterprise-grade AI-generated text detection suite"

# [AppSec] Rootless Execution: Create a non-privileged user
RUN groupadd -r slopbot && useradd -r -g slopbot -s /sbin/nologin slopbot

WORKDIR /app

# Copy python packages from builder
COPY --from=builder /root/.local /home/slopbot/.local
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages

# [AppSec] Infrastructure Isolation: Copy only necessary source code
COPY src/ src/
COPY models/ models/
COPY .env.example .env

# [Security] Enforce ownership for the non-root user
RUN chown -R slopbot:slopbot /app && \
    mkdir -p /app/data && chown -R slopbot:slopbot /app/data

# [AppSec] Environment Variables: Standardized defaults
ENV PATH="/home/slopbot/.local/bin:${PATH}" \
    PYTHONPATH="/app" \
    APP_ENV="production" \
    PORT=8000

USER slopbot

# [Security] Expose unprivileged port
EXPOSE 8000

# Healthcheck for container orchestration
HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

ENTRYPOINT ["uvicorn", "src.api.main:app"]
CMD ["--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
