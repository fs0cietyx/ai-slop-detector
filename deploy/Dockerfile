# Multi-stage build for production-grade image
FROM python:3.11-slim as builder

WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --user --no-cache-dir -r requirements.txt

# Stage 2: Production Runtime
FROM python:3.11-slim

WORKDIR /app
COPY --from=builder /root/.local /root/.local
COPY . .

RUN chmod +x src/predict.py

# Standard security practice: Avoid running as root
RUN useradd -m appuser && chown -R appuser /app
USER appuser

ENV PATH=/root/.local/bin:$PATH
ENV PYTHONPATH=/app

LABEL maintainer="Mainak Biswas"
LABEL version="1.0"
LABEL description="Enterprise AI-generated text classifier"

ENTRYPOINT ["python", "-m", "src.predict"]
