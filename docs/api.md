# API Integration

The AI Slop Detector provides a FastAPI-powered REST service designed for low-latency, high-throughput microservice architectures.

## Starting the Server

The server runs on Uvicorn and utilizes standard Python execution.

```bash
python -m src.slopguard.api.main
```
The API listens on `http://127.0.0.1:8000` by default.

## Authentication

All endpoints (except health checks) require an API key to be passed via the `X-API-KEY` HTTP header.

By default, the required key in production is `sk_dev_super_secret_999`.

## Endpoints

### Health Check

`GET /health`

Returns the operational status of the service and inference engine.

**Response**:
```json
{
  "status": "operational",
  "engine": "InferenceEngine",
  "version": "1.0.0"
}
```

### Predict

`POST /v1/predict`

Submit text for AI-generation analysis.

**Request Body** (`application/json`):
```json
{
  "text": "The string of text you wish to analyze. Must be at least 20 characters."
}
```

**Response**:
```json
{
  "label": "HUMAN-WRITTEN",
  "confidence": 0.985,
  "compute_time_ms": 12.4
}
```

**Status Codes**:
- `200 OK`: Successful inference.
- `401 Unauthorized`: Missing or invalid API key.
- `422 Unprocessable Entity`: Payload failed validation constraints.
- `429 Too Many Requests`: Rate limit exceeded (default 10 requests / minute / IP).
- `500 Internal Server Error`: Critical engine failure.
