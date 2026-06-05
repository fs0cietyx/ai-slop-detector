import time
from typing import Any, Awaitable, Callable, Dict, Optional, Union, cast

from fastapi import FastAPI, Header, HTTPException, Request, Response, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from ..core.config import config, logger
from ..core.engine import InferenceEngine

# --- API Security Infrastructure ---
limiter = Limiter(key_func=get_remote_address)
app = FastAPI(
    title=config.APP_NAME,
    version=config.APP_VERSION,
    description="Enterprise-grade AI-generated text detection gateway.",
)

# [AppSec] Rate Limiting: Handle abuse at the middleware level
app.state.limiter = limiter
app.add_exception_handler(
    RateLimitExceeded, 
    cast(Callable[[Request, Exception], Union[Response, Awaitable[Response]]], _rate_limit_exceeded_handler)
)

# [AppSec] CORS Hardening: In production, this should be restricted to specific origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# Singleton Engine Instance
engine = InferenceEngine()


# --- Pydantic Schemas (Pillar IV: Defensive Programming) ---
class DetectionRequest(BaseModel):
    """Secure request schema for text analysis."""
    text: str = Field(..., min_length=10, max_length=config.MAX_INPUT_CHARS)

    model_config = {
        "json_schema_extra": {
            "example": {"text": "This is a sample text that could be human-written or AI-generated."}
        }
    }


class DetectionResponse(BaseModel):
    """Standardized enterprise response schema."""
    label: str
    confidence: float
    status: str = "SUCCESS"
    audit_id: int = Field(default_factory=lambda: int(time.time()))
    latency_ms: float


class HealthResponse(BaseModel):
    """System health telemetry schema."""
    status: str
    version: str
    engine_online: bool


# --- Middleware: Security Audit ---
@app.middleware("http")
async def audit_middleware(request: Request, call_next: Callable[[Request], Awaitable[Response]]) -> Response:
    """
    Structured logging and security telemetry for every request.
    """
    start_time = time.time()
    response = await call_next(request)
    process_time = (time.time() - start_time) * 1000
    
    logger.info(
        f"API_AUDIT: method={request.method} path={request.url.path} "
        f"status={response.status_code} latency={process_time:.2f}ms"
    )
    return response


# --- Endpoints ---
@app.get("/health", response_model=HealthResponse)
async def health_check() -> Dict[str, Any]:
    """System health check and engine verification."""
    return {
        "status": "healthy",
        "version": config.APP_VERSION,
        "engine_online": hasattr(engine, "_initialized") and engine._initialized,
    }


@app.post(
    "/v1/predict", 
    response_model=DetectionResponse, 
    status_code=status.HTTP_200_OK
)
@limiter.limit(config.RATE_LIMIT_DEFAULT)
async def predict_text(
    request: Request,
    payload: DetectionRequest,
    x_api_key: Optional[str] = Header(None, alias="X-API-KEY")
) -> DetectionResponse:
    """
    Secure inference endpoint for AI slop detection.
    
    Adheres to Pillar II (Resource Discipline) and Pillar V (Abuse Prevention).
    """
    # [AppSec] Zero-Trust Auth: Simple header check (can be upgraded to JWT/OAuth)
    if config.ENV == "production" and x_api_key != config.API_KEY_INTERNAL.get_secret_value():
        logger.warning(f"UNAUTHORIZED_ACCESS_ATTEMPT: IP={get_remote_address(request)}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key.",
        )

    start_time = time.time()
    
    try:
        # [Optimization] Offload heavy compute to the engine
        label, confidence = engine.predict(payload.text)
        
        latency = (time.time() - start_time) * 1000
        
        return DetectionResponse(
            label=label,
            confidence=confidence,
            latency_ms=round(latency, 2)
        )

    except Exception as e:
        logger.error(f"API_INFERENCE_FAILURE: {str(e)}")
        # [AppSec] Safe Error Handling: Generic message for client
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An internal error occurred during analysis.",
        ) from e


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=config.HOST, port=config.PORT)
