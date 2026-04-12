"""
VisionaryAI Customer Support API with Security Features
Step 2: Serve the model through secure FastAPI
"""

from fastapi import FastAPI, HTTPException, Depends, Security, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from pydantic import BaseModel, validator, Field
import joblib
import json
import logging
import hashlib
import re
from datetime import datetime
import time
import os
from typing import Optional, List
import secrets

# Configure logging (don't log sensitive data)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Security configuration
VALID_API_KEYS = {
    "visionaryai-factory-001": {"name": "Factory Plant 001", "tier": "premium"},
    "visionaryai-support-team": {"name": "Internal Support", "tier": "internal"},
    "visionaryai-demo-key": {"name": "Demo Access", "tier": "basic"},
}

# Rate limiting setup
limiter = Limiter(key_func=get_remote_address)
app = FastAPI(
    title="VisionaryAI Customer Support Classifier",
    description="Secure AI-powered customer support ticket classification",
    version="1.0.0",
)

# Add rate limiting middleware
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# CORS middleware (restrict in production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://visionaryai.com"],  # Restrict to known domains
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# Security dependencies
security = HTTPBearer()


def verify_api_key(
    credentials: HTTPAuthorizationCredentials = Security(security),
) -> dict:
    """Verify API key and return user info"""
    api_key = credentials.credentials

    if api_key not in VALID_API_KEYS:
        logger.warning(f"Invalid API key attempted: {api_key[:8]}...")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API key"
        )

    user_info = VALID_API_KEYS[api_key]
    logger.info(f"Authenticated user: {user_info['name']}")
    return user_info


def sanitize_text(text: str) -> str:
    """Sanitize input text to prevent injection attacks"""
    if not text:
        return ""

    # Remove potential script injection patterns
    text = re.sub(r"<[^>]*>", "", text)  # Remove HTML tags
    text = re.sub(r"javascript:", "", text, flags=re.IGNORECASE)
    text = re.sub(r"on\w+\s*=", "", text, flags=re.IGNORECASE)  # Remove event handlers

    # Limit length and normalize
    text = text.strip()[:1000]  # Enforce max length
    return text


# Global variables for model and metadata
model = None
metadata = None


def load_model():
    """Load the trained model and metadata securely"""
    global model, metadata

    try:
        model_path = "models/support_classifier.joblib"
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")

        model = joblib.load(model_path)
        logger.info("✅ Model loaded successfully")

        with open("models/model_metadata.json", "r") as f:
            metadata = json.load(f)
        logger.info("✅ Model metadata loaded")

    except Exception as e:
        logger.error(f"❌ Error loading model: {str(e)}")
        raise e


@app.on_event("startup")
async def startup_event():
    load_model()


# Enhanced request/response models with security validation
class TicketRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=1000)
    customer_id: Optional[str] = Field(None, max_length=50)
    priority: str = Field("normal", pattern="^(low|normal|high|urgent)$")

    @validator("text")
    def validate_and_sanitize_text(cls, v):
        if not v or not v.strip():
            raise ValueError("Ticket text cannot be empty")

        # Sanitize the input
        sanitized = sanitize_text(v)
        if not sanitized:
            raise ValueError("Ticket text contains only invalid characters")

        return sanitized

    @validator("customer_id")
    def validate_customer_id(cls, v):
        if v is not None:
            # Simple validation - alphanumeric and dash only
            if not re.match(r"^[a-zA-Z0-9\-_]+", v):
                raise ValueError("Invalid customer ID format")
        return v


class PredictionResponse(BaseModel):
    category: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    categories_probabilities: dict
    metadata: dict


# API Endpoints with security
@app.get("/")
@limiter.limit("10/minute")
async def root(request, user: dict = Depends(verify_api_key)):
    """Health check endpoint - requires authentication"""
    return {
        "message": "VisionaryAI Customer Support Classifier API",
        "status": "healthy",
        "model_loaded": model is not None,
        "user": user["name"],
    }


@app.get("/model/info")
@limiter.limit("5/minute")
async def model_info(request, user: dict = Depends(verify_api_key)):
    """Get model information - requires authentication"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    # Don't expose sensitive model details to basic tier users
    if user["tier"] == "basic":
        return {
            "model_name": metadata["model_name"],
            "version": metadata["version"],
            "status": "ready",
        }

    return {"model_metadata": metadata, "status": "ready", "user_tier": user["tier"]}


@app.post("/classify", response_model=PredictionResponse)
@limiter.limit("100/minute")  # Rate limiting per IP
async def classify_ticket(
    request, ticket_request: TicketRequest, user: dict = Depends(verify_api_key)
):
    """Classify a customer support ticket with security measures"""

    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    start_time = time.time()

    try:
        # Additional input validation
        if len(ticket_request.text.split()) < 2:
            raise HTTPException(
                status_code=400,
                detail="Ticket text too short - minimum 2 words required",
            )

        # Make prediction
        prediction = model.predict([ticket_request.text])[0]
        probabilities = model.predict_proba([ticket_request.text])[0]
        confidence = max(probabilities)

        # Get all category probabilities
        categories = model.classes_
        category_probs = {
            cat: float(prob) for cat, prob in zip(categories, probabilities)
        }

        processing_time = (time.time() - start_time) * 1000

        # Create anonymized log entry (don't log full text for privacy)
        text_hash = hashlib.sha256(ticket_request.text.encode()).hexdigest()[:16]
        logger.info(
            f"Classification - Hash: {text_hash}, Category: {prediction}, "
            f"Confidence: {confidence:.3f}, User: {user['name']}"
        )

        response = PredictionResponse(
            category=prediction,
            confidence=float(confidence),
            categories_probabilities=category_probs,
            metadata={
                "model_version": metadata["version"],
                "processing_time_ms": round(processing_time, 2),
                "timestamp": datetime.now().isoformat(),
                "user_tier": user["tier"],
                # Don't include customer_id in response for privacy
                "request_id": secrets.token_hex(8),  # Unique request ID for tracking
            },
        )

        return response

    except HTTPException:
        raise  # Re-raise HTTP exceptions as-is
    except Exception as e:
        # Don't expose internal errors to users
        logger.error(f"❌ Classification error: {str(e)}")
        raise HTTPException(
            status_code=500, detail="Internal classification error. Please try again."
        )


@app.post("/classify/batch")
@limiter.limit("10/minute")  # Lower limit for batch operations
async def classify_batch(
    request, requests_list: List[TicketRequest], user: dict = Depends(verify_api_key)
):
    """Classify multiple tickets - requires premium tier"""

    # Check user permissions
    if user["tier"] == "basic":
        raise HTTPException(
            status_code=403, detail="Batch classification requires premium access"
        )

    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    # Enforce batch size limits based on tier
    max_batch_size = 100 if user["tier"] == "internal" else 50
    if len(requests_list) > max_batch_size:
        raise HTTPException(
            status_code=400,
            detail=f"Batch size too large (max {max_batch_size} for {user['tier']} tier)",
        )

    start_time = time.time()
    results = []

    try:
        texts = [req.text for req in requests_list]
        predictions = model.predict(texts)
        probabilities = model.predict_proba(texts)

        for i, (req, pred, probs) in enumerate(
            zip(requests_list, predictions, probabilities)
        ):
            confidence = max(probs)
            categories = model.classes_
            category_probs = {cat: float(prob) for cat, prob in zip(categories, probs)}

            results.append(
                {
                    "request_index": i,
                    "category": pred,
                    "confidence": float(confidence),
                    "categories_probabilities": category_probs,
                    "request_id": secrets.token_hex(8),
                }
            )

        processing_time = (time.time() - start_time) * 1000

        logger.info(
            f"Batch classification - Size: {len(requests_list)}, "
            f"User: {user['name']}, Time: {processing_time:.2f}ms"
        )

        return {
            "results": results,
            "metadata": {
                "model_version": metadata["version"],
                "batch_size": len(requests_list),
                "total_processing_time_ms": round(processing_time, 2),
                "timestamp": datetime.now().isoformat(),
                "user_tier": user["tier"],
            },
        }

    except Exception as e:
        logger.error(f"❌ Batch classification error: {str(e)}")
        raise HTTPException(
            status_code=500, detail="Batch classification error. Please try again."
        )


# Security monitoring endpoint
@app.get("/security/stats")
@limiter.limit("1/minute")
async def security_stats(request, user: dict = Depends(verify_api_key)):
    """Security statistics - internal access only"""

    if user["tier"] != "internal":
        raise HTTPException(status_code=403, detail="Access denied")

    # In production, this would query actual security metrics
    return {
        "rate_limit_violations_last_hour": 5,
        "invalid_api_key_attempts_last_hour": 12,
        "suspicious_input_patterns_detected": 2,
        "total_requests_last_hour": 1247,
        "timestamp": datetime.now().isoformat(),
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
