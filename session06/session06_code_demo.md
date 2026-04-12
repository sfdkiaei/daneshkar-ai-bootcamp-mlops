# Session 6: Code Demo - Model Packaging & Deployment

## Demo Overview

**Scenario:** VisionaryAI Customer Support Ticket Classification
**Progression:** Basic serialization → FastAPI service → Docker containerization
**Teaching Goal:** Show practical implementation of packaging and deployment concepts

---

## Demo Lesson Plan

### Phase 1: Model Serialization
**Points:**
- Different serialization formats and their use cases
- Importance of saving preprocessing along with models
- Model metadata and version tracking
- **Security consideration:** Secure model storage and access control

### Phase 2: FastAPI Service with Security
**Points:**
- Creating a simple ML API with built-in security features
- Input validation and sanitization (preventing injection attacks)
- Authentication via API keys
- Rate limiting to prevent abuse
- Error handling that doesn't leak sensitive information
- **Security focus:** Demonstrate multiple security layers

### Phase 3: Secure Docker Containerization
**Points:**
- Why containerization matters for ML security
- Creating a secure Dockerfile (non-root user, minimal base image)
- Environment variable management for secrets
- Container security best practices

### Phase 4: Security Testing & Validation
**Points:**
- Testing authentication and authorization
- Validating rate limiting and input sanitization
- Security monitoring basics
- **Security emphasis:** How to test your security measures

---

## Code Examples

### 1. Model Training and Serialization (`train_and_save.py`)

```python
"""
VisionaryAI Customer Support Ticket Classifier
Step 1: Train and serialize the model
"""
import joblib
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
import json
from datetime import datetime

# Sample training data (in reality, this would be much larger)
training_data = [
    ("My phone screen is cracked", "hardware_issue"),
    ("Cannot connect to wifi", "connectivity"),
    ("App keeps crashing", "software_issue"),
    ("Battery drains too fast", "hardware_issue"),
    ("Cannot log into my account", "account_issue"),
    ("Website is loading slowly", "connectivity"),
    ("Feature request for dark mode", "feature_request"),
    ("Billing question about my subscription", "billing"),
    ("How do I reset my password", "account_issue"),
    ("Phone gets very hot during use", "hardware_issue"),
]

# Separate features and labels
texts = [item[0] for item in training_data]
labels = [item[1] for item in training_data]

print("🤖 Training VisionaryAI Customer Support Classifier...")

# Create a pipeline with preprocessing and model
# This ensures preprocessing is saved with the model
classifier = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=1000, stop_words='english')),
    ('nb', MultinomialNB())
])

# Train the model
classifier.fit(texts, labels)

# Create model metadata
model_metadata = {
    "model_name": "customer_support_classifier",
    "version": "1.0.0",
    "training_date": datetime.now().isoformat(),
    "accuracy": "0.89",  # In reality, calculated from validation set
    "features": "TF-IDF vectors (max 1000 features)",
    "algorithm": "Multinomial Naive Bayes",
    "categories": list(set(labels))
}

print(f"✅ Model trained! Categories: {model_metadata['categories']}")

# Method 1: Save with joblib (recommended for sklearn)
print("\n💾 Saving model with joblib...")
joblib.dump(classifier, 'models/support_classifier.joblib')
print("✅ Saved as support_classifier.joblib")

# Method 2: Save with pickle (alternative)
print("\n💾 Saving model with pickle...")
with open('models/support_classifier.pkl', 'wb') as f:
    pickle.dump(classifier, f)
print("✅ Saved as support_classifier.pkl")

# Save metadata separately
print("\n📋 Saving model metadata...")
with open('models/model_metadata.json', 'w') as f:
    json.dump(model_metadata, f, indent=2)
print("✅ Metadata saved")

# Test the saved model
print("\n🧪 Testing saved model...")
loaded_model = joblib.load('models/support_classifier.joblib')
test_text = "My phone screen has a crack on it"
prediction = loaded_model.predict([test_text])[0]
confidence = max(loaded_model.predict_proba([test_text])[0])

print(f"Test input: '{test_text}'")
print(f"Prediction: {prediction}")
print(f"Confidence: {confidence:.3f}")
print("\n🎉 Model serialization complete!")
```

### 2. FastAPI Service with Security Features (`app.py`)

```python
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
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Security configuration
VALID_API_KEYS = {
    "visionaryai-factory-001": {"name": "Factory Plant 001", "tier": "premium"},
    "visionaryai-support-team": {"name": "Internal Support", "tier": "internal"},
    "visionaryai-demo-key": {"name": "Demo Access", "tier": "basic"}
}

# Rate limiting setup
limiter = Limiter(key_func=get_remote_address)
app = FastAPI(
    title="VisionaryAI Customer Support Classifier",
    description="Secure AI-powered customer support ticket classification",
    version="1.0.0"
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

def verify_api_key(credentials: HTTPAuthorizationCredentials = Security(security)) -> dict:
    """Verify API key and return user info"""
    api_key = credentials.credentials
    
    if api_key not in VALID_API_KEYS:
        logger.warning(f"Invalid API key attempted: {api_key[:8]}...")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key"
        )
    
    user_info = VALID_API_KEYS[api_key]
    logger.info(f"Authenticated user: {user_info['name']}")
    return user_info

def sanitize_text(text: str) -> str:
    """Sanitize input text to prevent injection attacks"""
    if not text:
        return ""
    
    # Remove potential script injection patterns
    text = re.sub(r'<[^>]*>', '', text)  # Remove HTML tags
    text = re.sub(r'javascript:', '', text, flags=re.IGNORECASE)
    text = re.sub(r'on\w+\s*=', '', text, flags=re.IGNORECASE)  # Remove event handlers
    
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
    priority: str = Field("normal", regex="^(low|normal|high|urgent)$")
    
    @validator('text')
    def validate_and_sanitize_text(cls, v):
        if not v or not v.strip():
            raise ValueError('Ticket text cannot be empty')
        
        # Sanitize the input
        sanitized = sanitize_text(v)
        if not sanitized:
            raise ValueError('Ticket text contains only invalid characters')
        
        return sanitized
    
    @validator('customer_id')
    def validate_customer_id(cls, v):
        if v is not None:
            # Simple validation - alphanumeric and dash only
            if not re.match(r'^[a-zA-Z0-9\-_]+', v):
                raise ValueError('Invalid customer ID format')
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
        "user": user["name"]
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
            "status": "ready"
        }
    
    return {
        "model_metadata": metadata,
        "status": "ready",
        "user_tier": user["tier"]
    }

@app.post("/classify", response_model=PredictionResponse)
@limiter.limit("100/minute")  # Rate limiting per IP
async def classify_ticket(
    request, 
    ticket_request: TicketRequest,
    user: dict = Depends(verify_api_key)
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
                detail="Ticket text too short - minimum 2 words required"
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
        logger.info(f"Classification - Hash: {text_hash}, Category: {prediction}, "
                   f"Confidence: {confidence:.3f}, User: {user['name']}")
        
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
                "request_id": secrets.token_hex(8)  # Unique request ID for tracking
            }
        )
        
        return response
        
    except HTTPException:
        raise  # Re-raise HTTP exceptions as-is
    except Exception as e:
        # Don't expose internal errors to users
        logger.error(f"❌ Classification error: {str(e)}")
        raise HTTPException(
            status_code=500, 
            detail="Internal classification error. Please try again."
        )

@app.post("/classify/batch")
@limiter.limit("10/minute")  # Lower limit for batch operations
async def classify_batch(
    request,
    requests_list: List[TicketRequest],
    user: dict = Depends(verify_api_key)
):
    """Classify multiple tickets - requires premium tier"""
    
    # Check user permissions
    if user["tier"] == "basic":
        raise HTTPException(
            status_code=403, 
            detail="Batch classification requires premium access"
        )
    
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    # Enforce batch size limits based on tier
    max_batch_size = 100 if user["tier"] == "internal" else 50
    if len(requests_list) > max_batch_size:
        raise HTTPException(
            status_code=400, 
            detail=f"Batch size too large (max {max_batch_size} for {user['tier']} tier)"
        )
    
    start_time = time.time()
    results = []
    
    try:
        texts = [req.text for req in requests_list]
        predictions = model.predict(texts)
        probabilities = model.predict_proba(texts)
        
        for i, (req, pred, probs) in enumerate(zip(requests_list, predictions, probabilities)):
            confidence = max(probs)
            categories = model.classes_
            category_probs = {
                cat: float(prob) for cat, prob in zip(categories, probs)
            }
            
            results.append({
                "request_index": i,
                "category": pred,
                "confidence": float(confidence),
                "categories_probabilities": category_probs,
                "request_id": secrets.token_hex(8)
            })
        
        processing_time = (time.time() - start_time) * 1000
        
        logger.info(f"Batch classification - Size: {len(requests_list)}, "
                   f"User: {user['name']}, Time: {processing_time:.2f}ms")
        
        return {
            "results": results,
            "metadata": {
                "model_version": metadata["version"],
                "batch_size": len(requests_list),
                "total_processing_time_ms": round(processing_time, 2),
                "timestamp": datetime.now().isoformat(),
                "user_tier": user["tier"]
            }
        }
        
    except Exception as e:
        logger.error(f"❌ Batch classification error: {str(e)}")
        raise HTTPException(
            status_code=500, 
            detail="Batch classification error. Please try again."
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
        "timestamp": datetime.now().isoformat()
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

### 3. Dockerfile

```dockerfile
# VisionaryAI Customer Support Classifier
# Step 3: Containerize the application

# Use Python 3.9 slim image for smaller size
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (for better caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app.py .
COPY train_and_save.py .

# Create models directory
RUN mkdir -p models

# Copy pre-trained model files
COPY models/ models/

# Create non-root user for security
RUN useradd --create-home --shell /bin/bash app
RUN chown -R app:app /app
USER app

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/ || exit 1

# Run the application
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
```

### 4. Requirements File (`requirements.txt`)

```txt
# Core FastAPI and ML dependencies
fastapi==0.104.1
uvicorn[standard]==0.24.0
scikit-learn==1.3.2
joblib==1.3.2
pydantic==2.5.0
python-multipart==0.0.6
numpy==1.24.3

# Security enhancements
slowapi==0.1.9  # Rate limiting
python-jose[cryptography]==3.3.0  # JWT handling
passlib[bcrypt]==1.7.4  # Password hashing
httpx==0.25.2  # Async HTTP client for testing

# Additional utilities
python-dotenv==1.0.0  # Environment variable management
requests==2.33.1
```

### 5. Docker Compose for Easy Deployment (`docker-compose.yml`)

```yaml
version: '3.8'

services:
  visionaryai-classifier:
    build: .
    ports:
      - "8000:8000"
    environment:
      - LOG_LEVEL=info
    volumes:
      # Mount models directory for development
      - ./models:/app/models
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s
```

### 6. Security Test Script (`test_api_security.py`)

```python
"""
VisionaryAI API Security Testing Script
Step 4: Test the deployed service with security features
"""
import requests
import json
import time
import threading
from concurrent.futures import ThreadPoolExecutor

# API configuration
BASE_URL = "http://localhost:8000"
VALID_API_KEY = "visionaryai-demo-key"
INVALID_API_KEY = "invalid-key-123"

# Headers for authenticated requests
AUTH_HEADERS = {"Authorization": f"Bearer {VALID_API_KEY}"}
INVALID_AUTH_HEADERS = {"Authorization": f"Bearer {INVALID_API_KEY}"}

def test_authentication():
    """Test API key authentication"""
    print("🔐 Testing Authentication...")
    
    # Test without API key
    print("\n--- Test 1: No API Key ---")
    response = requests.get(f"{BASE_URL}/")
    print(f"Status: {response.status_code} (Expected: 401/403)")
    
    # Test with invalid API key
    print("\n--- Test 2: Invalid API Key ---")
    response = requests.get(f"{BASE_URL}/", headers=INVALID_AUTH_HEADERS)
    print(f"Status: {response.status_code} (Expected: 401)")
    if response.status_code == 401:
        print("✅ Invalid API key correctly rejected")
    
    # Test with valid API key
    print("\n--- Test 3: Valid API Key ---")
    response = requests.get(f"{BASE_URL}/", headers=AUTH_HEADERS)
    print(f"Status: {response.status_code} (Expected: 200)")
    if response.status_code == 200:
        result = response.json()
        print(f"✅ Authenticated as: {result.get('user', 'Unknown')}")
    
    return response.status_code == 200

def test_input_validation():
    """Test input sanitization and validation"""
    print("\n🛡️ Testing Input Validation...")
    
    test_cases = [
        {
            "name": "Empty text",
            "data": {"text": "", "customer_id": "test_001"},
            "expected_status": 422
        },
        {
            "name": "Too long text", 
            "data": {"text": "x" * 1001, "customer_id": "test_002"},
            "expected_status": 422
        },
        {
            "name": "HTML injection attempt",
            "data": {"text": "Help me <script>alert('xss')</script>", "customer_id": "test_003"},
            "expected_status": 200  # Should be sanitized and processed
        },
        {
            "name": "Invalid customer ID format",
            "data": {"text": "Valid ticket text", "customer_id": "invalid@customer.id"},
            "expected_status": 422
        },
        {
            "name": "Invalid priority",
            "data": {"text": "Valid ticket text", "priority": "super_urgent"},
            "expected_status": 422
        }
    ]
    
    for test_case in test_cases:
        print(f"\n--- {test_case['name']} ---")
        response = requests.post(
            f"{BASE_URL}/classify", 
            json=test_case["data"],
            headers=AUTH_HEADERS
        )
        print(f"Status: {response.status_code} (Expected: {test_case['expected_status']})")
        
        if response.status_code == test_case["expected_status"]:
            print("✅ Input validation working correctly")
        else:
            print(f"❌ Unexpected response: {response.text[:200]}")

def test_rate_limiting():
    """Test rate limiting functionality"""
    print("\n⏱️ Testing Rate Limiting...")
    
    def make_request(request_id):
        """Make a single request"""
        try:
            response = requests.get(f"{BASE_URL}/", headers=AUTH_HEADERS, timeout=5)
            return {"id": request_id, "status": response.status_code}
        except Exception as e:
            return {"id": request_id, "error": str(e)}
    
    # Make rapid requests to trigger rate limiting
    print("Making 15 rapid requests (limit: 10/minute)...")
    
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(make_request, i) for i in range(15)]
        results = [future.result() for future in futures]
    
    # Count responses
    success_count = len([r for r in results if r.get("status") == 200])
    rate_limited = len([r for r in results if r.get("status") == 429])
    
    print(f"Successful requests: {success_count}")
    print(f"Rate limited (429): {rate_limited}")
    
    if rate_limited > 0:
        print("✅ Rate limiting is working")
    else:
        print("⚠️ Rate limiting may not be configured correctly")

def test_authorization_tiers():
    """Test different user tier access levels"""
    print("\n👥 Testing Authorization Tiers...")
    
    # Test model info access with basic tier
    print("\n--- Basic Tier Model Info ---")
    response = requests.get(f"{BASE_URL}/model/info", headers=AUTH_HEADERS)
    if response.status_code == 200:
        info = response.json()
        # Basic tier should get limited info
        if "model_metadata" not in info or len(info) <= 3:
            print("✅ Basic tier receives limited model info")
        else:
            print("⚠️ Basic tier might be getting too much info")
    
    # Test batch processing (should be limited for basic tier)
    print("\n--- Basic Tier Batch Access ---")
    batch_data = [
        {"text": "Test ticket 1"},
        {"text": "Test ticket 2"}
    ]
    response = requests.post(
        f"{BASE_URL}/classify/batch", 
        json=batch_data,
        headers=AUTH_HEADERS
    )
    
    if response.status_code == 403:
        print("✅ Basic tier correctly denied batch access")
    elif response.status_code == 200:
        print("⚠️ Basic tier has batch access (check if intended)")
    else:
        print(f"Unexpected status: {response.status_code}")

def test_security_monitoring():
    """Test security monitoring endpoint"""
    print("\n📊 Testing Security Monitoring...")
    
    response = requests.get(f"{BASE_URL}/security/stats", headers=AUTH_HEADERS)
    
    if response.status_code == 403:
        print("✅ Security stats correctly restricted to internal users")
    elif response.status_code == 200:
        print("⚠️ Security stats accessible to basic tier (potential security issue)")
    else:
        print(f"Unexpected status: {response.status_code}")

def test_error_handling():
    """Test that errors don't leak sensitive information"""
    print("\n🚨 Testing Secure Error Handling...")
    
    # Test with malformed JSON
    print("\n--- Malformed JSON ---")
    response = requests.post(
        f"{BASE_URL}/classify",
        data="invalid json",
        headers=AUTH_HEADERS
    )
    
    if response.status_code == 422:  # Unprocessable Entity
        error_detail = response.json().get("detail", "")
        # Should not expose internal paths or sensitive info
        if "models/" not in str(error_detail) and "traceback" not in str(error_detail).lower():
            print("✅ Error messages don't leak sensitive information")
        else:
            print("⚠️ Error messages might be exposing too much information")

def test_logging_privacy():
    """Verify that sensitive data isn't logged in plain text"""
    print("\n📝 Testing Privacy in Logging...")
    
    # Make a request with sensitive-looking data
    sensitive_data = {
        "text": "My credit card number is 1234-5678-9012-3456 and my SSN is 123-45-6789",
        "customer_id": "sensitive_customer_123"
    }
    
    response = requests.post(
        f"{BASE_URL}/classify",
        json=sensitive_data,
        headers=AUTH_HEADERS
    )
    
    if response.status_code == 200:
        result = response.json()
        # Check that sensitive data isn't echoed back in response
        response_text = str(result)
        if "1234-5678" not in response_text and "123-45-6789" not in response_text:
            print("✅ Sensitive data not echoed in response")
        else:
            print("⚠️ Sensitive data found in response")
    
    print("Note: Check server logs manually to ensure sensitive data isn't logged in plain text")

def main():
    """Run all security tests"""
    print("🚀 Starting VisionaryAI Security Tests...")
    print("=" * 60)
    
    try:
        # Wait for service
        print("⏳ Waiting for service to start...")
        time.sleep(3)
        
        # Run security tests
        auth_ok = test_authentication()
        if not auth_ok:
            print("❌ Authentication failed, skipping further tests")
            return
        
        test_input_validation()
        test_rate_limiting()
        test_authorization_tiers() 
        test_security_monitoring()
        test_error_handling()
        test_logging_privacy()
        
        print("\n" + "=" * 60)
        print("🎉 Security testing completed!")
        print("📋 Review any warnings above and check server logs for privacy compliance")
        
    except requests.exceptions.ConnectionError:
        print("❌ Could not connect to API. Is the service running?")
    except Exception as e:
        print(f"❌ Test error: {str(e)}")

if __name__ == "__main__":
    main()
```

---

## Demo Execution Commands

### Setup and Training
```bash
# Create directory structure
mkdir -p models
mkdir -p tests

# Train and save the model
python train_and_save.py
```

### Run FastAPI Locally
```bash
# Install dependencies
pip install -r requirements.txt

# Run the API
uvicorn app:app --reload --host 0.0.0.0 --port 8000

# Test in another terminal
python test_api_security.py
```

### Docker Build and Run
```bash
# Build the Docker image
docker build -t visionaryai-classifier .

# Run the container
docker run -p 8000:8000 visionaryai-classifier

# Or use docker-compose
docker-compose up --build
```

### Testing the Containerized Service
```bash
# Test the container
curl http://localhost:8000/

# Run full test suite
python test_api_security.py
```

---

## Key Points During Demo

### During Phase 1 (Serialization):
- **Point out:** How we save preprocessing with the model (Pipeline)
- **Explain:** Why metadata is important for tracking and security audits
- **Show:** Difference between joblib and pickle file sizes and security implications
- **Emphasize:** Testing loaded models before deployment
- **Security note:** Secure storage and access control for model files

### During Phase 2 (FastAPI with Security):
- **Highlight:** Multi-layered security approach (authentication, rate limiting, input validation)
- **Show:** How API keys work and user tier management
- **Explain:** Input sanitization prevents injection attacks
- **Demonstrate:** Rate limiting protection against abuse
- **Point out:** Privacy-preserving logging (hashing sensitive data)
- **Show:** Interactive API docs at `/docs` with security requirements

### During Phase 3 (Docker):
- **Explain:** Why containers solve "works on my machine" problems securely
- **Show:** Security-focused Dockerfile (non-root user, minimal base image)
- **Point out:** Environment variable management for secrets
- **Mention:** Container vulnerability scanning in production
- **Emphasize:** Principle of least privilege

### During Phase 4 (Security Testing):
- **Emphasize:** Security testing is as important as functional testing
- **Show:** How to test authentication, authorization, and input validation
- **Discuss:** Rate limiting verification and monitoring
- **Demonstrate:** Privacy protection in logging and responses
- **Mention:** Continuous security monitoring in production

---

## Questions

1. What happens if we forget to save the preprocessing pipeline?
2. Why might we choose joblib over pickle for this model?
3. What could go wrong if we don't validate API inputs?
4. How would you handle model updates without downtime?
5. What metrics would you monitor in production?
6. Why do we use API keys instead of just allowing open access?
7. What security risks exist if we don't implement rate limiting?
8. How does input sanitization protect against attacks?
9. Why shouldn't we log sensitive customer data in plain text?
10. What's the difference between authentication and authorization in our API?

---

## Extended Learning Resources

### For Students Who Want More:
- Complete working code with enhanced error handling and security
- Production-ready logging configuration with privacy protection
- Model performance monitoring examples with security metrics
- Kubernetes deployment manifests with security policies
- Load testing scripts with security stress testing
- Model A/B testing implementation with secure user segmentation
- Advanced authentication systems (OAuth 2.0, JWT tokens)
- Container security scanning and vulnerability management
- Comprehensive security monitoring and incident response procedures
- Compliance frameworks (GDPR, CCPA) implementation examples
