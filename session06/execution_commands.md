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