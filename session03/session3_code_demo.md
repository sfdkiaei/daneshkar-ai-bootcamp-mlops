# Session 3 Code Demo: Environment Management & Dependencies

## Demo Overview & Learning Objectives

### What you will learn:
1. How to create and manage virtual environments (venv and conda)
2. Practical differences between requirements.txt and conda environments
3. Basic Docker container creation for ML applications
4. Configuration management techniques

### Demo Structure:
- **Part 1:** Virtual Environments Hands-On
- **Part 2:** Requirements vs Conda Comparison
- **Part 3:** Basic Dockerfile Creation
- **Part 4:** Configuration Management Demo

---

## Part 1: Virtual Environments Hands-On

### Setup: VisionaryAI Computer Vision Project

**Demo Script:**

```bash
# Scenario: Setting up VisionaryAI's computer vision defect detection project
# We need TensorFlow, OpenCV, and some specific versions

# 1. First, let's see what happens WITHOUT virtual environments
python -c "import tensorflow as tf; print(f'TensorFlow: {tf.__version__}')"
python -c "import cv2; print(f'OpenCV: {cv2.__version__}')"

# Problem: We're using whatever versions are installed system-wide
# What if we need specific versions for our project?
```

### Creating Python venv Environment

```bash
# 2. Create a virtual environment for our computer vision project
python -m venv visionaryai_cv_env

# 3. Activate the environment (Linux/Mac)
source visionaryai_cv_env/bin/activate

# For Windows users (show both):
# visionaryai_cv_env\Scripts\activate

# 4. Notice the prompt change - you're now in the virtual environment
# Install specific versions our project needs
pip install tensorflow==2.8.0
pip install opencv-python==4.5.5.64
pip install numpy==1.21.6

# 5. Verify we're using the right versions
python -c "import tensorflow as tf; print(f'TensorFlow: {tf.__version__}')"
python -c "import cv2; print(f'OpenCV: {cv2.__version__}')"
python -c "import numpy as np; print(f'NumPy: {np.__version__}')"

# 6. See what's installed in this environment
pip list

# 7. Create requirements file for sharing
pip freeze > requirements.txt
cat requirements.txt
```

### Creating Conda Environment

```bash
# 8. Deactivate current environment first
deactivate

# 9. Create conda environment for NLP helpdesk agent project
conda create -n visionaryai_nlp python=3.9 -y

# 10. Activate conda environment
conda activate visionaryai_nlp

# 11. Install ML packages using conda (better dependency resolution)
conda install -c conda-forge transformers=4.20.0 -y
conda install pytorch=1.13.0 cpuonly -c pytorch -y
conda install pandas=1.4.4 -y

# 12. Verify installation
python -c "import transformers; print(f'Transformers: {transformers.__version__}')"
python -c "import torch; print(f'PyTorch: {torch.__version__}')"

# 13. Export conda environment
conda env export > environment.yml
cat environment.yml
```

**Points:**
- "Notice how the prompt changes when you activate environments"
- "See how pip freeze shows EXACT versions with hashes"
- "Conda handles complex dependencies better than pip"
- "Always activate before installing packages"

---

## Part 2: Requirements.txt vs Conda Environment Files

### Side-by-Side Comparison

**Create demo files to see the differences:**

```bash
# Create requirements.txt (from venv environment)
echo "# VisionaryAI Computer Vision Requirements
tensorflow==2.8.0
opencv-python==4.5.5.64
numpy==1.21.6
matplotlib==3.5.3
pillow==9.2.0" > cv_requirements.txt
```

```yaml
# Create environment.yml (from conda environment)
cat > nlp_environment.yml << EOF
name: visionaryai_nlp
channels:
  - conda-forge
  - pytorch
dependencies:
  - python=3.9
  - transformers=4.20.0
  - pytorch=1.13.0
  - pandas=1.4.4
  - numpy=1.21.0
  - pip
  - pip:
    - datasets==2.3.2
    - wandb==0.13.1
EOF
```

### Recreation

```bash
# Recreate from requirements.txt
python -m venv test_cv_env
source test_cv_env/bin/activate
pip install -r cv_requirements.txt

# Recreate from conda environment.yml
conda env create -f nlp_environment.yml
conda activate visionaryai_nlp
```

**Points:**
- "Requirements.txt is simpler but only handles Python packages"
- "Conda environments can include system dependencies and multiple languages"
- "Both approaches ensure reproducible environments"
- "Choose based on your project complexity"

---

## Part 3: Basic Dockerfile Creation

### VisionaryAI Recommender System Container

**Points:**
- "Containers go beyond virtual environments"
- "We're packaging the entire system, not just Python packages"
- "Same container runs anywhere Docker is installed"

```dockerfile
# Create Dockerfile for VisionaryAI's recommender system
cat > Dockerfile << EOF
# Start with Python base image
FROM python:3.9-slim

# Set working directory inside container
WORKDIR /app

# Install system dependencies (this is what venv can't do!)
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (Docker layer caching optimization)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Set environment variables
ENV PYTHONPATH=/app
ENV FLASK_ENV=production

# Expose port for the API
EXPOSE 5000

# Command to run the application
CMD ["python", "app.py"]
EOF
```

### Create Sample Application

```python
# Create simple Flask app (app.py)
cat > app.py << EOF
from flask import Flask, jsonify
import os
import numpy as np
from datetime import datetime

app = Flask(__name__)

@app.route('/health')
def health_check():
    return jsonify({
        'status': 'healthy',
        'service': 'VisionaryAI Recommender',
        'timestamp': datetime.now().isoformat(),
        'environment': os.getenv('FLASK_ENV', 'development')
    })

@app.route('/recommend/<user_id>')
def recommend(user_id):
    # Simulate ML recommendation logic
    recommendations = np.random.choice(['phone_case', 'charger', 'screen_protector'], 3)
    return jsonify({
        'user_id': user_id,
        'recommendations': recommendations.tolist(),
        'model_version': '1.2.3'
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
EOF
```

```bash
# Create requirements for the container
cat > requirements.txt << EOF
flask==2.2.2
numpy==1.21.6
scikit-learn==1.1.1
pandas==1.4.4
EOF
```

### Build and Run Container

```bash
# Build the Docker image
docker build -t visionaryai-recommender:v1.0 .

# Run the container
docker run -p 5000:5000 -e FLASK_ENV=production visionaryai-recommender:v1.0

# Test the API (in another terminal)
curl http://localhost:5000/health
curl http://localhost:5000/recommend/user123
```

**Points:**
- "FROM specifies base image - we start with Python pre-installed"
- "WORKDIR sets the working directory inside container"
- "RUN executes commands during build time"
- "COPY moves files from host to container"
- "ENV sets environment variables"
- "CMD specifies what runs when container starts"
- "This container will run identically on any machine with Docker"

---

## Part 4: Configuration Management

### Environment-Specific Configurations

**Points:**
- "Same code, different behavior through configuration"
- "Never hardcode environment-specific values"
- "Configuration should be externalized and secure"

```python
# Create config.py
cat > config.py << EOF
import os
from dataclasses import dataclass

@dataclass
class Config:
    # Database settings
    database_url: str = os.getenv('DATABASE_URL', 'sqlite:///local.db')
    
    # ML Model settings
    model_path: str = os.getenv('MODEL_PATH', './models/recommender_v1.pkl')
    batch_size: int = int(os.getenv('BATCH_SIZE', '32'))
    
    # API settings
    api_key: str = os.getenv('API_KEY', 'dev-key-123')
    max_workers: int = int(os.getenv('MAX_WORKERS', '2'))
    log_level: str = os.getenv('LOG_LEVEL', 'DEBUG')
    
    # Feature flags
    enable_caching: bool = os.getenv('ENABLE_CACHING', 'false').lower() == 'true'
    enable_monitoring: bool = os.getenv('ENABLE_MONITORING', 'false').lower() == 'true'

# Usage in application
config = Config()
print(f"Running with config: {config}")
EOF
```

### Environment Files

```bash
# Development environment
cat > .env.development << EOF
DATABASE_URL=sqlite:///dev.db
MODEL_PATH=./models/dev_model.pkl
BATCH_SIZE=16
API_KEY=dev-key-123
MAX_WORKERS=1
LOG_LEVEL=DEBUG
ENABLE_CACHING=false
ENABLE_MONITORING=false
EOF

# Production environment
cat > .env.production << EOF
DATABASE_URL=postgresql://prod.cluster.com:5432/recommendations
MODEL_PATH=/app/models/production_model.pkl
BATCH_SIZE=64
API_KEY=prod-key-456
MAX_WORKERS=8
LOG_LEVEL=INFO
ENABLE_CACHING=true
ENABLE_MONITORING=true
EOF
```

### Configuration Loading

```python
# Updated app.py with configuration
cat > app_with_config.py << EOF
from flask import Flask, jsonify
from config import Config
import os

# Load configuration
config = Config()
app = Flask(__name__)

@app.route('/config')
def get_config():
    # Return non-sensitive configuration
    return jsonify({
        'database_type': 'sqlite' if 'sqlite' in config.database_url else 'postgresql',
        'batch_size': config.batch_size,
        'max_workers': config.max_workers,
        'log_level': config.log_level,
        'caching_enabled': config.enable_caching,
        'monitoring_enabled': config.enable_monitoring
    })

if __name__ == '__main__':
    print(f"Starting with configuration: {config}")
    app.run(host='0.0.0.0', port=5000, debug=(config.log_level == 'DEBUG'))
EOF
```

### Test Different Configurations

```bash
# Test development configuration
export $(cat .env.development | xargs)
python app_with_config.py &
curl http://localhost:5000/config

# Kill the process
pkill -f app_with_config.py

# Test production configuration
export $(cat .env.production | xargs)
python app_with_config.py &
curl http://localhost:5000/config
```

**Points:**
- "Same Python code, completely different behavior"
- "Configuration loaded from environment variables"
- "Sensitive values (API keys) not hardcoded"
- "Easy to test different configurations"
- "Production secrets would come from secure secret management systems"

---

## Wrap-Up & Key Takeaways

### Key Demonstrations Completed:
1. ✅ Created virtual environments (venv and conda)
2. ✅ Compared requirements.txt vs environment.yml
3. ✅ Built and ran a Docker container
4. ✅ Implemented configuration management

### Takeaways:
- "Virtual environments isolate Python dependencies"
- "Containers isolate entire system environment"
- "Configuration enables same code to work across environments"
- "These tools work together to solve 'it works on my machine' problems"
