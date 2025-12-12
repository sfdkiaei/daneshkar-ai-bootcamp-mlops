# Session 2 Code Demo: ML Project Structure & Git Workflow

**Objective:** Transform a messy ML project into a professional, version-controlled structure

**Project:** VisionaryAI's Phone Defect Detection System

---

## Part 1: The Chaotic Starting Point

### Initial Messy Structure

**Point:** "Let's start with what we often see in real projects - complete chaos!"

```bash
# Create the messy project structure
mkdir chaotic_ml_project
cd chaotic_ml_project

# Create the messy files that students can relate to
touch notebook1.ipynb
touch notebook_copy.ipynb
touch final_model_REALLY_FINAL.ipynb
touch data.csv
touch model.pkl
touch test_stuff.py
touch utils.py
```

### Messy Notebook Content

**File: `final_model_REALLY_FINAL.ipynb`**

```python
# Cell 1 - Data loading with hardcoded paths
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Hardcoded path - only works on Sarah's machine!
data = pd.read_csv('/Users/sarah/Desktop/phone_images/defect_data.csv')

# Cell 5 - Model training (scattered across multiple cells)
from sklearn.ensemble import RandomForestClassifier

# No random seed - results not reproducible
model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)

# Cell 12 - Saving model (buried deep in notebook)
import pickle
pickle.dump(model, 'model.pkl')  # Which model? What parameters?

# Cell 15 - Different model attempt
# Wait, let me try a different approach...
from sklearn.svm import SVC
svm_model = SVC()
# This overwrites our previous results!
```

**Points:**
- "Notice the hardcoded paths - only works on one machine"
- "No random seeds - results change every run"  
- "Model saving is buried in the notebook"
- "No clear indication of which approach worked best"
- "New team member would be completely lost"

---

## Part 2: Building Professional Structure

### Step 1: Create Professional Directory Structure

**Point:** "Let's transform this chaos into something a professional team can work with"

```bash
# Start fresh with professional structure
mkdir phone_defect_detection
cd phone_defect_detection

# Create the professional structure
mkdir -p config data/{raw,interim,processed} src/{data,features,models} models reports notebooks tests

# Create essential files
touch README.md requirements.txt .gitignore
touch config/data_config.yaml config/model_config.yaml
```

### Step 2: Configuration Files

**File: `config/model_config.yaml`**

```yaml
# Model configuration - no more hardcoded parameters!
model:
  type: 'random_forest'
  n_estimators: 100
  random_state: 42
  max_depth: 10

training:
  test_size: 0.2
  validation_size: 0.1
  random_state: 42

paths:
  raw_data: 'data/raw/defect_data.csv'
  processed_data: 'data/processed/features.csv'  
  model_output: 'models/defect_detector.pkl'
```

**File: `config/data_config.yaml`**

```yaml
# Data processing configuration
preprocessing:
  image_size: [224, 224]
  normalize: true
  augmentation: true

features:
  extract_texture: true
  extract_color: true
  extract_shape: true

quality_checks:
  min_samples_per_class: 100
  max_missing_values: 0.05
```

**Points:**
- "Configuration files make experiments reproducible"
- "Easy to try different parameters without changing code"
- "Clear documentation of what parameters were used"

### Step 3: Organized Source Code

**File: `src/data/data_loader.py`**

```python
"""
Data loading and basic preprocessing for defect detection.
VisionaryAI - Phone Defect Detection System
"""

import pandas as pd
import yaml
from pathlib import Path

def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def load_raw_data(config: dict) -> pd.DataFrame:
    """
    Load raw defect data from configured path.
    
    Args:
        config: Configuration dictionary with data paths
        
    Returns:
        DataFrame with raw defect data
    """
    data_path = config['paths']['raw_data']
    
    # Use pathlib for cross-platform compatibility
    data_path = Path(data_path)
    
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")
    
    return pd.read_csv(data_path)

def validate_data(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """
    Validate data quality based on configuration rules.
    
    Args:
        df: Input DataFrame
        config: Configuration with quality check parameters
        
    Returns:
        Validated DataFrame
    """
    quality_config = config['quality_checks']
    
    # Check missing values
    missing_ratio = df.isnull().sum().sum() / (df.shape[0] * df.shape[1])
    max_missing = quality_config['max_missing_values']
    
    if missing_ratio > max_missing:
        raise ValueError(f"Too many missing values: {missing_ratio:.2%} > {max_missing:.2%}")
    
    print(f"✅ Data validation passed. Missing values: {missing_ratio:.2%}")
    return df

if __name__ == "__main__":
    # Example usage
    config = load_config('config/data_config.yaml')
    data = load_raw_data(config)
    validated_data = validate_data(data, config)
    print(f"Loaded {len(validated_data)} samples successfully")
```

**Points:**
- "Single responsibility - this module only handles data loading"
- "Configuration-driven - no hardcoded paths"
- "Error handling and validation built-in"
- "Docstrings make the code self-documenting"
- "Can be run independently for testing"

**File: `src/models/defect_classifier.py`**

```python
"""
Defect classification model for phone manufacturing.
VisionaryAI - Phone Defect Detection System
"""

import pickle
import yaml
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import pandas as pd

class DefectClassifier:
    """Defect detection classifier with configuration management."""
    
    def __init__(self, config_path: str):
        """Initialize classifier with configuration."""
        self.config = self._load_config(config_path)
        self.model = None
        self.is_trained = False
    
    def _load_config(self, config_path: str) -> dict:
        """Load model configuration."""
        with open(config_path, 'r') as file:
            return yaml.safe_load(file)
    
    def train(self, X: pd.DataFrame, y: pd.Series):
        """
        Train the defect classification model.
        
        Args:
            X: Feature matrix
            y: Target labels
        """
        model_config = self.config['model']
        training_config = self.config['training']
        
        # Split data with configured random state
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=training_config['test_size'],
            random_state=training_config['random_state']
        )
        
        # Initialize model with configured parameters
        self.model = RandomForestClassifier(
            n_estimators=model_config['n_estimators'],
            random_state=model_config['random_state'],
            max_depth=model_config['max_depth']
        )
        
        # Train model
        print("🚀 Training defect classifier...")
        self.model.fit(X_train, y_train)
        
        # Evaluate on test set
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"✅ Training completed. Test accuracy: {accuracy:.3f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        self.is_trained = True
    
    def save_model(self):
        """Save trained model to configured path."""
        if not self.is_trained:
            raise ValueError("Model must be trained before saving")
        
        model_path = Path(self.config['paths']['model_output'])
        model_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(model_path, 'wb') as file:
            pickle.dump(self.model, file)
        
        print(f"💾 Model saved to: {model_path}")
    
    def load_model(self):
        """Load saved model from configured path."""
        model_path = Path(self.config['paths']['model_output'])
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        with open(model_path, 'rb') as file:
            self.model = pickle.load(file)
        
        self.is_trained = True
        print(f"📁 Model loaded from: {model_path}")

if __name__ == "__main__":
    # Example usage
    classifier = DefectClassifier('config/model_config.yaml')
    
    # In a real scenario, you'd load your actual features here
    print("Defect classifier initialized and ready for training!")
```

**Points:**
- "Class-based design makes it easy to reuse and test"
- "Configuration is loaded once and used throughout"
- "Clear separation between training, saving, and loading"
- "Built-in evaluation and reporting"

### Step 4: Professional README

**File: `README.md`**

```markdown
    # Phone Defect Detection System

    VisionaryAI's automated defect detection system for mobile phone manufacturing.

    ## Project Overview

    This system uses computer vision and machine learning to automatically detect defects in mobile phone production lines, reducing manual inspection time by 80% while maintaining 99.5% accuracy.

    ## Quick Start

    ```bash
    # 1. Clone and setup
    git clone <repository-url>
    cd phone_defect_detection
    pip install -r requirements.txt

    # 2. Prepare data
    python src/data/data_loader.py

    # 3. Train model  
    python src/models/defect_classifier.py

    # 4. Run predictions
    python src/models/predict.py --input data/new_images/
    ```

    ## Project Structure

    ```
    phone_defect_detection/
    ├── config/              # Configuration files
    ├── data/               
    │   ├── raw/            # Original data (never modified)
    │   ├── interim/        # Intermediate processing results  
    │   └── processed/      # Final features for modeling
    ├── src/                # Source code
    │   ├── data/          # Data loading and preprocessing
    │   ├── features/      # Feature engineering
    │   └── models/        # Model training and prediction
    ├── models/            # Trained model artifacts
    ├── notebooks/         # Jupyter notebooks (exploration only)
    └── reports/           # Analysis and evaluation reports
    ```

    ## Configuration

    All parameters are managed through YAML files in the `config/` directory:

    - `model_config.yaml`: Model architecture and training parameters
    - `data_config.yaml`: Data processing and quality settings

    ## Development Workflow

    1. **Exploration**: Use notebooks in `notebooks/` for initial exploration
    2. **Development**: Implement production code in `src/`
    3. **Testing**: Run tests with `pytest tests/`
    4. **Training**: Execute training pipeline with configured parameters

    ## Team Members

    - Dr. Sarah Chen - Lead ML Engineer
    - Alex Rodriguez - Computer Vision Specialist  
    - Maya Patel - MLOps Engineer

    ## Contact

    For questions about this system, contact the VisionaryAI MLOps team at mlops@visionaryai.com
```

**Points:**
- "README tells the complete story of your project"
- "Anyone can understand and get started quickly"
- "Clear contact information for collaboration"
- "Explains the business value, not just technical details"

---

## Part 3: Git Workflow for ML Projects

### Initialize Git and Set Up ML-Specific Workflow

```bash
# Initialize git repository
git init

# Create ML-specific .gitignore
cat > .gitignore << EOF
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
venv/

# Jupyter Notebook
.ipynb_checkpoints

# ML specific
*.pkl
*.joblib
*.h5
*.pb

# Data files (too large for git)
data/raw/*.csv
data/raw/*.jpg
data/raw/*.png
data/interim/*.parquet
data/processed/*.npy

# Model artifacts (too large for git)
models/*.pkl
models/*.joblib
models/*.h5

# Experiment tracking
mlruns/
.mlflow/

# IDE
.vscode/
.idea/

# OS
.DS_Store
Thumbs.db
EOF
```

**Point:** "Notice we exclude model files and large data files - these need different versioning strategies"

### Create Initial Commit

```bash
# Add all code and configuration
git add .
git commit -m "Initial project structure

- Professional ML project organization
- Configuration-driven development  
- Separation of data/features/models
- Comprehensive documentation"
```

### ML Branching Workflow

```bash
# Create experiment branch for trying a new model architecture
git checkout -b experiment/svm-classifier

# Modify model configuration for experiment
cat > config/model_config.yaml << EOF
# Experiment: SVM classifier
model:
  type: 'svm'
  kernel: 'rbf'
  C: 1.0
  random_state: 42

training:
  test_size: 0.2
  validation_size: 0.1
  random_state: 42

paths:
  raw_data: 'data/raw/defect_data.csv'
  processed_data: 'data/processed/features.csv'  
  model_output: 'models/svm_defect_detector.pkl'
EOF

# Commit experiment
git add config/model_config.yaml
git commit -m "Experiment: Try SVM classifier

- Changed model type from RandomForest to SVM
- Updated output path to avoid conflicts
- Hypothesis: SVM might work better for high-dimensional image features"
```

**Points:**
- "Experiment branches capture the 'what' and 'why' of experiments"
- "Configuration changes are tracked just like code changes"
- "Commit messages explain the hypothesis being tested"

### Collaborative Workflow

```bash
# Switch back to main for a data update
git checkout main

# Create data pipeline update branch
git checkout -b data/add-validation-set

# Show how data changes are handled
mkdir -p data/raw/validation
echo "# Validation data from new factory line" > data/raw/validation/README.md

# Update data config
cat >> config/data_config.yaml << EOF

# New validation data
validation:
  path: 'data/raw/validation/'
  required_samples: 1000
EOF

git add .
git commit -m "Add validation dataset from Factory Line B

- New validation data from different production line
- Helps test model generalization across factories  
- Required for production deployment approval"
```

**Points:**
- "Data changes need careful coordination across team"
- "Clear documentation of what data changed and why"
- "Separate branches for data vs model experiments"

---

## Part 4: Handling Large Files

### The Large File Problem

```bash
# Simulate large files that can't go in Git
echo "# This represents a 2GB training dataset" > data/raw/large_dataset.csv
echo "# This represents a 500MB trained model" > models/production_model.pkl

# Show what happens when we try to add them
git add data/raw/large_dataset.csv
# Git will accept this, but it shouldn't!
```

### DVC Concept

**Point:** "For large files, we need specialized tools like DVC (Data Version Control)"

```bash
# This is conceptual - showing what DVC workflow looks like
# In real project, you'd install DVC first: pip install dvc

# DVC tracks large files separately from Git
dvc add data/raw/large_dataset.csv
# This creates: data/raw/large_dataset.csv.dvc (small metadata file)

# The .dvc file goes in Git, actual data goes to remote storage
git add data/raw/large_dataset.csv.dvc
git commit -m "Add training dataset v1.0

- 50,000 defect images from Factory Line A
- Balanced dataset: 25k defective, 25k normal
- Images preprocessed to 224x224 resolution"
```

### Version Control Summary

**Create summary file to demonstrate the complete workflow:**

**File: `VERSIONING_STRATEGY.md`**

```markdown
# VisionaryAI Versioning Strategy

## What Goes Where

### Git (Code Repository)
✅ **Include:**
- Python source code (.py files)
- Configuration files (.yaml, .json)  
- Documentation (.md files)
- Small sample datasets (< 10MB)
- Notebooks for exploration
- Test files
- Requirements and environment files

❌ **Exclude:**
- Large datasets (> 10MB)
- Trained models (> 10MB)
- Binary artifacts
- Cache files
- Personal IDE settings

### DVC (Data & Model Versioning)
✅ **Track with DVC:**
- Training datasets
- Validation datasets  
- Trained model files
- Feature engineering outputs
- Large intermediate files

### MLflow (Experiment Tracking)
✅ **Track with MLflow:**
- Model performance metrics
- Hyperparameters used
- Training duration
- Model artifacts (small)
- Experiment comparisons

## Branch Strategy

- `main` - Production-ready code
- `experiment/*` - Model and algorithm experiments
- `data/*` - Data pipeline changes
- `feature/*` - New functionality
- `bugfix/*` - Production bug fixes

## Collaboration Rules

1. **Data changes** - Coordinate with team before merging
2. **Experiment branches** - Can be deleted after learning is captured
3. **Model artifacts** - Use DVC, not Git
4. **Commit messages** - Explain the business hypothesis, not just technical changes
```

**Points:**
- "Different types of changes need different versioning strategies"
- "Clear rules prevent conflicts and confusion"
- "The goal is collaboration, not just storage"

---

## Wrap-Up & Key Takeaways

### What We Just Built

**Point:** "We transformed chaos into a professional ML system that:"

✅ **Anyone can understand and navigate**
- Clear directory structure
- Comprehensive documentation  
- Self-explanatory configuration

✅ **Supports team collaboration**
- Git workflow for different types of changes
- Clear branching strategy
- Proper handling of large files

✅ **Enables reproducible experiments**
- Configuration-driven development
- Documented parameters and decisions
- Version controlled methodology

✅ **Scales from research to production**
- Modular code organization
- Separation of concerns
- Professional development practices

### The Transformation

**Before:** 
- `defect_detection_final_v3_REALLY_FINAL.ipynb`
- Hardcoded paths and parameters
- No version control strategy
- Individual work only

**After:**
- Professional project structure
- Configuration-driven development
- ML-specific Git workflow  
- Team collaboration ready

**Key Message:** "The extra setup time pays for itself in the first week of collaboration or when you need to reproduce results!"
