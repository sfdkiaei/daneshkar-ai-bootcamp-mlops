# Session 1 Code Demo: From Messy to Organized ML Code
# VisionaryAI - Computer Vision Defect Detection Example

# =============================================================================
# PART 1: THE MESSY APPROACH (What NOT to do)
# =============================================================================

# messy_defect_detection.py (or more realistically, a Jupyter notebook cell)

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle
import cv2
import os

# Load data (hardcoded paths, no error handling)
data = pd.read_csv("/Users/john/Desktop/factory_data.csv")
images_path = "/Users/john/Desktop/images/"

# Quick data exploration (results not saved)
print(data.shape)
print(data.head())

# Data preprocessing (magic numbers everywhere)
data = data.dropna()
data = data[data["defect_score"] > 0.1]  # Why 0.1? Nobody knows!

# Feature extraction (no documentation)
features = []
labels = []
for idx, row in data.iterrows():
    img_path = images_path + row["image_filename"]
    if os.path.exists(img_path):
        img = cv2.imread(img_path)
        # Some magical feature extraction
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        hist = cv2.calcHist([gray], [0], None, [32], [0, 256])  # Why 32 bins?
        features.append(hist.flatten())
        labels.append(row["is_defective"])

# Convert to numpy (inefficient)
X = np.array(features)
y = np.array(labels)

# Split data (random seed? What's that?)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train model (hyperparameters chosen randomly)
model = RandomForestClassifier(n_estimators=50, max_depth=10)
model.fit(X_train, y_train)

# Evaluate (single metric, no context)
predictions = model.predict(X_test)
acc = accuracy_score(y_test, predictions)
print(f"Accuracy: {acc}")  # 0.87 - Is this good? Bad? We don't know!

# Save model (no versioning, overwrites previous models)
pickle.dump(model, open("defect_model.pkl", "wb"))

print("Done! Model saved.")

# =============================================================================
# PART 2: THE ORGANIZED APPROACH (MLOps Best Practices)
# =============================================================================

# 2.1: Project Structure
"""
visionaryai_defect_detection/
├── config/
│   ├── config.yaml
│   └── logging_config.yaml
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   └── data_loader.py
│   ├── features/
│   │   ├── __init__.py
│   │   └── feature_extractor.py
│   ├── models/
│   │   ├── __init__.py
│   │   └── defect_classifier.py
│   └── utils/
│       ├── __init__.py
│       └── logging_utils.py
├── experiments/
│   └── defect_detection_experiment.py
├── tests/
│   ├── test_data_loader.py
│   ├── test_feature_extractor.py
│   └── test_model.py
├── requirements.txt
├── setup.py
└── README.md
"""

# 2.2: Configuration Management (config/config.yaml)
"""
data:
  raw_data_path: "data/raw/factory_data.csv"
  images_path: "data/raw/images/"
  processed_data_path: "data/processed/"

preprocessing:
  min_defect_score: 0.1  # Threshold for filtering low-confidence samples
  image_size: [224, 224]  # Standardized image dimensions
  
features:
  histogram_bins: 32  # Number of bins for color histogram
  include_texture: true
  include_edge_features: true

model:
  name: "RandomForestClassifier"
  hyperparameters:
    n_estimators: 100
    max_depth: 15
    random_state: 42
    
training:
  test_size: 0.2
  validation_size: 0.2
  random_state: 42

paths:
  model_output: "models/"
  logs: "logs/"
  experiment_tracking: "experiments/mlruns/"
"""

# 2.3: Data Loading Module (src/data/data_loader.py)
import pandas as pd
import numpy as np
import cv2
import os
import logging
from typing import Tuple, List
from pathlib import Path


class DefectDataLoader:
    """
    Handles loading and basic validation of defect detection data.

    This class provides methods to load factory data and corresponding images,
    with proper error handling and logging.
    """

    def __init__(self, config: dict):
        self.config = config
        self.logger = logging.getLogger(__name__)

    def load_data(self) -> Tuple[pd.DataFrame, List[np.ndarray]]:
        """
        Load factory data and corresponding images.

        Returns:
            Tuple of (metadata_df, images_list)
        """
        try:
            # Load metadata
            data_path = Path(self.config["data"]["raw_data_path"])
            if not data_path.exists():
                raise FileNotFoundError(f"Data file not found: {data_path}")

            df = pd.read_csv(data_path)
            self.logger.info(f"Loaded {len(df)} records from {data_path}")

            # Validate data
            df_clean = self._validate_data(df)

            # Load images
            images = self._load_images(df_clean)

            return df_clean, images

        except Exception as e:
            self.logger.error(f"Error loading data: {str(e)}")
            raise

    def _validate_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply data validation and filtering."""
        initial_count = len(df)

        # Remove rows with missing critical columns
        required_columns = ["image_filename", "is_defective", "defect_score"]
        df_clean = df.dropna(subset=required_columns)

        # Apply defect score threshold
        min_score = self.config["preprocessing"]["min_defect_score"]
        df_clean = df_clean[df_clean["defect_score"] > min_score]

        final_count = len(df_clean)
        removed = initial_count - final_count
        self.logger.info(
            f"Data validation: removed {removed} records, kept {final_count}"
        )

        return df_clean

    def _load_images(self, df: pd.DataFrame) -> List[np.ndarray]:
        """Load and validate images."""
        images = []
        images_path = Path(self.config["data"]["images_path"])
        failed_loads = 0

        for filename in df["image_filename"]:
            img_path = images_path / filename
            try:
                if img_path.exists():
                    image = cv2.imread(str(img_path))
                    if image is not None:
                        images.append(image)
                    else:
                        failed_loads += 1
                        self.logger.warning(f"Could not read image: {img_path}")
                else:
                    failed_loads += 1
                    self.logger.warning(f"Image file not found: {img_path}")
            except Exception as e:
                failed_loads += 1
                self.logger.error(f"Error loading {img_path}: {str(e)}")

        self.logger.info(f"Loaded {len(images)} images, {failed_loads} failed")
        return images


# 2.4: Feature Extraction Module (src/features/feature_extractor.py)
import cv2
import numpy as np
from typing import List, Dict, Any
import logging


class DefectFeatureExtractor:
    """
    Extract features from factory images for defect detection.

    This class implements multiple feature extraction methods that can be
    combined for robust defect detection.
    """

    def __init__(self, config: dict):
        self.config = config
        self.logger = logging.getLogger(__name__)

    def extract_features(self, images: List[np.ndarray]) -> np.ndarray:
        """
        Extract features from a list of images.

        Args:
            images: List of OpenCV images (BGR format)

        Returns:
            Feature matrix of shape (n_images, n_features)
        """
        all_features = []

        for i, image in enumerate(images):
            try:
                features = self._extract_single_image_features(image)
                all_features.append(features)

                if (i + 1) % 100 == 0:
                    self.logger.info(f"Processed {i + 1}/{len(images)} images")

            except Exception as e:
                self.logger.error(f"Error processing image {i}: {str(e)}")
                # Use zero features for failed images
                feature_dim = self._get_feature_dimension()
                all_features.append(np.zeros(feature_dim))

        return np.array(all_features)

    def _extract_single_image_features(self, image: np.ndarray) -> np.ndarray:
        """Extract features from a single image."""
        features = []

        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Color histogram features
        hist_features = self._extract_histogram_features(gray)
        features.extend(hist_features)

        # Texture features (if enabled)
        if self.config["features"]["include_texture"]:
            texture_features = self._extract_texture_features(gray)
            features.extend(texture_features)

        # Edge features (if enabled)
        if self.config["features"]["include_edge_features"]:
            edge_features = self._extract_edge_features(gray)
            features.extend(edge_features)

        return np.array(features)

    def _extract_histogram_features(self, gray_image: np.ndarray) -> List[float]:
        """Extract histogram-based features."""
        bins = self.config["features"]["histogram_bins"]
        hist = cv2.calcHist([gray_image], [0], None, [bins], [0, 256])
        return hist.flatten().tolist()

    def _extract_texture_features(self, gray_image: np.ndarray) -> List[float]:
        """Extract texture-based features using Local Binary Patterns."""
        # Simplified texture features - in practice, you might use LBP or other methods
        # Standard deviation in local patches
        kernel_size = 5
        mean_filter = cv2.blur(gray_image, (kernel_size, kernel_size))
        variance = cv2.blur((gray_image - mean_filter) ** 2, (kernel_size, kernel_size))

        return [np.mean(variance), np.std(variance)]

    def _extract_edge_features(self, gray_image: np.ndarray) -> List[float]:
        """Extract edge-based features."""
        edges = cv2.Canny(gray_image, 50, 150)
        edge_density = np.sum(edges > 0) / edges.size

        return [edge_density]

    def _get_feature_dimension(self) -> int:
        """Calculate total feature dimension."""
        dim = self.config["features"]["histogram_bins"]  # Histogram features

        if self.config["features"]["include_texture"]:
            dim += 2  # Texture features

        if self.config["features"]["include_edge_features"]:
            dim += 1  # Edge features

        return dim


# 2.5: Model Training Module (src/models/defect_classifier.py)
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib
import logging
from pathlib import Path
from datetime import datetime
import json


class DefectClassifier:
    """
    Machine learning model for defect detection.

    This class handles model training, evaluation, and persistence with
    proper experiment tracking and model versioning.
    """

    def __init__(self, config: dict):
        self.config = config
        self.model = None
        self.logger = logging.getLogger(__name__)
        self.model_metadata = {}

    def train(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """
        Train the defect detection model.

        Args:
            X: Feature matrix
            y: Labels (1 for defective, 0 for normal)

        Returns:
            Dictionary of evaluation metrics
        """
        self.logger.info("Starting model training...")

        # Split data
        X_train, X_temp, y_train, y_temp = train_test_split(
            X,
            y,
            test_size=self.config["training"]["test_size"]
            + self.config["training"]["validation_size"],
            random_state=self.config["training"]["random_state"],
            stratify=y,
        )

        X_val, X_test, y_val, y_test = train_test_split(
            X_temp,
            y_temp,
            test_size=0.5,  # Split temp data equally between val and test
            random_state=self.config["training"]["random_state"],
            stratify=y_temp,
        )

        # Initialize model
        model_params = self.config["model"]["hyperparameters"]
        self.model = RandomForestClassifier(**model_params)

        # Train model
        self.model.fit(X_train, y_train)
        self.logger.info("Model training completed")

        # Evaluate on validation set
        metrics = self._evaluate_model(X_val, y_val, "validation")
        test_metrics = self._evaluate_model(X_test, y_test, "test")

        # Store metadata
        self.model_metadata = {
            "training_time": datetime.now().isoformat(),
            "model_type": self.config["model"]["name"],
            "hyperparameters": model_params,
            "training_samples": len(X_train),
            "validation_samples": len(X_val),
            "test_samples": len(X_test),
            "validation_metrics": metrics,
            "test_metrics": test_metrics,
        }

        return metrics

    def _evaluate_model(
        self, X: np.ndarray, y: np.ndarray, dataset_name: str
    ) -> Dict[str, float]:
        """Evaluate model performance on a dataset."""
        y_pred = self.model.predict(X)

        metrics = {
            "accuracy": accuracy_score(y, y_pred),
            "precision": precision_score(y, y_pred, average="weighted"),
            "recall": recall_score(y, y_pred, average="weighted"),
            "f1_score": f1_score(y, y_pred, average="weighted"),
        }

        self.logger.info(f"{dataset_name.capitalize()} metrics:")
        for metric, value in metrics.items():
            self.logger.info(f"  {metric}: {value:.4f}")

        return metrics

    def save_model(self, version: str = None) -> Path:
        """
        Save model with versioning and metadata.

        Args:
            version: Optional version string. If None, uses timestamp.

        Returns:
            Path where model was saved
        """
        if self.model is None:
            raise ValueError("No model to save. Train a model first.")

        # Create version string
        if version is None:
            version = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Create model directory
        model_dir = (
            Path(self.config["paths"]["model_output"]) / f"defect_model_v{version}"
        )
        model_dir.mkdir(parents=True, exist_ok=True)

        # Save model
        model_path = model_dir / "model.joblib"
        joblib.dump(self.model, model_path)

        # Save metadata
        metadata_path = model_dir / "metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(self.model_metadata, f, indent=2)

        # Save config
        config_path = model_dir / "config.json"
        with open(config_path, "w") as f:
            json.dump(self.config, f, indent=2)

        self.logger.info(f"Model saved to {model_dir}")
        return model_dir

    def load_model(self, model_path: Path):
        """Load a saved model."""
        model_file = model_path / "model.joblib"
        if not model_file.exists():
            raise FileNotFoundError(f"Model file not found: {model_file}")

        self.model = joblib.load(model_file)

        # Load metadata if available
        metadata_file = model_path / "metadata.json"
        if metadata_file.exists():
            with open(metadata_file, "r") as f:
                self.model_metadata = json.load(f)

        self.logger.info(f"Model loaded from {model_path}")


# 2.6: Main Experiment Script (experiments/defect_detection_experiment.py)
import yaml
import logging
from pathlib import Path
import sys

# Add src to path to import our modules
sys.path.append(str(Path(__file__).parent.parent / "src"))

from data.data_loader import DefectDataLoader
from features.feature_extractor import DefectFeatureExtractor
from models.defect_classifier import DefectClassifier


def setup_logging(config):
    """Setup logging configuration."""
    log_level = config.get("logging", {}).get("level", "INFO")
    logging.basicConfig(
        level=getattr(logging, log_level),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler("logs/defect_detection.log"),
            logging.StreamHandler(),
        ],
    )


def main():
    """Main experiment pipeline."""
    # Load configuration
    config_path = Path(__file__).parent.parent / "config" / "config.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Setup logging
    setup_logging(config)
    logger = logging.getLogger(__name__)

    logger.info("Starting defect detection experiment")

    try:
        # Load data
        logger.info("Loading data...")
        data_loader = DefectDataLoader(config)
        df, images = data_loader.load_data()

        # Extract features
        logger.info("Extracting features...")
        feature_extractor = DefectFeatureExtractor(config)
        X = feature_extractor.extract_features(images)
        y = df["is_defective"].values

        logger.info(f"Dataset shape: {X.shape}, Labels: {len(y)}")
        logger.info(f"Defective samples: {sum(y)}, Normal samples: {len(y) - sum(y)}")

        # Train model
        logger.info("Training model...")
        classifier = DefectClassifier(config)
        metrics = classifier.train(X, y)

        # Save model
        logger.info("Saving model...")
        model_path = classifier.save_model()

        logger.info("Experiment completed successfully!")
        logger.info(f"Final validation accuracy: {metrics['accuracy']:.4f}")

    except Exception as e:
        logger.error(f"Experiment failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()

# =============================================================================
# PART 3: ENVIRONMENT SETUP (requirements.txt)
# =============================================================================

"""
# requirements.txt
numpy==1.21.0
pandas==1.3.0
scikit-learn==1.0.2
opencv-python==4.5.3.56
pyyaml==5.4.1
joblib==1.0.1

# Development dependencies
pytest==6.2.4
black==21.6.0
flake8==3.9.2
"""

# =============================================================================
# PART 4: BASIC TESTING (tests/test_data_loader.py)
# =============================================================================

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import cv2


# This would normally be in a separate file
class TestDefectDataLoader:
    """Test suite for DefectDataLoader class."""

    def setup_method(self):
        """Setup test fixtures."""
        # Create temporary config
        self.config = {
            "data": {"raw_data_path": "test_data.csv", "images_path": "test_images/"},
            "preprocessing": {"min_defect_score": 0.1},
        }

    def test_data_validation_removes_low_scores(self):
        """Test that low defect scores are filtered out."""
        # Create test data
        test_df = pd.DataFrame(
            {
                "image_filename": ["img1.jpg", "img2.jpg", "img3.jpg"],
                "is_defective": [1, 0, 1],
                "defect_score": [0.05, 0.8, 0.9],  # First one should be filtered
            }
        )

        loader = DefectDataLoader(self.config)
        validated_df = loader._validate_data(test_df)

        # Should remove the first row
        assert len(validated_df) == 2
        assert all(validated_df["defect_score"] > 0.1)

    def test_handles_missing_files_gracefully(self):
        """Test that missing image files are handled properly."""
        # This test would create temporary files and test file handling
        pass  # Implementation would go here


print("\n" + "=" * 80)
print("SESSION 1 CODE DEMO SUMMARY")
print("=" * 80)
print("✅ Demonstrated progression from messy to organized code")
print("✅ Showed proper project structure for ML projects")
print("✅ Implemented configuration management")
print("✅ Added logging and error handling")
print("✅ Created modular, testable components")
print("✅ Included basic testing framework")
print("✅ Used proper dependency management")
print("=" * 80)
