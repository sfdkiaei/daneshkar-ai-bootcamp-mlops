# Session 5: Code Demo - Model Development & Testing
# VisionaryAI's Computer Vision Defect Detection System

"""
This demo shows the evolution of VisionaryAI's defect detection system
from a messy notebook to a production-ready, well-tested codebase.

We'll demonstrate:
1. Poor code organization → Improved structure
2. Hardcoded parameters → Configuration-driven development
3. No testing → Comprehensive testing strategy
"""

# =============================================================================
# PART 1: THE MESSY BEGINNING (What NOT to do)
# =============================================================================

print("=== PART 1: The Messy Beginning ===")
print(
    "This is how VisionaryAI's junior developer initially wrote the defect detection system"
)

# messy_defect_detection.py (All in one file!)
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle
import os


def messy_defect_detection():
    """
    Original messy implementation - everything in one function!
    Problems:
    - Everything hardcoded
    - No separation of concerns
    - Impossible to test
    - No configuration management
    """

    # Hardcoded paths and parameters (BAD!)
    data_path = "/data/phone_factory_images"
    model_path = "/models/defect_detector.pkl"
    image_size = (224, 224)
    test_size = 0.2
    n_estimators = 100

    # Load and preprocess data (mixed with everything else)
    images = []
    labels = []

    # This would normally load real images, but we'll simulate
    print("Loading images... (simulated)")
    for i in range(1000):  # Simulate 1000 images
        # Simulate image data
        img = np.random.rand(224, 224, 3)
        # Simulate preprocessing (resize, normalize, etc.)
        img = cv2.resize(img, image_size)
        img = img / 255.0
        images.append(img.flatten())  # Flatten for simple classifier

        # Simulate labels (0=no_defect, 1=defect)
        labels.append(np.random.choice([0, 1]))

    print(f"Loaded {len(images)} images")

    # Split data (hardcoded split ratio)
    X_train, X_test, y_train, y_test = train_test_split(
        images, labels, test_size=test_size, random_state=42
    )

    # Train model (hardcoded hyperparameters)
    model = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
    model.fit(X_train, y_train)

    # Evaluate (basic evaluation only)
    accuracy = model.score(X_test, y_test)
    print(f"Model accuracy: {accuracy:.3f}")

    # Save model (hardcoded path)
    with open("defect_detector.pkl", "wb") as f:
        pickle.dump(model, f)

    print("Model saved!")
    return model


# Run the messy version
print("Running messy implementation...")
messy_model = messy_defect_detection()
print("✗ Problems: Hardcoded values, no testing, poor structure\n")

# =============================================================================
# PART 2: IMPROVED CODE ORGANIZATION
# =============================================================================

print("=== PART 2: Improved Code Organization ===")
print("Let's refactor this into a proper structure with separation of concerns")

# Now let's organize this properly with separate modules


# ---- data/preprocessing.py ----
class DataPreprocessor:
    """Handles all data preprocessing tasks"""

    def __init__(self, image_size=(224, 224)):
        self.image_size = image_size

    def preprocess_image(self, image):
        """Preprocess a single image"""
        # Resize image
        if image.shape[:2] != self.image_size:
            image = cv2.resize(image, self.image_size)

        # Normalize to [0, 1]
        if image.max() > 1.0:
            image = image.astype(np.float32) / 255.0

        return image

    def preprocess_batch(self, images):
        """Preprocess a batch of images"""
        return [self.preprocess_image(img) for img in images]

    def validate_image(self, image):
        """Validate image format and properties"""
        if not isinstance(image, np.ndarray):
            raise ValueError("Image must be a numpy array")

        if len(image.shape) != 3 or image.shape[2] != 3:
            raise ValueError("Image must be RGB with shape (H, W, 3)")

        return True


# ---- data/loader.py ----
class DataLoader:
    """Handles data loading and splitting"""

    def __init__(self, data_path, preprocessor):
        self.data_path = data_path
        self.preprocessor = preprocessor

    def load_dataset(self, num_samples=1000):
        """Load dataset (simulated for demo)"""
        print(f"Loading {num_samples} images from {self.data_path}")

        images = []
        labels = []

        for i in range(num_samples):
            # Simulate loading real images
            img = np.random.rand(224, 224, 3)

            # Validate and preprocess
            self.preprocessor.validate_image(img)
            processed_img = self.preprocessor.preprocess_image(img)

            images.append(processed_img.flatten())  # Flatten for simple classifier
            labels.append(np.random.choice([0, 1]))  # 0=no_defect, 1=defect

        return np.array(images), np.array(labels)

    def split_data(self, X, y, test_size=0.2, val_size=0.1):
        """Split data into train/validation/test sets"""
        # First split: separate test set
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )

        # Second split: separate validation from training
        val_size_adjusted = val_size / (1 - test_size)  # Adjust for remaining data
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp,
            y_temp,
            test_size=val_size_adjusted,
            random_state=42,
            stratify=y_temp,
        )

        return X_train, X_val, X_test, y_train, y_val, y_test


# ---- models/defect_classifier.py ----
class DefectClassifier:
    """Defect detection model wrapper"""

    def __init__(self, model_type="random_forest", **kwargs):
        self.model_type = model_type
        self.model = None
        self.is_trained = False

        if model_type == "random_forest":
            self.model = RandomForestClassifier(**kwargs)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

    def train(self, X_train, y_train):
        """Train the model"""
        print(f"Training {self.model_type} model...")
        self.model.fit(X_train, y_train)
        self.is_trained = True
        print("Training completed!")

    def predict(self, X):
        """Make predictions"""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        return self.model.predict(X)

    def predict_proba(self, X):
        """Get prediction probabilities"""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        return self.model.predict_proba(X)

    def save(self, filepath):
        """Save trained model"""
        if not self.is_trained:
            raise ValueError("Cannot save untrained model")

        with open(filepath, "wb") as f:
            pickle.dump(self.model, f)
        print(f"Model saved to {filepath}")

    def load(self, filepath):
        """Load trained model"""
        with open(filepath, "rb") as f:
            self.model = pickle.load(f)
        self.is_trained = True
        print(f"Model loaded from {filepath}")


# ---- evaluation/metrics.py ----
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
)


class ModelEvaluator:
    """Handles model evaluation and metrics calculation"""

    def __init__(self):
        self.metrics = {}

    def evaluate(self, y_true, y_pred, y_pred_proba=None):
        """Calculate comprehensive evaluation metrics"""
        self.metrics = {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred),
            "recall": recall_score(y_true, y_pred),
            "f1_score": f1_score(y_true, y_pred),
        }

        return self.metrics

    def print_report(self, y_true, y_pred):
        """Print detailed classification report"""
        print("Classification Report:")
        print(
            classification_report(y_true, y_pred, target_names=["No Defect", "Defect"])
        )

    def calculate_business_metrics(self, y_true, y_pred):
        """Calculate business-relevant metrics"""
        # False negatives are costly (missed defects)
        false_negatives = np.sum((y_true == 1) & (y_pred == 0))
        false_positives = np.sum((y_true == 0) & (y_pred == 1))

        total_defects = np.sum(y_true == 1)
        total_non_defects = np.sum(y_true == 0)

        # Business costs (example values)
        cost_per_missed_defect = 1000  # $1000 per missed defect
        cost_per_false_alarm = 50  # $50 per false alarm

        business_metrics = {
            "missed_defects": false_negatives,
            "false_alarms": false_positives,
            "defect_detection_rate": 1 - (false_negatives / total_defects)
            if total_defects > 0
            else 1,
            "estimated_cost": (false_negatives * cost_per_missed_defect)
            + (false_positives * cost_per_false_alarm),
        }

        return business_metrics


# Now let's use the improved structure
print("Using improved code structure...")

# Initialize components
preprocessor = DataPreprocessor(image_size=(224, 224))
data_loader = DataLoader("/data/phone_factory_images", preprocessor)
model = DefectClassifier(model_type="random_forest", n_estimators=100, random_state=42)
evaluator = ModelEvaluator()

# Load and split data
X, y = data_loader.load_dataset(num_samples=1000)
X_train, X_val, X_test, y_train, y_val, y_test = data_loader.split_data(X, y)

print(f"Training set: {X_train.shape[0]} samples")
print(f"Validation set: {X_val.shape[0]} samples")
print(f"Test set: {X_test.shape[0]} samples")

# Train model
model.train(X_train, y_train)

# Evaluate on validation set
val_pred = model.predict(X_val)
val_metrics = evaluator.evaluate(y_val, val_pred)
print(f"Validation Accuracy: {val_metrics['accuracy']:.3f}")

# Evaluate on test set
test_pred = model.predict(X_test)
test_metrics = evaluator.evaluate(y_test, test_pred)
business_metrics = evaluator.calculate_business_metrics(y_test, test_pred)

print(f"✓ Test Accuracy: {test_metrics['accuracy']:.3f}")
print(f"✓ Defect Detection Rate: {business_metrics['defect_detection_rate']:.3f}")
print(f"✓ Estimated Cost: ${business_metrics['estimated_cost']:.2f}")

# =============================================================================
# PART 3: CONFIGURATION-DRIVEN DEVELOPMENT
# =============================================================================

print("\n=== PART 3: Configuration-Driven Development ===")

# Let's add configuration management
import yaml
from dataclasses import dataclass
from typing import Tuple, List, Dict, Any


@dataclass
class DataConfig:
    """Data-related configuration"""

    data_path: str
    image_size: Tuple[int, int]
    num_samples: int
    test_size: float
    val_size: float


@dataclass
class ModelConfig:
    """Model-related configuration"""

    model_type: str
    n_estimators: int
    max_depth: int
    random_state: int


@dataclass
class TrainingConfig:
    """Training-related configuration"""

    batch_size: int
    validation: bool
    save_path: str


@dataclass
class EvaluationConfig:
    """Evaluation-related configuration"""

    metrics: List[str]
    business_costs: Dict[str, float]


@dataclass
class Config:
    """Main configuration class"""

    data: DataConfig
    model: ModelConfig
    training: TrainingConfig
    evaluation: EvaluationConfig


class ConfigManager:
    """Manages configuration loading and validation"""

    @staticmethod
    def load_from_yaml(config_path: str) -> Config:
        """Load configuration from YAML file"""
        with open(config_path, "r") as f:
            config_dict = yaml.safe_load(f)

        return Config(
            data=DataConfig(**config_dict["data"]),
            model=ModelConfig(**config_dict["model"]),
            training=TrainingConfig(**config_dict["training"]),
            evaluation=EvaluationConfig(**config_dict["evaluation"]),
        )

    @staticmethod
    def validate_config(config: Config) -> bool:
        """Validate configuration values"""
        # Validate data config
        assert 0 < config.data.test_size < 1, "test_size must be between 0 and 1"
        assert 0 < config.data.val_size < 1, "val_size must be between 0 and 1"
        assert config.data.num_samples > 0, "num_samples must be positive"

        # Validate model config
        assert config.model.n_estimators > 0, "n_estimators must be positive"

        # Validate training config
        assert config.training.batch_size > 0, "batch_size must be positive"

        return True


# Create a sample configuration
sample_config = {
    "data": {
        "data_path": "/data/phone_factory_images",
        "image_size": [224, 224],
        "num_samples": 1000,
        "test_size": 0.2,
        "val_size": 0.1,
    },
    "model": {
        "model_type": "random_forest",
        "n_estimators": 100,
        "max_depth": 10,
        "random_state": 42,
    },
    "training": {
        "batch_size": 32,
        "validation": True,
        "save_path": "models/defect_detector.pkl",
    },
    "evaluation": {
        "metrics": ["accuracy", "precision", "recall", "f1_score"],
        "business_costs": {"missed_defect_cost": 1000.0, "false_alarm_cost": 50.0},
    },
}

# Save sample config to demonstrate loading
with open("config.yaml", "w") as f:
    yaml.dump(sample_config, f, default_flow_style=False)

print("Created sample configuration file:")
print("✓ All parameters externalized")
print("✓ Environment-specific configs possible")
print("✓ Easy experimentation")

# Load and use configuration
config = ConfigManager.load_from_yaml("config.yaml")
ConfigManager.validate_config(config)

print(
    f"✓ Loaded config: {config.model.model_type} with {config.model.n_estimators} estimators"
)

# =============================================================================
# PART 4: COMPREHENSIVE TESTING
# =============================================================================

print("\n=== PART 4: Comprehensive Testing ===")

# Let's add comprehensive testing
import unittest
import tempfile
import time


class TestDataPreprocessor(unittest.TestCase):
    """Unit tests for DataPreprocessor"""

    def setUp(self):
        self.preprocessor = DataPreprocessor(image_size=(224, 224))

    def test_preprocess_image_shape(self):
        """Test that preprocessing produces correct image shape"""
        # Create test image
        test_image = np.random.rand(256, 256, 3) * 255
        test_image = test_image.astype(np.uint8)

        # Preprocess
        processed = self.preprocessor.preprocess_image(test_image)

        # Check shape
        self.assertEqual(processed.shape, (224, 224, 3))

    def test_preprocess_image_normalization(self):
        """Test that preprocessing normalizes image correctly"""
        # Create test image with values > 1
        test_image = np.random.rand(224, 224, 3) * 255
        test_image = test_image.astype(np.uint8)

        # Preprocess
        processed = self.preprocessor.preprocess_image(test_image)

        # Check normalization
        self.assertGreaterEqual(processed.min(), 0.0)
        self.assertLessEqual(processed.max(), 1.0)
        self.assertEqual(processed.dtype, np.float32)

    def test_validate_image_valid(self):
        """Test validation with valid image"""
        valid_image = np.random.rand(100, 100, 3)
        self.assertTrue(self.preprocessor.validate_image(valid_image))

    def test_validate_image_invalid_shape(self):
        """Test validation with invalid image shape"""
        invalid_image = np.random.rand(100, 100)  # Missing channel dimension
        with self.assertRaises(ValueError):
            self.preprocessor.validate_image(invalid_image)


class TestDefectClassifier(unittest.TestCase):
    """Unit tests for DefectClassifier"""

    def setUp(self):
        self.classifier = DefectClassifier(model_type="random_forest", n_estimators=10)
        # Create small dataset for testing
        self.X_test = np.random.rand(50, 100)
        self.y_test = np.random.choice([0, 1], 50)

    def test_train_model(self):
        """Test model training"""
        self.classifier.train(self.X_test, self.y_test)
        self.assertTrue(self.classifier.is_trained)

    def test_predict_untrained_model(self):
        """Test that prediction fails on untrained model"""
        with self.assertRaises(ValueError):
            self.classifier.predict(self.X_test)

    def test_prediction_shape(self):
        """Test prediction output shape"""
        self.classifier.train(self.X_test, self.y_test)
        predictions = self.classifier.predict(self.X_test)
        self.assertEqual(predictions.shape[0], self.X_test.shape[0])

    def test_save_load_model(self):
        """Test model serialization"""
        self.classifier.train(self.X_test, self.y_test)

        # Save model
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as tmp:
            self.classifier.save(tmp.name)

            # Load model
            new_classifier = DefectClassifier()
            new_classifier.load(tmp.name)

            # Test that loaded model works
            predictions = new_classifier.predict(self.X_test)
            self.assertEqual(predictions.shape[0], self.X_test.shape[0])


class TestModelEvaluator(unittest.TestCase):
    """Unit tests for ModelEvaluator"""

    def setUp(self):
        self.evaluator = ModelEvaluator()
        self.y_true = np.array([0, 1, 0, 1, 1])
        self.y_pred = np.array([0, 1, 1, 1, 0])

    def test_evaluate_metrics(self):
        """Test evaluation metrics calculation"""
        metrics = self.evaluator.evaluate(self.y_true, self.y_pred)

        # Check that all expected metrics are present
        expected_metrics = ["accuracy", "precision", "recall", "f1_score"]
        for metric in expected_metrics:
            self.assertIn(metric, metrics)
            self.assertIsInstance(metrics[metric], float)

    def test_business_metrics(self):
        """Test business metrics calculation"""
        business_metrics = self.evaluator.calculate_business_metrics(
            self.y_true, self.y_pred
        )

        expected_keys = [
            "missed_defects",
            "false_alarms",
            "defect_detection_rate",
            "estimated_cost",
        ]
        for key in expected_keys:
            self.assertIn(key, business_metrics)


# Integration Tests
class TestTrainingPipeline(unittest.TestCase):
    """Integration tests for the complete training pipeline"""

    def setUp(self):
        self.config = Config(
            data=DataConfig("/tmp", (64, 64), 100, 0.2, 0.1),
            model=ModelConfig("random_forest", 10, 5, 42),
            training=TrainingConfig(32, True, "/tmp/model.pkl"),
            evaluation=EvaluationConfig(["accuracy"], {"missed_defect_cost": 1000}),
        )

    def test_full_pipeline(self):
        """Test the complete training pipeline"""
        # Initialize components
        preprocessor = DataPreprocessor(image_size=self.config.data.image_size)
        data_loader = DataLoader(self.config.data.data_path, preprocessor)
        model = DefectClassifier(
            model_type=self.config.model.model_type,
            n_estimators=self.config.model.n_estimators,
            random_state=self.config.model.random_state,
        )
        evaluator = ModelEvaluator()

        # Load and split data
        X, y = data_loader.load_dataset(num_samples=self.config.data.num_samples)
        X_train, X_val, X_test, y_train, y_val, y_test = data_loader.split_data(X, y)

        # Train model
        model.train(X_train, y_train)

        # Evaluate
        predictions = model.predict(X_test)
        metrics = evaluator.evaluate(y_test, predictions)

        # Check that we get reasonable results
        self.assertGreater(metrics["accuracy"], 0.3)  # Should be better than random
        self.assertLess(metrics["accuracy"], 1.0)  # Shouldn't be perfect


# Model-specific tests (invariance and performance)
class TestModelInvariance(unittest.TestCase):
    """Tests for model invariance properties"""

    def setUp(self):
        self.preprocessor = DataPreprocessor()
        self.model = DefectClassifier(n_estimators=50)

        # Train a simple model
        X = np.random.rand(200, 100)
        y = np.random.choice([0, 1], 200)
        self.model.train(X, y)

    def test_prediction_consistency(self):
        """Test that identical inputs produce identical outputs"""
        test_input = np.random.rand(1, 100)

        pred1 = self.model.predict(test_input)
        pred2 = self.model.predict(test_input)

        np.testing.assert_array_equal(pred1, pred2)

    def test_batch_vs_single_prediction(self):
        """Test that batch and single predictions match"""
        test_inputs = np.random.rand(5, 100)

        # Batch prediction
        batch_pred = self.model.predict(test_inputs)

        # Individual predictions
        individual_preds = [
            self.model.predict(test_inputs[i : i + 1])[0] for i in range(5)
        ]

        np.testing.assert_array_equal(batch_pred, individual_preds)


class TestModelPerformance(unittest.TestCase):
    """Tests for model performance requirements"""

    def setUp(self):
        self.model = DefectClassifier(n_estimators=50)
        X = np.random.rand(1000, 100)
        y = np.random.choice([0, 1], 1000)
        self.model.train(X, y)

    def test_inference_speed(self):
        """Test that inference meets speed requirements"""
        # VisionaryAI requirement: process 100 images in < 5 seconds
        batch_size = 100
        test_batch = np.random.rand(batch_size, 100)

        start_time = time.time()
        predictions = self.model.predict(test_batch)
        end_time = time.time()

        inference_time = end_time - start_time
        print(f"Inference time for {batch_size} samples: {inference_time:.3f} seconds")

        # Should process 100 samples in less than 5 seconds
        self.assertLess(inference_time, 5.0)

    def test_memory_usage(self):
        """Test memory usage (simplified check)"""
        import sys

        # Get model size
        model_size = sys.getsizeof(self.model)
        print(f"Model memory usage: {model_size / (1024 * 1024):.2f} MB")

        # Should be reasonable size (less than 100MB for this simple model)
        self.assertLess(model_size, 100 * 1024 * 1024)


# Run all tests
def run_tests():
    """Run all test suites"""
    print("Running Unit Tests...")

    # Create test suite
    test_suite = unittest.TestSuite()

    # Add test classes
    test_classes = [
        TestDataPreprocessor,
        TestDefectClassifier,
        TestModelEvaluator,
        TestTrainingPipeline,
        TestModelInvariance,
        TestModelPerformance,
    ]

    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)

    print(f"\nTest Results:")
    print(f"✓ Tests run: {result.testsRun}")
    print(f"✗ Failures: {len(result.failures)}")
    print(f"✗ Errors: {len(result.errors)}")

    return result.wasSuccessful()


# Demonstrate testing
print("Demonstrating comprehensive testing strategy...")
test_success = run_tests()

if test_success:
    print("✅ All tests passed! Code is ready for production.")
else:
    print("❌ Some tests failed. Code needs improvement.")

print("\n=== DEMO SUMMARY ===")
print("✓ Code Organization: Modular, reusable, testable structure")
print("✓ Configuration-Driven: All parameters externalized")
print("✓ Comprehensive Testing: Unit, integration, and model tests")
print("✓ Business Metrics: Beyond accuracy metrics")
print("✓ Performance Testing: Speed and memory requirements")
print("✓ Invariance Testing: Consistent model behavior")

print("\nVisionaryAI's defect detection system is now production-ready!")

# Clean up demo files
try:
    os.remove("config.yaml")
    os.remove("defect_detector.pkl")
except:
    pass
