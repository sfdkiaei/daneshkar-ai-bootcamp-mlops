# -----------------------------------------------------------------------------
# File: tests/model_validation.py
# Point: Model performance validation in CI/CD
# -----------------------------------------------------------------------------

import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import cross_val_score
import joblib
import sys
import logging
from typing import Dict, Any, Tuple

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelValidator:
    """Comprehensive model validation for CI/CD pipelines."""

    def __init__(self, model_path: str, test_data_path: str):
        self.model_path = model_path
        self.test_data_path = test_data_path
        self.model = None
        self.test_data = None

    def load_model_and_data(self):
        """Load model and test data."""
        try:
            logger.info(f"Loading model from {self.model_path}")
            self.model = joblib.load(self.model_path)

            logger.info(f"Loading test data from {self.test_data_path}")
            self.test_data = pd.read_csv(self.test_data_path)

            return True
        except Exception as e:
            logger.error(f"Failed to load model or data: {e}")
            return False

    def validate_defect_detection_model(
        self, threshold: float = 0.95
    ) -> Dict[str, Any]:
        """Validate computer vision defect detection model."""

        # Separate features and labels
        X_test = self.test_data.drop(["defect_type"], axis=1)
        y_test = self.test_data["defect_type"]

        # Make predictions
        y_pred = self.model.predict(X_test)
        y_proba = self.model.predict_proba(X_test)

        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average="weighted")
        precision = precision_score(y_test, y_pred, average="weighted")
        recall = recall_score(y_test, y_pred, average="weighted")

        # Performance tests
        results = {
            "accuracy": accuracy,
            "f1_score": f1,
            "precision": precision,
            "recall": recall,
            "meets_threshold": accuracy >= threshold,
            "confidence_distribution": np.mean(np.max(y_proba, axis=1)),
        }

        # Additional ML-specific tests
        results.update(self._run_ml_property_tests(X_test, y_pred))

        return results

    def validate_helpdesk_nlp_model(self, threshold: float = 0.90) -> Dict[str, Any]:
        """Validate NLP helpdesk classification model."""

        X_test = self.test_data["customer_query"]
        y_test = self.test_data["intent_label"]

        # Make predictions
        y_pred = self.model.predict(X_test)

        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average="weighted")

        results = {
            "accuracy": accuracy,
            "f1_score": f1,
            "meets_threshold": f1 >= threshold,
        }

        # NLP-specific tests
        results.update(
            self._test_text_robustness(X_test.iloc[:100])
        )  # Sample for speed

        return results

    def validate_recommendation_model(self, threshold: float = 0.85) -> Dict[str, Any]:
        """Validate recommendation system model."""

        # For recommendation systems, we might test NDCG, precision@k, recall@k
        # This is a simplified version

        X_test = self.test_data[["user_id", "item_id", "category", "price"]]
        y_test = self.test_data["rating"]

        y_pred = self.model.predict(X_test)

        # Calculate RMSE for rating prediction
        rmse = np.sqrt(np.mean((y_test - y_pred) ** 2))

        # Convert to a "goodness" metric (lower RMSE is better)
        normalized_score = max(0, 1 - (rmse / 4))  # Assuming 1-5 rating scale

        results = {
            "rmse": rmse,
            "normalized_score": normalized_score,
            "meets_threshold": normalized_score >= threshold,
        }

        return results

    def _run_ml_property_tests(
        self, X_test: pd.DataFrame, y_pred: np.ndarray
    ) -> Dict[str, bool]:
        """Run property-based tests for ML models."""

        tests = {}

        # Test 1: Prediction consistency
        # Same input should give same output
        sample_idx = np.random.choice(len(X_test), min(10, len(X_test)), replace=False)
        sample_data = X_test.iloc[sample_idx]

        pred1 = self.model.predict(sample_data)
        pred2 = self.model.predict(sample_data)
        tests["prediction_consistency"] = np.array_equal(pred1, pred2)

        # Test 2: No NaN predictions
        tests["no_nan_predictions"] = not np.any(pd.isna(y_pred))

        # Test 3: Valid prediction range (for classification)
        if hasattr(self.model, "classes_"):
            valid_predictions = np.all(np.isin(y_pred, self.model.classes_))
            tests["valid_prediction_range"] = valid_predictions

        return tests

    def _test_text_robustness(self, text_samples: pd.Series) -> Dict[str, bool]:
        """Test NLP model robustness to text variations."""

        tests = {}

        try:
            # Test with minor text variations
            original_preds = self.model.predict(text_samples)

            # Test case sensitivity
            lower_case_texts = text_samples.str.lower()
            lower_case_preds = self.model.predict(lower_case_texts)

            # Most predictions should be the same (allowing some variation)
            consistency_rate = np.mean(original_preds == lower_case_preds)
            tests["case_sensitivity_robust"] = consistency_rate > 0.8

            # Test with extra whitespace
            whitespace_texts = text_samples.apply(lambda x: f"  {x}  ")
            whitespace_preds = self.model.predict(whitespace_texts)

            consistency_rate = np.mean(original_preds == whitespace_preds)
            tests["whitespace_robust"] = consistency_rate > 0.9

        except Exception as e:
            logger.warning(f"Robustness tests failed: {e}")
            tests["case_sensitivity_robust"] = False
            tests["whitespace_robust"] = False

        return tests


def main():
    """Main function for CI/CD pipeline integration."""
    import argparse

    parser = argparse.ArgumentParser(description="Validate ML model performance")
    parser.add_argument("--model-path", required=True, help="Path to trained model")
    parser.add_argument("--test-data", required=True, help="Path to test dataset")
    parser.add_argument(
        "--model-type",
        required=True,
        choices=["defect_detection", "helpdesk", "recommendation"],
    )
    parser.add_argument(
        "--threshold", type=float, default=0.9, help="Performance threshold"
    )

    args = parser.parse_args()

    # Initialize validator
    validator = ModelValidator(args.model_path, args.test_data)

    if not validator.load_model_and_data():
        sys.exit(1)

    # Run appropriate validation
    validation_methods = {
        "defect_detection": validator.validate_defect_detection_model,
        "helpdesk": validator.validate_helpdesk_nlp_model,
        "recommendation": validator.validate_recommendation_model,
    }

    results = validation_methods[args.model_type](args.threshold)

    # Log results
    logger.info("=== Model Validation Results ===")
    for metric, value in results.items():
        logger.info(f"{metric}: {value}")

    # Check if model meets requirements
    if results.get("meets_threshold", False):
        logger.info("✅ Model validation PASSED")
        sys.exit(0)
    else:
        logger.error("❌ Model validation FAILED")
        sys.exit(1)


if __name__ == "__main__":
    main()
