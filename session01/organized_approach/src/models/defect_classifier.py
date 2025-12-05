from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib
import logging
from pathlib import Path
from datetime import datetime
import json
import numpy as np
from typing import Dict


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
