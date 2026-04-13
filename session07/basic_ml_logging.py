import json
import logging
import time
from datetime import datetime
from typing import Dict, Any
import numpy as np


class MLLogger:
    """Simple ML logging system for VisionaryAI models"""

    def __init__(self, model_name: str, version: str):
        self.model_name = model_name
        self.version = version

        # Set up structured logging
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )
        self.logger = logging.getLogger(f"ML-{model_name}")

    def log_prediction(
        self,
        request_id: str,
        input_data: Dict[str, Any],
        prediction: Any,
        confidence: float,
        processing_time: float,
    ):
        """Log a model prediction with all relevant context"""

        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "request_id": request_id,
            "model_name": self.model_name,
            "model_version": self.version,
            "input_features": {
                # Log feature statistics, not raw data for privacy
                "feature_count": len(input_data),
                "has_missing_values": any(v is None for v in input_data.values()),
                "numeric_feature_stats": self._compute_feature_stats(input_data),
            },
            "prediction": prediction,
            "confidence": confidence,
            "processing_time_ms": processing_time * 1000,
            "system_info": {
                "cpu_usage_percent": 45.2,  # In real system, get actual metrics
                "memory_usage_mb": 512.1,
            },
        }

        # Log as JSON for easy parsing
        self.logger.info(json.dumps(log_entry))

        return log_entry

    def _compute_feature_stats(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """Compute basic statistics for numeric features"""
        numeric_features = {
            k: v
            for k, v in features.items()
            if isinstance(v, (int, float)) and v is not None
        }

        if not numeric_features:
            return {}

        values = list(numeric_features.values())
        return {
            "mean": np.mean(values),
            "std": np.std(values),
            "min": min(values),
            "max": max(values),
        }


# Example usage for VisionaryAI systems
def demo_basic_logging():
    print("=== Part 1: Basic ML Logging ===")

    # Initialize loggers for different VisionaryAI systems
    defect_logger = MLLogger("defect_detection", "v1.2.3")
    helpdesk_logger = MLLogger("helpdesk_nlp", "v2.1.0")

    # Example 1: Defect Detection System
    print("\n1. Logging defect detection prediction:")
    image_features = {
        "brightness": 0.65,
        "contrast": 0.8,
        "edge_density": 0.42,
        "color_variance": 0.33,
    }

    start_time = time.time()
    prediction = "no_defect"  # Simulated prediction
    confidence = 0.94
    processing_time = time.time() - start_time

    log_entry = defect_logger.log_prediction(
        request_id="IMG_12345",
        input_data=image_features,
        prediction=prediction,
        confidence=confidence,
        processing_time=processing_time,
    )

    # Example 2: Helpdesk NLP System
    print("\n2. Logging helpdesk classification:")
    ticket_features = {
        "text_length": 150,
        "question_marks": 2,
        "exclamation_marks": 0,
        "sentiment_score": 0.3,
        "urgency_keywords": 1,
    }

    start_time = time.time()
    prediction = "billing_issue"
    confidence = 0.87
    processing_time = time.time() - start_time

    helpdesk_logger.log_prediction(
        request_id="TKT_67890",
        input_data=ticket_features,
        prediction=prediction,
        confidence=confidence,
        processing_time=processing_time,
    )

    print("\n✅ Key Point: Structure your logs as JSON for easy analysis")
    print("✅ Key Point: Log feature stats, not raw data for privacy")
    print("✅ Key Point: Always include model version and confidence scores")


if __name__ == "__main__":
    demo_basic_logging()
