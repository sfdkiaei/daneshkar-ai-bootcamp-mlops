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
