import cv2
import numpy as np
from typing import List
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
