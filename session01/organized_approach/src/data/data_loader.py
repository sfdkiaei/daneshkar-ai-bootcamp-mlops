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
