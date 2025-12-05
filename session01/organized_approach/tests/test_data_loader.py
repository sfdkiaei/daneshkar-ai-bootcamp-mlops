import pytest
import pandas as pd
from pathlib import Path
import sys

# Add src to path to import our modules
sys.path.append(str(Path(__file__).parent.parent / "src"))

from data.data_loader import DefectDataLoader


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
