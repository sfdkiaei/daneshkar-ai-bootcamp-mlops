# -----------------------------------------------------------------------------
# File: tests/validate_data_schema.py
# Point: Data validation in CI/CD pipelines
# -----------------------------------------------------------------------------

import pandas as pd
import pandera as pa
from pandera import Column, DataFrameSchema, Check
import sys
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_defect_detection_schema():
    """Schema for mobile phone factory defect detection data."""
    return DataFrameSchema(
        {
            "image_path": Column(str, nullable=False),
            "defect_type": Column(
                str, Check.isin(["scratch", "dent", "discoloration", "none"])
            ),
            "defect_severity": Column(float, Check.between(0, 1), nullable=True),
            "inspector_id": Column(str, nullable=False),
            "timestamp": Column(pd.Timestamp, nullable=False),
            "production_line": Column(str, Check.isin(["line_1", "line_2", "line_3"])),
        }
    )


def create_helpdesk_schema():
    """Schema for NLP helpdesk data."""
    return DataFrameSchema(
        {
            "ticket_id": Column(str, nullable=False, unique=True),
            "customer_query": Column(
                str,
                Check.str_length(min_val=10),
            ),
            "intent_label": Column(
                str,
                Check.isin(
                    ["technical_support", "billing", "returns", "general_inquiry"]
                ),
            ),
            "priority": Column(str, Check.isin(["low", "medium", "high", "urgent"])),
            "resolution_time": Column(float, Check.greater_than(0), nullable=True),
            "customer_satisfaction": Column(float, Check.between(1, 5), nullable=True),
        }
    )


def create_recommendation_schema():
    """Schema for recommendation engine data."""
    return DataFrameSchema(
        {
            "user_id": Column(str, nullable=False),
            "item_id": Column(str, nullable=False),
            "interaction_type": Column(str, Check.isin(["view", "cart", "purchase"])),
            "rating": Column(float, Check.between(1, 5), nullable=True),
            "timestamp": Column(pd.Timestamp, nullable=False),
            "category": Column(str, nullable=False),
            "price": Column(float, Check.greater_than(0)),
        }
    )


def validate_dataset(data_path: str, schema_type: str) -> bool:
    """Validate dataset against predefined schema."""

    try:
        # Load data
        logger.info(f"Loading data from {data_path}")
        df = pd.read_csv(data_path)

        # Select appropriate schema
        schemas = {
            "defect_detection": create_defect_detection_schema(),
            "helpdesk": create_helpdesk_schema(),
            "recommendation": create_recommendation_schema(),
        }

        if schema_type not in schemas:
            raise ValueError(f"Unknown schema type: {schema_type}")

        schema = schemas[schema_type]

        # Validate data
        logger.info(f"Validating {len(df)} rows against {schema_type} schema")
        validated_df = schema.validate(df)

        logger.info(f"✅ Data validation successful for {schema_type}")
        return True

    except pa.errors.SchemaError as e:
        logger.error(f"❌ Schema validation failed: {e}")
        return False
    except Exception as e:
        logger.error(f"❌ Validation error: {e}")
        return False


if __name__ == "__main__":
    # Example usage in CI/CD pipeline
    import argparse

    parser = argparse.ArgumentParser(description="Validate data schema")
    parser.add_argument("--data-path", required=True, help="Path to data file")
    parser.add_argument(
        "--schema-type",
        required=True,
        choices=["defect_detection", "helpdesk", "recommendation"],
    )

    args = parser.parse_args()

    success = validate_dataset(args.data_path, args.schema_type)
    sys.exit(0 if success else 1)
