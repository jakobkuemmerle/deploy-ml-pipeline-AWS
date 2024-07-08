import sys
import logging
from pathlib import Path
from typing import List
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

def read_enriched_dataset(file_path: str) -> pd.DataFrame:
    """Reads the enriched dataset from disk."""
    try:
        # Read the enriched dataset from disk
        enriched_dataset = pd.read_csv(file_path)
        return enriched_dataset
    except FileNotFoundError as e:
        error_msg = f"File '{file_path}' not found."
        logger.error(error_msg)
        raise FileNotFoundError(error_msg) from e
    except pd.errors.ParserError as e:
        error_msg = f"Error parsing enriched dataset from '{file_path}': {e}."
        logger.error(error_msg)
        raise pd.errors.ParserError(error_msg) from e
    except Exception as e:
        error_msg = f"An unexpected error occurred while reading enriched dataset from '{file_path}': {e}."
        logger.error(error_msg)
        raise Exception(error_msg) from e

def convert_columns_to_float(features: pd.DataFrame) -> pd.DataFrame:
    """Convert all columns in the DataFrame to float."""
    return features.astype(float)

def check_columns_existence(features: pd.DataFrame, columns: List[str]) -> None:
    """Check if the specified columns exist in the DataFrame."""
    for column in columns:
        if column not in features.columns:
            error_message = f"Column '{column}' required for feature calculation is missing."
            logger.error(error_message)
            raise ValueError(error_message)

def calculate_range_features(features: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    """Calculate range features."""
    for column in columns:
        max_col = f"{column}_max"
        min_col = f"{column}_min"
        check_columns_existence(features, [max_col, min_col])
        features[f"{column}_range"] = features[max_col] - features[min_col]
    return features

def calculate_normalized_range_features(features: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    """Calculate normalized range features."""
    for column in columns:
        max_col = f"{column}_max"
        min_col = f"{column}_min"
        mean_col = f"{column}_mean"
        check_columns_existence(features, [max_col, min_col, mean_col])
        features[f"{column}_norm_range"] = calculate_norm_range(features, min_col, max_col, mean_col)
    return features

def perform_log_transformation(features: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    """Perform log transformation."""
    for column in columns:
        check_columns_existence(features, [column])
        features[f"log_{column}"] = np.log(features[column])
    return features

def perform_feature_multiplication(features: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    """Perform multiplication of features."""
    for i in range(len(columns) - 1):
        for j in range(i + 1, len(columns)):
            col_a = columns[i]
            col_b = columns[j]
            check_columns_existence(features, [col_a, col_b])
            new_feature_name = f"{col_a}_x_{col_b}"
            if new_feature_name in features.columns:
                warning_message = (
                    f"New feature '{new_feature_name}' already exists in the DataFrame. "
                    "Will be overwritten"
                )

                logger.warning(warning_message)
            features[new_feature_name] = features[col_a] * features[col_b]
    return features

def generate_features(data: pd.DataFrame, feature_config: dict) -> pd.DataFrame:
    """Generate additional features from the input data."""
    logger.debug("Generating additional features from input data.")

    # Copy the DataFrame to avoid modifying the original data
    features = data.copy()

    # Convert columns to float
    features = convert_columns_to_float(features)

    # Iterate over each feature type to generate
    for feature_type, columns in feature_config.items():
        if feature_type == "calculate_range":
            logger.debug("Calculating range features.")
            features = calculate_range_features(features, columns)
        elif feature_type == "calculate_norm_range":
            logger.debug("Calculating normalized range features.")
            features = calculate_normalized_range_features(features, columns)
        elif feature_type == "log_transform":
            logger.debug("Performing log transformation.")
            features = perform_log_transformation(features, columns)
        elif feature_type == "multiply":
            logger.debug("Performing multiplication of features.")
            features = perform_feature_multiplication(features, columns)
        else:
            raise KeyError(f"Invalid feature type: {feature_type}")

    logger.info("Feature generation completed.")
    return features

def calculate_norm_range(features: pd.DataFrame, min_col: str, max_col: str, mean_col: str) -> pd.Series:
    """Calculate normalized range feature."""
    logger.debug("Calculating normalized range feature.")

    # Check if mean_col is 0
    if (features[mean_col] == 0).any():
        raise ValueError(f"Column '{mean_col}' has zero mean value.")

    # Check for missing values
    if features[min_col].isnull().any() or features[max_col].isnull().any() or features[mean_col].isnull().any():
        raise ValueError("One or more columns have missing values.")

    return (features[max_col] - features[min_col]) / features[mean_col]

def save_enriched_dataset(dataset: pd.DataFrame, save_path: Path) -> None:
    """Save structured dataset to disk.

    Args:
        dataset (pd.DataFrame): DataFrame containing the structured dataset.
        save_path (Path): Path to save the dataset.
    """
    logger.debug("Saving enriched dataset to disk.")
    try:
        dataset.to_csv(save_path, index=False)
        logger.info("Dataset saved to %s", save_path)
    except Exception as e:
        logger.error("Error occurred while trying to save dataset: %s", e)
        sys.exit(1)
