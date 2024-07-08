import logging
import time
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

# Define logger
logger = logging.getLogger(__name__)

def split_data(features: pd.DataFrame, target: pd.Series, test_size: float = 0.4) -> tuple:
    """Split data into training and testing sets."""
    logger.debug("Splitting data into training and testing sets.")
    logger.debug("Test size: %f.", test_size)
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=test_size)
    logger.info("Data split completed.")
    return X_train, X_test, y_train, y_test

def train_model(X_train: pd.DataFrame, y_train: pd.Series, initial_features: list, n_estimators: int = 10, max_depth: int = 10) -> RandomForestClassifier:
    """Train a random forest classifier."""
    logger.debug("Training random forest classifier.")
    start_time = time.time()
    rf_model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth)
    rf_model.fit(X_train[initial_features], y_train)
    end_time = time.time()
    logger.info("Training completed.")
    logger.debug("Training completed in %.2f seconds.", end_time - start_time)

    # Log basic model specs
    logger.debug("Model specifications - n_estimators: %d, max_depth: %d.", n_estimators, max_depth)

    # Log warning if training time exceeds threshold
    threshold_time = 60  # in seconds
    if end_time - start_time > threshold_time:
        logger.warning("Training time exceeded threshold.")

    # Log warning if training size is very small compared to the depth of the model
    if len(X_train) < max_depth * 10:
        logger.warning("Training size is very small compared to the depth of the model.")

    return rf_model

def save_model(model: RandomForestClassifier, model_path: Path) -> None:
    """Save the trained model to disk.

    Args:
        model (RandomForestClassifier): Trained model to be saved.
        model_path (Path): Path to save the trained model.
    """
    try:
        logger.debug("Saving trained model to disk at %s.", model_path)
        joblib.dump(model, model_path)
        logger.info("Model saved.")
    except Exception as e:
        error_msg = f"An unexpected error occurred while saving the model to '{model_path}': {e}."
        logger.error(error_msg)
        raise Exception(error_msg) from e

def read_model_and_data(model_path: str, train_data_path: str, test_data_path: str) -> tuple:
    """Reads the trained model and data from disk."""
    logger.info("Reading trained model and data from disk.")
    try:
        # Read the trained model from disk
        trained_model = joblib.load(model_path)
        logger.info("Trained model loaded.")

        # Read the training and testing data from disk
        train_data = pd.read_csv(train_data_path)
        test_data = pd.read_csv(test_data_path)
        logger.info("Training and testing data loaded.")

        return trained_model, train_data, test_data
    except Exception as e:
        logger.error("Error occurred while reading model and data from disk: %s", e)
        raise

def save_data(X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.Series, y_test: pd.Series, artifacts_dir: Path) -> None:
    """Save the train and test datasets to disk."""
    logger.debug("Saving train and test datasets to disk.")
    try:
        # Save train and test datasets to disk
        X_train.to_csv(artifacts_dir / "X_train.csv", index=False)
        X_test.to_csv(artifacts_dir / "X_test.csv", index=False)
        y_train.to_csv(artifacts_dir / "y_train.csv", index=False)
        y_test.to_csv(artifacts_dir / "y_test.csv", index=False)
        logger.info("Train and test datasets saved.")
    except Exception as e:
        logger.error("Error occurred while saving train and test datasets to disk: %s", e)
        raise
