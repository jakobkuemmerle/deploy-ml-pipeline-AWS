import logging
import time
import pandas as pd

# Define logger
logger = logging.getLogger(__name__)

def score_model(test: pd.DataFrame, y_test: pd.Series, model, initial_features: list) -> pd.DataFrame:
    """Score the model on the test set and return a DataFrame with true labels, 
    predicted probabilities, and binary predictions.

    Args:
        test (pd.DataFrame): DataFrame containing the test set features.
        y_test (pd.Series): Series containing the true labels for the test set.
        model: Trained machine learning model.
        initial_features (list): List of initial features used for prediction.

    Returns:
        pd.DataFrame: DataFrame containing true labels, predicted probabilities, and binary predictions.
    """
    logger.debug("Scoring the model on the test set.")
    start_time = time.time()
    y_pred_proba = model.predict_proba(test[initial_features])[:, 1]
    y_pred_bin = model.predict(test[initial_features])
    end_time = time.time()
    logger.info("Scoring completed.")
    logger.debug("Scoring completed in %.2f seconds.", end_time - start_time)

    # Create DataFrame with scores
    scores = pd.DataFrame({
        "true_labels": y_test,
        "predicted_probabilities": y_pred_proba,
        "predicted_labels": y_pred_bin
    })

    return scores


def save_scores(scores: pd.DataFrame, save_path: str) -> None:
    """Save the model scores to disk.

    Args:
        scores (pd.DataFrame): DataFrame containing model scores.
        save_path (str): Path to save the scores.
    """
    logger.debug("Saving model scores to %s.", save_path)
    try:
        scores.to_csv(save_path, index=False)
        logger.info("Scores saved.")
    except Exception as e:
        logger.error("Error occurred while saving scores to disk: %s", e)
        raise

def read_scores(scores_path: str) -> pd.DataFrame:
    """Reads the model scores from disk.

    Args:
        scores_path (str): Path to the scores file.

    Returns:
        pd.DataFrame: DataFrame containing the model scores.
    """
    logger.debug("Reading model scores from %s.", scores_path)
    try:
        # Read the scores from disk
        scores = pd.read_csv(scores_path)
        logger.info("Scores read.")
        return scores
    except Exception as e:
        logger.error("Error occurred while reading scores from disk: %s", e)
        raise
