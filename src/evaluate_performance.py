import time
from pathlib import Path
import logging
from sklearn import metrics
import pandas as pd
import matplotlib.pyplot as plt
import yaml

logger = logging.getLogger(__name__)

def evaluate_performance(scores: pd.DataFrame, evaluation_metrics: list) -> dict:
    """Evaluate the model performance metrics.

    Args:
        scores (pd.DataFrame): DataFrame containing model scores.
        evaluation_metrics (list): List of evaluation metrics to compute.

    Returns:
        dict: Dictionary containing the computed evaluation metrics.
    """
    logger.debug("Evaluating model performance.")
    start_time = time.time()

    evaluation_results = {}

    y_true = scores["true_labels"]
    y_pred_proba = scores["predicted_probabilities"]
    y_pred_bin = scores["predicted_labels"]

    if "auc" in evaluation_metrics:
        evaluation_results["auc"] = metrics.roc_auc_score(y_true, y_pred_proba)
    if "accuracy" in evaluation_metrics:
        accuracy = metrics.accuracy_score(y_true, y_pred_bin)
        evaluation_results["accuracy"] = accuracy
        logger.info("Model accuracy: %.2f%%", accuracy)
        if accuracy < 0.9:
            logger.warning("Model accuracy is below 90%.")

    if "confusion_matrix" in evaluation_metrics:
        evaluation_results["confusion_matrix"] = pd.DataFrame(
            metrics.confusion_matrix(y_true, y_pred_bin),
            index=["Actual negative", "Actual positive"],
            columns=["Predicted negative", "Predicted positive"]
        ).to_dict()
    if "classification_report" in evaluation_metrics:
        evaluation_results["classification_report"] = metrics.classification_report(y_true, y_pred_bin,
                                                                                    output_dict=True)

    end_time = time.time()
    logger.debug("Model performance evaluation completed in %.2f seconds.", end_time - start_time)
    logger.info("Model performance evaluation completed.")

    return evaluation_results

def plot_metrics_bar_chart(used_metrics: dict, save_dir: str) -> None:
    """Plot and save selected evaluation metrics as a bar chart."""

    logger.debug("Creating Evaluation metrics bar chart.")
    # Extract relevant metrics for plotting
    accuracy = used_metrics["accuracy"]
    auc = used_metrics["auc"]
    f1_score = used_metrics["classification_report"]["macro avg"]["f1-score"]
    precision = used_metrics["classification_report"]["macro avg"]["precision"]
    recall = used_metrics["classification_report"]["macro avg"]["recall"]

    # Plot the bar chart
    metrics_names = ["Accuracy", "AUC", "F1-Score", "Precision", "Recall"]
    metrics_values = [accuracy, auc, f1_score, precision, recall]

    plt.figure(figsize=(8, 6))
    plt.bar(metrics_names, metrics_values, color="skyblue")
    plt.xlabel("Metrics")
    plt.ylabel("Value")
    plt.title("Evaluation Metrics")
    plt.grid(axis="y")

    # Set y-axis limit to truncate at 0.8
    plt.ylim(bottom=0.8)

    # Save the chart
    chart_path = save_dir / "metrics_bar_chart.png"
    plt.savefig(chart_path)

    logger.info("Evaluation metrics bar chart saved at: %s", chart_path)

    # Close the plot to release memory
    plt.close()

def save_metrics(metrics_object: dict, save_path: str) -> None:
    """Save the evaluation metrics to disk.

    Args:
        metrics (dict): Dictionary containing the evaluation metrics.
        save_path (str): Path to save the metrics.
    """
    logger.debug("Saving evaluation metrics to %s.", save_path)
    try:
        with open(save_path, "w") as f:
            yaml.dump(metrics_object, f)
        logger.info("Evaluation metrics saved.")
    except Exception as e:
        logger.error("Error occurred while saving evaluation metrics to disk: %s", e)
        raise

    # Plot and save the bar chart
    plot_metrics_bar_chart(metrics_object, Path(save_path).parent)
