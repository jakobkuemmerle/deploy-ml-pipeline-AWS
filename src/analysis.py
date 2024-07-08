import datetime
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from cycler import cycler

logger = logging.getLogger(__name__)

def update_matplotlib_defaults():
    """Update matplotlib defaults to a predefined style."""
    mpl_update = {
        "font.size": 16,
        "axes.prop_cycle": cycler("color", ["#0085ca", "#888b8d", "#00c389", "#f4364c", "#e56db1"]),
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "figure.figsize": [12.0, 8.0],
        "axes.labelsize": 20,
        "axes.labelcolor": "#677385",
        "axes.titlesize": 20,
        "lines.color": "#0055A7",
        "lines.linewidth": 3,
        "text.color": "#677385",
        "font.family": "sans-serif",
        "font.sans-serif": "Tahoma",
    }
    plt.rcParams.update(mpl_update)
    logger.debug("Matplotlib defaults updated.")


def dateplus(x: str) -> str:
    """Prepend the current date to a string."""
    now = datetime.datetime.now().strftime("%Y-%m-%d")
    return f"{now}-{x}"


def save_figures(data: pd.DataFrame, dir: Path) -> list[Path]:
    """Save figures for each feature in the DataFrame to the specified directory.

    Args:
        data (pd.DataFrame): DataFrame containing features.
        dir (Path): Directory to save the figures to.

    Returns:
        list[Path]: List of paths to the saved figures.
    """
    saved_paths = []

    # Create the directory if it doesn't exist
    dir.mkdir(parents=True, exist_ok=True)

    # Update matplotlib defaults
    update_matplotlib_defaults()

    for feat in data.columns:
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.hist(
            [
                data[feat][data["class"] == 0].values,
                data[feat][data["class"] == 1].values,
            ]
        )
        ax.set_xlabel(" ".join(feat.split("_")).capitalize())
        ax.set_ylabel("Number of observations")

        # Save the figure with prepended date
        fig_name = dateplus(f"{feat}.png")
        fig_path = dir / fig_name
        try:
            fig.savefig(fig_path)
            saved_paths.append(fig_path)
            logger.debug("Figure saved: %s", fig_path)
        except Exception as e:
            logger.error("Error occurred while saving figure %s: %s", fig_path, e)

        # Close the figure to release memory
        plt.close(fig)
    logger.info("All figures saved successfully")

    if not saved_paths:
        logger.warning("No figures were saved.")

    return saved_paths
