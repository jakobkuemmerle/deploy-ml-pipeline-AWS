import logging
import sys
from pathlib import Path
import pandas as pd

logger = logging.getLogger(__name__)

def read_dataset(file_path: str) -> pd.DataFrame:
    """Reads the structured dataset from disk.

    Args:
        file_path (str): Path to the file containing the structured dataset.

    Returns:
        pd.DataFrame: DataFrame containing the structured dataset.
    """
    try:
        # Read the dataset from disk
        dataset = pd.read_csv(file_path)
        return dataset
    except FileNotFoundError as e:
        error_msg = f"File '{file_path}' not found."
        logger.error(error_msg)
        raise FileNotFoundError(error_msg) from e
    except pd.errors.ParserError as e:
        error_msg = f"Error parsing dataset from '{file_path}': {e}."
        logger.error(error_msg)
        raise pd.errors.ParserError(error_msg) from e
    except Exception as e:
        error_msg = f"An unexpected error occurred while reading dataset from '{file_path}': {e}."
        logger.error(error_msg)
        raise Exception(error_msg) from e

def create_dataset(file_path: str, class_indices: tuple, columns: list) -> pd.DataFrame:
    """Imports data from file and splits it into two classes.

    Args:
        file_path (str): Path to the file containing the data.
        class_indices (tuple): Tuple containing the start and end indices of the two classes.
        columns (list): List of column names for the DataFrame.

    Returns:
        pd.DataFrame: DataFrame containing the imported data with class labels.
    """
    # Log information about the function call
    logger.debug("Creating dataset from file: %s", file_path)
    logger.debug("Columns used: %s", columns)

    with open(file_path, "r") as f:
        try:
            data = [[s for s in line.split(" ") if s!=""] for line in f.readlines()]
        except Exception as e:
            logger.error("Error occurred while importing data from file: %s", e, exc_info=True)


    # Extract first class
    first_class_data = data[class_indices[0][0]:class_indices[0][1]]
    first_class_df = pd.DataFrame(first_class_data, columns=columns)
    first_class_df["class"] = 0

    # Extract second class
    second_class_data = data[class_indices[1][0]:class_indices[1][1]]
    second_class_df = pd.DataFrame(second_class_data, columns=columns)
    second_class_df["class"] = 1

    # Concatenate dataframes
    merged_df = pd.concat([first_class_df, second_class_df], ignore_index=True)

    # Log the size of the resulting DataFrame
    logger.info("Dataset created.")
    logger.debug("Dataset created. Size: %s", merged_df.shape)

    # Log the count of rows for each class
    class_counts = merged_df["class"].value_counts()
    logger.debug("Class 0 count: %d", class_counts[0])
    logger.debug("Class 1 count: %d", class_counts[1])

    if merged_df.empty:
        logger.warning("The created dataset is empty.")

    return merged_df

def save_dataset(dataset: pd.DataFrame, save_path: Path) -> None:
    """Save structured dataset to disk.

    Args:
        dataset (pd.DataFrame): DataFrame containing the structured dataset.
        save_path (Path): Path to save the dataset.
    """

    logger.debug("Saving dataset to path: %s", save_path)
    try:
        dataset.to_csv(save_path, index=False)
        logger.info("Dataset saved successfully.")
    except Exception as e:
        logger.error("Error occurred while trying to save dataset: %s", e)
        sys.exit(1)

