import logging
import sys
import time
from pathlib import Path

import requests
from requests.exceptions import ConnectionError, Timeout, HTTPError

logger = logging.getLogger(__name__)

def acquire_data(url: str, save_path: Path) -> None:
    """Acquires data from specified URL.

    Args:
        url (str): URL from where data is to be acquired.
        save_path (Path): Local path to write data to.
    """
    url_contents = get_data(url)
    try:
        write_data(url_contents, save_path)
        logger.info("Data written to %s", save_path)
    except FileNotFoundError:
        logger.error("Please provide a valid file location to save dataset to.")
        sys.exit(1)
    except OSError as e:
        logger.error("Error occurred while trying to write dataset to file: %s", e)
        sys.exit(1)

def get_data(url: str, attempts: int = 4, wait: int = 3, wait_multiple: int = 2) -> bytes:
    """Acquires data from URL.

    Parameters:
    url (str): The URL from which to acquire the data.
    attempts (int): Number of retry attempts in case of failure (default is 4).
    wait (int): Initial waiting time between retry attempts in seconds (default is 3).
    wait_multiple (int): Factor by which the wait time is multiplied after each attempt (default is 2).

    Returns:
    bytes: The data acquired from the URL.

    Raises:
    ValueError: If the URL is invalid.
    ConnectionError: If unable to establish a connection.
    Timeout: If the request times out.
    """
    if not url.startswith("http"):
        logger.error("Invalid URL. URL must start with 'http' or 'https'.")
        raise ValueError("Invalid URL. URL must start with 'http' or 'https'.")

    for attempt in range(1, attempts + 1):
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()  # Raise error for non-200 status codes
            logger.info("Data acquired successfully from %s", url)
            return response.content
        except (ConnectionError, Timeout, HTTPError) as e:
            if attempt == attempts:
                logger.error("Failed to acquire data from %s after %d attempts.", url, attempts)
                raise e
            logger.warning("Attempt %d/%d failed. Retrying in %d seconds...", attempt, attempts, wait)
            time.sleep(wait)
            wait *= wait_multiple  # Increase wait time exponentially for next attempt
    return b""  # Explicitly return empty bytes if all attempts fail

class WriteDataError(Exception):
    """Exception raised when an error occurs while writing data to a file."""
    pass

def write_data(data: bytes, save_path: Path) -> None:
    """Writes data to the specified file path.

    Args:
        data (bytes): Data to be written.
        save_path (Path): Local path to write the data to.

    Raises:
        WriteDataError: If an error occurs while writing data to the file.
    """
    try:
        with open(save_path, "wb") as f:
            f.write(data)
    except FileNotFoundError as e:
        error_msg = f"File '{save_path}' not found."
        logger.error(error_msg)
        raise WriteDataError(error_msg) from e
    except PermissionError as e:
        error_msg = f"Permission denied to write to '{save_path}'."
        logger.error(error_msg)
        raise WriteDataError(error_msg) from e
    except IsADirectoryError as e:
        error_msg = f"'{save_path}' is a directory, cannot write data to it."
        logger.error(error_msg)
        raise WriteDataError(error_msg) from e
    except OSError as e:
        error_msg = f"Failed to write data to '{save_path}': {e}."
        logger.error(error_msg)
        raise WriteDataError(error_msg) from e
