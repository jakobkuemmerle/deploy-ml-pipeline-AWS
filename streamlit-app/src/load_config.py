import yaml
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_config(file_path):
    """
    Load the configuration file.

    Parameters:
        file_path (str): The path to the configuration file.
    
    Returns:
        dict: The configuration dictionary, or None if an error occurs.
    """
    try:
        with open(file_path, 'r') as yaml_file:
            config = yaml.safe_load(yaml_file)
            logger.info("Configuration file loaded successfully.")
        return config
    except Exception as err:
        logger.error(f"Error loading configuration: {err}")
        return None
