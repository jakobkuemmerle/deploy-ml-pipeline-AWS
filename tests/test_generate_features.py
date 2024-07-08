import logging
import logging.config
import numpy as np
import pandas as pd
import pytest
from src import generate_features as gf

# Load logging configuration from file
logging.config.fileConfig("config/logging_test.conf")

# Log a separator to indicate the start of a new logging session
logging.info("========================================")
logging.info("New test-logging session started")
logging.info("========================================")

# Set up logging for testing
logger = logging.getLogger(__name__)
logger.info("Test Starts")

# Fixture for sample data
@pytest.fixture
def sample_data():
    """
    Fixture for sample data.
    """
    logger.debug("Creating sample data fixture")
    return pd.DataFrame({
        "A": [1, 2, 3],
        "B_min": [4, 5, 6],
        "B_max": [7, 8, 9],
        "C_min": [10, 11, 12],
        "C_max": [13, 14, 15],
        "C_mean": [16, 17, 18],
        "D": [5, 3, 9]
    })

# Fixture for sample data with conflicting names
@pytest.fixture
def sample_data_2():
    """
    Fixture for sample data with conflicting names.
    """
    logger.debug("Creating sample data 2 fixture")
    return pd.DataFrame({
        "A": [1, 2, 3],
        "B_min": [4, 5, 6],
        "B_max": [7, 8, 9],
        "C_min": [10, 11, 12],
        "C_max": [13, 14, 15],
        "C_mean": [16, 17, 18],
        "D": [5, 3, 9],
        "A_x_D": [5, 3, 9],
    })

# Fixture for feature configuration
@pytest.fixture
def feature_config():
    """
    Fixture for feature configuration.
    """
    logger.debug("Creating feature configuration fixture")
    return {
        "calculate_range": ["B", "C"],
        "calculate_norm_range": ["C"],
        "log_transform": ["B_range"],
        "multiply": ["A", "D"]
    }

# Happy path test for generate_features
def test_generate_features_happy_path(sample_data, feature_config):
    """
    Happy path test for generate_features.
    """
    logger.debug("Running happy path test for expected columns")
    expected_columns = [
        "A", "B_min", "B_max", "C_min", "C_max", "C_mean", "D",
        "B_range", "C_range", "C_norm_range", "log_B_range", "A_x_D"
    ]
    result = gf.generate_features(sample_data, feature_config)

    # Check if all expected columns are present in the result
    missing_columns = [col for col in expected_columns if col not in result.columns]
    if missing_columns:
        logger.error("The following expected columns are missing in the result: %s", missing_columns)
        pytest.fail("Expected columns are missing in the result")

    # Check if there are any extra columns in the result that are not in the expected columns
    extra_columns = [col for col in result.columns if col not in expected_columns]
    if extra_columns:
        logger.error("The following extra columns are present in the result: %s", extra_columns)
        pytest.fail("Extra columns are present in the result")

    logger.info("Happy path test for expected columns successful")

# Unhappy path test for generate_features
def test_generate_features_unhappy_path(sample_data):
    """
    Unhappy path test for generate_features.
    """
    logger.debug("Running unhappy path test for invalid feature")
    try:
        feature_config_invalid = {
            "invalid_feature": ["B"]
        }
        with pytest.raises(KeyError):
            gf.generate_features(sample_data, feature_config_invalid)
        logger.info("Unhappy path test for invalid feature successful")
    except AssertionError:
        logger.error("Unhappy path test for invalid feature failed")
        raise

# Happy path test for calculate_norm_range
def test_calculate_norm_range_happy_path(sample_data):
    """
    Happy path test for calculate_norm_range.
    """
    logger.debug("Running happy path test for calculate_norm_range")
    try:
        result = gf.calculate_norm_range(sample_data, "C_min", "C_max", "C_mean")
        expected_result = (sample_data["C_max"] - sample_data["C_min"]) / sample_data["C_mean"]
        pd.testing.assert_series_equal(expected_result, result)
        logger.info("Happy path test for calculate_norm_range successful")
    except AssertionError:
        logger.error("Happy path test for calculate_norm_range failed")
        raise

# Unhappy path test for calculate_norm_range
def test_calculate_norm_range_unhappy_path(sample_data):
    logger.debug("Running unhappy path test for calculate_norm_range")
    data_invalid = pd.DataFrame({
        "A": [1, 2, 3],
        "B_min": [4, 5, 6],
        "B_max": [7, 8, 9],
        "C_min": [10, 11, None],  # Missing one value
        "C_max": [13, 14, 15],
        "C_mean": [16, 17, 18]
    })
    try:
        with pytest.raises(ValueError):
            gf.calculate_norm_range(data_invalid, "C_min", "C_max", "C_mean")
        logger.info("Unhappy path test for calculate_norm_range successful")
    except ValueError:
        logger.error("Unhappy path test for calculate_norm_range failed: ValueError not raised")
        raise

# Fixture for missing columns in feature configuration
@pytest.fixture
def feature_config_missing_columns():
    """
    Fixture for missing columns in feature configuration.
    """
    logger.debug("Creating feature configuration missing columns fixture")
    return {
        "calculate_range": ["B", "D"],  # 'D_max' and 'D_min' are missing
        "calculate_norm_range": ["C"],
        "log_transform": ["A"],
        "multiply": ["A", "B"]
    }


# Fixture for conflicting names in feature configuration
@pytest.fixture
def feature_config_conflicting_names():
    """
    Fixture for conflicting names in feature configuration.
    """
    logger.debug("Creating feature configuration conflicting names fixture")
    return {
        "multiply": ["A", "D"]  # 'A_x_D' conflicts with existing 'A_x_D' column
    }

# Test for missing columns in generate_features
def test_generate_features_missing_columns(sample_data, feature_config_missing_columns):
    """
    Test for missing columns in generate_features.
    """
    logger.debug("Running test for missing columns in generate_features")
    with pytest.raises(ValueError):
        gf.generate_features(sample_data, feature_config_missing_columns)
    logger.info("Test for missing columns in generate_features successful")

# Test for conflicting names in generate_features
def test_generate_features_conflicting_names(sample_data_2, feature_config_conflicting_names, caplog):
    """
    Test for conflicting names in generate_features.
    """
    logger.info("Running test for conflicting names in generate_features")
    with caplog.at_level(logging.DEBUG):
        result = gf.generate_features(sample_data_2, feature_config_conflicting_names)
    assert "New feature 'A_x_D' already exists in the DataFrame. Will be overwritten"
    assert "A_x_D" in result.columns  # Verify that the feature is still created despite the warning
    if "A_x_D" not in result.columns:
        logger.error("Test for conflicting names in generate_features failed: 'A_x_D' column not found in result")
        raise AssertionError("Test for conflicting names in generate_features failed: 'A_x_D' column not found in result")
    logger.info("Test for conflicting names in generate_features successful")

# Fixture for log transformation feature configuration
@pytest.fixture
def feature_config_log():
    """
    Fixture for log transformation feature configuration.
    """
    logger.debug("Creating log transformation feature configuration fixture")
    return {
        "log_transform": ["A"]
    }

# Happy path test for log transformation in generate_features
def test_generate_features_log_happy_path(sample_data, feature_config_log):
    """
    Happy path test for log transformation in generate_features.
    """
    logger.info("Running happy path test for log transformation in generate_features")
    try:
        result = gf.generate_features(sample_data, feature_config_log)
        assert "log_A" in result.columns, "Column 'log_A' not found in result columns"
        assert np.array_equal(result["log_A"], np.log(sample_data["A"])), "Values in 'log_A' do not match expected log values"
        logger.info("Happy path test for log transformation in generate_features successful")
    except AssertionError as e:
        logger.error("Happy path test for log transformation in generate_features failed: %s", e)
        raise

# Unhappy path test for log transformation in generate_features
def test_generate_features_log_unhappy_path(sample_data):
    """
    Unhappy path test for log transformation in generate_features.
    """
    logger.debug("Running unhappy path test for log transformation in generate_features")
    feature_config_invalid = {
        "log_transform": ["E"]  # 'E' column does not exist
    }
    try:
        gf.generate_features(sample_data, feature_config_invalid)
        logger.error("Unhappy path test for log transformation in generate_features failed")
    except ValueError:
        logger.info("Unhappy path test for log transformation in generate_features successful")
