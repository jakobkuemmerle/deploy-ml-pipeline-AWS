import pytest
from unittest.mock import patch, MagicMock
from src.aws_utils import fetch_model_from_s3

@pytest.fixture
def mock_s3_client():
    with patch("boto3.client") as mock_client:
        s3 = MagicMock()
        mock_client.return_value = s3
        yield s3

@pytest.fixture
def bucket_name():
    return "test-bucket"

@pytest.fixture
def prefix():
    return "test-prefix"

@pytest.fixture
def model_name():
    return "example_model.joblib"
