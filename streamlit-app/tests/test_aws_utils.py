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

def test_fetch_model_from_s3_failure(mock_s3_client, bucket_name, prefix, model_name, caplog):
    """Test to ensure None is returned on model loading failure."""
    mock_s3_client.get_object.side_effect = Exception('Mocked S3 error')

    model = fetch_model_from_s3(bucket_name, prefix, model_name)

    assert model is None
    assert "Failed to load model 'example_model.joblib' from S3" in caplog.text
