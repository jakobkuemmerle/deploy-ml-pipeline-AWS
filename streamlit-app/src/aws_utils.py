import boto3
import logging
import joblib
from io import BytesIO

# Set up logging
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

def fetch_model_from_s3(bucket_name, prefix, model_name):
    """
    Retrieve the specified model from the AWS S3 bucket.
    
    Parameters:
        bucket_name (str): The name of the S3 bucket.
        prefix (str): The prefix path in the bucket.
        model_name (str): The name of the model file.
    
    Returns:
        model: The loaded model object, or None if an error occurs.
    """
    s3_client = boto3.client('s3')

    try:
        model_path = f"{prefix}/{model_name}"
        response = s3_client.get_object(Bucket=bucket_name, Key=model_path)
        log.info(f"Successfully loaded model '{model_name}' from bucket '{bucket_name}' with prefix '{prefix}'.")
        
        model_data = response['Body'].read()
        model = joblib.load(BytesIO(model_data))

        return model
    
    except Exception as e:
        log.error(f"Failed to load model '{model_name}' from S3: {e}")
        return None
