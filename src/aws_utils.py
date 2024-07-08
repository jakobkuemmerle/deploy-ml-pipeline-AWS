from pathlib import Path
import logging
import boto3

# Set up logging
logger = logging.getLogger(__name__)

# Set the desired log level for boto3 and s3transfer to WARN
logging.getLogger("botocore").setLevel(logging.WARN)
logging.getLogger("boto3").setLevel(logging.WARN)
logging.getLogger("s3transfer").setLevel(logging.WARN)

def upload_artifacts(artifacts: Path, config: dict, timestamp: int) -> list[str]:
    """Upload all the artifacts in the specified directory to S3

    Args:
        artifacts: Directory containing all the artifacts from a given experiment
        config: Config required to upload artifacts to S3; see example config file for structure
        timestamp: Timestamp to use as a subfolder in S3

    Returns:
        List of S3 uri's for each file that was uploaded
    """
    logger.debug("Uploading Artifacts to S3.")
    try:
        # Check if AWS credentials are set
        session = boto3.Session()
        s3 = session.client("s3")
        if not s3:
            logger.error("s3 session could not be established. Check access keys")
            raise ValueError("s3 session could not be established. Check access keys")

        bucket_name = config.get("bucket_name")
        if not bucket_name:
            logger.error("Bucket name not specified in the config.")
            raise ValueError("Bucket name not specified in the config.")

        prefix = config.get("prefix", "")
        # Add the timestamp as a subfolder under the prefix
        prefix = f"{prefix}/{timestamp}"

        uploaded_files = []

        for file_path in artifacts.glob("**/*"):
            if file_path.is_file():
                s3_key = f"{prefix}/{file_path.relative_to(artifacts)}"
                s3.upload_file(str(file_path), bucket_name, s3_key)
                s3_uri = f"s3://{bucket_name}/{s3_key}"
                uploaded_files.append(s3_uri)

        return uploaded_files

    except ValueError as ve:
        logger.error(f"ValueError: {ve}")
        raise
    except Exception as e:
        logger.error(f"An error occurred while uploading artifacts to S3: {e}")
        raise
