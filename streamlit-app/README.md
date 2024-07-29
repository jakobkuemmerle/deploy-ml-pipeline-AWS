# For Local

docker build --file dockerfile/Dockerfile --tag name .
docker run  -v ~/.aws:/root/.aws -e AWS_PROFILE=profile -p 8501:8501 --name cloud_streamlit name


# For AWS

- added --platform=linux/x86_64 to dockerfile

docker build --file dockerfile/Dockerfile -t your-path .

## from your ECR commands
aws ecr-public get-login-password --region us-east-1 | docker login --username AWS --password-stdin
