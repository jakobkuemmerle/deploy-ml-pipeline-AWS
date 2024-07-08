# Assignment2: Cloud - Jakob Kuemmerle

### Use Case Description

In this project we want to deploy a Machine Learning model into production.
Anyone with access to the code can use this model to classify clouds in the sky into two distinct cloud types.
The user can define specifications for the data acquisition, data processing, model training and evaluation to find the most suitable configuration for their dataset. Extensive logging functionality, unit tests and configurations to run the code inside of docker containers facilitate the use.

## Instructions to execute ML model and Tests in Docker:

The dockerfiles specify the installation of the requirements.txt file. Therefore, no separate installation is necessary.
To fetch data the user has to specify the path to a online directory from where data should be acquired.
Paths to specify the location of the desired data can be adjusted in the configuration files.
The s3 bucket needs to also be specified in the config file, more information is available under the AWS section in README.
It is assumed that the library code of the project changes more frequently than the requirements.txt, therefore the dockerfiles are optimised to leverage caching functionality by first copying only the requirements and subsequently the other files to the container.

### Build the Docker image

Build the Docker image for the model pipeline:
```bash
docker build -t cloud_hw2_pipeline -f dockerfiles/Dockerfile.pipeline .
```

### Run the entire model pipeline
Run the entire model pipeline using the built Docker image.
Store artifacts locally as well as in AWS S3 bucket:
```bash
docker run -v ${HOME}/.aws/:/root/.aws/:ro -v $(pwd)/runs:/app/runs -v $(pwd)/logs:/app/logs cloud_hw2_pipeline
```

### Build the Docker image for tests
Build the Docker image for running tests:
```bash
docker build -t cloud_hw2_test -f dockerfiles/Dockerfile.test . 
```

### Run the tests
Run the tests using the built Docker image.
Store log file of tests locally:
```bash
docker run -v $(pwd)/logs:/app/logs cloud_hw2_test
```

## Project Overview

The following tree structure describes the project:
```bash
├── README.md
├── config
│   ├── config.yaml
│   ├── logging.conf
│   ├── logging_test.conf
│   └── requirements.txt
├── dockerfiles
│   ├── Dockerfile.pipeline
│   └── Dockerfile.test
├── logs
│   ├── pipeline.log
│   └── test.log
├── pipeline_log.py
├── runs
│   └── timestamp
├── src
│   ├── __init__.py
│   ├── acquire_data.py
│   ├── analysis.py
│   ├── aws_utils.py
│   ├── create_dataset.py
│   ├── evaluate_performance.py
│   ├── generate_features.py
│   ├── score_model.py
│   └── train_model.py
└── tests
    ├── __init__.py
    └── test_generate_features.py
```

- in the config directory can all configurations be defined that are necessary to run tests as well as the pipeline to train and evaluate the model or specify logging activities. 
- two dockerfiles are defined to seperate test and prod.
- logs on information, debug messages, error code and warnings are stored in the logs directory, one file for pipeline and one for testing purposes.
- all relevant artifacts created during the machine learning process are stored under the runs directory. Each run automatically creates a subdirectory with the current timestamp.
- the scr directory comprises all individual moduls that contain the functions to execute the pipeline.
- the test directory includes python files to execute all necessary unit tests.
- pipeline.py calls all src modules and their functions and executes the entire ML process with one command.

## Pipeline Functionality

The intend of the pipeline is to orchestrate the individual modules and their functions. The entire pipeline can be executed with one single command. The pipeline calls each function using the specified arguments and parameter. Each function stores its progress as an artifacts and saves it to the local file specified. The next function takes the artifacts as input. This chain continues until every step is completed. The pipeline gives us an overview of the structure and serves as the backbone; new functions can be implemented depending on the user requirements.

Functionality
The script performs the following steps:

Setting up Logging: The script initializes logging configuration from a file and prints the directory where logs will be saved. It starts a new logging session and logs each step of the pipeline for traceability and debugging.

Main Function (main): The main function orchestrates the data processing pipeline. It:
- Parses command-line arguments to get the path to the configuration file.
- Loads the configuration file to retrieve parameters and run configuration.
- Sets up an output directory for saving artifacts (e.g., trained model, generated features, evaluation metrics).
- Acquires data from an online repository and saves it to disk.
- Creates a structured dataset from the raw data and saves it to disk.
- Generates features from the dataset and saves them to disk.
- Performs exploratory data analysis (EDA) and saves figures.
- Splits the data into training and testing sets.
- Trains a machine learning model on the training set and saves the trained model.
- Scores the model on the test set and saves the scores.
- Evaluates model performance metrics and saves the metrics.
- Logs the completion of the pipeline.

Error Handling: The script includes error handling to catch any exceptions that occur during pipeline execution. If an exception occurs, it logs an error message with details of the exception for debugging purposes.

## Unit tests

The provided unit tests validate the functionality of the generate_features module in the project. These tests cover various scenarios to ensure the correctness and robustness of the feature generation process.

#### Happy Path Tests

Happy Path Test for Expected Columns: This test verifies that the generated features contain all the expected columns based on the feature configuration.

Happy Path Test for Calculate Normalized Range: This test checks the correctness of the calculation of the normalized range feature.

Happy Path Test for Log Transformation: This test validates the log transformation feature by ensuring the generated log-transformed feature matches the expected values.

#### Unhappy Path Tests

Unhappy Path Test for Invalid Feature: This test ensures that an error is raised when an invalid feature is specified in the feature configuration.

Unhappy Path Test for Calculate Normalized Range: This test checks that an error is raised when data with missing values is passed to the calculate_norm_range function.

Unhappy Path Test for Log Transformation: This test verifies that an error is raised when a non-existent column is specified for log transformation.

#### Additional Tests

Test for Missing Columns in Generate Features: This test ensures that an error is raised when the feature configuration contains columns that do not exist in the dataset.

Test for Conflicting Names in Generate Features: This test checks the behavior when a new feature conflicts with an existing column name in the dataset.

Test for Log Transformation Feature: These tests validate the log transformation feature by checking both happy and unhappy path scenarios.

These unit tests are essential for maintaining the integrity of the feature generation process and catching potential errors or discrepancies in the data processing pipeline.

## Logging

#### Logging Configuration:
Logging is configured for both the pipeline and unit tests.
Configuration for the pipeline logging is specified in config/logging.config, while the configuration for unit test logging is in config/logging_test.config.
The user can specify the desired log level in these configuration files.

#### Log Levels:
Four log levels are used: DEBUG, INFO, WARNING, and ERROR.
Each log level provides different levels of detail about the execution flow and potential issues.

#### Output:
Logs are written to log files for both the pipeline and unit tests.
Logs are stored locally and in s3 artifacts
Additionally, relevant high-level information is displayed on the console during pipeline execution.

#### Exception Handling:
Logging is integrated into exception handling mechanisms.
Errors and exceptions are logged with appropriate severity levels to provide insight into any encountered issues during execution.

#### Summary:
The logging setup in the project ensures detailed tracking of the execution flow, potential issues, and errors encountered during both pipeline execution and unit testing. The use of different log levels allows for fine-grained control over the verbosity of logging output. Users can configure the desired log level according to their preference or debugging needs.

## AWS
The user can specify their AWS bucket and folder name in the config/config.yaml file.
For each run a new folder in s3 is created with the respective timestamp. This helps debug and track progress over time.
Credentials:
The user has to be logged in with the default user and that user must have access to the s3 bucket.