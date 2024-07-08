import argparse
import datetime
import logging.config
from pathlib import Path
import os
import shutil
import yaml

import src.acquire_data as ad
import src.analysis as eda
import src.create_dataset as cd
import src.generate_features as gf
import src.train_model as tm
import src.score_model as sm
import src.evaluate_performance as ep
import src.aws_utils as aws

def setup_logging():
    """Set up logging configuration."""
    # Load logging configuration from file
    logging.config.fileConfig("config/logging.conf")

    # Get the absolute path of the logs directory
    logs_dir = os.path.abspath("logs")
    print(f"Logs directory: {logs_dir}")

    # Log a separator to indicate the start of a new logging session
    logging.info("========================================")
    logging.info("New logging session started")
    logging.info("========================================")

def main():
    """Main function to run the data processing pipeline."""
    # Set up logging
    setup_logging()
    logger = logging.getLogger("pipeline_logger")

    try:
        parser = argparse.ArgumentParser(
            description="Acquire, clean, and create features from clouds data"
        )
        parser.add_argument(
            "--config", default="config/config.yaml", help="Path to configuration file"
        )
        args = parser.parse_args()

        # Load configuration file for parameters and run config
        with open(args.config, "r") as f:
            config = yaml.load(f, Loader=yaml.FullLoader)

        run_config = config.get("run_config", {})

        # Set up output directory for saving artifacts
        now = int(datetime.datetime.now().timestamp())
        artifacts = Path(run_config.get("output", "runs")) / str(now)
        artifacts.mkdir(parents=True)

        # Save config file to artifacts directory for traceability
        with (artifacts / "config.yaml").open("w") as f:
            yaml.dump(config, f)
        logger.info("Configuration file saved to artifacts directory.")

        # Acquire data from online repository and save to disk
        ad.acquire_data(run_config["data_source"], artifacts / "clouds.data")
        logger.info("Data acquisition completed successfully.")

        # Create structured dataset from raw data
        data = cd.create_dataset(
            artifacts / "clouds.data",
            config["create_dataset"]["class_indices"],
            config["create_dataset"]["columns"])
        cd.save_dataset(data, artifacts / "clouds.csv")
        logger.info("Dataset creation completed successfully.")

        # Generate features and save to disk
        features = gf.generate_features(data, config["generate_features"])
        gf.save_enriched_dataset(features, artifacts / "enriched_clouds.csv")
        logger.info("Feature generation completed successfully.")

        # Perform exploratory data analysis and save figures
        figures = artifacts / "figures"
        figures.mkdir()
        eda.save_figures(features, figures)
        logger.info("Exploratory data analysis completed successfully.")

        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = tm.split_data(features, features["class"])

        # Train model and save trained model
        selected_features = config["train_model"]["selected_features"]
        tmo = tm.train_model(X_train=X_train, y_train=y_train, initial_features=selected_features)
        tm.save_model(tmo, artifacts / "trained_model_object.pkl")
        # Save the train and test datasets
        tm.save_data(X_train, X_test, y_train, y_test, artifacts)
        logger.info("Model training completed successfully.")

        # Score model on test set and save scores
        scores = sm.score_model(X_test, y_test, tmo, selected_features)
        #scores = sm.score_model(features, tmo, config["train_model"]["selected_features"])
        sm.save_scores(scores, artifacts / "scores.csv")
        logger.info("Model scoring completed successfully.")

        # Evaluate model performance metrics and save metrics
        evaluation_results = ep.evaluate_performance(scores, config["evaluate_performance"])
        ep.save_metrics(evaluation_results, artifacts / "metrics.yaml")
        logger.info("Model evaluation completed successfully.")

        # Copy log file to artifacts directory
        log_file_path = Path("logs/pipeline.log")
        if log_file_path.exists():
            shutil.copy(log_file_path, artifacts / "pipeline.log")
            logger.info("Log file copied to artifacts directory.")

        # Upload all artifacts to S3
        aws_config = config.get("aws")
        if aws_config.get("upload", False):
            aws.upload_artifacts(artifacts, aws_config, now)
            logger.info("Artifacts successfully uploaded to S3.")

        logger.info("Pipeline completed - logging end.")

    except Exception as e:
        logger.exception("An error occurred: %s", str(e))


if __name__ == "__main__":
    main()
