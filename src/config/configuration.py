import sys
from pathlib import Path

from src.constants import CONFIG_FILE_PATH, SCHEMA_FILE_PATH
from src.entity.config_entity import (DataIngestionConfig,
                                      DataTransformationConfig,
                                      DataValidationConfig,
                                      ModelEvaluationConfig,
                                      ModelTrainerConfig,
                                      TrainingPipelineConfig)
from src.exception import CreditRiskException
from src.logger import logger
from src.utils import create_directories, read_yaml


class ConfigurationManager:
    """
    Manages loading, parsing, and retrieving of all pipeline configuration settings.
    """

    def __init__(
        self,
        config_filepath: Path = CONFIG_FILE_PATH,
        schema_filepath: Path = SCHEMA_FILE_PATH,
    ):
        try:
            self.config = read_yaml(config_filepath)
            self.schema_filepath = schema_filepath

            # Retrieve artifacts root and initialize it
            self.artifacts_dir = Path(self.config["artifacts_dir"])
            create_directories([self.artifacts_dir])

            logger.info("Configuration Manager initialized successfully")
        except Exception as e:
            raise CreditRiskException(e, sys)

    def get_training_pipeline_config(self) -> TrainingPipelineConfig:
        """
        Retrieves root training pipeline configurations.
        """
        try:
            return TrainingPipelineConfig(artifacts_dir=self.artifacts_dir)
        except Exception as e:
            raise CreditRiskException(e, sys)

    def get_data_ingestion_config(self) -> DataIngestionConfig:
        """
        Retrieves configurations for the Data Ingestion pipeline stage.
        """
        try:
            config_info = self.config["data_ingestion"]

            # Automatically create raw and ingestion output directories
            raw_data_dir = Path(config_info["raw_data_dir"])
            ingested_dir = Path(config_info["ingested_dir"])
            create_directories([raw_data_dir, ingested_dir])

            return DataIngestionConfig(
                source_data_path=Path(config_info["source_data_path"]),
                raw_data_dir=raw_data_dir,
                ingested_dir=ingested_dir,
                train_file_name=config_info["train_file_name"],
                test_file_name=config_info["test_file_name"],
                test_size=float(config_info["test_size"]),
                random_state=int(config_info["random_state"]),
            )
        except Exception as e:
            raise CreditRiskException(e, sys)

    def get_data_validation_config(self) -> DataValidationConfig:
        """
        Retrieves configurations for the Data Validation pipeline stage.
        """
        try:
            config_info = self.config["data_validation"]

            # Automatically create validation output directories
            validation_report_dir = Path(config_info["validation_report_dir"])
            create_directories([validation_report_dir])

            return DataValidationConfig(
                validation_report_dir=validation_report_dir,
                drift_report_file_name=config_info["drift_report_file_name"],
                schema_file_path=Path(config_info["schema_file_path"]),
            )
        except Exception as e:
            raise CreditRiskException(e, sys)

    def get_data_transformation_config(self) -> DataTransformationConfig:
        """
        Retrieves configurations for the Data Transformation pipeline stage.
        """
        try:
            config_info = self.config["data_transformation"]

            # Automatically create transformation output directories
            transformed_dir = Path(config_info["transformed_dir"])
            create_directories([transformed_dir])

            return DataTransformationConfig(
                transformed_dir=transformed_dir,
                transformed_train_file_name=config_info["transformed_train_file_name"],
                transformed_test_file_name=config_info["transformed_test_file_name"],
                preprocessing_pipeline_file_name=config_info[
                    "preprocessing_pipeline_file_name"
                ],
            )
        except Exception as e:
            raise CreditRiskException(e, sys)

    def get_model_trainer_config(self) -> ModelTrainerConfig:
        """
        Retrieves configurations for the Model Training pipeline stage.
        """
        try:
            config_info = self.config["model_trainer"]

            # Automatically create model training output directories
            model_dir = Path(config_info["model_dir"])
            create_directories([model_dir])

            return ModelTrainerConfig(
                model_dir=model_dir,
                model_file_name=config_info["model_file_name"],
                expected_score_threshold=float(config_info["expected_score_threshold"]),
                overfitting_threshold=float(config_info["overfitting_threshold"]),
            )
        except Exception as e:
            raise CreditRiskException(e, sys)

    def get_model_evaluation_config(self) -> ModelEvaluationConfig:
        """
        Retrieves configurations for the Model Evaluation pipeline stage.
        """
        try:
            config_info = self.config["model_evaluation"]

            # Automatically create model evaluation output directories
            evaluation_report_dir = Path(config_info["evaluation_report_dir"])
            create_directories([evaluation_report_dir])

            return ModelEvaluationConfig(
                evaluation_report_dir=evaluation_report_dir,
                mlflow_tracking_uri=config_info["mlflow_tracking_uri"],
                mlflow_experiment_name=config_info["mlflow_experiment_name"],
                target_column=config_info["target_column"],
            )
        except Exception as e:
            raise CreditRiskException(e, sys)
