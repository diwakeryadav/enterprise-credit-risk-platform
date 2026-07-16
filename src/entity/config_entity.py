from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class TrainingPipelineConfig:
    """
    Configuration for the root training pipeline directory structure.
    """
    artifacts_dir: Path


@dataclass(frozen=True)
class DataIngestionConfig:
    """
    Configuration for the data ingestion stage.
    """
    source_data_path: Path
    raw_data_dir: Path
    ingested_dir: Path
    train_file_name: str
    test_file_name: str
    test_size: float
    random_state: int


@dataclass(frozen=True)
class DataValidationConfig:
    """
    Configuration for the data validation stage.
    """
    validation_report_dir: Path
    drift_report_file_name: str
    schema_file_path: Path


@dataclass(frozen=True)
class DataTransformationConfig:
    """
    Configuration for the data transformation stage.
    """
    transformed_dir: Path
    transformed_train_file_name: str
    transformed_test_file_name: str
    preprocessing_pipeline_file_name: str


@dataclass(frozen=True)
class ModelTrainerConfig:
    """
    Configuration for the model training stage.
    """
    model_dir: Path
    model_file_name: str
    expected_score_threshold: float
    overfitting_threshold: float


@dataclass(frozen=True)
class ModelEvaluationConfig:
    """
    Configuration for the model evaluation stage.
    """
    evaluation_report_dir: Path
    mlflow_tracking_uri: str
    mlflow_experiment_name: str
    target_column: str
