from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DataIngestionArtifact:
    """
    Artifact representing the outputs of the data ingestion stage.
    """
    raw_data_path: Path
    train_file_path: Path
    test_file_path: Path


@dataclass(frozen=True)
class DataValidationArtifact:
    """
    Artifact representing the outputs of the data validation stage.
    """
    validation_status: bool
    validation_report_dir: Path
    drift_report_file_path: Path


@dataclass(frozen=True)
class DataTransformationArtifact:
    """
    Artifact representing the outputs of the data transformation stage.
    """
    transformed_train_file_path: Path
    transformed_test_file_path: Path
    preprocessing_pipeline_file_path: Path


@dataclass(frozen=True)
class ModelTrainerArtifact:
    """
    Artifact representing the outputs of the model training stage.
    """
    trained_model_file_path: Path
    train_metric_score: float
    test_metric_score: float
    is_trained: bool


@dataclass(frozen=True)
class ModelEvaluationArtifact:
    """
    Artifact representing the outputs of the model evaluation stage.
    """
    is_model_accepted: bool
    improved_accuracy: float
    best_model_path: Path
    mlflow_run_id: str
