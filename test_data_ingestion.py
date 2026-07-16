from pathlib import Path

from src.components.data_ingestion import(
    DataIngestion,
    DataIngestionConfig,
)

config = DataIngestionConfig(
    source_data_path = Path(r"C:\Users\diwak\Downloads\application_train.csv"),
    raw_data_path =Path("data/raw"),
)

ingestion = DataIngestion(config)

dataset_path = ingestion.copy_dataset()
print(dataset_path)