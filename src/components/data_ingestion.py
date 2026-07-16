from dataclasses import dataclass
from pathlib import Path
import shutil
import sys
from src.logger import logger
from src.exception import CreditRiskException


@dataclass
class DataIngestionConfig:
    source_data_path : Path
    raw_data_path : Path


class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config

    def copy_dataset(self):
        logger.info("starting data ingestion process.")

        try:
            self.config.raw_data_path.mkdir(
                parents =True,
                exist_ok = True
            )
            destination = (
                self.config.raw_data_path /
                self.config.source_data_path.name
            )
            shutil.copy(
                self.config.source_data_path,
                destination
            )
            logger.info("Data Ingestion completed successfully.")
            return destination
        except Exception as e:
            logger.error("Data Ingestion Failed")
            raise CreditRiskException(e, sys)