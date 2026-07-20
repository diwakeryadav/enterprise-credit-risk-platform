import sys
from pathlib import Path

import pandas as pd

from src.entity.config_entity import DataValidationConfig
from src.exception import CreditRiskException
from src.logger import logger


class DataDriftDetector:
    def __init__(
        self,
        config: DataValidationConfig,
        drift_threshold: float = 0.1,
        p_value: float = 0.05,
    ):
        self.config = config
        self.drift_threshold = drift_threshold
        self.p_value = p_value

    def detect_drift(
        self, reference_df: pd.DataFrame, current_df: pd.DataFrame
    ) -> bool:
        """
        Runs Evidently AI to compute feature drift between a reference dataset and live inference data.
        Returns:
            bool: True if drift percentage exceeds threshold (indicating major drift), False otherwise.
        """
        try:
            from evidently.metric_preset import DataDriftPreset
            from evidently.report import Report

            logger.info("Initializing Evidently AI Data Drift report")

            # Select common columns (excluding target column if present)
            common_cols = list(set(reference_df.columns) & set(current_df.columns))
            if "TARGET" in common_cols:
                common_cols.remove("TARGET")

            ref_data = reference_df[common_cols]
            curr_data = current_df[common_cols]

            drift_report = Report(
                metrics=[DataDriftPreset(stattest_threshold=self.p_value)]
            )

            logger.info("Calculating drift metrics...")
            drift_report.run(reference_data=ref_data, current_data=curr_data)

            report_dict = drift_report.dict()

            # Extract summary statistics
            metrics = report_dict["metrics"][0]["result"]
            number_of_columns = metrics["number_of_columns"]
            number_of_drifted_columns = metrics["number_of_drifted_columns"]
            share_of_drifted_columns = metrics["share_of_drifted_columns"]
            dataset_drift = metrics["dataset_drift"]

            logger.info(
                f"Feature Drift Result: Out of {number_of_columns} features, "
                f"{number_of_drifted_columns} have drifted ({share_of_drifted_columns:.2%}). "
                f"Dataset-level drift detected: {dataset_drift}"
            )

            # Save report to artifacts for auditability
            report_output_path = (
                Path(self.config.validation_report_dir)
                / self.config.drift_report_file_name
            )
            drift_report.save_json(str(report_output_path))
            logger.info(f"Evidently AI JSON Drift Report saved to {report_output_path}")

            # Check thresholds and trigger alerts
            if share_of_drifted_columns >= self.drift_threshold:
                logger.warning(
                    f"[ALERT] High Feature Drift Detected! "
                    f"Drifted feature share ({share_of_drifted_columns:.2%}) "
                    f"exceeds the maximum threshold of ({self.drift_threshold:.2%})."
                )
                return True

            return False

        except Exception as e:
            logger.error("Error during drift detection execution")
            raise CreditRiskException(e, sys)
