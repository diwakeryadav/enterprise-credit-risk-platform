from pathlib import Path

import pandas as pd
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from src.components.data_validation import DataDriftDetector
from src.config.configuration import ConfigurationManager
from src.logger import logger

router = APIRouter()


class DriftCheckRequest(BaseModel):
    batch_data_path: str


class DriftCheckResponse(BaseModel):
    drift_detected: bool
    message: str


@router.post("/check-drift", response_model=DriftCheckResponse)
async def check_drift(request: DriftCheckRequest):
    """
    Triggers feature drift detection by comparing training baseline data
    against the provided batch data path.
    """
    try:
        config_manager = ConfigurationManager()
        validation_config = config_manager.get_data_validation_config()

        # Load Baseline Reference Data (e.g. ingested training data)
        ingestion_config = config_manager.get_data_ingestion_config()
        reference_path = (
            Path(ingestion_config.ingested_dir) / ingestion_config.train_file_name
        )

        if not reference_path.exists():
            raise HTTPException(
                status_code=400,
                detail=f"Baseline reference data not found at {reference_path}. Please ingest data first.",
            )

        current_path = Path(request.batch_data_path)
        if not current_path.exists():
            raise HTTPException(
                status_code=400,
                detail=f"Inference batch data not found at {current_path}.",
            )

        # Read datasets
        reference_df = pd.read_csv(reference_path)
        current_df = pd.read_csv(current_path)

        # Detect Drift
        drift_config = config_manager.config["monitoring"]
        detector = DataDriftDetector(
            config=validation_config,
            drift_threshold=drift_config["drift_threshold"],
            p_value=drift_config["p_value_threshold"],
        )

        drift_detected = detector.detect_drift(reference_df, current_df)

        message = (
            "Significant feature drift detected! Check drift report."
            if drift_detected
            else "Data distribution within safe thresholds."
        )
        return DriftCheckResponse(drift_detected=drift_detected, message=message)

    except Exception as e:
        logger.error(f"Error during API drift check: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Failed to perform drift check: {str(e)}"
        )


@router.post("/setup-demo")
async def setup_demo():
    """
    Creates demo baseline and batch files (both drifted and clean)
    so the user can test drift check out-of-the-box.
    """
    try:
        import numpy as np
        
        # Ensure directories exist
        baseline_dir = Path("artifacts/data_ingestion")
        baseline_dir.mkdir(parents=True, exist_ok=True)
        reference_path = baseline_dir / "train.csv"
        
        demo_dir = Path("data/demo")
        demo_dir.mkdir(parents=True, exist_ok=True)
        clean_batch_path = demo_dir / "batch_clean.csv"
        drifted_batch_path = demo_dir / "batch_drifted.csv"
        
        # Generate baseline data (100 rows)
        np.random.seed(42)
        baseline_data = {
            "EXT_SOURCE_3": np.random.uniform(0.4, 0.8, 100),
            "AMT_CREDIT": np.random.normal(500000, 100000, 100),
            "DAYS_EMPLOYED": np.random.normal(-1500, 500, 100),
            "TARGET": np.random.choice([0, 1], size=100, p=[0.9, 0.1])
        }
        df_baseline = pd.DataFrame(baseline_data)
        df_baseline.to_csv(reference_path, index=False)
        
        # Generate clean batch (no drift)
        clean_data = {
            "EXT_SOURCE_3": np.random.uniform(0.4, 0.8, 100),
            "AMT_CREDIT": np.random.normal(500000, 100000, 100),
            "DAYS_EMPLOYED": np.random.normal(-1500, 500, 100),
        }
        df_clean = pd.DataFrame(clean_data)
        df_clean.to_csv(clean_batch_path, index=False)
        
        # Generate drifted batch
        drifted_data = {
            "EXT_SOURCE_3": np.random.uniform(0.1, 0.3, 100),  # lower bureau score
            "AMT_CREDIT": np.random.normal(800000, 150000, 100),  # higher credit requested
            "DAYS_EMPLOYED": np.random.normal(-300, 100, 100),  # shorter employment duration
        }
        df_drifted = pd.DataFrame(drifted_data)
        df_drifted.to_csv(drifted_batch_path, index=False)
        
        return {
            "reference_path": str(reference_path.resolve()),
            "clean_batch_path": str(clean_batch_path.resolve()),
            "drifted_batch_path": str(drifted_batch_path.resolve()),
            "message": "Demo datasets created successfully!"
        }
    except Exception as e:
        logger.error(f"Error during setup demo data: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Failed to generate demo data: {str(e)}"
        )

