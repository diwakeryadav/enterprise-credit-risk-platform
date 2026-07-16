import os
import sys
import yaml
import json
import dill
from pathlib import Path
from typing import Any, List
from src.logger import logger
from src.exception import CreditRiskException


def read_yaml(path_to_yaml: Path) -> dict:
    """
    Reads a YAML file and returns its content as a dictionary.

    Args:
        path_to_yaml (Path): Absolute/relative path to the YAML file.

    Raises:
        CreditRiskException: If the file is not found or is empty.

    Returns:
        dict: Parsed content of the YAML file.
    """
    try:
        logger.info(f"Reading YAML file from path: {path_to_yaml}")
        with open(path_to_yaml, "r") as yaml_file:
            content = yaml.safe_load(yaml_file)
            if not content:
                raise ValueError("YAML file is empty")
            logger.info(f"YAML file: {path_to_yaml} loaded successfully")
            return content
    except Exception as e:
        logger.error(f"Failed to read YAML file from: {path_to_yaml}")
        raise CreditRiskException(e, sys)


def create_directories(path_to_directories: List[Path], verbose: bool = True) -> None:
    """
    Creates multiple directories if they do not exist.

    Args:
        path_to_directories (List[Path]): List of directory paths.
        verbose (bool): If True, logs directory creation messages.
    """
    try:
        for path in path_to_directories:
            if not os.path.exists(path):
                os.makedirs(path, exist_ok=True)
                if verbose:
                    logger.info(f"Created directory at: {path}")
            else:
                if verbose:
                    logger.info(f"Directory already exists at: {path}")
    except Exception as e:
        logger.error(f"Failed to create directories: {path_to_directories}")
        raise CreditRiskException(e, sys)


def save_json(path: Path, data: dict) -> None:
    """
    Saves a dictionary as a JSON file.

    Args:
        path (Path): Path to output file.
        data (dict): Dictionary to save.
    """
    try:
        logger.info(f"Saving JSON file to path: {path}")
        # Ensure directory exists
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(data, f, indent=4)
        logger.info(f"JSON file saved successfully at: {path}")
    except Exception as e:
        logger.error(f"Failed to save JSON to: {path}")
        raise CreditRiskException(e, sys)


def load_json(path: Path) -> dict:
    """
    Loads a JSON file.

    Args:
        path (Path): Path to JSON file.

    Returns:
        dict: Loaded dictionary.
    """
    try:
        logger.info(f"Loading JSON file from: {path}")
        with open(path, "r") as f:
            content = json.load(f)
        logger.info(f"JSON file loaded successfully from: {path}")
        return content
    except Exception as e:
        logger.error(f"Failed to load JSON from: {path}")
        raise CreditRiskException(e, sys)


def save_bin(file_path: Path, data: Any) -> None:
    """
    Saves binary data using dill.

    Args:
        file_path (Path): File output path.
        data (Any): Python object to serialize.
    """
    try:
        logger.info(f"Saving binary object to: {file_path}")
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, "wb") as f:
            dill.dump(data, f)
        logger.info(f"Binary object saved successfully at: {file_path}")
    except Exception as e:
        logger.error(f"Failed to save binary object to: {file_path}")
        raise CreditRiskException(e, sys)


def load_bin(file_path: Path) -> Any:
    """
    Loads serialized binary data using dill.

    Args:
        file_path (Path): Path to binary file.

    Returns:
        Any: Unpickled Python object.
    """
    try:
        logger.info(f"Loading binary object from: {file_path}")
        with open(file_path, "rb") as f:
            object_ = dill.load(f)
        logger.info(f"Binary object loaded successfully from: {file_path}")
        return object_
    except Exception as e:
        logger.error(f"Failed to load binary object from: {file_path}")
        raise CreditRiskException(e, sys)
