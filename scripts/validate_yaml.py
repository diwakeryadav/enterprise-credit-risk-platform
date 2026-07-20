import sys
import yaml
from pathlib import Path

def validate_config(config_path: Path, schema_path: Path):
    print(f"Validating configuration file: {config_path} against schema: {schema_path}")
    
    if not config_path.exists():
        print(f"Error: Config file not found at {config_path}")
        sys.exit(1)
    if not schema_path.exists():
        print(f"Error: Schema file not found at {schema_path}")
        sys.exit(1)

    try:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
            
        with open(schema_path, "r") as f:
            schema = yaml.safe_load(f)
            
        # Validate core keys
        for key in schema["required_keys"]:
            if key not in config:
                print(f"Validation Failure: Missing required key '{key}' in config.")
                sys.exit(1)
                
        # Validate nested data ingestion keys
        for key in schema["data_ingestion_keys"]:
            if key not in config["data_ingestion"]:
                print(f"Validation Failure: Missing sub-key '{key}' under 'data_ingestion'.")
                sys.exit(1)
                
        print("Success: Configuration YAML is valid!")
        
    except Exception as e:
        print(f"Error during parsing and validation: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    validate_config(
        config_path=Path("configs/config.yaml"),
        schema_path=Path("configs/schema.yaml")
    )
