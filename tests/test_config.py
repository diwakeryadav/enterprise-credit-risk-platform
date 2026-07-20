from pathlib import Path

def test_yaml_validation():
    # Verify configs/config.yaml and configs/schema.yaml exist
    config_path = Path("configs/config.yaml")
    schema_path = Path("configs/schema.yaml")
    assert config_path.exists()
    assert schema_path.exists()
