import yaml
from dagster import asset

@asset
def load_config(context, file_path="configs/training_config.yaml"):
    """Load configuration from a YAML file."""
    with open(file_path, 'r') as file:
        config = yaml.safe_load(file)
    
    # You can use context for logging purposes if needed
    context.log.info(f"Config loaded from {file_path}")
    
    return config
