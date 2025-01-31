import pandas as pd
from dagster import op

from dagster import asset

@asset
def load_raw_data(context, config: dict) -> pd.DataFrame:
    """Load raw fruit dataset from file."""
    raw_data_path = config['data']['raw_data_path']
    df = pd.read_csv(raw_data_path, sep='\t')
    context.log.info(f"Loaded data from {raw_data_path}")
    return df
