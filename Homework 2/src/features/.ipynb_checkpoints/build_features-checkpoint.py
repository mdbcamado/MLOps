import pandas as pd
import numpy as np
from dagster import asset

@asset
def create_features(context, df: pd.DataFrame) -> pd.DataFrame:
    """Create engineered features for the fruit dataset."""
    df['area'] = df['width'] * df['height']
    df['density'] = df['mass'] / df['area']
    df['aspect_ratio'] = df['height'] / df['width']
    context.log.info("Engineered features created.")
    return df
