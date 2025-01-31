from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import joblib
from dagster import asset

@asset
def preprocess_data(context, df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Preprocess raw data with scaling and cleaning."""
    
    # You can access loggers via context for logging
    context.log.info("Starting preprocessing")

    scaler = StandardScaler()
    numerical_features = ['mass', 'width', 'height', 'color_score']
    df[numerical_features] = scaler.fit_transform(df[numerical_features])
    
    # Save the scaler
    joblib.dump(scaler, config['paths']['scaler_save_path'])
    
    context.log.info("Preprocessing completed successfully")
    return df
