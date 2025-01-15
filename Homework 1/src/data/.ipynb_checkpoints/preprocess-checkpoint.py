from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """Preprocess raw data with scaling and cleaning."""
    scaler = StandardScaler()
    numerical_features = ['mass', 'width', 'height', 'color_score']
    df[numerical_features] = scaler.fit_transform(df[numerical_features])
    return df