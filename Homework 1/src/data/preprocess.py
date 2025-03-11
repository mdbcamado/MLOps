from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

# Get the absolute path to the Project directory
BASE_DIR = Path(__file__).resolve().parent.parent  # Moves up to src/

# Adjust the file path relative to diabetesProj
filepath = BASE_DIR.parent / "models/scaler.joblib"

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """Preprocess raw data with scaling and cleaning."""
    scaler = StandardScaler()
    numerical_features = ['mass', 'width', 'height', 'color_score']
    df[numerical_features] = scaler.fit_transform(df[numerical_features])
    joblib.dump(scaler, filepath)
    return df