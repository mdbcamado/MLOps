import numpy as np
from typing import List
from dagster import asset

@asset
def predict_fruit(context, model, features: np.ndarray) -> List[str]:
    """Make fruit predictions with probability scores."""
    context.log.info(f"Making predictions with model: {model}")
    
    # Make the prediction
    predictions = model.predict(features)
    
    # Calculate the probabilities
    probabilities = model.predict_proba(features)
    
    # Map predictions to fruit labels
    label_map = {1: 'apple', 2: 'mandarin', 3: 'orange', 4: 'lemon'}
    fruit_predictions = [label_map[pred] for pred in predictions]
    
    # Log the predictions and probabilities
    context.log.info(f"Predictions: {fruit_predictions}")
    context.log.info(f"Probabilities: {probabilities}")
    
    return fruit_predictions
