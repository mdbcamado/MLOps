from sklearn.metrics import classification_report, confusion_matrix
import json
from dagster import asset

@asset
def evaluate_model(context, model, X_test, y_test, config: dict, save_path: str = None):
    """Evaluate model performance and save metrics."""
    # Logging the start of the evaluation process
    context.log.info("Starting model evaluation")
    
    predictions = model.predict(X_test)
    
    metrics = {
        'classification_report': classification_report(y_test, predictions, output_dict=True),
        'confusion_matrix': confusion_matrix(y_test, predictions).tolist()
    }
    
    if save_path:
        with open(save_path, 'w') as f:
            json.dump(metrics, f, indent=4)
    else:
        save_path = config['paths']['performance_path']
        with open(save_path, 'w') as f:
            json.dump(metrics, f, indent=4)

    # Logging the completion of the evaluation process
    context.log.info("Model evaluation completed successfully")
    
    return metrics
