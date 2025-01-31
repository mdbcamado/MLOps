from dagster import job
from src.data.load_data import load_raw_data
from src.data.preprocess import preprocess_data
from src.features.build_features import create_features
from src.models.train_model import train_model
from src.models.evaluate_model import evaluate_model
from src.models.predict import predict_fruit
from src.utils.config import load_config  # Import the new config solid
import joblib

@job
def fruit_classification_pipeline():
    # Step 1: Load config using the new config solid
    config = load_config()

    model_save_path = config['paths']['model_save_path']
    
    # Step 2: Load raw data
    df = load_raw_data(config)
    
    # Step 3: Preprocess the data
    processed_df = preprocess_data(df, config)
    
    # Step 4: Create features
    df_with_features = create_features(processed_df)
    
    # Step 5: Define X and y
    X = df_with_features.drop(columns=['fruit_label'])
    y = df_with_features['fruit_label']
    
    # Step 6: Train the model and save it (no need to pass the path in train_model)
    model, model_data = train_model(X, y, config)
    
    # Step 7: Make predictions by loading the saved model
    model = joblib.load(model_save_path)  # Load the saved model for prediction
    predictions = predict_fruit(model, model_data['X_test'])
    
    # Step 8: Evaluate the model
    metrics = evaluate_model(model, model_data['X_test'], model_data['y_test'], config)
    
    # Optional: Return values for further use
    return model, predictions, metrics
