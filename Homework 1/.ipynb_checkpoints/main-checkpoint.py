import logging
from src.utils.config import load_config
from src.data.load_data import load_raw_data
from src.data.preprocess import preprocess_data
from src.features.build_features import create_features
from src.models.train_model import train_model
from src.models.evaluate_model import evaluate_model
from src.models.predict import predict_fruit
import joblib

# Set up logging
config = load_config('configs/training_config.yaml')
logging.basicConfig(level=logging.INFO, filename=config['paths']['log_path'], 
                    format='%(asctime)s - %(levelname)s - %(message)s')


def main():
    logging.info("Starting the data pipeline.")
    
    # Step 1: Load raw data
    logging.info("Loading raw data.")
    df = load_raw_data(config['data']['raw_data_path'])

    # Step 2: Preprocess data
    logging.info("Preprocessing data.")
    df_processed = preprocess_data(df)

    # Step 3: Feature engineering
    logging.info("Creating engineered features.")
    df_features = create_features(df_processed)

    # Step 4: Check the columns and sample data
    logging.info("Inspecting features:")
    print(df_features.head())  # Print first few rows of the dataframe
    
    # Step 5: Define features and target
    X = df_features[config['features']['numerical']]
    y = df_features['fruit_label']  # Use 'fruit_label' for target variable
    
    # Step 6: Train the model
    logging.info("Training the model.")
    model, _ = train_model(X, y, config)
    joblib.dump(model, config['paths']['model_save_path'])

    logging.info("Evaluating the model.")
    metrics = evaluate_model(model, _['X_test'], _['y_test'], save_path=config['paths']['performance_path'])

    logging.info("Pipeline execution completed.")
