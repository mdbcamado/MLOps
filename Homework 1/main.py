import logging
import subprocess
from src.utils.config import load_config
from src.data.load_data import load_raw_data
from src.data.preprocess import preprocess_data
from src.features.build_features import create_features
from src.models.train_model import train_model
from src.models.evaluate_model import evaluate_model
from src.models.predict import predict_fruit
import joblib

def main():
    config = load_config('configs/training_config.yaml')
    logging.basicConfig(level=logging.INFO, filename=config['paths']['log_path'], 
                    format='%(asctime)s - %(levelname)s - %(message)s')
    logging.info("Starting the data pipeline.")

    logging.info("Loading raw data.")
    df = load_raw_data(config['data']['raw_data_path'])
    print("\n \nSample DF contents")
    print(df.head(2))

    logging.info("Preprocessing data.")
    df_processed = preprocess_data(df, config)
    print("\n \nProcessed DF Contents")
    print(df_processed.head(2))

    logging.info("Creating engineered features.")
    df_features = create_features(df_processed)
    print("\n \nTransformed DF Contents")
    print(df_features.head(2))

    X = df_features[config['features']['numerical']]
    y = df_features['fruit_label']
    print("\n \nSample Input Features")
    print(X.head(2))
    print("\n \nSample Labels")
    print(y.head(2))
    
    logging.info("Training the model.")
    model, _ = train_model(X, y, config)
    joblib.dump(model, config['paths']['model_save_path'])
    print(f"\n \nModel saved at {config['paths']['model_save_path']}")

    logging.info("Evaluating the model.")
    metrics = evaluate_model(model, _['X_test'], _['y_test'], save_path=config['paths']['performance_path'])
    
    print(f"\n \nModel saved at {config['paths']['performance_path']}")
    
    
    logging.info("Running Streamlit app.")
    subprocess.run(["streamlit", "run", "deployment/streamlit/app.py"])

    logging.info("Pipeline execution completed.")
    print("\n \nExecuted successfully.")
    
    
if __name__ == "__main__":
    main()
