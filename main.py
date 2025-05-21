import os
import yaml
from src.data_retrieval.nasa_api import NASADataRetriever
from src.preprocessing.data_cleaner import DataCleaner
from src.modeling.rnn_model import ModelTrainer
from src.insights.web_interface import WebInterface
from src.region_selection.coordinate_handler import CoordinateHandler

def main():
    # Load configuration
    with open('config/config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    # Initialize components
    print("Selecting region...")
    coord_handler = CoordinateHandler()
    region = coord_handler.select_region(
        lat=12.97,  # Bengaluru latitude
        lon=77.59,  # Bengaluru longitude
        radius_km=50  # 50km radius around Bengaluru
    )
    nasa_api = NASADataRetriever()
    
    # Get all available parameters
    parameters = nasa_api.get_available_parameters()
    print(f"Processing parameters: {parameters}")
    
    # Retrieve and preprocess all data
    print("Preprocessing data...")
    data_cleaner = DataCleaner()
    start_date = config['data']['temporal']['default_start_date']
    end_date = config['data']['temporal']['default_end_date']
    
    # Get and clean all parameter data
    raw_data = nasa_api.get_environmental_data(region, parameters, start_date=start_date, end_date=end_date)
    cleaned_data = {}
    for param in parameters:
        cleaned_data[param] = data_cleaner.preprocess_dataset(raw_data[param], param)
    
    # Train models and make predictions for each parameter
    print("Training predictive models...")
    input_size = 100 * 100  # Assuming a 100x100 grid for spatial dimensions
    hidden_size = config['model']['rnn']['hidden_size']
    num_layers = config['model']['rnn']['num_layers']
    output_size = input_size  # Match output size to input size for prediction
    
    predictions = {}
    for param in parameters:
        print(f"\n=== Processing {param} ===")
        model_trainer = ModelTrainer(input_size, hidden_size, num_layers, output_size)
        
        # Prepare data for training
        sequence_length = config['model']['rnn']['sequence_length']
        X, y = model_trainer.prepare_data(cleaned_data[param], sequence_length, param)

        # Train the model
        epochs = config['model']['rnn']['epochs']
        model_trainer.train(X, y, epochs)

        # Make predictions
        prediction_horizon = config['model']['rnn']['prediction_horizon']
        predictions[param] = model_trainer.predict(X, prediction_horizon)
    
    print("Starting web interface...")
    web_interface = WebInterface(cleaned_data, predictions)
    web_interface.run(debug=False, port=5000)

if __name__ == "__main__":
    main() 