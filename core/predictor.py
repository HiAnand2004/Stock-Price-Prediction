import os
import sys
import numpy as np
import pandas as pd

# Django setup for data access if needed
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'stock_predictor_project.settings')
import django
django.setup()

# Import model loading functions instead of training functions directly
from core.model_train import prepare_data, load_model, load_scaler

def load_latest_data(sequence_length=10):
    """
    Loads the latest sequence of data needed for prediction.
    It prepares the data and returns the last sequence and the scaler used.
    """
    X, y, scaler = prepare_data(sequence_length=sequence_length)
    if X.size == 0 or scaler is None:
        print("Error: No data available or scaler not loaded for prediction.")
        return None, None
    # Get the most recent sequence for prediction
    latest_X = X[-1:]
    return latest_X, scaler

def predict_next_close(model, latest_X, scaler, model_type='lstm'):
    """
    Makes a prediction using the given model and inverse transforms the result.
    Args:
        model: The loaded trained model.
        latest_X (numpy.array): The latest sequence of features for prediction.
        scaler (MinMaxScaler): The scaler used during training.
        model_type (str): 'lstm', 'gru', or 'xgboost'.
    Returns:
        float: The predicted next close price.
    """
    if model is None or latest_X is None or scaler is None:
        print("Error: Model, data, or scaler is missing for prediction.")
        return np.nan # Return Not a Number for failed prediction

    # Check input for NaN/Inf
    if np.isnan(latest_X).any() or np.isinf(latest_X).any():
        print("Error: Input to model contains NaN or Inf!")
        return np.nan

    if model_type == 'xgboost':
        # XGBoost expects a 2D array (samples, features)
        pred = model.predict(latest_X.reshape(latest_X.shape[0], -1))
    else:
        # Keras models (LSTM, GRU) expect 3D array (samples, sequence_length, features)
        pred = model.predict(latest_X)

    # Check output for NaN/Inf
    if np.isnan(pred).any() or np.isinf(pred).any():
        print("Error: Model prediction is NaN or Inf!")
        return np.nan

    # Inverse transform to get actual close price
    # The scaler expects the full feature set (number of original features),
    # so we create a dummy array and place the predicted 'Close' value into its original position.
    
    # The number of features in the scaler can be obtained from scaler.n_features_in_
    # Ensure this matches the number of features used in prepare_data
    num_features = scaler.n_features_in_ 
    dummy = np.zeros((1, num_features))
    
    # Place the predicted close in the correct index
    # This requires knowing the index of 'Close' feature when the scaler was fit.
    # From prepare_data, we know features = ['Open', 'High', 'Low', 'Close', 'Volume', 'sentiment_score']
    # So 'Close' is at index 3 (0-indexed).
    close_idx = 3 
    dummy[0, close_idx] = float(pred.flatten()[0])

    inv = scaler.inverse_transform(dummy)
    predicted_close = inv[0, close_idx]
    
    return predicted_close

if __name__ == "__main__":
    # This block is for testing the predictor script directly.
    # In a real Django setup, views.py will call these functions.

    sequence_length = 10
    
    # Ensure models are trained and saved by running core/model_train.py once before this.
    print("Attempting to load latest data and models...")
    latest_X, scaler = load_latest_data(sequence_length=sequence_length)
    lstm_model = load_model('lstm')
    gru_model = load_model('gru')
    xgb_model = load_model('xgboost')

    if latest_X is not None and scaler is not None:
        if lstm_model:
            lstm_pred = predict_next_close(lstm_model, latest_X, scaler, model_type='lstm')
            print(f"Predicted next Close price (LSTM): {lstm_pred:.2f}")
        else:
            print("LSTM model not loaded, skipping prediction.")

        if gru_model:
            gru_pred = predict_next_close(gru_model, latest_X, scaler, model_type='gru')
            print(f"Predicted next Close price (GRU): {gru_pred:.2f}")
        else:
            print("GRU model not loaded, skipping prediction.")

        if xgb_model:
            xgb_pred = predict_next_close(xgb_model, latest_X, scaler, model_type='xgboost')
            print(f"Predicted next Close price (XGBoost): {xgb_pred:.2f}")
        else:
            print("XGBoost model not loaded, skipping prediction.")
    else:
        print("Could not load data or scaler. Please ensure data is available and models are trained/saved.")
        
df = pd.DataFrame()  # Assuming df is defined somewhere earlier in the actual code
if 'title' in df.columns:
    # safe to use df['title']
    print("Column 'title' found in DataFrame")
else:
    print("Column 'title' not found in DataFrame")
