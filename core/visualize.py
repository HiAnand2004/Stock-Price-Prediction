import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Django setup for data access if needed
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'stock_predictor_project.settings')
import django
django.setup()

# Import model loading functions instead of training functions directly
from core.model_train import prepare_data, load_model, load_scaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error # Needed for evaluation, if you add it back

def get_predictions(sequence_length=10):
    """
    Prepares data and loads pre-trained models to get predictions for visualization.
    Returns:
        tuple: (y_test, lstm_pred, gru_pred, xgb_pred)
    """
    X, y, scaler = prepare_data(sequence_length=sequence_length)
    if X.size == 0 or y.size == 0:
        print("Warning: No sufficient data to generate predictions for visualization.")
        return np.array([]), np.array([]), np.array([]), np.array([]) # Return empty arrays

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    # Load pre-trained models
    lstm_model = load_model('lstm')
    gru_model = load_model('gru')
    xgb_model = load_model('xgboost')

    lstm_pred = np.array([])
    gru_pred = np.array([])
    xgb_pred = np.array([])

    if lstm_model:
        lstm_pred = lstm_model.predict(X_test).flatten()
    else:
        print("Warning: LSTM model not loaded. Skipping LSTM prediction.")

    if gru_model:
        gru_pred = gru_model.predict(X_test).flatten()
    else:
        print("Warning: GRU model not loaded. Skipping GRU prediction.")

    if xgb_model:
        # XGBoost expects 2D array (samples, features)
        xgb_pred = xgb_model.predict(X_test.reshape(X_test.shape[0], -1)).flatten()
    else:
        print("Warning: XGBoost model not loaded. Skipping XGBoost prediction.")

    return y_test, lstm_pred, gru_pred, xgb_pred

def plot_predictions(y_test, lstm_pred, gru_pred, xgb_pred):
    """
    Plots actual vs. predicted stock close prices.
    Saves the plot as a PNG image in the static directory.
    """
    if y_test.size == 0:
        print("No data to plot.")
        return

    plt.figure(figsize=(14, 7))
    plt.plot(y_test, label='Actual Close', color='black')
    
    if lstm_pred.size > 0:
        plt.plot(lstm_pred, label='LSTM Prediction', linestyle='--', color='blue', alpha=0.7)
    if gru_pred.size > 0:
        plt.plot(gru_pred, label='GRU Prediction', linestyle='--', color='green', alpha=0.7)
    if xgb_pred.size > 0:
        plt.plot(xgb_pred, label='XGBoost Prediction', linestyle='--', color='red', alpha=0.7)
        
    plt.title('Stock Close Price Prediction', fontsize=16)
    plt.xlabel('Time Step', fontsize=12)
    plt.ylabel('Normalized Close Price', fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()

    # Define the path to save the plot in your static directory
    static_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'static', 'core')
    os.makedirs(static_dir, exist_ok=True) # Ensure the directory exists
    plot_path = os.path.join(static_dir, 'prediction_plot.png')
    
    plt.savefig(plot_path)
    plt.close() # Close the plot to free up memory
    print(f"Plot saved to: {plot_path}")
    return plot_path # Return the path where the plot was saved

if __name__ == "__main__":
    # Example usage for generating and saving a plot locally.
    # Remember to run core/model_train.py first to save models.
    print("Generating predictions for visualization...")
    y_test, lstm_pred, gru_pred, xgb_pred = get_predictions(sequence_length=10)
    plot_predictions(y_test, lstm_pred, gru_pred, xgb_pred)

