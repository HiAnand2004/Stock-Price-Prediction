import os
import sys
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import xgboost as xgb
from keras.models import Sequential
from keras.layers import LSTM, GRU, Dense
from keras.models import load_model as keras_load_model

# Django setup for data access if needed
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'stock_predictor_project.settings')
import django
django.setup()

from core.merge_data import merge_stock_and_sentiment

# Directory to save models
model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'saved_models')
os.makedirs(model_dir, exist_ok=True)

def prepare_data(sequence_length=10):
    # Merge and clean data
    df = merge_stock_and_sentiment()
    # Standardize column names
    rename_map = {
        'Open_^NSEI': 'open',
        'High_^NSEI': 'high',
        'Low_^NSEI': 'low',
        'Close_^NSEI': 'close',
        'Volume_^NSEI': 'volume',
        'Date': 'date'
    }
    for old, new in rename_map.items():
        if old in df.columns:
            df = df.rename(columns={old: new})

    features = ['open', 'high', 'low', 'close', 'volume', 'sentiment_score']

    # Fill missing sentiment with 0, drop other NaN/Inf
    df['sentiment_score'] = df['sentiment_score'].fillna(0)
    df = df[features].replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=features)

    # Print feature std for debugging
    print("Feature std:\n", df[features].std())

    # Fit scaler on 2D data
    X = df[features].values
    scaler = MinMaxScaler()
    scaler.fit(X)
    joblib.dump(scaler, os.path.join(model_dir, 'scaler.pkl'))

    # Transform data
    X_scaled = scaler.transform(X)

    # Create sequences for LSTM/GRU
    X_seq = []
    y_seq = []
    for i in range(len(X_scaled) - sequence_length):
        X_seq.append(X_scaled[i:i+sequence_length])
        y_seq.append(X_scaled[i+sequence_length][features.index('close')])
    X_seq = np.array(X_seq)
    y_seq = np.array(y_seq)

    # Debug prints
    print("X_seq shape:", X_seq.shape)
    print("y_seq shape:", y_seq.shape)
    print("NaN in X_seq:", np.isnan(X_seq).sum())
    print("Inf in X_seq:", np.isinf(X_seq).sum())
    print("NaN in y_seq:", np.isnan(y_seq).sum())
    print("Inf in y_seq:", np.isinf(y_seq).sum())

    return X_seq, y_seq, scaler

def train_lstm(X, y, epochs=10, batch_size=32):
    try:
        if X.shape[0] > 0 and y.shape[0] > 0:
            model = Sequential([
                LSTM(32, input_shape=(X.shape[1], X.shape[2])),
                Dense(1)
            ])
            model.compile(optimizer='adam', loss='mse')
            model.fit(X, y, epochs=epochs, batch_size=batch_size)
            print("LSTM model trained.")
            return model
        else:
            print("Not enough data to train the LSTM model.")
            return None
    except Exception as e:
        print("Error during LSTM model training:", e)
        return None

def train_gru(X, y, epochs=10, batch_size=32):
    try:
        if X.shape[0] > 0 and y.shape[0] > 0:
            model = Sequential([
                GRU(32, input_shape=(X.shape[1], X.shape[2])),
                Dense(1)
            ])
            model.compile(optimizer='adam', loss='mse')
            model.fit(X, y, epochs=epochs, batch_size=batch_size)
            print("GRU model trained.")
            return model
        else:
            print("Not enough data to train the GRU model.")
            return None
    except Exception as e:
        print("Error during GRU model training:", e)
        return None

def train_xgboost(X, y):
    try:
        # Flatten 3D sequence to 2D for XGBoost
        X_flat = X.reshape(X.shape[0], -1)
        model = xgb.XGBRegressor(n_estimators=100, max_depth=3)
        model.fit(X_flat, y)
        print("XGBoost model trained.")
        return model
    except Exception as e:
        print("Error during XGBoost model training:", e)
        return None

def load_model(model_path, compile=False):
    return keras_load_model(model_path, compile=compile)

def load_scaler(scaler_path):
    import joblib
    return joblib.load(scaler_path)

def predict_with_all_models(latest_X, scaler, model_dir=None):
    """
    latest_X: shape (1, sequence_length, n_features) - the latest sequence for prediction
    scaler: fitted MinMaxScaler
    model_dir: directory where models are saved
    Returns: dict with predictions from LSTM, GRU, XGBoost
    """
    if model_dir is None:
        model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'saved_models')

    results = {}

    # LSTM
    lstm_path = os.path.join(model_dir, 'lstm_model.h5')
    if os.path.exists(lstm_path):
        lstm_model = load_model(lstm_path, compile=False)
        pred = lstm_model.predict(latest_X)
        # For LSTM and GRU
        close_idx = 3  # index of 'close' in your features
        num_features = scaler.n_features_in_
        dummy = np.zeros((1, num_features))
        dummy[0, close_idx] = pred.flatten()[0]
        inv = scaler.inverse_transform(dummy)
        results['lstm'] = round(float(inv[0, close_idx]), 2)
    else:
        results['lstm'] = None

    # GRU
    gru_path = os.path.join(model_dir, 'gru_model.h5')
    if os.path.exists(gru_path):
        gru_model = load_model(gru_path, compile=False)
        pred = gru_model.predict(latest_X)
        # For LSTM and GRU
        close_idx = 3  # index of 'close' in your features
        num_features = scaler.n_features_in_
        dummy = np.zeros((1, num_features))
        dummy[0, close_idx] = pred.flatten()[0]
        inv = scaler.inverse_transform(dummy)
        results['gru'] = round(float(inv[0, close_idx]), 2)
    else:
        results['gru'] = None

    # XGBoost
    xgb_path = os.path.join(model_dir, 'xgboost_model.pkl')
    print("Looking for XGBoost model at:", xgb_path)
    if os.path.exists(xgb_path):
        print("XGBoost model found.")
        try:
            xgb_model = joblib.load(xgb_path)
            print("XGBoost model loaded.")
            # XGBoost expects 2D input: (1, sequence_length * n_features)
            print("latest_X shape:", latest_X.shape)
            xgb_input = latest_X.reshape(1, -1)
            print("xgb_input shape:", xgb_input.shape)
            xgb_pred = xgb_model.predict(xgb_input)
            print("XGBoost raw prediction:", xgb_pred)
            # Inverse transform to get actual price
            close_idx = 3  # index of 'close'
            num_features = scaler.n_features_in_
            dummy = np.zeros((1, num_features))
            dummy[0, close_idx] = xgb_pred.flatten()[0]
            inv = scaler.inverse_transform(dummy)
            results['xgboost'] = round(float(inv[0, close_idx]), 2)
            print("XGBoost final prediction:", results['xgboost'])
        except Exception as e:
            print("XGBoost prediction error:", e)
            results['xgboost'] = None
    else:
        print("XGBoost model file does not exist.")
        results['xgboost'] = None

    return results

def main():
    print("Preparing data...")
    X, y, scaler = prepare_data(sequence_length=10)

    if X.size > 0 and y.size > 0:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

        print("\nTraining LSTM...")
        lstm_model = train_lstm(X_train, y_train)
        if lstm_model:
            lstm_model.save(os.path.join(model_dir, 'lstm_model.h5'))
            print("LSTM model saved.")

        print("\nTraining GRU...")
        gru_model = train_gru(X_train, y_train)
        if gru_model:
            gru_model.save(os.path.join(model_dir, 'gru_model.h5'))
            print("GRU model saved.")

        print("\nTraining XGBoost...")
        xgb_model = train_xgboost(X_train, y_train)
        if xgb_model:
            joblib.dump(xgb_model, os.path.join(model_dir, 'xgboost_model.pkl'))
            print("XGBoost model saved.")

        # Scaler already saved in prepare_data
    else:
        print("Skipping model training: No sufficient data to train models.")

    # Load your DataFrame with the latest data (replace this with your actual data source)
    # For example, get the last N rows from your merged DataFrame:
    features = ['open', 'high', 'low', 'close', 'volume', 'sentiment_score']
    sequence_length = 10

    df = merge_stock_and_sentiment()
    df = df.sort_values('date')
    df['sentiment_score'] = df['sentiment_score'].fillna(0)
    df = df[features].replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=features)

    # Load scaler
    from core.model_train import load_scaler
    scaler = load_scaler('core/saved_models/scaler.pkl')

    # Scale the features
    X = df[features].values
    X_scaled = scaler.transform(X)

    # Prepare the latest sequence
    if len(X_scaled) >= sequence_length:
        latest_X = X_scaled[-sequence_length:].reshape(1, sequence_length, len(features))
    else:
        # Not enough data for a full sequence
        latest_X = None

    # Now you can safely call:
    if latest_X is not None:
        from core.model_train import predict_with_all_models
        results = predict_with_all_models(latest_X, scaler)
    else:
        results = {"error": "Not enough data for prediction."}

if __name__ == "__main__":
    main()
