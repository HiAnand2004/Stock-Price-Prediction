import os
import sys
import numpy as np
import pandas as pd

# Django setup for data access if needed
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'stock_predictor_project.settings')
import django
django.setup()

from core.model_train import prepare_data, train_lstm, train_gru, train_xgboost
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def evaluate_model(model, X_test, y_test, model_type='lstm'):
    if model_type in ['lstm', 'gru']:
        preds = model.predict(X_test)
    elif model_type == 'xgboost':
        preds = model.predict(X_test.reshape(X_test.shape[0], -1))
    else:
        raise ValueError("Unknown model type")
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)
    return {'RMSE': rmse, 'MAE': mae, 'R2': r2}

def compare_models(sequence_length=10):
    X, y, scaler = prepare_data(sequence_length=sequence_length)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    print("Training LSTM...")
    lstm_model = train_lstm(X_train, y_train)
    lstm_metrics = evaluate_model(lstm_model, X_test, y_test, model_type='lstm')

    print("Training GRU...")
    gru_model = train_gru(X_train, y_train)
    gru_metrics = evaluate_model(gru_model, X_test, y_test, model_type='gru')

    print("Training XGBoost...")
    xgb_model = train_xgboost(X_train, y_train)
    xgb_metrics = evaluate_model(xgb_model, X_test, y_test, model_type='xgboost')

    results = pd.DataFrame([lstm_metrics, gru_metrics, xgb_metrics],
                           index=['LSTM', 'GRU', 'XGBoost'])
    print("\nModel Comparison:")
    print(results)
    return results

if __name__ == "__main__":
    compare_models(sequence_length=10)