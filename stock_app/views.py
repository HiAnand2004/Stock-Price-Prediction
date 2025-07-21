from django.shortcuts import render, redirect
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.decorators import login_required
from django.http import JsonResponse
from .forms import RegistrationForm # Import the updated RegistrationForm
from django.contrib import messages
from .models import User

# Import functions for model loading
from core.model_train import prepare_data, train_lstm, train_gru, train_xgboost, load_model, load_scaler, predict_with_all_models

# Fix feature names to lowercase in views where features are used
FEATURES = ['open', 'high', 'low', 'close', 'volume', 'sentiment_score']

@login_required
def dashboard(request):
    # Your existing dashboard view code
    return render(request, 'core/dashboard.html')

@login_required
def prediction(request):
    predictions = predict_with_all_models()
    if predictions is None:
        return JsonResponse({'error': 'Models not trained or data unavailable'}, status=400)
    return render(request, 'core/prediction.html', {'predictions': predictions})

@login_required
def visualize(request):
    # Your existing visualize view code
    return render(request, 'core/visualize.html')

from core.predictor import load_latest_data, predict_next_close
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
from django.conf import settings

def home(request):
    """Renders the base home page."""
    return render(request, 'core/base.html')


def signup_view(request):
    """Handles user registration."""
    if request.user.is_authenticated:
        # Redirect authenticated users away from the registration page
        return redirect('home')

    if request.method == 'POST':
        form = RegistrationForm(request.POST)
        if form.is_valid():
            user = form.save()
            # Log the user in immediately after registration, or redirect to login
            login(request, user) 
            messages.success(request, f"Account created for {user.username}! You are now logged in.")
            return redirect('dashboard') # Redirect to dashboard after successful registration and login
        else:
            # Display form errors
            for field, errors in form.errors.items():
                for error in errors:
                    messages.error(request, f"{field.replace('_', ' ').title()}: {error}")
    else:
        form = RegistrationForm()
    return render(request, 'core/signup.html', {'form': form})

def login_view(request):
    """Handles user login."""
    if request.user.is_authenticated:
        # Redirect authenticated users away from the login page
        return redirect('home')

    if request.method == 'POST':
        email = request.POST.get('email')
        password = request.POST.get('password')

        # Check if email exists first for better user feedback
        if not email:
            messages.error(request, "Please enter your email.")
            return render(request, 'core/login.html')
        if not User.objects.filter(email=email).exists():
            messages.error(request, "Email not registered. Please sign up.")
            return render(request, 'core/login.html')
            
        # Authenticate user using email and password
        user = authenticate(request, email=email, password=password)
        
        if user:
            login(request, user)
            messages.success(request, f"Welcome back, {user.username}!")
            return redirect('dashboard') # Redirect to dashboard after successful login
        else:
            messages.error(request, "Invalid email or password.")
    return render(request, 'core/login.html')

def logout_view(request):
    """Logs out the current user."""
    logout(request)
    messages.info(request, "You have been logged out.")
    return redirect('home')

@login_required
def prediction_view(request):
    features = ['open', 'high', 'low', 'close', 'volume', 'sentiment_score']
    sequence_length = 10

    # Prepare latest data
    df = merge_stock_and_sentiment()
    df = df.sort_values('date')
    df['sentiment_score'] = df['sentiment_score'].fillna(0)
    df = df[features].replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=features)

    scaler = load_scaler('core/saved_models/scaler.pkl')
    X = df[features].values
    X_scaled = scaler.transform(X)

    if len(X_scaled) >= sequence_length:
        latest_X = X_scaled[-sequence_length:].reshape(1, sequence_length, len(features))
        predictions = predict_with_all_models(latest_X, scaler)
        error = None
    else:
        predictions = None
        error = "Not enough data for prediction."

    return render(request, 'core/prediction.html', {'predictions': predictions, 'error': error})

@login_required
def visualize_view(request):
    """
    Renders the visualization page.
    Note: For actual plot display, you would typically generate the plot 
    as a static image (e.g., PNG) and serve it, or use a JS charting library.
    """
    # Example data, replace with your actual data
    x = [1, 2, 3, 4, 5]
    y_actual = [10, 12, 14, 13, 15]
    y_lstm = [11, 13, 13, 14, 16]
    y_gru = [10, 12, 15, 13, 14]
    y_xgb = [9, 11, 14, 12, 15]

    plt.figure(figsize=(10, 5))
    plt.plot(x, y_actual, label='Actual')
    plt.plot(x, y_lstm, label='LSTM')
    plt.plot(x, y_gru, label='GRU')
    plt.plot(x, y_xgb, label='XGBoost')
    plt.legend()
    plt.title('Stock Price Prediction')
    plt.xlabel('Time')
    plt.ylabel('Price')

    # Ensure media directory exists
    os.makedirs(settings.MEDIA_ROOT, exist_ok=True)
    plot_path = os.path.join(settings.MEDIA_ROOT, 'plot.png')
    plt.savefig(plot_path)
    plt.close()

    plot_url = settings.MEDIA_URL + 'plot.png'
    return render(request, 'core/visualize.html', {'plot_url': plot_url})

import numpy as np
from django.shortcuts import render
from core.merge_data import merge_stock_and_sentiment
from core.model_train import predict_with_all_models, load_scaler

def predict_view(request):
    features = ['open', 'high', 'low', 'close', 'volume', 'sentiment_score']
    sequence_length = 10

    df = merge_stock_and_sentiment()
    df = df.sort_values('date')
    df['sentiment_score'] = df['sentiment_score'].fillna(0)
    df = df[features].replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=features)

    scaler = load_scaler('core/saved_models/scaler.pkl')
    X = df[features].values
    X_scaled = scaler.transform(X)

    if len(X_scaled) >= sequence_length:
        latest_X = X_scaled[-sequence_length:].reshape(1, sequence_length, len(features))
        results = predict_with_all_models(latest_X, scaler)
    else:
        results = {"error": "Not enough data for prediction."}

    return render(request, 'predict.html', {'results': results})
