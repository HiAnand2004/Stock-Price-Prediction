# Stock Price Prediction

Predicting stock prices using deep learning (LSTM, GRU) and machine learning (XGBoost) models, with sentiment analysis from financial news.

---

## 🚀 Project Overview

This project aims to predict stock prices by combining historical stock data with sentiment analysis from financial news headlines. It leverages advanced deep learning models (LSTM, GRU) and XGBoost for robust time series forecasting.

---

## 📂 Features

- **Data Collection:**  
  - Fetches historical stock data (e.g., from Yahoo Finance).
  - Collects financial news headlines from Google News and NewsAPI.
- **Sentiment Analysis:**  
  - Analyzes news headlines to generate sentiment scores.
- **Data Preprocessing:**  
  - Cleans, merges, and scales stock and sentiment data.
- **Model Training:**  
  - Trains LSTM, GRU, and XGBoost models for price prediction.
- **Prediction:**  
  - Predicts future stock prices using the trained models.
- **Web Interface:**  
  - Django-based web app for user interaction and visualization.

---

## 🛠️ Tech Stack

- **Backend:** Python, Django
- **Machine Learning:** TensorFlow/Keras (LSTM, GRU), XGBoost, scikit-learn
- **Data Collection:** yfinance, requests, newspaper3k, NewsAPI
- **Sentiment Analysis:** TextBlob or VADER
- **Frontend:** HTML, CSS (Tailwind), JavaScript (optional)
- **Database:** SQLite (default, can be changed)
- **Other:** Git, Git LFS (for large model files)

---

## 📊 Model Architectures

- **LSTM (Long Short-Term Memory):**  
  For capturing long-term dependencies in time series data.
- **GRU (Gated Recurrent Unit):**  
  Efficient alternative to LSTM for sequential data.
- **XGBoost:**  
  Powerful tree-based model for tabular data.

---

## 📁 Project Structure

```
stock_predictor_project/
│
├── core/
│   ├── data_collection.py
│   ├── fetch_news_data.py
│   ├── model_train.py
│   ├── ...
│
├── stock_app/
│   ├── models.py
│   ├── views.py
│   ├── ...
│
├── templates/
│   └── core/
│       └── base.html
│
├── manage.py
├── requirements.txt
├── .gitignore
├── README.md
```

---

## ⚙️ Setup Instructions

1. **Clone the repository:**
   ```sh
   git clone https://github.com/HiAnand2004/Stock-Price-Prediction.git
   cd Stock-Price-Prediction
   ```

2. **Create and activate a virtual environment:**
   ```sh
   python -m venv tfenv
   # On Windows:
   tfenv\Scripts\activate
   # On Mac/Linux:
   source tfenv/bin/activate
   ```

3. **Install dependencies:**
   ```sh
   pip install -r requirements.txt
   ```

4. **Set up environment variables:**
   - Copy `.env.example` to `.env` and fill in your API keys and settings.

5. **Apply migrations:**
   ```sh
   python manage.py migrate
   ```

6. **Collect data and train models:**
   ```sh
   python core/data_collection.py
   python core/model_train.py
   ```

7. **Run the development server:**
   ```sh
   python manage.py runserver
   ```

---

## 📝 Usage

- Access the web interface at [http://127.0.0.1:8000/](http://127.0.0.1:8000/)
- View predictions, charts, and sentiment analysis results.

---

## 📦 Model & Data Files

- **Large model files** (`.h5`, `.pkl`) are tracked with Git LFS or provided via external download links.
- **Do not commit your virtual environment, database, or large raw data files.**

---

## 🧠 How It Works

1. **Data is collected** from Yahoo Finance and news APIs.
2. **Sentiment analysis** is performed on news headlines.
3. **Data is merged, cleaned, and scaled.**
4. **Models are trained** on the processed data.
5. **Predictions** are made and displayed via the web app.

---

## 📝 Contributing

Contributions are welcome! Please open an issue or submit a pull request.

---

## 📄 License

This project is licensed under the MIT License.

---

## 🙏 Acknowledgements

- [Yahoo Finance](https://finance.yahoo.com/)
- [NewsAPI](https://newsapi.org/)
- [Google News](https://news.google.com/)
- [TensorFlow](https://www.tensorflow.org/)
- [XGBoost](https://xgboost.readthedocs.io/)
- [Django](https://www.djangoproject.com/)

---

*For any questions or support, please contact [74980anand@gmail.com]*
