import os
import sys
import django

# 1. Set up Django
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'stock_predictor_project.settings')
django.setup()

# 2. Imports after Django setup
from stock_app.models import NewsHeadline
from core.preprocessing import get_cleaned_news_dataframe
from textblob import TextBlob

import pandas as pd

# 3. Sentiment function using TextBlob (-1 to 1)
def analyze_sentiment(text):
    return TextBlob(text).sentiment.polarity  # returns a float between -1 (negative) and 1 (positive)

# 4. Load and clean data
def run_sentiment_pipeline():
    df = get_cleaned_news_dataframe()

    # Check if DataFrame is empty
    if df.empty:
        print("No news headlines found.")
        return pd.DataFrame()

    # Ensure we are using the correct field for sentiment analysis
    df["sentiment_score"] = df["text"].apply(analyze_sentiment)

    # Print few results
    print(df[["title", "sentiment_score"]].head())

    # Optional: Save to CSV or return
    df.to_csv("sentiment_output.csv", index=False)
    return df

def get_news_sentiment_dataframe():
    df = get_cleaned_news_dataframe()
    if df.empty:
        return pd.DataFrame()
    df["sentiment_score"] = df["text"].apply(analyze_sentiment)
    # Only use fields that exist
    return df[["title", "date", "sentiment_score"]]

# 5. Entry point
if __name__ == "__main__":
    run_sentiment_pipeline()
