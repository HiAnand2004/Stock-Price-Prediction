import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'stock_predictor_project.settings')
import django
django.setup()

import pandas as pd
import re

def clean_text(text):
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r"http\S+", "", text)
    return text.lower()

def get_cleaned_news_dataframe():
    from stock_app.models import NewsHeadline
    # Query date instead of 'published_at'
    headlines = NewsHeadline.objects.all().values('title', 'date', 'sentiment_score')
    df = pd.DataFrame(list(headlines))
    if df.empty or 'title' not in df.columns:
        print("No news headlines or 'title' column missing.")
        return pd.DataFrame()

    # Rename date to date for consistency downstream (no rename needed)
    # if 'published_at' in df.columns:
    #     df.rename(columns={'published_at': 'date'}, inplace=True)

    df['text'] = df['title']
    df['text'] = df['text'].apply(clean_text)
    return df

features = ['open', 'high', 'low', 'close', 'volume', 'sentiment_score']
