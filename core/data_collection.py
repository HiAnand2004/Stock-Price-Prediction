import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'stock_predictor_project.settings')
import django
django.setup()

import yfinance as yf
from googlesearch import search
from newspaper import Article
import pandas as pd
from datetime import datetime, timedelta
import time
from django.utils import timezone
from stock_app.models import NewsHeadline  # Model we'll define in models.py

def get_stock_data(ticker, start_date, end_date):
    df = yf.download(ticker, start=start_date, end=end_date)
    df.reset_index(inplace=True)
    return df

def fetch_google_news_headlines(query, num_results=10):
    headlines = []
    for url in search(query + " stock news", num_results=num_results):
        try:
            article = Article(url)
            article.download()
            article.parse()
            headlines.append({
                'title': article.title,
                'published': article.publish_date or timezone.now(),
                'source': article.source_url,
                'url': url
            })
            time.sleep(1)
        except:
            continue
    return headlines

def save_headlines_to_db(headlines, ticker):
    for item in headlines:
        NewsHeadline.objects.create(
            ticker=ticker,
            title=item['title'],
            published_at=item['published'],  # changed here
            source=item['source'],
            url=item['url']
        )

