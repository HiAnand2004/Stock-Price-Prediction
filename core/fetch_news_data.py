import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import os
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'stock_predictor_project.settings')
import django
django.setup()

import requests
from stock_app.models import NewsHeadline
from django.utils import timezone

NEWS_API_KEY = 'f5912f4dd2d74613b57a96055a8886ba'

def fetch_and_save_news(query='stock market', page_size=10):
    url = f'https://newsapi.org/v2/everything?q={query}&language=en&pageSize={page_size}&apiKey={NEWS_API_KEY}'
    response = requests.get(url)
    data = response.json()
    for article in data.get('articles', []):
        # Fix here: use 'publishedAt' mapped to 'published_at'
        published_at = article.get('publishedAt')
        if published_at is None:
            published_at = timezone.now()
        else:
            # Convert publishedAt string to datetime if needed (assume ISO format)
            from django.utils.dateparse import parse_datetime
            dt = parse_datetime(published_at)
            published_at = dt if dt else timezone.now()

        NewsHeadline.objects.update_or_create(
            title=article['title'][:200],
            date=article.get('publishedAt', timezone.now()),
            defaults={
                'sentiment_score': 0.0,  # placeholder, sentiment updated later
                'ticker': '^NSEI'  # default ticker
            }
        )
    print("Saved news headlines.")

features = ['open', 'high', 'low', 'close', 'volume', 'sentiment_score']

if __name__ == "__main__":
    fetch_and_save_news()

from django.db import models

class NewsHeadline(models.Model):
    title = models.CharField(max_length=200)
    date = models.DateTimeField()
    sentiment_score = models.FloatField(default=0.0)
    ticker = models.CharField(max_length=20, default='^NSEI')  # <-- add this line

