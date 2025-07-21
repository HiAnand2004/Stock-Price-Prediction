import os
import sys
from datetime import datetime, timedelta

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'stock_predictor_project.settings')
import django

django.setup()

from core.preprocessing import get_cleaned_news_dataframe
from stock_app.models import NewsHeadline
import pandas as pd
from core.data_collection import get_stock_data
from core.sentiment_analysis import get_news_sentiment_dataframe


def period_to_dates(period):
    end_date = datetime.today()
    if period.endswith('mo'):
        months = int(period[:-2])
        start_date = end_date - timedelta(days=30 * months)
    elif period.endswith('y'):
        years = int(period[:-1])
        start_date = end_date - timedelta(days=365 * years)
    else:
        # Default to 6 months if period is not recognized
        start_date = end_date - timedelta(days=180)
    return start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d')


def merge_stock_and_sentiment(ticker='^NSEI', period='6mo'):
    # Calculate start and end dates
    start_date, end_date = period_to_dates(period)

    # Get stock data
    stock_df = get_stock_data(ticker, start_date, end_date)

    # Get sentiment data
    news_df = get_news_sentiment_dataframe()

    # Convert published date to date only (drop time)
    if 'date' in news_df.columns:
        news_df['date'] = pd.to_datetime(news_df['date']).dt.date
    else:
        news_df['date'] = pd.NaT

    # Group by date to get average sentiment per day
    sentiment_daily = news_df.groupby('date')['sentiment_score'].mean().reset_index()

    # Ensure columns are flattened if it's a MultiIndex
    stock_df = stock_df.reset_index()

    if isinstance(stock_df.columns, pd.MultiIndex):
        stock_df.columns = ['_'.join([str(i) for i in col if i]) for col in stock_df.columns.values]

    # Fix date columns - keep only one date column named 'Date'
    date_cols = [col for col in stock_df.columns if 'date' in col.lower()]
    if len(date_cols) > 1:
        keep = 'Date' if 'Date' in date_cols else date_cols[0]
        for col in date_cols:
            if col != keep:
                stock_df.drop(columns=col, inplace=True)
        if keep != 'Date':
            stock_df.rename(columns={keep: 'Date'}, inplace=True)
    elif len(date_cols) == 1 and date_cols[0] != 'Date':
        stock_df.rename(columns={date_cols[0]: 'Date'}, inplace=True)
    elif 'Date' not in stock_df.columns:
        raise KeyError("No date column found in stock data after reset_index()")

    stock_df['date'] = pd.to_datetime(stock_df['Date']).dt.date
    sentiment_daily['date'] = pd.to_datetime(sentiment_daily['date']).dt.date

    # Merge on lowercased 'date'
    merged = pd.merge(stock_df, sentiment_daily, on='date', how='left')

    # Remove duplicate 'date' columns if any
    merged = merged.loc[:, ~merged.columns.duplicated()]

    # Or, to debug:
    print("Merged columns:", merged.columns.tolist())
    if merged.columns.duplicated().any():
        print("Duplicate columns found:", merged.columns[merged.columns.duplicated()].tolist())

    # Debug prints (can comment out in production)
    # print("stock_df index:", stock_df.index)
    # print("stock_df columns:", stock_df.columns)
    # print("sentiment_daily index:", sentiment_daily.index)
    # print("sentiment_daily columns:", sentiment_daily.columns)
    # print("Merged DataFrame columns:", merged.columns)

    if 'date' not in merged.columns:
        print("Error: 'date' column missing after merge.")
        return merged  # or handle appropriately

    rename_map = {
        'Open_^NSEI': 'open',
        'High_^NSEI': 'high',
        'Low_^NSEI': 'low',
        'Close_^NSEI': 'close',
        'Volume_^NSEI': 'volume',
        'Date': 'date'
    }
    for old, new in rename_map.items():
        if old in merged.columns:
            merged = merged.rename(columns={old: new})

    # Remove duplicate columns if any
    merged = merged.loc[:, ~merged.columns.duplicated()]

    return merged
