import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'stock_predictor_project.settings')
django.setup()

import pandas as pd
import yfinance as yf
from stock_app.models import StockPrice

def fetch_and_save_stock_data(ticker='^NSEI', period='10y'):
    df = yf.download(ticker, period=period)
    df = df.reset_index()

    # Flatten MultiIndex columns if needed
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = ['_'.join([str(i) for i in col if i]) for col in df.columns]

    # Lowercase all column names
    df.columns = [str(col).lower() for col in df.columns]

    # Map columns like 'open_^nsei' to 'open'
    rename_map = {}
    for col in df.columns:
        if col.endswith('_^nsei'):
            base = col.replace('_^nsei', '')
            rename_map[col] = base
    df = df.rename(columns=rename_map)

    print("Columns after rename:", df.columns.tolist())

    for _, row in df.iterrows():
        date_val = row['date']
        if hasattr(date_val, 'date'):
            date_val = date_val.date()
        else:
            date_val = pd.to_datetime(date_val).date()
        StockPrice.objects.update_or_create(
            ticker=ticker,
            date=date_val,
            defaults={
                'open': row['open'],
                'high': row['high'],
                'low': row['low'],
                'close': row['close'],
                'volume': int(row['volume']),
            }
        )

    print(f"Saved stock data for {ticker} from {df['date'].min().date()} to {df['date'].max().date()}")

if __name__ == "__main__":
    fetch_and_save_stock_data()
