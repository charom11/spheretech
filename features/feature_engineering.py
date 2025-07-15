import pandas as pd
import numpy as np


def extract_features(data_path):
    """
    Loads stock data from CSV, adds technical indicators, and returns as DataFrame.
    Args:
        data_path (str): Path to the stock data CSV file.
    Returns:
        features (pd.DataFrame): Stock data with technical indicators.
    """
    if data_path.endswith('Bitcoin_1_1_2008-7_15_2025_historical_data_coinmarketcap.csv'):
        df = pd.read_csv(data_path, delimiter=';')
        # Rename columns to match expected names
        df = df.rename(columns={
            'timeOpen': 'Date',
            'open': 'Open',
            'high': 'High',
            'low': 'Low',
            'close': 'Close',
            'volume': 'Volume'
        })
        # Only keep necessary columns
        df = df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]
    else:
        df = pd.read_csv(data_path)
    print("Columns after loading:", df.columns.tolist())
    # Rename 'Open Time' to 'Date' for Binance CSV compatibility
    if 'Open Time' in df.columns and 'Date' not in df.columns:
        df = df.rename(columns={'Open Time': 'Date'})
    print("Columns after loading and renaming:", df.columns.tolist())
    # Convert all columns except Date to numeric
    for col in df.columns:
        if col != 'Date':
            df[col] = pd.to_numeric(df[col], errors='coerce')
    # Ensure 'Close' column exists for technical indicators
    if 'Close' not in df.columns and 'Price' in df.columns:
        df = df.rename(columns={'Price': 'Close'})
    df['Close'] = pd.Series(df['Close'])
    # Technical indicators are calculated below on DataFrame columns only
    df['MA_5'] = df['Close'].rolling(window=5).mean()
    df['MA_10'] = df['Close'].rolling(window=10).mean()
    # RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / (loss + 1e-9)
    df['RSI_14'] = 100 - (100 / (1 + rs))
    # MACD
    ema12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = ema12 - ema26
    df['MACD_signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    # EMA20, EMA50
    df['EMA_20'] = df['Close'].ewm(span=20, adjust=False).mean()
    df['EMA_50'] = df['Close'].ewm(span=50, adjust=False).mean()
    # Bollinger Bands
    df['BB_MID'] = df['Close'].rolling(window=20).mean()
    df['BB_UPPER'] = df['BB_MID'] + 2 * df['Close'].rolling(window=20).std()
    df['BB_LOWER'] = df['BB_MID'] - 2 * df['Close'].rolling(window=20).std()
    # Volume is already present
    return df
