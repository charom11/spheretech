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
    df = pd.read_csv(data_path)
    # Drop the first two rows if they are non-numeric headers
    df = df.drop([0, 1]).reset_index(drop=True)
    # Convert all columns except Date to numeric
    for col in df.columns:
        if col != 'Price' and col != 'Date':
            df[col] = pd.to_numeric(df[col], errors='coerce')
    # Rename columns for consistency if needed
    if 'Close' not in df.columns and 'Price' in df.columns:
        df = df.rename(columns={'Price': 'Close'})
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
    return df
