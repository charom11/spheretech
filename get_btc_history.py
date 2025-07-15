import os
import pandas as pd
from binance.client import Client
from datetime import datetime

# If you don't have python-binance, install it with: pip install python-binance

# Load Binance API credentials from environment variables
API_KEY = os.getenv('BINANCE_API_KEY')
API_SECRET = os.getenv('BINANCE_API_SECRET')

# Debug: print the environment variable values
print("BINANCE_API_KEY:", repr(API_KEY))
print("BINANCE_API_SECRET:", repr(API_SECRET))

if not API_KEY or not API_SECRET:
    raise ValueError("Please set BINANCE_API_KEY and BINANCE_API_SECRET environment variables.")

# Initialize Binance client
client = Client(API_KEY, API_SECRET)

# Parameters
symbol = 'BTCUSDT'  # BTC-USD equivalent on Binance
interval = Client.KLINE_INTERVAL_1DAY  # Daily candles

# Set your date range here (format: 'YYYY-MM-DD'). Leave as None to fetch all available data.
start_str = '2009-01-01'  # e.g., '2009-01-01'
end_str = '2025-07-01'    # e.g., '2025-07-01'

# Fetch historical klines
if start_str and end_str:
    klines = client.get_historical_klines(symbol, interval, start_str, end_str)
else:
    # Fetch maximum available data
    klines = client.get_historical_klines(symbol, interval, "1 Jan, 2017")

# Prepare DataFrame
columns = [
    'Open Time', 'Open', 'High', 'Low', 'Close', 'Volume',
    'Close Time', 'Quote Asset Volume', 'Number of Trades',
    'Taker Buy Base Asset Volume', 'Taker Buy Quote Asset Volume', 'Ignore'
]
# type: ignore
# The linter error here is a false positive; this is correct usage for pandas.
df = pd.DataFrame(klines, columns=columns)

# Convert timestamps to readable dates
for col in ['Open Time', 'Close Time']:
    df[col] = pd.to_datetime(df[col], unit='ms')

# Save to CSV
csv_filename = f"{symbol}_binance_historical_data.csv"
df.to_csv(csv_filename, index=False)

print(f"Historical data for {symbol} saved to {csv_filename}")
print(f"Rows: {len(df)} | Columns: {len(df.columns)}")