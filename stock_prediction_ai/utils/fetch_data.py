import yfinance as yf

def fetch_and_save_stock_data(ticker, start_date, end_date, filename):
    """
    Fetches historical stock data and saves it as a CSV file.
    Args:
        ticker (str): Stock ticker symbol (e.g., 'AAPL').
        start_date (str): Start date in 'YYYY-MM-DD'.
        end_date (str): End date in 'YYYY-MM-DD'.
        filename (str): Path to save the CSV file.
    """
    df = yf.download(ticker, start=start_date, end=end_date)
    df.to_csv(filename)
    print(f"Saved data for {ticker} to {filename}")