import yfinance as yf

def download_data(ticker="RELIANCE.NS", start="2018-01-01", end="2024-12-31"):
    df = yf.download(ticker, start=start, end=end)
    return df
