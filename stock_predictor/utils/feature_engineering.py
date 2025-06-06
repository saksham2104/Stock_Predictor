def create_features(df):
    df['Return'] = df['Close'].pct_change()
    df['Lag_1'] = df['Close'].shift(1)
    df['MA_5'] = df['Close'].rolling(window=5).mean()
    df['Volatility_5'] = df['Return'].rolling(window=5).std()
    df = df.dropna()
    return df
