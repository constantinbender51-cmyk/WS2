import pandas as pd
import numpy as np
from binance import Client
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from flask import Flask, Response
import io
import base64
from datetime import datetime, timedelta

# Initialize Binance client (no API keys needed for public data)
client = Client()

# 1. Fetch 1h OHLC data from Binance since 2020
def fetch_binance_data(symbol='BTCUSDT', start_date='2020-01-01'):
    klines = client.get_historical_klines(
        symbol,
        Client.KLINE_INTERVAL_1HOUR,
        start_date
    )
    df = pd.DataFrame(klines, columns=[
        'timestamp', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'quote_asset_volume', 'number_of_trades',
        'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
    ])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('tdf['close'] = df['close'].astype(float)
    return df[['open', 'high', 'low', 'close', 'volume']]

# 2. Compute SMAs and shift them
def add_sma_features(df):
    # Calculate SMAs
    df['sma_365'] = df['close'].rolling(window=365).mean()
    df['sma_730'] = df['close'].rolling(window=730).mean()
    df['sma_1460'] = df['close'].rolling(window=1460).mean()
    
    # Shift SMAs to future (negative shift = future values)
    df['sma_365_shifted'] = df['sma_365'].shift(-365)
    df['sma_730_shifted'] = df['sma_730'].shift(-730)
    df['sma_1460_shifted'] = df['sma_1460'].shift(-1460)
    
    return df

# 3. Prepare data for training
def prepare_ml_data(df):
    # Features: current price and SMAs
    feature_cols = ['close', 'sma_365', 'sma_730', 'sma_1460']
    target_cols = ['sma_365_shifted', 'sma_730_shifted', 'sma_1460_shifted']
    
    X = df[feature_cols].copy()
    y = df[target_cols].copy()
    
    # Remove rows with NaN targets (future periods)
    valid_mask = y.notna().all(axis=1)
    X_train = X[valid_mask]
    y_train = y[valid_mask]
    
    return X_train, y_train, feature_cols, target_cols

# 4. Train Random Forest model
def train_model(X_train, y_train):
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

# 5. Web server with plotting
app = Flask(__name__)

@app.route('/')
def plot_results():
    # Generate prediction for last available point
    last_idx = X_train.index[-1]
    last_features = X_train.loc[last_idx].values.reshape(1, -1)
    prediction = model.predict(last_features)[0]
    
    # Create plot
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Plot actual closing prices
    ax.plot(df.index[-2000:], df['close'][-2000:], label='Actual Price', alpha=0.7)
    
    # Plot actual SMAs (where available)
    ax.plot(df.index[-2000:], df['sma_365'][-2000:], label='365-SMA', alpha=0.8)
    ax.plot(df.index[-2000:], df['sma_730'][-2000:], label='730-SMA', alpha=0.8)
    ax.plot(df.index[-2000:], df['sma_1460'][-2000:], label='1460-SMA', alpha=0.8)
    
    # Plot predictions as future points
    future_dates = [df.index[-1] + timedelta(hours=i) for i in [365, 730, 1460]]
    ax.scatter(future_dates, prediction, color='red', s=100, zorder=5, label='Predictions')
    
    ax.set_title('Binance BTC/USDT Price with SMA Predictions')
    ax.set_xlabel('Date')
    ax.set_ylabel('Price (USDT)')
    ax.legend()
    ax.grid(True)
    
    # Save to buffer
    img = io.BytesIO()
    plt.savefig(img, format='png', bbox_inches='tight')
    img.seek(0)
    plt.close()
    
    return Response(img.getvalue(), mimetype='image/png')

if __name__ == '__main__':
    print("Fetching data...")
    df = fetch_binance_data()
    print(f"Data shape: {df.shape}")
    
    print("Adding SMA features...")
    df = add_sma_features(df)
    
    print("Preparing ML data...")
    X_train, y_train, feature_cols, target_cols = prepare_ml_data(df)
    print(f"Training samples: {len(X_train)}")
    
    print("Training model...")
    model = train_model(X_train, y_train)
    
    # Evaluate model
    y_pred = model.predict(X_train)
    mse = mean_squared_error(y_train, y_pred)
    print(f"Model MSE: {mse:.2f}")
    
    print("Starting web server on http://localhost:8080")
    app.run(host='0.0.0.0', port=8080, debug=False)