import pandas as pd
import numpy as np
from binance import Client
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from flask import Flask, Response
import io
import warnings
warnings.filterwarnings('ignore')

# Initialize Binance client (no API keys needed for public data)
client = Client()

# 1. Fetch 1h OHLC data from Binance since 2020
def fetch_binance_data(symbol='BTCUSDT', start_date='2020-01-01'):
    print(f"Fetching {symbol} 1h data from {start_date}...")
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
    
    # Convert timestamp to datetime and set as index
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)  # CORRECTED LINE
    
    # Convert price/volume columns to numeric
    for col in ['open', 'high', 'low', 'close', 'volume']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    print(f"Fetched {len(df)} rows")
    return df[['open', 'high', 'low', 'close', 'volume']]

# 2. Compute SMAs and shift them
def add_sma_features(df):
    print("Calculating SMAs...")
    df = df.copy()
    df['sma_365'] = df['close'].rolling(window=365).mean()
    df['sma_730'] = df['close'].rolling(window=730).mean()
    df['sma_1460'] = df['close'].rolling(window=1460).mean()
    
    # Shift SMAs to future (negative shift = future values)
    df['sma_365_future'] = df['sma_365'].shift(-365)
    df['sma_730_future'] = df['sma_730'].shift(-730)
    df['sma_1460_future'] = df['sma_1460'].shift(-1460)
    
    return df

# 3. Prepare data for training
def prepare_ml_data(df):
    feature_cols = ['close', 'sma_365', 'sma_730', 'sma_1460']
    target_cols = ['sma_365_future', 'sma_730_future', 'sma_1460_future']
    
    X = df[feature_cols].copy()
    y = df[target_cols].copy()
    
    # Remove rows with NaN in targets (future periods without data)
    mask = y.notna().all(axis=1)
    X_train = X[mask]
    y_train = y[mask]
    
    print(f"Training samples after NaN removal: {len(X_train)}")
    return X_train, y_train, feature_cols, target_cols

# 4. Train Random Forest model
def train_model(X_train, y_train):
    print("Training Random Forest model...")
    model = RandomForestRegressor(n_estimators=100, max_depth=20, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_train)
    mse = mean_squared_error(y_train, y_pred)
    print(f"Training MSE: {mse:.2f}")
    return model

# 5. Web server with plotting
app = Flask(__name__)

@app.route('/')
def plot_results():
    # Get last valid prediction point
    last_valid_idx = X_train.index[-1]
    last_features = X_train.loc[last_valid_idx].values.reshape(1, -1)
    predictions = model.predict(last_features)[0]
    
    # Create visualization
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Plot recent price history (last 2000 hours ≈ 83 days)
    recent_df = df[-2000:].copy()
    ax.plot(recent_df.index, recent_df['close'], label='BTC Price', color='blue', alpha=0.7, linewidth=1.5)
    
    # Plot SMAs where available
    ax.plot(recent_df.index, recent_df['sma_365'], label='365h SMA (~15d)', color='orange')
    ax.plot(recent_df.index, recent_df['sma_730'], label='730h SMA (~30d)', color='green')
    ax.plot(recent_df.index, recent_df['sma_1460'], label='1460h SMA (~60d)', color='red')
    
    # Plot predictions as future points
    future_offsets = [pd.Timedelta(hours=365), pd.Timedelta(hours=730), pd.Timedelta(hours=1460)]
    future_dates = [last_valid_idx + offset for offset in future_offsets]
    colors = ['orange', 'green', 'red']
    
    for date, pred, color in zip(future_dates, predictions, colors):
        ax.scatter(date, pred, color=color, s=200, marker='*', 
                  edgecolors='black', linewidths=1.5, zorder=5, 
                  label=f'Predicted SMA @ {date.strftime("%Y-%m-%d")}')
    
    ax.set_title('BTC/USDT Price with SMA Predictions (Random Forest)', fontsize=16, pad=20)
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Price (USDT)', fontsize=12)
    ax.legend(loc='upper left', fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # Return as PNG image
    img = io.BytesIO()
    plt.savefig(img, format='png', dpi=100)
    img.seek(0)
    plt.close()
    
    return Response(img.getvalue(), mimetype='image/png')

@app.route('/health')
def health():
    return {
        'status': 'ok',
        'data_points': len(df),
        'training_samples': len(X_train),
        'last_update': df.index[-1].strftime('%Y-%m-%d %H:%M')
    }

if __name__ == '__main__':
    # Fetch and prepare data
    df = fetch_binance_data()
    df = add_sma_features(df)
    X_train, y_train, feature_cols, target_cols = prepare_ml_data(df)
    
    # Train model
    model = train_model(X_train, y_train)
    
    # Start server
    print("\n🚀 Server starting at http://localhost:8080")
    print("   Endpoints:")
    print("   - /      : Chart visualization")
    print("   - /health: API health check")
    app.run(host='0.0.0.0', port=8080, debug=False, threaded=True)