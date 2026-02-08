import pandas as pd
import numpy as np
from binance import Client
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
from flask import Flask, Response
import io
import warnings
warnings.filterwarnings('ignore')

# Initialize Binance client
client = Client()

# 1. Fetch hourly data from 2018 and resample to daily closes
def fetch_daily_data(symbol='BTCUSDT', start_date='2018-01-01'):
    print(f"Fetching {symbol} hourly data from {start_date}...")
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
    df['close'] = df['close'].astype(float)
    
    # Resample to daily closes (using last hourly close of each day)
    df.set_index('timestamp', inplace=True)
    daily = df['close'].resample('D').last().dropna()
    print(f"Got {len(daily)} daily closes from {daily.index[0].date()} to {daily.index[-1].date()}")
    return daily

# 2. Compute day-based SMAs and prepare features for prediction
def prepare_data(daily_close):
    df = pd.DataFrame({'close': daily_close})
    
    # Day-based SMAs (365 days = 365 periods since we're daily)
    df['sma_365'] = df['close'].rolling(365).mean()
    df['sma_730'] = df['close'].rolling(730).mean()
    df['sma_1460'] = df['close'].rolling(1460).mean()
    
    # Create features: recent price action + current SMA slope
    df['slope_365'] = df['sma_365'].diff(30)  # 30-day slope
    df['slope_730'] = df['sma_730'].diff(60)
    df['slope_1460'] = df['sma_1460'].diff(120)
    df['price_vs_sma365'] = df['close'] / df['sma_365'] - 1
    
    df.dropna(inplace=True)
    return df

# 3. Train model to predict NEXT DAY's SMA values
def train_models(df):
    features = ['close', 'sma_365', 'sma_730', 'sma_1460', 'slope_365', 'slope_730', 'slope_1460', 'price_vs_sma365']
    targets = ['sma_365', 'sma_730', 'sma_1460']
    
    # Shift targets forward by 1 day (predict tomorrow's SMA based on today's features)
    X = df[features]
    y = df[targets].shift(-1)
    
    # Drop NaN from shift
    mask = y.notna().all(axis=1)
    X_train, y_train = X[mask], y[mask]
    
    models = {}
    for i, target in enumerate(targets):
        model = RandomForestRegressor(n_estimators=200, max_depth=25, random_state=42, n_jobs=-1)
        model.fit(X_train, y_train.iloc[:, i])
        models[target] = model
        print(f"Trained {target} model | Feature importance: {model.feature_importances_[1]:.3f} on current SMA")
    
    return models, features

# 4. Generate smooth continuation (predict 365 days forward)
def predict_continuation(df, models, features, days_forward=365):
    # Start from last known state
    current = df.iloc[-1].copy()
    future = []
    
    for _ in range(days_forward):
        X = pd.DataFrame([current[features]])
        next_sma365 = models['sma_365'].predict(X)[0]
        next_sma730 = models['sma_730'].predict(X)[0]
        next_sma1460 = models['sma_1460'].predict(X)[0]
        
        # Update state for next iteration (simple price assumption: follow SMA365)
        current['close'] = next_sma365 * (1 + current['price_vs_sma365'])
        current['sma_365'] = next_sma365
        current['sma_730'] = next_sma730
        current['sma_1460'] = next_sma1460
        current['slope_365'] = next_sma365 - current['sma_365']  # rough slope update
        current['slope_730'] = next_sma730 - current['sma_730']
        current['slope_1460'] = next_sma1460 - current['sma_1460']
        
        future.append({
            'sma_365': next_sma365,
            'sma_730': next_sma730,
            'sma_1460': next_sma1460
        })
    
    # Create future index
    last_date = df.index[-1]
    future_index = pd.date_range(last_date + pd.Timedelta(days=1), periods=days_forward, freq='D')
    return pd.DataFrame(future, index=future_index)

# 5. Web server
app = Flask(__name__)

@app.route('/')
def plot():
    # Plot actual + predicted continuation
    fig, ax = plt.subplots(figsize=(16, 9))
    
    # Actual data (last 4 years for clarity)
    plot_start = -1460  # last 4 years
    ax.plot(df.index[plot_start:], df['close'][plot_start:], 
            color='gray', alpha=0.4, label='BTC Price', linewidth=1)
    
    # Actual SMAs
    ax.plot(df.index[plot_start:], df['sma_365'][plot_start:], 
            color='blue', label='365d SMA (actual)', linewidth=2.5)
    ax.plot(df.index[plot_start:], df['sma_730'][plot_start:], 
            color='green', label='730d SMA (actual)', linewidth=2.5)
    ax.plot(df.index[plot_start:], df['sma_1460'][plot_start:], 
            color='red', label='1460d SMA (actual)', linewidth=2.5)
    
    # Predicted continuation
    ax.plot(future_df.index, future_df['sma_365'], 
            color='blue', linestyle='--', label='365d SMA (predicted)', linewidth=2.5, alpha=0.8)
    ax.plot(future_df.index, future_df['sma_730'], 
            color='green', linestyle='--', label='730d SMA (predicted)', linewidth=2.5, alpha=0.8)
    ax.plot(future_df.index, future_df['sma_1460'], 
            color='red', linestyle='--', label='1460d SMA (predicted)', linewidth=2.5, alpha=0.8)
    
    # Visual separator at prediction start
    ax.axvline(df.index[-1], color='black', linestyle=':', alpha=0.7, linewidth=2, label='Prediction start')
    
    ax.set_title('BTC Daily Price with SMA Continuation (Random Forest)', fontsize=18, pad=20)
    ax.set_xlabel('Date', fontsize=14)
    ax.set_ylabel('Price (USDT)', fontsize=14)
    ax.legend(loc='upper left', fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # Output as image
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=120, bbox_inches='tight')
    buf.seek(0)
    plt.close()
    return Response(buf.getvalue(), mimetype='image/png')

if __name__ == '__main__':
    # Data pipeline
    daily_close = fetch_daily_data(start_date='2018-01-01')
    df = prepare_data(daily_close)
    
    # Train
    models, features = train_models(df)
    
    # Predict 365 days forward
    future_df = predict_continuation(df, models, features, days_forward=365)
    print(f"Predicted {len(future_df)} days of SMA continuation")
    
    # Serve
    print("\n✅ Server running at http://localhost:8080")
    print("   Shows: Actual SMAs (solid) + Predicted continuation (dashed)")
    app.run(host='0.0.0.0', port=8080, debug=False, threaded=True)