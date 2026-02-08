import io
import time
import numpy as np
import pandas as pd
import ccxt
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from flask import Flask, Response
from prophet import Prophet

app = Flask(__name__)

# --- 1. Data Pipeline ---
def fetch_raw_history():
    exchange = ccxt.binance()
    symbol = 'BTC/USDT'
    # Start 2017 to capture full previous cycle context
    since = exchange.parse8601('2017-01-01T00:00:00Z') 
    all_ohlcv = []
    
    print("Fetching Raw Data...")
    while True:
        try:
            ohlcv = exchange.fetch_ohlcv(symbol, '1d', since, limit=1000)
            if not ohlcv: break
            all_ohlcv.extend(ohlcv)
            since = ohlcv[-1][0] + 86400000
            if since > exchange.milliseconds(): break
            time.sleep(0.05)
        except: break
            
    df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['date'] = pd.to_datetime(df['timestamp'], unit='ms')
    return df

# --- 2. Prophet Modeling Engine ---
# Runs once on startup to cache the heavy optimization
def generate_forecast():
    df = fetch_raw_history()
    
    # Prophet requires specific column names: 'ds' (Date) and 'y' (Value)
    data = df[['date', 'close']].rename(columns={'date': 'ds', 'close': 'y'})
    
    # Configuration
    # seasonality_mode='multiplicative': The swings get bigger as price gets higher (Log-like behavior)
    # changepoint_prior_scale: Flexibility of the trend line (0.05 is default, 0.1 allows more shifts)
    m = Prophet(seasonality_mode='multiplicative', 
                changepoint_prior_scale=0.1,
                yearly_seasonality=True,
                weekly_seasonality=False,
                daily_seasonality=False)
    
    # Explicitly model the "Halving Cycle" (Approx 4 years / 1460 days)
    m.add_seasonality(name='4_year_cycle', period=1460, fourier_order=8)
    
    print("Fitting Prophet Model (MCMC)...")
    m.fit(data)
    
    # Forecast Horizon
    future = m.make_future_dataframe(periods=365 * 2) # 2 Years
    forecast = m.predict(future)
    
    return m, forecast, data

# Precompute
model, forecast, history = generate_forecast()

@app.route('/')
def plot_prophet():
    # Visualization: Decomposition
    # We will plot the Main Forecast + The Isolated 4-Year Component
    
    fig = plt.figure(figsize=(14, 12))
    gs = fig.add_gridspec(3, 1, height_ratios=[2, 1, 1])
    
    # Plot 1: Main Forecast
    ax1 = fig.add_subplot(gs[0])
    
    # Actual Data
    ax1.scatter(history['ds'], history['y'], color='black', s=2, alpha=0.3, label='Actual Daily Close')
    
    # Model Trend
    ax1.plot(forecast['ds'], forecast['yhat'], color='#0072B2', linewidth=2, label='Model Forecast')
    
    # Uncertainty
    ax1.fill_between(forecast['ds'], forecast['yhat_lower'], forecast['yhat_upper'], 
                     color='#0072B2', alpha=0.2, label='Uncertainty Interval')
    
    ax1.set_title('Prophet Time Sequence Decomposition: BTC/USDT', fontsize=12)
    ax1.set_ylabel('Price (USDT)')
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: The Trend Component (The "Fair Value" line without noise)
    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    ax2.plot(forecast['ds'], forecast['trend'], color='red', linewidth=2)
    ax2.set_title('Component 1: Underlying Growth Trend (Piecewise Linear)', fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: The 4-Year Cycle Component (Extracted Seasonality)
    ax3 = fig.add_subplot(gs[2], sharex=ax1)
    # Plotting the multiplicative factor (percentage impact)
    ax3.plot(forecast['ds'], forecast['4_year_cycle'], color='green', linewidth=1.5)
    ax3.set_title('Component 2: Extracted 4-Year Halving Cycle (Multiplicative Factor)', fontsize=10)
    ax3.set_ylabel('Impact %')
    ax3.grid(True, alpha=0.3)
    
    # Formatting
    plt.tight_layout()
    
    output = io.BytesIO()
    fig.savefig(output, format='png', dpi=100)
    plt.close(fig)
    output.seek(0)
    
    return Response(output.getvalue(), mimetype='image/png')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
