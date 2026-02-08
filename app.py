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
from prophet.plot import add_changepoints_to_plot

app = Flask(__name__)

# --- 1. Data Pipeline ---
def fetch_raw_history():
    exchange = ccxt.binance()
    symbol = 'BTC/USDT'
    since = exchange.parse8601('2017-01-01T00:00:00Z') 
    all_ohlcv = []
    
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

# --- 2. Complex Event Modeling ---
def get_halving_events():
    # Defining exact dates where the supply logic changed
    halvings = pd.DataFrame({
        'holiday': 'halving',
        'ds': pd.to_datetime(['2016-07-09', '2020-05-11', '2024-04-19']),
        'lower_window': 0,
        'upper_window': 180, # The "shock" lasts ~6 months post-event
    })
    return halvings

# --- 3. Modeling Engine ---
def generate_complex_forecast():
    df = fetch_raw_history()
    data = df[['date', 'close']].rename(columns={'date': 'ds', 'close': 'y'})
    
    # 1. Holiday Injection
    halvings = get_halving_events()
    
    # 2. Hyperparameter Tuning (Manual Complexity)
    # changepoint_prior_scale=0.5: (Very Flexible) Allows the trend to react violently to crashes.
    # seasonality_prior_scale=10.0: (Rigid) Forces the cycle to be a strong predictor.
    m = Prophet(seasonality_mode='multiplicative',
                changepoint_prior_scale=0.5,
                seasonality_prior_scale=10.0,
                holidays=halvings,
                interval_width=0.95, # 95% Confidence Interval
                daily_seasonality=False,
                weekly_seasonality=False)
    
    # 3. High-Fidelity Cycle Extraction
    # Fourier Order 15 allows the wave to be "spiky" (parabolic top) rather than smooth
    m.add_seasonality(name='4_year_cycle', period=1460, fourier_order=15)
    m.add_seasonality(name='yearly', period=365.25, fourier_order=10)
    
    print("Fitting Complex Prophet Model...")
    m.fit(data)
    
    future = m.make_future_dataframe(periods=365 * 2) 
    forecast = m.predict(future)
    
    return m, forecast, data

# Precompute
model, forecast, history = generate_complex_forecast()

@app.route('/')
def plot_complex_prophet():
    fig = plt.figure(figsize=(16, 14))
    gs = fig.add_gridspec(4, 1, height_ratios=[3, 1, 1, 1])
    plt.subplots_adjust(hspace=0.4)
    
    # --- Plot 1: The Master Forecast ---
    ax1 = fig.add_subplot(gs[0])
    
    # Trend Line and Uncertainty
    ax1.plot(forecast['ds'], forecast['yhat'], color='#004488', linewidth=2, label='Ensemble Forecast')
    ax1.fill_between(forecast['ds'], forecast['yhat_lower'], forecast['yhat_upper'], 
                     color='#004488', alpha=0.15, label='95% Uncertainty')
    
    # Actual Data
    ax1.scatter(history['ds'], history['y'], color='black', s=3, alpha=0.4, label='Price Action', zorder=3)
    
    # Corrected Changepoint call: Passing all as keywords to avoid positional conflicts
    add_changepoints_to_plot(ax=ax1, m=model, fcst=forecast)
    
    # Halving Markers
    halvings = ['2016-07-09', '2020-05-11', '2024-04-19']
    for h in halvings:
        ax1.axvline(pd.to_datetime(h), color='red', linestyle='--', alpha=0.6, linewidth=1)
        ax1.text(pd.to_datetime(h), ax1.get_ylim()[1]*0.9, ' HALVING', color='red', fontsize=8, rotation=90)

    ax1.set_title('Bayesian Structural Time Series: BTC/USDT', fontsize=12)
    ax1.set_ylabel('Price (USDT)')
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.2)
    
    # --- Plot 2: 4-Year Fractal ---
    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    ax2.plot(forecast['ds'], forecast['4_year_cycle'], color='purple', linewidth=1.5)
    ax2.set_title('Component: 4-Year Cycle', fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    # --- Plot 3: Yearly Seasonality ---
    ax3 = fig.add_subplot(gs[2], sharex=ax1)
    ax3.plot(forecast['ds'], forecast['yearly'], color='green', linewidth=1.5)
    ax3.set_title('Component: Annual Seasonality', fontsize=10)
    ax3.grid(True, alpha=0.3)

    # --- Plot 4: Halving Shock ---
    ax4 = fig.add_subplot(gs[3], sharex=ax1)
    ax4.plot(forecast['ds'], forecast['halving'], color='red', linewidth=1.5)
    ax4.fill_between(forecast['ds'], 0, forecast['halving'], color='red', alpha=0.1)
    ax4.set_title('Component: Supply Shock Events', fontsize=10)
    ax4.grid(True, alpha=0.3)
    
    output = io.BytesIO()
    fig.savefig(output, format='png', dpi=100)
    plt.close(fig)
    output.seek(0)
    
    return Response(output.getvalue(), mimetype='image/png')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
