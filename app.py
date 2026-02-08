import io
import time
import numpy as np
import pandas as pd
import ccxt
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from flask import Flask, Response
from gplearn.genetic import SymbolicRegressor
from datetime import datetime, timedelta

app = Flask(__name__)

# --- 1. Data Acquisition (Binance via CCXT) ---
def fetch_btc_data():
    exchange = ccxt.binance()
    symbol = 'BTC/USDT'
    timeframe = '1d'
    since = exchange.parse8601('2018-01-01T00:00:00Z')
    all_ohlcv = []
    
    # Pagination loop (Binance limit is usually 500-1000)
    while True:
        try:
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since, limit=1000)
            if not ohlcv:
                break
            all_ohlcv.extend(ohlcv)
            since = ohlcv[-1][0] + 86400000 # Advance timestamp
            if since > exchange.milliseconds():
                break
            time.sleep(0.1) # Rate limit respect
        except Exception as e:
            print(f"Data fetch error: {e}")
            break
            
    df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['date'] = pd.to_datetime(df['timestamp'], unit='ms')
    return df

# --- 2. Processing & SR Modeling ---
print("Initializing Data and Model...")
df = fetch_btc_data()
df['sma_365'] = df['close'].rolling(window=365).mean()
data = df.dropna(subset=['sma_365']).reset_index(drop=True)

# Feature Engineering: X is days since start of SMA
X = np.arange(len(data)).reshape(-1, 1)
y = data['sma_365'].values

# Symbolic Regression Config
# Constraints: Limited generations for execution speed. 
# Functions: Basic arithmetic + trigs for seasonality/cycles.
est_gp = SymbolicRegressor(population_size=1000,
                           generations=20,
                           stopping_criteria=0.01,
                           p_crossover=0.7,
                           p_subtree_mutation=0.1,
                           p_hoist_mutation=0.05,
                           p_point_mutation=0.1,
                           max_samples=0.9,
                           verbose=1,
                           function_set=['add', 'sub', 'mul', 'div', 'sin', 'cos', 'log', 'sqrt'],
                           random_state=42)

print("Training Symbolic Regressor (this may take a moment)...")
est_gp.fit(X, y)
print(f"Best Equation: {est_gp._program}")

# --- 3. Extrapolation ---
future_days = 365
X_future = np.arange(len(data) + future_days).reshape(-1, 1)
y_gp = est_gp.predict(X_future)

# --- 4. Server ---
@app.route('/')
def plot_sr():
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Historical Data
    ax.plot(X, y, 'k', label='Actual BTC 365-SMA (2018-2026)', linewidth=2)
    
    # SR Model Fit & Extrapolation
    # Split color to show where history ends and future begins
    ax.plot(X_future[:len(X)], y_gp[:len(X)], 'r--', label='SR Model Fit (In-Sample)', alpha=0.7)
    ax.plot(X_future[len(X):], y_gp[len(X):], 'g-', label='SR Extrapolation (Next 365 Days)', linewidth=2)
    
    ax.set_title(f'Symbolic Regression on BTC 365-SMA\nFormula: {est_gp._program}', fontsize=10)
    ax.set_xlabel('Days since SMA start')
    ax.set_ylabel('Price (USDT)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Buffer
    output = io.BytesIO()
    fig.savefig(output, format='png')
    plt.close(fig)
    output.seek(0)
    
    return Response(output.getvalue(), mimetype='image/png')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
