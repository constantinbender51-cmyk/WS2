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
    
    while True:
        try:
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since, limit=1000)
            if not ohlcv:
                break
            all_ohlcv.extend(ohlcv)
            since = ohlcv[-1][0] + 86400000 
            if since > exchange.milliseconds():
                break
            time.sleep(0.05) 
        except Exception as e:
            print(f"Data fetch error: {e}")
            break
            
    df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['date'] = pd.to_datetime(df['timestamp'], unit='ms')
    return df

# --- 2. Processing & High-Intensity SR Modeling ---
print("Initializing Data...")
df = fetch_btc_data()
df['sma_365'] = df['close'].rolling(window=365).mean()
data = df.dropna(subset=['sma_365']).reset_index(drop=True)

X = np.arange(len(data)).reshape(-1, 1)
y = data['sma_365'].values

print("Configuring High-Intensity Symbolic Regressor...")
# EXPANDED SEARCH PARAMETERS
est_gp = SymbolicRegressor(
    population_size=5000,       # Increased from 1000
    generations=50,             # Increased from 20
    tournament_size=40,         # High selection pressure
    stopping_criteria=0.0001,
    const_range=(-10000, 10000),# Wider constant search
    p_crossover=0.6,
    p_subtree_mutation=0.15,
    p_hoist_mutation=0.1,
    p_point_mutation=0.1,
    max_samples=0.95,
    verbose=1,
    parsimony_coefficient=0.01, # Low penalty for complexity
    function_set=['add', 'sub', 'mul', 'div', 
                  'sin', 'cos', 'tan', 'sqrt', 'log', 'abs', 'neg'],
    n_jobs=-1,                  # Parallel processing
    random_state=42
)

print("Training started (High Intensity)...")
est_gp.fit(X, y)
print(f"Best Equation Found: {est_gp._program}")

# --- 3. Extrapolation ---
future_days = 730 # 2 Years
X_total = np.arange(len(data) + future_days).reshape(-1, 1)
y_gp = est_gp.predict(X_total)

# --- 4. Server ---
@app.route('/')
def plot_sr_high_intensity():
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Historical Data
    ax.plot(X, y, 'k', label='Actual BTC 365-SMA', linewidth=2)
    
    # SR Model Fit (In-Sample)
    ax.plot(X_total[:len(X)], y_gp[:len(X)], 'r--', label='Model Fit (Reconstruction)', alpha=0.6)
    
    # SR Extrapolation (Out-of-Sample)
    ax.plot(X_total[len(X):], y_gp[len(X):], 'b-', label=f'Extrapolation (+{future_days} days)', linewidth=2.5)
    
    # Text formatting for equation
    eq_str = str(est_gp._program)
    # Wrap text if too long
    title_text = f"High-Intensity Symbolic Regression\nEquation: {eq_str[:80]}..." if len(eq_str) > 80 else f"Equation: {eq_str}"
    
    ax.set_title(title_text, fontsize=9)
    ax.set_xlabel('Days since SMA start')
    ax.set_ylabel('Price (USDT)')
    ax.legend()
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    
    output = io.BytesIO()
    fig.savefig(output, format='png', dpi=150)
    plt.close(fig)
    output.seek(0)
    
    return Response(output.getvalue(), mimetype='image/png')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
