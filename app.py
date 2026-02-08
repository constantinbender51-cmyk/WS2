import io
import time
import numpy as np
import pandas as pd
import ccxt
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from flask import Flask, Response
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, DotProduct, ExpSineSquared, RationalQuadratic, ConstantKernel as C

app = Flask(__name__)

# GLOBAL CACHE
CHART_BUFFER = None

# --- 1. Data Pipeline ---
def fetch_full_history():
    print("   [1/4] Connecting to Binance API...")
    exchange = ccxt.binance()
    symbol = 'BTC/USDT'
    # Start from 2018 to capture the full cycle structure
    since = exchange.parse8601('2018-01-01T00:00:00Z')
    all_ohlcv = []
    
    while True:
        try:
            ohlcv = exchange.fetch_ohlcv(symbol, '1d', since, limit=1000)
            if not ohlcv:
                break
            all_ohlcv.extend(ohlcv)
            since = ohlcv[-1][0] + 86400000 
            if since > exchange.milliseconds():
                break
            time.sleep(0.05) 
        except:
            break
            
    df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['date'] = pd.to_datetime(df['timestamp'], unit='ms')
    print(f"   [1/4] Complete. Fetched {len(df)} days of history.")
    return df

# --- 2. The Heavy Engine (Runs ONCE) ---
def precompute_intelligence():
    print("--- INITIALIZING INTELLIGENCE STACK ---")
    df = fetch_full_history()
    
    smas = [21, 45, 90, 180, 365]
    future_days = 365 * 1 # 1 Year forecast
    
    # Create Figure
    print("   [2/4] Configuring Canvas (5 Subplots)...")
    fig, axes = plt.subplots(len(smas), 1, figsize=(14, 24), sharex=True)
    plt.subplots_adjust(hspace=0.25, top=0.95, bottom=0.05)
    
    # Process Each SMA
    print("   [3/4] Training Gaussian Processes (This will take time)...")
    
    for i, window in enumerate(smas):
        ax = axes[i]
        print(f"      > Training Model {i+1}/5: {window}-Day SMA...")
        
        # Calculation
        col_name = f'sma_{window}'
        df[col_name] = df['close'].rolling(window=window).mean()
        data = df.dropna(subset=[col_name]).reset_index(drop=True)
        
        # Adaptive Downsampling
        step = 8 if window < 90 else 15
        X_train = np.arange(len(data))[::step].reshape(-1, 1)
        y_train = data[col_name].values[::step]
        
        # Kernel Architecture
        # Short SMAs (21d) = High Texture, High Noise
        # Long SMAs (365d) = High Trend, Low Noise
        noise_level = 50.0 if window < 90 else 10.0
        
        k1 = C(10.0) * DotProduct(sigma_0=0.0) 
        k2 = C(1.0) * ExpSineSquared(length_scale=100.0, periodicity=1400.0) * RBF(length_scale=1000.0)
        k3 = C(1.0) * RationalQuadratic(length_scale=50.0, alpha=1.0)
        k4 = WhiteKernel(noise_level=noise_level)
        
        kernel = k1 + k2 + k3 + k4
        
        # Fitting
        gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=2, normalize_y=True)
        gp.fit(X_train, y_train)
        
        # Prediction
        X_total = np.arange(len(data) + future_days).reshape(-1, 1)
        y_pred, sigma = gp.predict(X_total, return_std=True)
        
        # Visualization
        # 1. Uncertainty Cone
        ax.fill_between(X_total.ravel(), y_pred - 1.96 * sigma, y_pred + 1.96 * sigma, 
                        alpha=0.15, color='blue', label='95% Confidence')
        
        # 2. Prediction Line
        ax.plot(X_total, y_pred, color='navy', linewidth=1.5, label='GP Projection')
        
        # 3. Actual History
        ax.plot(np.arange(len(data)), data[col_name], 
                color='crimson', linewidth=2, label=f'Actual {window}-SMA')
        
        # Formatting
        ax.set_title(f"Gaussian Process: {window}-Day SMA", fontsize=10, loc='left', pad=5)
        ax.grid(True, linestyle=':', alpha=0.6)
        ax.legend(loc='upper left', fontsize='small')

    axes[-1].set_xlabel("Days (Index)", fontsize=10)
    fig.suptitle(f"Multi-Scale Bitcoin Price Projection (Generated: {pd.Timestamp.now()})", fontsize=14)
    
    print("   [4/4] Rendering Image to Memory...")
    output = io.BytesIO()
    fig.savefig(output, format='png', dpi=90) # Standard DPI for web
    plt.close(fig)
    output.seek(0)
    
    print("--- SERVER READY ---")
    return output.getvalue()

# --- 3. Execution (Before Server Start) ---
if __name__ == '__main__':
    # Force computation immediately
    CHART_BUFFER = precompute_intelligence()
    
    @app.route('/')
    def serve_chart():
        return Response(CHART_BUFFER, mimetype='image/png')
    
    app.run(host='0.0.0.0', port=8080)
