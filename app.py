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

# --- 1. Data Pipeline (Single Fetch) ---
def fetch_full_history():
    exchange = ccxt.binance()
    symbol = 'BTC/USDT'
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
    return df

# --- 2. Multi-SMA Processing Engine ---
@app.route('/')
def plot_multi_sma():
    print("Fetching Data...")
    df = fetch_full_history()
    
    # Configuration
    smas = [21, 45, 90, 180, 365]
    future_days = 365 * 1  # 1 Year forecast for clearer short-term detail
    
    fig, axes = plt.subplots(len(smas), 1, figsize=(12, 20), sharex=True)
    plt.subplots_adjust(hspace=0.3)
    
    for i, window in enumerate(smas):
        ax = axes[i]
        print(f"Processing {window}-Day SMA...")
        
        # Calculate SMA
        col_name = f'sma_{window}'
        df[col_name] = df['close'].rolling(window=window).mean()
        data = df.dropna(subset=[col_name]).reset_index(drop=True)
        
        # Adaptive Downsampling
        # Short SMAs (21d) have high freq noise, need more density -> Every 8th point
        # Long SMAs (365d) are smooth -> Every 15th point
        step = 8 if window < 90 else 15
        
        X_train = np.arange(len(data))[::step].reshape(-1, 1)
        y_train = data[col_name].values[::step]
        
        # Kernel Definition (Re-instantiated for each model to avoid contamination)
        # We allow higher noise_level for shorter SMAs
        noise_start = 50.0 if window < 90 else 10.0
        
        k1 = C(10.0) * DotProduct(sigma_0=0.0) # Trend
        k2 = C(1.0) * ExpSineSquared(length_scale=100.0, periodicity=1400.0) * RBF(length_scale=1000.0) # Cycle
        k3 = C(1.0) * RationalQuadratic(length_scale=50.0, alpha=1.0) # Texture
        k4 = WhiteKernel(noise_level=noise_start) # Noise
        
        kernel = k1 + k2 + k3 + k4
        
        # Fit
        gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=3, normalize_y=True)
        gp.fit(X_train, y_train)
        
        # Predict
        X_total = np.arange(len(data) + future_days).reshape(-1, 1)
        y_pred, sigma = gp.predict(X_total, return_std=True)
        
        # --- Plotting ---
        # 1. Uncertainty
        ax.fill_between(X_total.ravel(), y_pred - 1.96 * sigma, y_pred + 1.96 * sigma, 
                        alpha=0.15, color='blue', label='95% Conf')
        
        # 2. Model Mean
        ax.plot(X_total, y_pred, color='navy', linewidth=1.5, label='GP Projection')
        
        # 3. Actual Data
        ax.plot(np.arange(len(data)), data[col_name], 
                color='crimson', linewidth=2, label=f'Actual {window}-SMA')
        
        ax.set_title(f"{window}-Day SMA Gaussian Process", fontsize=10, pad=3)
        ax.grid(True, linestyle=':', alpha=0.6)
        
        if i == 0:
            ax.legend(loc='upper left', fontsize='small')

    axes[-1].set_xlabel("Days (Index)", fontsize=10)
    
    output = io.BytesIO()
    fig.savefig(output, format='png', dpi=80)
    plt.close(fig)
    output.seek(0)
    
    return Response(output.getvalue(), mimetype='image/png')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
