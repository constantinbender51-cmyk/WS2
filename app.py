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
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, DotProduct, ExpSineSquared, ConstantKernel as C

app = Flask(__name__)

# --- 1. Full Data Acquisition (Pagination Restored) ---
def fetch_full_history():
    exchange = ccxt.binance()
    symbol = 'BTC/USDT'
    timeframe = '1d'
    # Explicit start date to capture the full curve structure
    since = exchange.parse8601('2018-01-01T00:00:00Z')
    all_ohlcv = []
    
    print("Fetching full history from 2018...")
    while True:
        try:
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since, limit=1000)
            if not ohlcv:
                break
            all_ohlcv.extend(ohlcv)
            since = ohlcv[-1][0] + 86400000 
            # Stop if we reach the present
            if since > exchange.milliseconds():
                break
            time.sleep(0.05) # Rate limit respect
        except Exception as e:
            print(f"Fetch Error: {e}")
            break
            
    df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['date'] = pd.to_datetime(df['timestamp'], unit='ms')
    return df

# --- 2. Processing ---
df = fetch_full_history()
df['sma_365'] = df['close'].rolling(window=365).mean()
data = df.dropna(subset=['sma_365']).reset_index(drop=True)

# Downsample: GP cannot handle 2500+ points efficiently. 
# We take every 15th point to capture the shape while keeping N < 500.
X_train = np.arange(len(data))[::15].reshape(-1, 1)
y_train = data['sma_365'].values[::15]

print(f"Training Data Points: {len(X_train)}")

# --- 3. Kernel & Model ---
# Complex Kernel:
# 1. DotProduct: Long-term Growth Trend
# 2. RBF: Local non-linear deviations (The hills and valleys)
# 3. WhiteKernel: Noise handling
kernel = C(1.0) * DotProduct(sigma_0=1.0) + \
         C(1.0) * RBF(length_scale=100.0) + \
         WhiteKernel(noise_level=10.0)

gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=5, normalize_y=True)

print("Fitting GP on full history...")
gp.fit(X_train, y_train)
print(f"Learned Kernel: {gp.kernel_}")

# --- 4. Extrapolation ---
future_days = 730 # 2 Years
X_total = np.arange(len(data) + future_days).reshape(-1, 1)

# Predict in chunks to save memory if needed, but here we do one pass
y_pred, sigma = gp.predict(X_total, return_std=True)

# --- 5. Visualization ---
@app.route('/')
def plot_full_gp():
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # 1. The GP Model (Mean)
    ax.plot(X_total, y_pred, color='navy', linewidth=2, label='GP Model Mean', zorder=5)
    
    # 2. Uncertainty Bands
    ax.fill_between(X_total.ravel(), 
                    y_pred - 1.96 * sigma, 
                    y_pred + 1.96 * sigma, 
                    alpha=0.2, color='blue', label='95% Confidence', zorder=4)
    
    # 3. The ACTUAL Data (Full History)
    # We plot the full resolution data (not downsampled) to prove it matches the screenshot
    ax.plot(np.arange(len(data)), data['sma_365'], 
            color='crimson', linewidth=2.5, label='Actual BTC 365-SMA (2018-Present)', zorder=6)
    
    ax.set_title(f"Gaussian Process on Full History (2018-2026)\nKernel: {gp.kernel_}", fontsize=10)
    ax.set_xlabel('Days since SMA Valid (approx Jan 2019)')
    ax.set_ylabel('Price (USDT)')
    ax.legend(loc='upper left')
    ax.grid(True, which='major', linestyle='--', alpha=0.6)
    
    output = io.BytesIO()
    fig.savefig(output, format='png', dpi=100)
    plt.close(fig)
    output.seek(0)
    
    return Response(output.getvalue(), mimetype='image/png')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
