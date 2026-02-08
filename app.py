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

# --- 1. Data Pipeline ---
def fetch_full_history():
    exchange = ccxt.binance()
    symbol = 'BTC/USDT'
    # Start early enough to capture the 2017/2018 cycle top for context if possible, 
    # but 2018-01-01 is safe for consistent API data.
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

# --- 2. Processing ---
print("Fetching Data...")
df = fetch_full_history()
df['sma_365'] = df['close'].rolling(window=365).mean()
data = df.dropna(subset=['sma_365']).reset_index(drop=True)

# Downsampling: Using every 10th point. 
# Complex kernels are expensive; 250 data points is the sweet spot for a quick server response.
X_train = np.arange(len(data))[::10].reshape(-1, 1)
y_train = data['sma_365'].values[::10]

# --- 3. Complex Kernel Construction ---
# K1: Long Term Trend (Rising)
k1 = C(10.0) * DotProduct(sigma_0=0.0) + RBF(length_scale=2000.0)

# K2: The Halving Cycle (Quasi-Periodic)
# We multiply by RBF to allow the cycle to evolve/decay rather than stay perfectly rigid forever
k2 = C(1.0) * ExpSineSquared(length_scale=100.0, periodicity=1400.0, periodicity_bounds=(1200, 1600)) * \
     RBF(length_scale=1000.0)

# K3: Medium Term Irregularities (The "Texture")
k3 = C(1.0) * RationalQuadratic(length_scale=100.0, alpha=1.0)

# K4: Noise
k4 = WhiteKernel(noise_level=10.0, noise_level_bounds=(1e-5, 1e5))

kernel = k1 + k2 + k3 + k4

gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, normalize_y=True)

print("Fitting Complex GP...")
gp.fit(X_train, y_train)
print(f"Learned Kernel: {gp.kernel_}")

# --- 4. Extrapolation ---
future_days = 365 * 3 # 3 Years
X_total = np.arange(len(data) + future_days).reshape(-1, 1)

y_pred, sigma = gp.predict(X_total, return_std=True)

# --- 5. Visualization ---
@app.route('/')
def plot_complex_gp():
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Uncertainty Bands (Outer vs Inner)
    ax.fill_between(X_total.ravel(), 
                    y_pred - 1.96 * sigma, 
                    y_pred + 1.96 * sigma, 
                    alpha=0.1, color='blue', label='95% Confidence')
    
    ax.fill_between(X_total.ravel(), 
                    y_pred - 0.67 * sigma, 
                    y_pred + 0.67 * sigma, 
                    alpha=0.2, color='blue', label='50% Confidence')

    # The Model Mean
    ax.plot(X_total, y_pred, color='navy', linewidth=2, label='Posterior Mean', zorder=5)
    
    # Actual Data
    ax.plot(np.arange(len(data)), data['sma_365'], 
            color='crimson', linewidth=2, label='Actual BTC 365-SMA', zorder=6)
    
    # Vertical line for "Today"
    ax.axvline(x=len(data), color='k', linestyle='--', alpha=0.5, label='Forecast Start')

    ax.set_title(f"Multi-Kernel Gaussian Process (Trend + Cycle + Texture)\n{gp.kernel_}", fontsize=8)
    ax.set_ylabel('Price (USDT)')
    ax.legend(loc='upper left')
    ax.grid(True, linestyle='--', alpha=0.5)
    
    output = io.BytesIO()
    fig.savefig(output, format='png', dpi=100)
    plt.close(fig)
    output.seek(0)
    
    return Response(output.getvalue(), mimetype='image/png')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
