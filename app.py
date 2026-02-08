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
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, DotProduct, ConstantKernel as C

app = Flask(__name__)

# --- 1. Data Pipeline ---
def fetch_btc_data():
    exchange = ccxt.binance()
    symbol = 'BTC/USDT'
    try:
        # Fetch sufficient history
        ohlcv = exchange.fetch_ohlcv(symbol, '1d', limit=1500)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        return df
    except:
        return pd.DataFrame()

# --- 2. Gaussian Process Configuration ---
print("Initializing Gaussian Process...")
df = fetch_btc_data()
df['sma_365'] = df['close'].rolling(window=365).mean()
data = df.dropna(subset=['sma_365']).reset_index(drop=True)

# Downsample for GP speed (O(N^3) complexity)
X_train = np.arange(len(data))[::5].reshape(-1, 1)
y_train = data['sma_365'].values[::5]

# Kernel Construction
kernel = C(1.0) * DotProduct(sigma_0=1.0) + C(1.0) * RBF(length_scale=100.0) + WhiteKernel(noise_level=1.0)

gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=5, normalize_y=True)

print("Fitting GP...")
gp.fit(X_train, y_train)
print(f"Learned Kernel: {gp.kernel_}")

# --- 3. Extrapolation ---
future_days = 730 # 2 Years
X_future = np.arange(len(data) + future_days).reshape(-1, 1)

y_pred, sigma = gp.predict(X_future, return_std=True)

# --- 4. Server ---
@app.route('/')
def plot_gp_visible():
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # GP Mean Prediction
    ax.plot(X_future, y_pred, color='navy', linewidth=2, label='GP Posterior Mean', zorder=5)
    
    # Confidence Intervals
    ax.fill_between(X_future.ravel(), 
                    y_pred - 1.96 * sigma, 
                    y_pred + 1.96 * sigma, 
                    alpha=0.2, color='blue', label='95% Confidence Interval', zorder=4)
    
    # Actual Data (High Visibility Settings)
    # Plotting every point (not downsampled) to ensure density is visible
    ax.scatter(np.arange(len(data)), data['sma_365'], 
               color='crimson',    # High contrast Red
               s=25,               # Larger size
               alpha=1.0,          # Solid opacity
               label='Actual Data (BTC 365-SMA)', 
               zorder=10)          # Draw ON TOP of lines
    
    ax.set_title(f"Gaussian Process Extrapolation (High Visibility)\nKernel: {gp.kernel_}", fontsize=10)
    ax.set_xlabel('Days')
    ax.set_ylabel('BTC 365-SMA (USDT)')
    ax.legend(loc='upper left', framealpha=1.0)
    ax.grid(True, linestyle='--', alpha=0.5)
    
    output = io.BytesIO()
    fig.savefig(output, format='png', dpi=100)
    plt.close(fig)
    output.seek(0)
    
    return Response(output.getvalue(), mimetype='image/png')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
