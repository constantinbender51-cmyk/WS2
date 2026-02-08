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
    # Fetching limited history for speed. In prod, fetch full history.
    try:
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

# Downsample for GP speed (GPs are O(N^3) complexity)
# We take every 5th point to keep the matrix inversion fast
X = np.arange(len(data))[::5].reshape(-1, 1)
y = data['sma_365'].values[::5]

# Kernel Construction:
# 1. Trend: DotProduct() allows the model to extrapolate linearly/polynomially.
# 2. Local: RBF() captures the smooth curvature of the SMA.
# 3. Noise: WhiteKernel() handles sensor noise/irregularities.
kernel = C(1.0) * DotProduct(sigma_0=1.0) + C(1.0) * RBF(length_scale=100.0) + WhiteKernel(noise_level=1.0)

gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=5, normalize_y=True)

print("Fitting GP (O(N^3) operation)...")
gp.fit(X, y)
print(f"Learned Kernel: {gp.kernel_}")

# --- 3. Extrapolation ---
future_days = 365 * 2 # 2 Years
X_future = np.arange(len(data) + future_days).reshape(-1, 1)

# Predict Mean AND Standard Deviation (Confidence Interval)
y_pred, sigma = gp.predict(X_future, return_std=True)

# --- 4. Server ---
@app.route('/')
def plot_gp():
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Historical Data (All points, not just downsampled)
    ax.scatter(np.arange(len(data)), data['sma_365'], c='k', s=5, label='Actual Data', alpha=0.5)
    
    # GP Mean Prediction
    ax.plot(X_future, y_pred, 'b-', label='GP Posterior Mean', linewidth=2)
    
    # Confidence Intervals (95% = 1.96 std dev)
    ax.fill_between(X_future.ravel(), 
                    y_pred - 1.96 * sigma, 
                    y_pred + 1.96 * sigma, 
                    alpha=0.2, color='blue', label='95% Confidence Interval')
    
    ax.set_title(f"Gaussian Process Extrapolation\nKernel: {gp.kernel_}", fontsize=10)
    ax.set_xlabel('Days')
    ax.set_ylabel('BTC 365-SMA')
    ax.legend(loc='upper left')
    ax.grid(True, linestyle='--')
    
    output = io.BytesIO()
    fig.savefig(output, format='png')
    plt.close(fig)
    output.seek(0)
    
    return Response(output.getvalue(), mimetype='image/png')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
