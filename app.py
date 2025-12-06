import ccxt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.collections
from datetime import datetime
from flask import Flask, send_file
import os

# 1. CONFIGURATION
# ----------------
symbol = 'BTC/USDT'
timeframe = '1d'
start_date_str = '2018-01-01 00:00:00'

# Strategy Params
SMA_FAST = 40
SMA_SLOW = 120
SL_PCT = 0.02
TP_PCT = 0.16

# Efficiency Calculation Window
III_WINDOW = 14 

def fetch_binance_history(symbol, start_str):
    print(f"Fetching data for {symbol} starting from {start_str}...")
    exchange = ccxt.binance()
    since = exchange.parse8601(start_str)
    all_ohlcv = []
    
    while True:
        try:
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since, limit=1000)
            if not ohlcv: break
            all_ohlcv.extend(ohlcv)
            since = ohlcv[-1][0] + 1
            if since > exchange.milliseconds(): break
        except Exception as e:
            print(f"Error fetching: {e}")
            break
            
    df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    df = df[~df.index.duplicated(keep='first')]
    print(f"Fetched {len(df)} days of data.")
    return df

# 2. PREPARE DATA
# ---------------
df = fetch_binance_history(symbol, start_date_str)

# --- A. CALCULATE INVERTED INEFFICIENCY INDEX (III) ---
# Formula: Abs(Net Change) / Sum(Abs(Changes))
# Range: 0 to 1

df['log_ret'] = np.log(df['close'] / df['close'].shift(1))

# Numerator: Absolute Net Direction over window
# We use rolling sum of log returns = log(Close_t / Close_t-n)
df['net_direction'] = df['log_ret'].rolling(III_WINDOW).sum().abs()

# Denominator: Sum of absolute individual moves (Path Length)
df['path_length'] = df['log_ret'].abs().rolling(III_WINDOW).sum()

# The Index
epsilon = 1e-8
df['iii'] = df['net_direction'] / (df['path_length'] + epsilon)

# --- B. BASE STRATEGY BACKTEST ---
df['sma_fast'] = df['close'].rolling(SMA_FAST).mean()
df['sma_slow'] = df['close'].rolling(SMA_SLOW).mean()

df['strategy_equity'] = 1.0
df['buy_hold_equity'] = 1.0
df['daily_ret'] = 0.0

equity = 1.0
hold_equity = 1.0
start_idx = max(SMA_SLOW, III_WINDOW)

for i in range(start_idx, len(df)):
    today = df.index[i]
    
    # Yesterday's Signals
    prev_close = df['close'].iloc[i-1]
    prev_fast = df['sma_fast'].iloc[i-1]
    prev_slow = df['sma_slow'].iloc[i-1]
    
    # Today's Execution
    open_p = df['open'].iloc[i]
    high_p = df['high'].iloc[i]
    low_p = df['low'].iloc[i]
    close_p = df['close'].iloc[i]
    
    daily_ret = 0.0
    
    # Long
    if prev_close > prev_fast and prev_close > prev_slow:
        entry = open_p
        sl = entry * (1 - SL_PCT)
        tp = entry * (1 + TP_PCT)
        if low_p <= sl: daily_ret = -SL_PCT
        elif high_p >= tp: daily_ret = TP_PCT
        else: daily_ret = (close_p - entry) / entry
        
    # Short
    elif prev_close < prev_fast and prev_close < prev_slow:
        entry = open_p
        sl = entry * (1 + SL_PCT)
        tp = entry * (1 - TP_PCT)
        if high_p >= sl: daily_ret = -SL_PCT
        elif low_p <= tp: daily_ret = TP_PCT
        else: daily_ret = (entry - close_p) / entry
        
    equity *= (1 + daily_ret)
    
    # Buy & Hold
    bh_ret = (close_p - df['close'].iloc[i-1]) / df['close'].iloc[i-1]
    hold_equity *= (1 + bh_ret)
    
    df.at[today, 'strategy_equity'] = equity
    df.at[today, 'buy_hold_equity'] = hold_equity
    df.at[today, 'daily_ret'] = daily_ret

# 3. VISUALIZATION
# ----------------
plt.figure(figsize=(14, 14))

# Filter data
plot_data = df.iloc[start_idx:].copy()

# Plot 1: Strategy Performance
ax1 = plt.subplot(3, 1, 1)
ax1.plot(plot_data.index, plot_data['strategy_equity'], label='Strategy Equity', color='blue', linewidth=2)
ax1.plot(plot_data.index, plot_data['buy_hold_equity'], label='Buy & Hold', color='gray', alpha=0.5)
ax1.set_yscale('log')
ax1.set_title('Strategy vs Buy & Hold (No Circuit Breakers)')
ax1.legend()
ax1.grid(True, which='both', linestyle='--', alpha=0.3)

# Plot 2: The Inverted Inefficiency Index (III)
ax2 = plt.subplot(3, 1, 2, sharex=ax1)
# Color code the line:
# High values (Efficiency) = Green
# Low values (Chop) = Red
points = np.array([plot_data.index.astype(np.int64) // 10**9, plot_data['iii']]).T.reshape(-1, 1, 2)
segments = np.concatenate([points[:-1], points[1:]], axis=1)

# Create a continuous norm to map from 0 to 1
norm = plt.Normalize(0, 1)
lc = matplotlib.collections.LineCollection(segments, cmap='RdYlGn', norm=norm)
lc.set_array(plot_data['iii'])
lc.set_linewidth(1.5)
ax2.add_collection(lc)
ax2.autoscale_view()

# Add Threshold lines
ax2.axhline(0.6, color='green', linestyle='--', alpha=0.5, label='High Efficiency (Trend)')
ax2.axhline(0.2, color='red', linestyle='--', alpha=0.5, label='Low Efficiency (Chop)')
ax2.set_ylabel('Efficiency (0=Chop, 1=Trend)')
ax2.set_title('Inverted Inefficiency Index (III)')
ax2.legend(loc='upper left')
ax2.grid(True, alpha=0.3)

# Plot 3: Scatter Analysis (Does III predict Volatility/Big Moves?)
# We plot III (x-axis) vs Absolute Next Day Return (y-axis)
ax3 = plt.subplot(3, 1, 3)
# Shift returns back by 1 to align "Today's III" with "Tomorrow's Move"
x_scatter = plot_data['iii'][:-1]
y_scatter = plot_data['daily_ret'].abs().shift(-1).dropna()
# Ensure alignment
common_idx = x_scatter.index.intersection(y_scatter.index)
x_scatter = x_scatter.loc[common_idx]
y_scatter = y_scatter.loc[common_idx]

# Color points by positive (profit) vs negative (loss) return
raw_ret = plot_data['daily_ret'].shift(-1).loc[common_idx]
colors = ['green' if r > 0 else 'red' for r in raw_ret]

ax3.scatter(x_scatter, y_scatter, c=colors, alpha=0.4, s=10)
ax3.set_xlabel('III Value (Today)')
ax3.set_ylabel('Absolute Return (Tomorrow)')
ax3.set_title('Correlation: Does High Efficiency Predict Big Moves Tomorrow?')
# Add trendline
z = np.polyfit(x_scatter, y_scatter, 1)
p = np.poly1d(z)
ax3.plot(x_scatter, p(x_scatter), "k--", alpha=0.5, label=f'Trend Line')
ax3.legend()
ax3.grid(True, alpha=0.3)

plt.tight_layout()



# Save
plot_dir = '/app/static'
if not os.path.exists(plot_dir): os.makedirs(plot_dir)
plot_path = os.path.join(plot_dir, 'plot.png')
plt.savefig(plot_path, dpi=300, bbox_inches='tight')

# Flask
app = Flask(__name__)
@app.route('/')
def serve_plot(): return send_file(plot_path, mimetype='image/png')
@app.route('/health')
def health(): return 'OK', 200

if __name__ == '__main__':
    print(f"Final Strategy Equity: {equity:.2f}x")
    print("\nStarting Web Server...")
    app.run(host='0.0.0.0', port=8080, debug=False)
