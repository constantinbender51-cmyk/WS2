import ccxt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
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
III_WINDOW = 14 

# Grid Params
STEP = 0.05
RANGE_MIN = 0.05
RANGE_MAX = 0.95

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
    return df

# 2. PRE-CALCULATION
# ------------------
df = fetch_binance_history(symbol, start_date_str)

# Calculate III (Vectorized)
df['log_ret'] = np.log(df['close'] / df['close'].shift(1))
df['net_direction'] = df['log_ret'].rolling(III_WINDOW).sum().abs()
df['path_length'] = df['log_ret'].abs().rolling(III_WINDOW).sum()
epsilon = 1e-8
df['iii'] = df['net_direction'] / (df['path_length'] + epsilon)

# Indicators
df['sma_fast'] = df['close'].rolling(SMA_FAST).mean()
df['sma_slow'] = df['close'].rolling(SMA_SLOW).mean()

# 3. GENERATE BASE RETURNS (1x)
# -----------------------------
# We calculate the daily return of the strategy at 1x leverage ONCE.
# Later we just multiply this array by the leverage vector.

print("Calculating Base Strategy Returns...")
base_returns = []
start_idx = max(SMA_SLOW, III_WINDOW)

# We need a loop here because SL/TP logic is path-dependent intraday
# However, this only runs ONCE, not for every grid point.
for i in range(len(df)):
    if i < start_idx:
        base_returns.append(0.0)
        continue
        
    prev_close = df['close'].iloc[i-1]
    prev_fast = df['sma_fast'].iloc[i-1]
    prev_slow = df['sma_slow'].iloc[i-1]
    
    open_p = df['open'].iloc[i]
    high_p = df['high'].iloc[i]
    low_p = df['low'].iloc[i]
    close_p = df['close'].iloc[i]
    
    daily_ret = 0.0
    
    if prev_close > prev_fast and prev_close > prev_slow:
        # Long
        entry = open_p
        sl = entry * (1 - SL_PCT)
        tp = entry * (1 + TP_PCT)
        if low_p <= sl: daily_ret = -SL_PCT
        elif high_p >= tp: daily_ret = TP_PCT
        else: daily_ret = (close_p - entry) / entry
        
    elif prev_close < prev_fast and prev_close < prev_slow:
        # Short
        entry = open_p
        sl = entry * (1 + SL_PCT)
        tp = entry * (1 - TP_PCT)
        if high_p >= sl: daily_ret = -SL_PCT
        elif low_p <= tp: daily_ret = TP_PCT
        else: daily_ret = (entry - close_p) / entry
        
    base_returns.append(daily_ret)

df['base_ret'] = base_returns

# 4. GRID SEARCH EXECUTION
# ------------------------
print("Starting Grid Search...")

# We use numpy arrays for speed
base_ret_arr = df['base_ret'].values
iii_prev_arr = df['iii'].shift(1).fillna(0).values # Use Yesterday's III

results = []
thresholds = np.arange(RANGE_MIN, RANGE_MAX + 0.01, STEP)

# Iterate High Threshold
for high in thresholds:
    # Iterate Low Threshold
    for low in thresholds:
        if low >= high: continue # Logical constraint
        
        # Vectorized Leverage Logic
        # Default 4x
        lev_arr = np.full(len(df), 4.0)
        
        # Apply 2x where III < high
        lev_arr[iii_prev_arr < high] = 2.0
        
        # Apply 1x where III < low
        lev_arr[iii_prev_arr < low] = 1.0
        
        # Calculate Equity Curve
        final_rets = base_ret_arr * lev_arr
        
        # Approximate Liquidation Check (Vectorized)
        # If any single day return is <= -100%, we bust.
        # With 4x leverage, a -25% base move kills us.
        # Our SL is 2%, so max loss is 8%. Gaps > 25% are rare but possible.
        # We assume standard compounding for ranking.
        
        cum_ret = np.cumprod(1 + final_rets)
        total_ret = cum_ret[-1]
        
        results.append({
            'Low': round(low, 2),
            'High': round(high, 2),
            'Total_Return': total_ret
        })

# 5. ANALYSIS & HEATMAP
# ---------------------
res_df = pd.DataFrame(results)

# Find Best
best_res = res_df.loc[res_df['Total_Return'].idxmax()]
print("\n" + "="*30)
print(f"BEST COMBINATION FOUND")
print(f"Low Threshold:  {best_res['Low']}")
print(f"High Threshold: {best_res['High']}")
print(f"Total Return:   {best_res['Total_Return']:.2f}x")
print("="*30 + "\n")

# Pivot for Heatmap
pivot_table = res_df.pivot(index='Low', columns='High', values='Total_Return')

# Visualization
plt.figure(figsize=(12, 10))
ax = sns.heatmap(pivot_table, annot=True, fmt=".1f", cmap="viridis", cbar_kws={'label': 'Total Equity Multiple'})
plt.title('Grid Search: Total Return Multiplier\n(X=High Threshold, Y=Low Threshold)')
plt.gca().invert_yaxis() # Standard Y axis direction

# Highlight Best
# We need to find coordinates for the rectangle
# This is tricky in Seaborn heatmaps, so we just print it clearly.

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
    print("Starting Web Server...")
    app.run(host='0.0.0.0', port=8080, debug=False)
