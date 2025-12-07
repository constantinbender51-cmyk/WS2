import ccxt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from flask import Flask, send_file
import os
import itertools

# --- CONFIGURATION ---
SYMBOL = 'BTC/USDT'
START_DATE = '2017-01-01 00:00:00' # Need early data for training 2018
TIMEFRAME = '1d'

# --- OPTIMIZATION GRID (The "Search Space") ---
# We will search ALL these combinations every single year.
GRID_SMA_FAST = [30, 35, 40, 45, 50]
GRID_SMA_SLOW = [105, 110, 115, 120, 125, 130]
GRID_III_WINDOW = [14, 35]

# Leverage Profiles (The "Modes" the algo can switch between)
LEVERAGE_PROFILES = [
    # Label: (Low_Lev, Mid_Lev, High_Lev)
    {'name': 'Aggressive (Original)', 'vals': [0.5, 4.5, 2.45]}, 
    {'name': 'Defensive (User Idea)', 'vals': [0.0, 1.75, 1.0]},
    {'name': 'Balanced',              'vals': [0.25, 3.0, 1.5]} 
]

# Threshold Profiles (Low, High)
THRESHOLD_PROFILES = [
    (0.13, 0.18), # Original
    (0.10, 0.20), # Wide
    (0.15, 0.25)  # High Shift
]

def fetch_data():
    print(f"Fetching {SYMBOL}...")
    exchange = ccxt.binance()
    since = exchange.parse8601(START_DATE)
    all_ohlcv = []
    while True:
        try:
            ohlcv = exchange.fetch_ohlcv(SYMBOL, TIMEFRAME, since, limit=1000)
            if not ohlcv: break
            all_ohlcv.extend(ohlcv)
            since = ohlcv[-1][0] + 1
            if since > exchange.milliseconds(): break
        except: break
    df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    df = df[~df.index.duplicated(keep='first')]
    return df

def precompute_indicators(df):
    """
    Pre-calculates all possible indicators to make the grid search instant.
    """
    print("Pre-computing indicator library...")
    # 1. SMAs
    for f in GRID_SMA_FAST:
        df[f'sma_fast_{f}'] = df['close'].rolling(f).mean()
    for s in GRID_SMA_SLOW:
        df[f'sma_slow_{s}'] = df['close'].rolling(s).mean()
        
    # 2. III Variations
    df['log_ret'] = np.log(df['close'] / df['close'].shift(1))
    for w in GRID_III_WINDOW:
        net = df['log_ret'].rolling(w).sum().abs()
        path = df['log_ret'].abs().rolling(w).sum()
        df[f'iii_{w}'] = net / (path + 1e-9)
        
    return df

def run_backtest_vectorized(df, fast_per, slow_per, iii_per, t_low, t_high, lev_profile):
    """
    Runs a fast vectorized backtest on a specific slice of data.
    """
    # 1. Base Signal (Trend)
    c = df['close'].values
    o = df['open'].values
    # Shift indicators by 1 to prevent lookahead
    fast = df[f'sma_fast_{fast_per}'].shift(1).fillna(0).values
    slow = df[f'sma_slow_{slow_per}'].shift(1).fillna(0).values
    prev_c = df['close'].shift(1).fillna(0).values
    
    # 0 = Neutral, 1 = Long, -1 = Short (Short logic simplified to inverse for speed)
    # Note: Using your specific "Close > Fast AND Close > Slow" logic
    signal = np.zeros(len(df))
    long_cond = (prev_c > fast) & (prev_c > slow)
    short_cond = (prev_c < fast) & (prev_c < slow)
    
    # Calculate Raw Daily Returns (Unlevered)
    # We approximate daily PnL as (Close - Open) / Open for speed in grid search
    # (Exact SL/TP is too slow for 10,000 combos, this proxies "capture" well)
    daily_raw = (c - o) / o
    
    base_pnl = np.zeros(len(df))
    base_pnl[long_cond] = daily_raw[long_cond]
    base_pnl[short_cond] = -daily_raw[short_cond]
    
    # 2. Apply III & Leverage
    iii_prev = df[f'iii_{iii_per}'].shift(1).fillna(0).values
    
    l_low, l_mid, l_high = lev_profile['vals']
    lev_vector = np.full(len(df), l_high) # Default High
    lev_vector[(iii_prev >= t_low) & (iii_prev < t_high)] = l_mid
    lev_vector[iii_prev < t_low] = l_low
    
    final_rets = base_pnl * lev_vector
    return final_rets

def get_sharpe(rets):
    if np.std(rets) == 0: return -999
    return (np.mean(rets) / np.std(rets)) * np.sqrt(365)

# --- MAIN EXECUTION ---
df = fetch_data()
df = precompute_indicators(df)

# Define WFO Years
years = sorted(list(set(df.index.year)))
years = [y for y in years if y >= 2019] # Start trading in 2019

wfo_equity = [1.0]
wfo_dates = [df.index[0]]
current_equity = 1.0
wfo_log = []

# Generate all parameter combinations once
param_grid = list(itertools.product(
    GRID_SMA_FAST, GRID_SMA_SLOW, GRID_III_WINDOW, THRESHOLD_PROFILES, LEVERAGE_PROFILES
))
print(f"Grid Space Size: {len(param_grid)} combinations per year.")

# Start WFO Loop
start_idx_global = df.index.get_loc(df[df.index.year == 2019].index[0])
# Fill pre-2019 with 1.0
wfo_equity = [1.0] * start_idx_global

for year in years:
    print(f"\n--- OPTIMIZING FOR YEAR {year} ---")
    
    # 1. Define Training Data (Expanding Window: Start -> Year-1)
    train_mask = (df.index.year < year) & (df.index.year >= (year-2)) # Use 2 year rolling window for better adaptability? 
    # Or use full history? Let's use Full History to be robust.
    train_mask = (df.index.year < year)
    
    train_df = df.loc[train_mask].copy()
    if len(train_df) < 300: continue
    
    best_score = -999
    best_params = None
    
    # 2. Grid Search
    # This might take 5-10 seconds per year
    for params in param_grid:
        f_sma, s_sma, iii_w, (t_l, t_h), lev_prof = params
        
        if f_sma >= s_sma: continue # Invalid SMA
        
        rets = run_backtest_vectorized(train_df, f_sma, s_sma, iii_w, t_l, t_h, lev_prof)
        score = get_sharpe(rets)
        
        if score > best_score:
            best_score = score
            best_params = params
            
    # Log the winner
    f, s, i_w, (tl, th), lp = best_params
    log_msg = f"Year {year}: Best SMA {f}/{s} | III {i_w} | Th {tl}/{th} | Mode: {lp['name']} (Sharpe {best_score:.2f})"
    print(log_msg)
    wfo_log.append(log_msg)
    
    # 3. Trade the "Test" Year
    test_mask = (df.index.year == year)
    test_df = df.loc[test_mask].copy()
    
    # Run logic with winning params
    # Note: We use the SLOW logic here (or re-use vector) for precision
    # For simplicity, re-use vectorized since it maps 1:1
    test_rets = run_backtest_vectorized(test_df, f, s, i_w, tl, th, lp)
    
    # Accumulate Equity
    for r in test_rets:
        current_equity *= (1 + r)
        wfo_equity.append(current_equity)

# --- PLOTTING ---
plt.figure(figsize=(12, 10))

# 1. Equity Curve
ax1 = plt.subplot(2, 1, 1)
# Align Equity Array to DF
plot_df = df.iloc[:len(wfo_equity)].copy()
plot_df['wfo_equity'] = wfo_equity

# Create a comparison "Static" strategy (The one you are worried about: 40/120)
static_rets = run_backtest_vectorized(
    plot_df, 40, 120, 35, 0.13, 0.18, 
    {'name': 'Static', 'vals': [0.5, 4.5, 2.45]}
)
plot_df['static_equity'] = (1 + static_rets).cumprod()
# Normalize to start of WFO
start_date_wfo = plot_df[plot_df.index.year == 2019].index[0]
val_at_start_wfo = plot_df.loc[start_date_wfo, 'wfo_equity']
val_at_start_static = plot_df.loc[start_date_wfo, 'static_equity']

plot_df['wfo_equity_norm'] = plot_df['wfo_equity'] / val_at_start_wfo
plot_df['static_equity_norm'] = plot_df['static_equity'] / val_at_start_static

ax1.plot(plot_df.index, plot_df['wfo_equity_norm'], color='blue', label='Walk-Forward Optimized (Dynamic)', linewidth=2)
ax1.plot(plot_df.index, plot_df['static_equity_norm'], color='gray', linestyle='--', label='Static 40/120 (Hindsight)', alpha=0.7)
ax1.set_yscale('log')
ax1.set_title('Walk-Forward vs Static Optimization')
ax1.legend()
ax1.grid(True, which='both', linestyle='--', alpha=0.3)

# 2. Parameter Stability Text
ax2 = plt.subplot(2, 1, 2)
ax2.axis('off')
text_content = "Yearly Optimized Parameters:\n\n" + "\n".join(wfo_log)
ax2.text(0.1, 0.9, text_content, fontsize=10, verticalalignment='top', fontfamily='monospace')

plt.tight_layout()
plot_path = os.path.join('/app/static', 'wfo_results.png')
if not os.path.exists('/app/static'): os.makedirs('/app/static')
plt.savefig(plot_path)

app = Flask(__name__)
@app.route('/')
def serve_image(): return send_file(plot_path, mimetype='image/png')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
