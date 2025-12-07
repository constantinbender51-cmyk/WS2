import ccxt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from flask import Flask, send_file
import os
import io

# 1. CONFIGURATION
symbol = 'BTC/USDT'
timeframe = '1d'
start_date_str = '2018-01-01 00:00:00'

# Fixed Strategy Params
SMA_FAST = 40
SMA_SLOW = 120
SL_PCT = 0.02
TP_PCT = 0.16
III_WINDOW = 35 

# The "Center" of our search (Your current best params)
CENTER_LOW = 0.13
CENTER_HIGH = 0.18

# The Leverages to test (Fixed based on your configuration)
# Zone 0 (III < Low): Noise -> 0.5x
# Zone 1 (Low < III < High): "Wall of Worry" -> 4.5x
# Zone 2 (III > High): "Euphoria/Knife" -> 2.45x
LEV_LOW = 0.5
LEV_MID = 4.5
LEV_HIGH = 2.45

# Grid Settings
GRID_STEPS = 10  # How many steps in each direction
GRID_STEP_SIZE = 0.01

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

# 2. DATA PREP
df = fetch_binance_history(symbol, start_date_str)

# Calculate III
df['log_ret'] = np.log(df['close'] / df['close'].shift(1))
df['net_direction'] = df['log_ret'].rolling(III_WINDOW).sum().abs()
df['path_length'] = df['log_ret'].abs().rolling(III_WINDOW).sum()
epsilon = 1e-8
df['iii'] = df['net_direction'] / (df['path_length'] + epsilon)

# Indicators
df['sma_fast'] = df['close'].rolling(SMA_FAST).mean()
df['sma_slow'] = df['close'].rolling(SMA_SLOW).mean()

# Pre-calculate Base Returns (1x)
print("Pre-calculating base strategy returns...")
base_returns = []
start_idx = max(SMA_SLOW, III_WINDOW)

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
    
    # Simple Trend Logic
    if prev_close > prev_fast and prev_close > prev_slow:
        entry = open_p; sl = entry * (1 - SL_PCT); tp = entry * (1 + TP_PCT)
        if low_p <= sl: daily_ret = -SL_PCT
        elif high_p >= tp: daily_ret = TP_PCT
        else: daily_ret = (close_p - entry) / entry
        
    elif prev_close < prev_fast and prev_close < prev_slow:
        entry = open_p; sl = entry * (1 + SL_PCT); tp = entry * (1 - TP_PCT)
        if high_p >= sl: daily_ret = -SL_PCT
        elif low_p <= tp: daily_ret = TP_PCT
        else: daily_ret = (entry - close_p) / entry
        
    base_returns.append(daily_ret)

df['base_ret'] = base_returns
df_calc = df.iloc[start_idx:].copy() # Work with valid data only
base_rets_arr = df_calc['base_ret'].values
iii_prev_arr = df_calc['iii'].shift(1).fillna(0).values

# 3. GRID SEARCH
print("Running Grid Search for Parameter Stability...")

# Create ranges
low_range = np.linspace(CENTER_LOW - (GRID_STEPS*GRID_STEP_SIZE), 
                        CENTER_LOW + (GRID_STEPS*GRID_STEP_SIZE), 
                        num=GRID_STEPS*2+1)
high_range = np.linspace(CENTER_HIGH - (GRID_STEPS*GRID_STEP_SIZE), 
                         CENTER_HIGH + (GRID_STEPS*GRID_STEP_SIZE), 
                         num=GRID_STEPS*2+1)

sharpe_grid = np.zeros((len(low_range), len(high_range)))
valid_mask = np.zeros((len(low_range), len(high_range)))

for i, t_low in enumerate(low_range):
    for j, t_high in enumerate(high_range):
        # Constraint: Low threshold must be lower than High threshold
        # We also enforce a small gap to prevent logic collapse
        if t_low >= t_high - 0.01:
            sharpe_grid[i, j] = np.nan
            continue
            
        valid_mask[i, j] = 1
        
        # Vectorized Leverage Application
        # Default to High Efficiency/Euphoria (Level 2)
        lev_vector = np.full(len(iii_prev_arr), LEV_HIGH)
        
        # Apply Mid Efficiency/Wall of Worry (Level 1)
        # Logic: t_low <= III < t_high
        lev_vector[(iii_prev_arr >= t_low) & (iii_prev_arr < t_high)] = LEV_MID
        
        # Apply Low Efficiency/Noise (Level 0)
        # Logic: III < t_low
        lev_vector[iii_prev_arr < t_low] = LEV_LOW
        
        # Calculate Returns
        strategy_rets = base_rets_arr * lev_vector
        
        # Calculate Sharpe
        mean_ret = np.mean(strategy_rets)
        std_ret = np.std(strategy_rets)
        
        if std_ret > 1e-9:
            sharpe = (mean_ret / std_ret) * np.sqrt(365)
        else:
            sharpe = 0.0
            
        sharpe_grid[i, j] = sharpe

# 4. PLOTTING
plt.figure(figsize=(10, 8))

# Use imshow for the heatmap
# Origin 'lower' puts the low values at the bottom-left
plt.imshow(sharpe_grid, origin='lower', cmap='viridis', aspect='auto', interpolation='nearest')

# Set ticks
x_ticks = np.arange(0, len(high_range), 2)
y_ticks = np.arange(0, len(low_range), 2)

plt.xticks(x_ticks, np.round(high_range[x_ticks], 2))
plt.yticks(y_ticks, np.round(low_range[y_ticks], 2))

cbar = plt.colorbar()
cbar.set_label('Sharpe Ratio')

# Mark the center (Your current config)
center_x = GRID_STEPS
center_y = GRID_STEPS
plt.scatter(center_x, center_y, color='red', marker='x', s=100, label='Current Config (0.13 / 0.18)')

plt.title(f'Stability Analysis: III Thresholds\nLeverages: <Low:{LEV_LOW}x> -- <Mid:{LEV_MID}x> -- <High:{LEV_HIGH}x>')
plt.xlabel('High Threshold (Switch from Mid -> High Leverage)')
plt.ylabel('Low Threshold (Switch from Low -> Mid Leverage)')
plt.legend()

# Save plot
plot_dir = '/app/static'
if not os.path.exists(plot_dir): os.makedirs(plot_dir)
plot_path = os.path.join(plot_dir, 'sensitivity_heatmap.png')
plt.savefig(plot_path, dpi=100, bbox_inches='tight')

# Flask Server
app = Flask(__name__)

@app.route('/')
def serve_heatmap():
    return send_file(plot_path, mimetype='image/png')

if __name__ == '__main__':
    print("Starting Web Server...")
    app.run(host='0.0.0.0', port=8080, debug=False)
