import ccxt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from flask import Flask, send_file
import os
import io
import base64

# 1. CONFIGURATION
symbol = 'BTC/USDT'
timeframe = '1d'
start_date_str = '2018-01-01 00:00:00'

# Fixed Params for this Config
III_WINDOW = 35 
L_LOW = 0.5
L_MID = 4.5
L_HIGH = 2.45
SL_PCT = 0.02
TP_PCT = 0.16

# --- SCENARIOS ---
scenarios = {
    "0. BASELINE (40/120, 0.13/0.18)": {
        "SMA_F": 40, "SMA_S": 120,
        "T_LOW": 0.13, "T_HIGH": 0.18
    },
    "1. SMA +5 (45/125)": {
        "SMA_F": 45, "SMA_S": 125, # Robustness Check 1
        "T_LOW": 0.13, "T_HIGH": 0.18
    },
    "2. SMA -5 (35/115)": {
        "SMA_F": 35, "SMA_S": 115, # Robustness Check 2
        "T_LOW": 0.13, "T_HIGH": 0.18
    },
    "3. THRESH +0.01 (0.14/0.19)": {
        "SMA_F": 40, "SMA_S": 120,
        "T_LOW": 0.14, "T_HIGH": 0.19 # Micro-nudge up
    },
    "4. THRESH -0.01 (0.12/0.17)": {
        "SMA_F": 40, "SMA_S": 120,
        "T_LOW": 0.12, "T_HIGH": 0.17 # Micro-nudge down
    },
}

def fetch_binance_history(symbol, start_str):
    print(f"Fetching data for {symbol} starting from {start_str}...")
    exchange = ccxt.binance()
    since = exchange.parse8601(start_str)
    all_ohlcv = []
    while True:
        try:
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since, limit=1000)
            if not ohlcv:
                print('No more data to fetch.')
                break
            all_ohlcv.extend(ohlcv)
            since = ohlcv[-1][0] + 1 # Move 'since' to the end of the fetched candle
            # Optional: Add a small delay to avoid hitting rate limits too quickly
            # time.sleep(0.1) 
        except ccxt.NetworkError as e:
            print(f"Network error: {e}. Retrying...")
            # Implement retry logic here if desired
            break # Or continue, depending on desired behavior
        except ccxt.ExchangeError as e:
            print(f"Exchange error: {e}. Stopping.")
            break
        except Exception as e:
            print(f"An unexpected error occurred: {e}. Stopping.")
            break
        # Check if we've fetched up to the current time or a reasonable limit
        if since >= exchange.milliseconds():
            print('Reached current time, stopping fetch.')
            break
    df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    if not df.empty:
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        df = df[~df.index.duplicated(keep='first')]
    return df

# Metrics
def calculate_metrics(equity_series, base_returns):
    ret = equity_series.pct_change().fillna(0)
    days = (equity_series.index[-1] - equity_series.index[0]).days
    cagr = (equity_series.iloc[-1] / equity_series.iloc[0]) ** (365.0 / days) - 1
    roll_max = equity_series.cummax()
    drawdown = (equity_series - roll_max) / roll_max
    max_dd = drawdown.min()
    sharpe = (ret.mean() / ret.std()) * np.sqrt(365) if ret.std() != 0 else 0
    
    # Buy & Hold
    bh_series = pd.Series(np.cumprod(1 + base_returns), index=equity_series.index)
    bh_cagr = (bh_series.iloc[-1] / bh_series.iloc[0]) ** (365.0 / days) - 1
    
    return cagr, max_dd, sharpe, bh_cagr

# 2. DATA PREP (Common)
df = fetch_binance_history(symbol, start_date_str)

# Calculate III (Constant for all tests)
df['log_ret'] = np.log(df['close'] / df['close'].shift(1))
df['net_direction'] = df['log_ret'].rolling(III_WINDOW).sum().abs()
df['path_length'] = df['log_ret'].abs().rolling(III_WINDOW).sum()
epsilon = 1e-8
df['iii'] = df['net_direction'] / (df['path_length'] + epsilon)
iii_prev = df['iii'].shift(1).fillna(0).values

# 3. RUN SCENARIOS
print("\nRunning Robustness Tests (SMA & Threshold Sensitivity)...")
comparison_data = []

# To ensure fair comparison, find max start index across all possible SMAs
max_sma = 125 
start_idx_max = max(max_sma, III_WINDOW)

for name, params in scenarios.items():
    # A. Calculate SMA for this scenario
    sma_f_val = params['SMA_F']
    sma_s_val = params['SMA_S']
    
    sma_fast = df['close'].rolling(sma_f_val).mean()
    sma_slow = df['close'].rolling(sma_s_val).mean()
    
    # B. Calculate Base Returns (Trend Logic)
    # We must do this inside loop because signals change with SMA
    temp_base_rets = np.zeros(len(df))
    
    # Vectorized signal calculation for speed
    # Note: Using shift(1) for prev close/sma
    prev_c = df['close'].shift(1)
    prev_f = sma_fast.shift(1)
    prev_s = sma_slow.shift(1)
    
    # Entry conditions
    long_cond = (prev_c > prev_f) & (prev_c > prev_s)
    short_cond = (prev_c < prev_f) & (prev_c < prev_s)
    
    # We need to apply SL/TP logic. 
    # Since we need high accuracy, we iterate.
    # (Optimization: We iterate only from start_idx_max)
    
    scenario_equity = [1.0] * start_idx_max # Placeholder
    equity = 1.0
    
    # Tiers
    t_low, t_high = params['T_LOW'], params['T_HIGH']
    
    # Leverage Array
    tier_mask = np.full(len(df), 2, dtype=int) 
    tier_mask[iii_prev < t_high] = 1
    tier_mask[iii_prev < t_low] = 0
    lookup = np.array([L_LOW, L_MID, L_HIGH])
    lev_arr = lookup[tier_mask]

    for i in range(start_idx_max, len(df)):
        daily_ret = 0.0
        
        # Check Signal
        is_long = long_cond.iloc[i]
        is_short = short_cond.iloc[i]
        
        open_p = df['open'].iloc[i]
        high_p = df['high'].iloc[i]
        low_p = df['low'].iloc[i]
        close_p = df['close'].iloc[i]
        
        if is_long:
            entry = open_p; sl = entry*(1-SL_PCT); tp = entry*(1+TP_PCT)
            if low_p <= sl: daily_ret = -SL_PCT
            elif high_p >= tp: daily_ret = TP_PCT
            else: daily_ret = (close_p - entry)/entry
            
        elif is_short:
            entry = open_p; sl = entry*(1+SL_PCT); tp = entry*(1-TP_PCT)
            if high_p >= sl: daily_ret = -SL_PCT
            elif low_p <= tp: daily_ret = TP_PCT
            else: daily_ret = (entry - close_p)/entry
            
        # Apply Leverage
        lev = lev_arr[i]
        final_ret = daily_ret * lev
        equity *= (1 + final_ret)
        
        # Check liquidation (simplistic)
        if equity < 0.05: equity = 0
            
        scenario_equity.append(equity)
        
    # C. Metrics
    eq_series = pd.Series(scenario_equity, index=df.index[start_idx_max-len(scenario_equity):])
    # Recalculate pure Buy Hold for comparison on same timeframe
    bh_series = df['close'].iloc[start_idx_max:] / df['close'].iloc[start_idx_max]
    
    # Metrics
    # Filter for the valid trading period
    valid_eq = eq_series.iloc[start_idx_max:] 
    
    # Calculate Base Returns for BH CAGR (using close price)
    base_bh_ret = df['close'].pct_change().fillna(0)
    
    cagr, mdd, sharpe, bh_cagr = calculate_metrics(valid_eq, base_bh_ret.iloc[start_idx_max:].values)
    
    comparison_data.append({
        'Scenario': name,
        'CAGR': cagr,
        'MDD': mdd,
        'Sharpe': sharpe
    })

results_df = pd.DataFrame(comparison_data)

# 4. OUTPUT
print("\n" + "="*85)
print(f"ROBUSTNESS TEST: HIGH AGGRESSION CONFIG (35d / 4.5x)")
print("-" * 85)
print(f"{'Scenario':<30} | {'Sharpe':<8} | {'CAGR':<8} | {'MDD':<8} | {'Change vs Base'}")
print("-" * 85)

base_sharpe = results_df.iloc[0]['Sharpe']

for index, row in results_df.iterrows():
    change = (row['Sharpe'] - base_sharpe) / base_sharpe
    color_code = ""
    if abs(change) < 0.10: status = "ROBUST"
    else: status = "SENSITIVE"
    
    print(f"{row['Scenario']:<30} | {row['Sharpe']:.2f}<{' ':<2} | {row['CAGR']*100:.1f}%<{' ':<1} | {row['MDD']*100:.1f}%<{' ':<1} | {change*100:+.1f}% ({status})")

print("="*85 + "\n")

# 5. VISUALIZATION
plt.figure(figsize=(14, 8))
colors = ['blue', 'orange', 'green', 'purple', 'red']
for (name, params), color in zip(scenarios.items(), colors):
    # Re-run loop briefly to get curve (inefficient but cleaner code structure)
    # ... (Logic identical to above loop, omitted for brevity in print, included in exec)
    # Actually, let's just create a quick helper to grab the curve data from the main loop logic if we stored it
    # For now, we just rely on the table metrics which are the definitive proof.
    pass

# We will generate the plot image using the logic inside the Flask route to save state
pass

# Plotting Logic recreated for the file save
plt.figure(figsize=(14, 8))
for (name, params), color in zip(scenarios.items(), colors):
    # Re-calc for plot
    sma_f = df['close'].rolling(params['SMA_F']).mean()
    sma_s = df['close'].rolling(params['SMA_S']).mean()
    prev_c = df['close'].shift(1); prev_f = sma_f.shift(1); prev_s = sma_s.shift(1)
    
    t_low, t_high = params['T_LOW'], params['T_HIGH']
    tier_mask = np.full(len(df), 2, dtype=int) 
    tier_mask[iii_prev < t_high] = 1; tier_mask[iii_prev < t_low] = 0
    lookup = np.array([L_LOW, L_MID, L_HIGH]); lev_arr = lookup[tier_mask]
    
    eq = [1.0] * start_idx_max
    curr_eq = 1.0
    
    for i in range(start_idx_max, len(df)):
        daily_ret = 0
        if prev_c.iloc[i] > prev_f.iloc[i] and prev_c.iloc[i] > prev_s.iloc[i]:
            entry = df['open'].iloc[i]; sl=entry*(1-SL_PCT); tp=entry*(1+TP_PCT)
            if df['low'].iloc[i]<=sl: daily_ret=-SL_PCT
            elif df['high'].iloc[i]>=tp: daily_ret=TP_PCT
            else: daily_ret=(df['close'].iloc[i]-entry)/entry
        elif prev_c.iloc[i] < prev_f.iloc[i] and prev_c.iloc[i] < prev_s.iloc[i]:
            entry = df['open'].iloc[i]; sl=entry*(1+SL_PCT); tp=entry*(1-TP_PCT)
            if df['high'].iloc[i]>=sl: daily_ret=-SL_PCT
            elif df['low'].iloc[i]<=tp: daily_ret=TP_PCT
            else: daily_ret=(entry-df['close'].iloc[i])/entry
            
        curr_eq *= (1 + daily_ret * lev_arr[i])
        eq.append(curr_eq)
        
    plt.plot(df.index[start_idx_max-len(eq):], eq, label=name, color=color, linewidth=2 if "BASELINE" in name else 1)

plt.yscale('log')
plt.title('Robustness Check: SMA & Threshold Sensitivity')
plt.legend()
plt.grid(True, which='both', linestyle='--', alpha=0.3)

plot_dir = '/app/static'
if not os.path.exists(plot_dir): os.makedirs(plot_dir)
plot_path = os.path.join(plot_dir, 'robustness_final.png')
plt.savefig(plot_path, dpi=300, bbox_inches='tight')

app = Flask(__name__)
@app.route('/')
def serve_plot(): 
    # Just serving the image for simplicity in this final step
    return f"""
    <html><body>
    <h2>Robustness Test Results</h2>
    <img src="/static/robustness_final.png" style="width:100%; max-width:1000px;">
    </body></html>
    """
@app.route('/health')
def health(): return 'OK', 200

if __name__ == '__main__':
    print("Starting Web Server...")
    app.run(host='0.0.0.0', port=8080, debug=False)
