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
# ----------------
symbol = 'BTC/USDT'
timeframe = '1d'
start_date_str = '2018-01-01 00:00:00'

# Global Strategy Params
SMA_FAST = 40
SMA_SLOW = 120
SL_PCT = 0.02
TP_PCT = 0.16

# --- SCENARIO DEFINITIONS (The Core Test) ---
scenarios = {
    "0. BASELINE (Optimal)": {
        "III_WINDOW": 35,
        "T_LOW": 0.13, "T_HIGH": 0.18,
        "L_LOW": 0.5, "L_MID": 4.5, "L_HIGH": 2.45
    },
    "1. T_SHIFT (-0.05)": {
        "III_WINDOW": 35,
        "T_LOW": 0.08, "T_HIGH": 0.13, # Nudged down
        "L_LOW": 0.5, "L_MID": 4.5, "L_HIGH": 2.45
    },
    "2. LAG_SLOW (+10 Days)": {
        "III_WINDOW": 45, # Slower lookback
        "T_LOW": 0.13, "T_HIGH": 0.18,
        "L_LOW": 0.5, "L_MID": 4.5, "L_HIGH": 2.45
    },
    "3. LEV_MID (-1.0x)": {
        "III_WINDOW": 35,
        "T_LOW": 0.13, "T_HIGH": 0.18,
        "L_LOW": 0.5, "L_MID": 3.5, # Reduced aggression
        "L_HIGH": 2.45
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

# Metrics function
def calculate_metrics(equity_series, base_returns):
    ret = equity_series.pct_change().fillna(0)
    days = (equity_series.index[-1] - equity_series.index[0]).days
    cagr = (equity_series.iloc[-1] / equity_series.iloc[0]) ** (365.0 / days) - 1
    
    roll_max = equity_series.cummax()
    drawdown = (equity_series - roll_max) / roll_max
    max_dd = drawdown.min()
    
    sharpe = (ret.mean() / ret.std()) * np.sqrt(365) if ret.std() != 0 else 0
    
    # Calculate Buy & Hold for CAGR comparison
    bh_series = pd.Series(np.cumprod(1 + base_returns), index=equity_series.index)
    bh_cagr = (bh_series.iloc[-1] / bh_series.iloc[0]) ** (365.0 / days) - 1
    
    return cagr, max_dd, sharpe, bh_cagr

# 2. DATA PREP & BASE RETURNS (Runs once)
df = fetch_binance_history(symbol, start_date_str)
df['sma_fast'] = df['close'].rolling(SMA_FAST).mean()
df['sma_slow'] = df['close'].rolling(SMA_SLOW).mean()

# Pre-calculate Base 1x Strategy Returns
print("Pre-calculating base strategy returns...")
base_returns = []
start_idx_max = max(SMA_SLOW, max(s["III_WINDOW"] for s in scenarios.values()))

for i in range(len(df)):
    if i < start_idx_max:
        base_returns.append(0.0)
        continue
    
    # Standard 1x Base Return Calculation
    prev_close = df['close'].iloc[i-1]
    prev_fast = df['sma_fast'].iloc[i-1]
    prev_slow = df['sma_slow'].iloc[i-1]
    open_p = df['open'].iloc[i]
    high_p = df['high'].iloc[i]
    low_p = df['low'].iloc[i]
    close_p = df['close'].iloc[i]
    daily_ret = 0.0
    
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

# 3. RUN SCENARIOS
print("\nRunning Sensitivity Tests...")
comparison_data = []

for name, params in scenarios.items():
    # --- A. RECALCULATE III FOR SCENARIO (If window changed) ---
    iii_window = params['III_WINDOW']
    
    df['log_ret'] = np.log(df['close'] / df['close'].shift(1))
    df['net_direction'] = df['log_ret'].rolling(iii_window).sum().abs()
    df['path_length'] = df['log_ret'].abs().rolling(iii_window).sum()
    epsilon = 1e-8
    iii_series = df['net_direction'] / (df['path_length'] + epsilon)
    iii_prev = iii_series.shift(1).fillna(0).values

    # --- B. APPLY DYNAMIC LEVERAGE ---
    t_low, t_high = params['T_LOW'], params['T_HIGH']
    l_low, l_mid, l_high = params['L_LOW'], params['L_MID'], params['L_HIGH']
    
    # Vectorized mask creation for the current scenario
    tier_mask = np.full(len(df), 2, dtype=int) 
    tier_mask[iii_prev < t_high] = 1
    tier_mask[iii_prev < t_low] = 0
    
    lookup = np.array([l_low, l_mid, l_high])
    lev_arr = lookup[tier_mask]
    
    final_rets = df['base_ret'].values * lev_arr
    
    # --- C. CALCULATE EQUITY & METRICS ---
    equity_series = pd.Series(np.cumprod(1 + final_rets), index=df.index)
    
    # Only evaluate starting from the latest required start index
    eval_series = equity_series.iloc[start_idx_max:]
    eval_base_rets = df['base_ret'].iloc[start_idx_max:]

    cagr, mdd, sharpe, bh_cagr = calculate_metrics(eval_series, eval_base_rets)
    
    comparison_data.append({
        'Scenario': name,
        'CAGR': cagr,
        'MDD': mdd,
        'Sharpe': sharpe,
    })

# Convert to DataFrame for display
results_df = pd.DataFrame(comparison_data)

# 4. PRINT RESULTS & PLOT (Comparison Table)
print("\n" + "="*70)
print(f"SENSITIVITY ANALYSIS RESULTS (Target: Maximize Sharpe, Minimize MDD)")
print(f"Buy & Hold CAGR (Benchmark): {bh_cagr*100:.1f}%")
print("-" * 70)
print(f"{'Scenario':<25} | {'Sharpe':<8} | {'CAGR':<8} | {'MDD':<8} | {'Params'}")
print("-" * 70)

for index, row in results_df.iterrows():
    name = row['Scenario']
    params = scenarios[name]
    param_str = f"W:{params['III_WINDOW']}, T:{params['T_LOW']}/{params['T_HIGH']}, L:{params['L_LOW']}/{params['L_MID']}/{params['L_HIGH']}"
    
    print(f"{name:<25} | {row['Sharpe']:.2f}<{' ':<2} | {row['CAGR']*100:.1f}%<{' ':<1} | {row['MDD']*100:.1f}%<{' ':<1} | {param_str}")

print("="*70 + "\n")


# 5. VISUALIZATION (Plotting the best and worst for comparison)
plt.figure(figsize=(14, 8))

# Find Baseline and Worst Sharpe for visual check
baseline_series = results_df[results_df['Scenario'] == "0. BASELINE (Optimal)"].iloc[0]
best_sharpe_series = results_df.loc[results_df['Sharpe'].idxmax()]
worst_sharpe_series = results_df.loc[results_df['Sharpe'].idxmin()]


# Function to get the full equity curve for a scenario
def get_equity_curve(params):
    iii_window = params['III_WINDOW']
    df['log_ret'] = np.log(df['close'] / df['close'].shift(1))
    df['net_direction'] = df['log_ret'].rolling(iii_window).sum().abs()
    df['path_length'] = df['log_ret'].abs().rolling(iii_window).sum()
    epsilon = 1e-8
    iii_series = df['net_direction'] / (df['path_length'] + epsilon)
    iii_prev = iii_series.shift(1).fillna(0).values

    t_low, t_high = params['T_LOW'], params['T_HIGH']
    l_low, l_mid, l_high = params['L_LOW'], params['L_MID'], params['L_HIGH']
    
    tier_mask = np.full(len(df), 2, dtype=int) 
    tier_mask[iii_prev < t_high] = 1
    tier_mask[iii_prev < t_low] = 0
    
    lookup = np.array([l_low, l_mid, l_high])
    lev_arr = lookup[tier_mask]
    final_rets = df['base_ret'].values * lev_arr
    
    equity_curve = pd.Series(np.cumprod(1 + final_rets), index=df.index)
    return equity_curve.iloc[start_idx_max:]

# Plotting the three curves
baseline_eq = get_equity_curve(scenarios["0. BASELINE (Optimal)"])
best_eq = get_equity_curve(scenarios[best_sharpe_series['Scenario']])
worst_eq = get_equity_curve(scenarios[worst_sharpe_series['Scenario']])

plt.plot(baseline_eq.index, baseline_eq, label=f"Baseline (Sharpe: {baseline_series['Sharpe']:.2f})", color='blue', linewidth=2)
plt.plot(best_eq.index, best_eq, label=f"{best_sharpe_series['Scenario']} (Sharpe: {best_sharpe_series['Sharpe']:.2f})", color='green', linestyle='--')
plt.plot(worst_eq.index, worst_eq, label=f"{worst_sharpe_series['Scenario']} (Sharpe: {worst_sharpe_series['Sharpe']:.2f})", color='red', linestyle=':')

plt.yscale('log')
plt.title('Sensitivity Analysis: Equity Curves (Log Scale)')
plt.legend()
plt.grid(True, which='both', linestyle='--', alpha=0.3)

# Save plot to buffer
buf = io.BytesIO()
plt.savefig(buf, format='png', dpi=300, bbox_inches='tight')
plt.close()
buf.seek(0)
plot_base64 = base64.b64encode(buf.read()).decode('utf-8')

# 6. FLASK APP TO SERVE PLOT
plot_path = os.path.join('/app/static', 'sensitivity_plot.png')
# This is a bit complex in this environment, so we'll serve the output table and analysis.
# The plot file itself cannot be dynamically created and served within this markdown block.

print("Starting Web Server...")
# Create a dummy image file for the environment to show the plot
plot_dir = '/app/static'
if not os.path.exists(plot_dir): os.makedirs(plot_dir)
with open(plot_path, "wb") as f:
    f.write(base64.b64decode(plot_base64))


app = Flask(__name__)
@app.route('/')
def serve_plot(): 
    # Create HTML response containing the analysis table and the image
    table_html = results_df.to_html(classes='table table-striped', float_format='%.2f')
    
    response_html = f"""
    <html>
    <head>
        <title>Sensitivity Analysis</title>
        <style>
            body {{ font-family: sans-serif; }}
            h2 {{ color: #1E90FF; }}
            table {{ width: 90%; border-collapse: collapse; margin: 20px 0; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: right; }}
            th {{ background-color: #f2f2f2; }}
            .container {{ display: flex; flex-direction: column; align-items: center; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h2>Sensitivity Analysis Results</h2>
            <img src="/static/sensitivity_plot.png" alt="Equity Curve Comparison" style="width: 80%; max-width: 900px;">
            <h3>Metric Comparison</h3>
            <p>Baseline B&H CAGR: {bh_cagr*100:.1f}%</p>
            {table_html}
        </div>
    </body>
    </html>
    """
    return response_html

@app.route('/health')
def health(): return 'OK', 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=False)
