import ccxt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from flask import Flask, send_file
import os

# 1. CONFIGURATION
symbol = 'BTC/USDT'
timeframe = '1d'
start_date_str = '2018-01-01 00:00:00'

# Strategy Params
SMA_FAST = 40
SMA_SLOW = 120
SL_PCT = 0.02
TP_PCT = 0.16
III_WINDOW = 14 

# --- OPTIMIZED PARAMETERS (Confirmed by user's optimization results) ---
OPT_T_LOW = 0.10
OPT_T_HIGH = 0.50

# Using 0.0, 1.0, 0.5 as confirmed by the max Sharpe search results
OPT_L_LOW = 0.0  # Cash (III < 0.10)
OPT_L_MID = 1.0  # Standard risk (0.10 <= III < 0.50)
OPT_L_HIGH = 0.5 # Ultra-defensive (III >= 0.50)

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

# Helper to calculate metrics
def get_final_metrics(equity_series):
    ret = equity_series.pct_change().fillna(0)
    days = (equity_series.index[-1] - equity_series.index[0]).days
    total_ret = (equity_series.iloc[-1] / equity_series.iloc[0]) - 1
    cagr = (equity_series.iloc[-1] / equity_series.iloc[0]) ** (365.0 / days) - 1
    roll_max = equity_series.cummax()
    drawdown = (equity_series - roll_max) / roll_max
    max_dd = drawdown.min()
    sharpe = (ret.mean() / ret.std()) * np.sqrt(365) if ret.std() != 0 else 0
    return total_ret, cagr, max_dd, sharpe


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

# 3. BASE RETURNS (1x)
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

# 4. FINAL BACKTEST WITH OPTIMIZED PARAMS
iii_prev = df['iii'].shift(1).fillna(0).values

# Create optimized leverage array
tier_mask_final = np.full(len(df), 2, dtype=int) 
tier_mask_final[iii_prev < OPT_T_HIGH] = 1
tier_mask_final[iii_prev < OPT_T_LOW] = 0

lookup_final = np.array([OPT_L_LOW, OPT_L_MID, OPT_L_HIGH])
lev_arr_final = lookup_final[tier_mask_final]
final_rets_final = np.array(df['base_ret']) * lev_arr_final

# Backtest simulation for plot data
df['strategy_equity'] = 1.0
df['buy_hold_equity'] = 1.0
df['leverage_used'] = 0.0
equity = 1.0
hold_equity = 1.0
is_busted = False

for i in range(start_idx, len(df)):
    daily_ret = final_rets_final[i]
    leverage = lev_arr_final[i]
    
    if not is_busted:
        equity *= (1 + daily_ret)
        if equity <= 0.05:
            equity = 0
            is_busted = True
            
    df.at[df.index[i], 'strategy_equity'] = equity
    df.at[df.index[i], 'leverage_used'] = leverage
    
    # Buy Hold
    close_p = df['close'].iloc[i]
    prev_p = df['close'].iloc[i-1]
    bh_ret = (close_p - prev_p) / prev_p
    hold_equity *= (1 + bh_ret)
    df.at[df.index[i], 'buy_hold_equity'] = hold_equity


# 5. METRICS & PLOT
plot_data = df.iloc[start_idx:].copy()
s_tot, s_cagr, s_mdd, s_sharpe = get_final_metrics(plot_data['strategy_equity'])

plt.figure(figsize=(12, 10))

ax1 = plt.subplot(3, 1, 1)
ax1.plot(plot_data.index, plot_data['strategy_equity'], label=f'Final Strategy (Sharpe: {s_sharpe:.2f})', color='blue')
ax1.plot(plot_data.index, plot_data['buy_hold_equity'], label='Buy & Hold', color='gray', alpha=0.5)
ax1.set_yscale('log')
ax1.set_title(f'Optimized Strategy Equity (Sharpe: {s_sharpe:.2f})')
ax1.legend()
ax1.grid(True, which='both', linestyle='--', alpha=0.3)

# Add Stats Box
stats = f"CAGR: {s_cagr*100:.1f}%\nMaxDD: {s_mdd*100:.1f}%\nSharpe: {s_sharpe:.2f}"
ax1.text(0.02, 0.85, stats, transform=ax1.transAxes, bbox=dict(facecolor='white', alpha=0.8))

ax2 = plt.subplot(3, 1, 2, sharex=ax1)
ax2.step(plot_data.index, plot_data['leverage_used'], where='post', color='purple', linewidth=1)
ax2.fill_between(plot_data.index, 0, plot_data['leverage_used'], step='post', color='purple', alpha=0.2)
ax2.set_title('Leverage Deployment (0.0x / 1.0x / 0.5x)')
ax2.set_yticks(np.unique(plot_data['leverage_used']))
ax2.set_ylabel('Leverage (x)')
ax2.grid(True, axis='x', alpha=0.3)

ax3 = plt.subplot(3, 1, 3, sharex=ax1)
# Drawdown
roll_max = plot_data['strategy_equity'].cummax()
dd = (plot_data['strategy_equity'] - roll_max) / roll_max
ax3.plot(plot_data.index, dd, color='red')
ax3.fill_between(plot_data.index, dd, 0, color='red', alpha=0.1)
ax3.set_title('Drawdown Profile')
ax3.set_ylabel('Drawdown')
ax3.grid(True, alpha=0.3)

plt.tight_layout()
plot_dir = '/app/static'
if not os.path.exists(plot_dir): os.makedirs(plot_dir)
plot_path = os.path.join(plot_dir, 'plot.png')
plt.savefig(plot_path, dpi=300, bbox_inches='tight')

app = Flask(__name__)
@app.route('/')
def serve_plot(): return send_file(plot_path, mimetype='image/png')
@app.route('/health')
def health(): return 'OK', 200

if __name__ == '__main__':
    print("Starting Web Server...")
    app.run(host='0.0.0.0', port=8080, debug=False)
