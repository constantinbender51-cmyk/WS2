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

# OPTIMIZED PARAMETERS (From Grid Search)
THRESH_LOW = 0.05
THRESH_HIGH = 0.10

# LEVERAGE TIERS
LEV_LOW = 1.0
LEV_MID = 2.0
LEV_HIGH = 4.0

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

# 3. BACKTEST
df['strategy_equity'] = 1.0
df['buy_hold_equity'] = 1.0
df['leverage_used'] = 1.0
equity = 1.0
hold_equity = 1.0
start_idx = max(SMA_SLOW, III_WINDOW)
is_busted = False

for i in range(start_idx, len(df)):
    today = df.index[i]
    prev_close = df['close'].iloc[i-1]
    prev_fast = df['sma_fast'].iloc[i-1]
    prev_slow = df['sma_slow'].iloc[i-1]
    
    # --- OPTIMIZED LEVERAGE LOGIC ---
    prev_iii = df['iii'].iloc[i-1]
    
    if prev_iii < THRESH_LOW:
        leverage = LEV_LOW
    elif prev_iii < THRESH_HIGH:
        leverage = LEV_MID
    else:
        leverage = LEV_HIGH # We are here most of the time!
        
    df.at[today, 'leverage_used'] = leverage
    
    # Execution
    open_p = df['open'].iloc[i]
    high_p = df['high'].iloc[i]
    low_p = df['low'].iloc[i]
    close_p = df['close'].iloc[i]
    
    base_ret = 0.0
    
    # Trend Logic
    if prev_close > prev_fast and prev_close > prev_slow:
        entry = open_p
        sl = entry * (1 - SL_PCT)
        tp = entry * (1 + TP_PCT)
        if low_p <= sl: base_ret = -SL_PCT
        elif high_p >= tp: base_ret = TP_PCT
        else: base_ret = (close_p - entry) / entry
        
    elif prev_close < prev_fast and prev_close < prev_slow:
        entry = open_p
        sl = entry * (1 + SL_PCT)
        tp = entry * (1 - TP_PCT)
        if high_p >= sl: base_ret = -SL_PCT
        elif low_p <= tp: base_ret = TP_PCT
        else: base_ret = (entry - close_p) / entry
        
    if not is_busted:
        daily_ret = base_ret * leverage
        equity *= (1 + daily_ret)
        if equity <= 0.05:
            equity = 0
            is_busted = True
            
    # Buy Hold
    bh_ret = (close_p - df['close'].iloc[i-1]) / df['close'].iloc[i-1]
    hold_equity *= (1 + bh_ret)
    
    df.at[today, 'strategy_equity'] = equity
    df.at[today, 'buy_hold_equity'] = hold_equity

# 4. METRICS & PLOT
def get_metrics(equity_series):
    ret = equity_series.pct_change().fillna(0)
    total_ret = (equity_series.iloc[-1] / equity_series.iloc[0]) - 1
    days = (equity_series.index[-1] - equity_series.index[0]).days
    cagr = (equity_series.iloc[-1] / equity_series.iloc[0]) ** (365.0 / days) - 1
    roll_max = equity_series.cummax()
    drawdown = (equity_series - roll_max) / roll_max
    max_dd = drawdown.min()
    sharpe = (ret.mean() / ret.std()) * np.sqrt(365) if ret.std() != 0 else 0
    return total_ret, cagr, max_dd, sharpe

s_tot, s_cagr, s_mdd, s_sharpe = get_metrics(df['strategy_equity'])

print(f"OPTIMIZED RESULTS (Low={THRESH_LOW}, High={THRESH_HIGH})")
print(f"Sharpe Ratio: {s_sharpe:.2f}")
print(f"Total Return: {s_tot:.2f}x")
print(f"Max Drawdown: {s_mdd*100:.2f}%")

plt.figure(figsize=(12, 10))

ax1 = plt.subplot(2, 1, 1)
plot_data = df.iloc[start_idx:]
ax1.plot(plot_data.index, plot_data['strategy_equity'], label='Optimized Strategy', color='blue')
ax1.plot(plot_data.index, plot_data['buy_hold_equity'], label='Buy & Hold', color='gray', alpha=0.5)
ax1.set_yscale('log')
ax1.set_title(f'Optimized Strategy (Sharpe: {s_sharpe:.2f})')
ax1.legend()
ax1.grid(True, alpha=0.3)

ax2 = plt.subplot(2, 1, 2, sharex=ax1)
ax2.plot(plot_data.index, plot_data['leverage_used'], color='purple', label='Leverage')
ax2.set_title('Leverage Deployment')
ax2.grid(True, alpha=0.3)

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
