import ccxt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from flask import Flask, send_file
import os

# --- CONFIGURATION ---
SYMBOL = 'BTC/USDT'
START_DATE = '2018-01-01 00:00:00'
TIMEFRAME = '1d'

# --- ROBUST PARAMETERS (Derived from WFO & Heatmap Analysis) ---
# SMA: WFO selected 50/120 or 50/110 consistently.
SMA_FAST = 50
SMA_SLOW = 120

# Thresholds: WFO selected 0.15/0.25 for majority of history.
# Heatmap confirms 0.25 is safe.
T_LOW = 0.15
T_HIGH = 0.25

# Leverage: WFO rejected "Aggressive" (4.5x) for "Balanced" (3.0x).
L_LOW = 0.25   # Defensive / Noise
L_MID = 3.00   # Wall of Worry (The Sweet Spot)
L_HIGH = 1.50  # Euphoria (Safety de-leverage)

# III Window: 35 was the unanimous winner.
III_WINDOW = 35

def fetch_data():
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

def calculate_metrics(series):
    total_ret = (series.iloc[-1] / series.iloc[0]) - 1
    days = (series.index[-1] - series.index[0]).days
    cagr = (series.iloc[-1] / series.iloc[0]) ** (365/days) - 1
    
    # Max Drawdown
    roll_max = series.cummax()
    dd = (series - roll_max) / roll_max
    max_dd = dd.min()
    
    # Sharpe
    ret_pct = series.pct_change().fillna(0)
    sharpe = (ret_pct.mean() / ret_pct.std()) * np.sqrt(365) if ret_pct.std() != 0 else 0
    
    return total_ret, cagr, max_dd, sharpe

# --- EXECUTION ---
df = fetch_data()

# 1. Indicators
df['sma_fast'] = df['close'].rolling(SMA_FAST).mean()
df['sma_slow'] = df['close'].rolling(SMA_SLOW).mean()

df['log_ret'] = np.log(df['close'] / df['close'].shift(1))
df['iii'] = (df['log_ret'].rolling(III_WINDOW).sum().abs() / 
             (df['log_ret'].abs().rolling(III_WINDOW).sum() + 1e-9))

# 2. Strategy Logic
base_returns = []
leverage_log = []

# Shift vars for signal generation to avoid lookahead
prev_close = df['close'].shift(1)
prev_fast = df['sma_fast'].shift(1)
prev_slow = df['sma_slow'].shift(1)
prev_iii = df['iii'].shift(1)

# Vectorized Signal (Much faster)
# 1. Trend Filter
trend_long = (prev_close > prev_fast) & (prev_close > prev_slow)
trend_short = (prev_close < prev_fast) & (prev_close < prev_slow)

# 2. Leverage Sizing
# Default to High Efficiency Mode (Euphoria)
lev_series = pd.Series(L_HIGH, index=df.index)

# Mid Efficiency (Wall of Worry)
mask_mid = (prev_iii >= T_LOW) & (prev_iii < T_HIGH)
lev_series[mask_mid] = L_MID

# Low Efficiency (Noise)
mask_low = (prev_iii < T_LOW)
lev_series[mask_low] = L_LOW

# 3. Apply Returns
daily_ret = df['close'].pct_change().fillna(0)
strategy_ret = pd.Series(0.0, index=df.index)

# Long Logic
strategy_ret[trend_long] = daily_ret[trend_long] * lev_series[trend_long]
# Short Logic (Inverse return)
strategy_ret[trend_short] = -daily_ret[trend_short] * lev_series[trend_short]

# 4. Simulation
df['equity'] = (1 + strategy_ret).cumprod()
df['bh_equity'] = (1 + daily_ret).cumprod()

# --- METRICS & PLOT ---
tot, cagr, mdd, sharpe = calculate_metrics(df['equity'])
bh_tot, bh_cagr, bh_mdd, bh_sharpe = calculate_metrics(df['bh_equity'])

print(f"ROBUST STRATEGY RESULTS (50/120 | III 35 | {T_LOW}-{T_HIGH})")
print(f"CAGR: {cagr*100:.2f}% (vs BH {bh_cagr*100:.2f}%)")
print(f"MaxDD: {mdd*100:.2f}% (vs BH {bh_mdd*100:.2f}%)")
print(f"Sharpe: {sharpe:.2f} (vs BH {bh_sharpe:.2f})")

plt.figure(figsize=(12, 8))
ax1 = plt.subplot(2, 1, 1)
ax1.plot(df.index, df['equity'], label=f'Robust Strategy (Sharpe: {sharpe:.2f})', color='blue')
ax1.plot(df.index, df['bh_equity'], label='Buy & Hold', color='gray', alpha=0.5)
ax1.set_yscale('log')
ax1.set_title(f'Robust Strategy vs Buy & Hold\nParams: SMA 50/120 | III {T_LOW}-{T_HIGH} | Lev {L_LOW}x-{L_MID}x-{L_HIGH}x')
ax1.legend()
ax1.grid(True, which='both', linestyle='--', alpha=0.3)

ax2 = plt.subplot(2, 1, 2, sharex=ax1)
ax2.plot(df.index, lev_series, color='purple', label='Leverage Used', linewidth=0.8)
ax2.set_ylabel('Leverage')
ax2.set_title('Leverage Regime Switching')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plot_path = os.path.join('/app/static', 'robust_test.png')
if not os.path.exists('/app/static'): os.makedirs('/app/static')
plt.savefig(plot_path)

app = Flask(__name__)
@app.route('/')
def serve_plot(): return send_file(plot_path, mimetype='image/png')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
