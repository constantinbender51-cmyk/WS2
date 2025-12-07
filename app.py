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

# 1. ENSEMBLE PARAMETERS (The "Cloud")
# We will generate a mesh of trend combinations
FAST_MAS = [30, 35, 40, 45, 50, 55]
SLOW_MAS = [100, 105, 110, 115, 120, 125, 130]

# III Parameters (Robust Base)
III_WINDOW = 35
# We use the wider, more tolerant thresholds found in the WFO
T_LOW = 0.15
T_HIGH = 0.25

# 2. LEVERAGE SCALING
# Instead of fixed tiers, we scale based on Trend Confidence + III Zone
# Maximum theoretical leverage if EVERYTHING aligns
MAX_LEV = 4.0 
MIN_LEV = 0.0

def fetch_data():
    exchange = ccxt.binance()
    since = exchange.parse8601(start_date_str=START_DATE)
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
    roll_max = series.cummax()
    dd = (series - roll_max) / roll_max
    max_dd = dd.min()
    ret_pct = series.pct_change().fillna(0)
    sharpe = (ret_pct.mean() / ret_pct.std()) * np.sqrt(365) if ret_pct.std() != 0 else 0
    return total_ret, cagr, max_dd, sharpe

# --- EXECUTION ---
df = fetch_data()

# 1. BUILD THE ENSEMBLE
print(f"Building Ensemble with {len(FAST_MAS) * len(SLOW_MAS)} Strategy Pairs...")

# We create a dataframe just for votes
votes_df = pd.DataFrame(index=df.index)
vote_count = 0

for f in FAST_MAS:
    for s in SLOW_MAS:
        if f >= s: continue # Skip invalid pairs
        
        # Calculate MAs
        sma_f = df['close'].rolling(f).mean()
        sma_s = df['close'].rolling(s).mean()
        
        # Vote Logic: 1 if Bullish, 0 if Bearish/Neutral
        # Using strict Close > Fast > Slow
        vote = ((df['close'] > sma_f) & (df['close'] > sma_s)).astype(int)
        
        votes_df[f'vote_{f}_{s}'] = vote
        vote_count += 1

# Calculate "Trend Confidence" (0.0 to 1.0)
# We shift by 1 to avoid lookahead bias!
df['trend_consensus'] = votes_df.sum(axis=1).shift(1) / vote_count
df['trend_consensus'] = df['trend_consensus'].fillna(0)

# 2. CALCULATE III (Market Efficiency)
df['log_ret'] = np.log(df['close'] / df['close'].shift(1))
df['net_direction'] = df['log_ret'].rolling(III_WINDOW).sum().abs()
df['path_length'] = df['log_ret'].abs().rolling(III_WINDOW).sum()
df['iii'] = df['net_direction'] / (df['path_length'] + 1e-9)
prev_iii = df['iii'].shift(1).fillna(0) # Avoid lookahead

# 3. DYNAMIC LEVERAGE LOGIC
# We combine Trend Consensus (Direction) with III (Quality)

leverage_signal = []

for i in range(len(df)):
    consensus = df['trend_consensus'].iloc[i]
    eff = prev_iii.iloc[i]
    
    # Base Leverage based on Trend Strength
    # If consensus < 50%, we are mostly out.
    # If consensus > 80%, we are fully committed.
    
    if consensus < 0.50:
        # NO TREND / BEARISH
        # Only take tiny positions if Efficiency is very low (Mean Reversion attempts)
        if eff < 0.12: lev = 0.5 # Defensive noise trading
        else: lev = 0.0          # Cash
        
    else:
        # BULLISH TREND
        # We scale leverage based on HOW efficient the trend is
        
        if eff < T_LOW: 
            # Trend exists, but it's noisy (Choppy up-trend)
            # Wall of Worry starting?
            lev = 1.0 * (consensus) # Max 1x
            
        elif eff >= T_LOW and eff < T_HIGH:
            # THE SWEET SPOT (Wall of Worry)
            # This is where we want maximum exposure
            # Scale from 2x to MAX_LEV based on Consensus
            lev = 2.0 + (MAX_LEV - 2.0) * ((consensus - 0.5) / 0.5)
            
        else: # eff >= T_HIGH
            # EUPHORIA (High Efficiency)
            # Trend is perfect, but crash risk is high.
            # We cap leverage to avoid the "knife"
            lev = 1.5
            
    leverage_signal.append(lev)

df['leverage'] = leverage_signal

# 4. CALCULATE RETURNS
df['daily_ret'] = df['close'].pct_change().fillna(0)
df['strat_ret'] = df['daily_ret'] * df['leverage']
df['equity'] = (1 + df['strat_ret']).cumprod()
df['bh_equity'] = (1 + df['daily_ret']).cumprod()

# --- PLOTTING ---
tot, cagr, mdd, sharpe = calculate_metrics(df['equity'])
bh_tot, bh_cagr, bh_mdd, bh_sharpe = calculate_metrics(df['bh_equity'])

print(f"ENSEMBLE STRATEGY RESULTS")
print(f"Sharpe: {sharpe:.2f}")
print(f"CAGR: {cagr*100:.2f}%")
print(f"MaxDD: {mdd*100:.2f}%")

plt.figure(figsize=(12, 12))

# Equity
ax1 = plt.subplot(3, 1, 1)
ax1.plot(df.index, df['equity'], color='blue', label=f'Ensemble Strategy (Sharpe: {sharpe:.2f})')
ax1.plot(df.index, df['bh_equity'], color='gray', alpha=0.5, label='Buy & Hold')
ax1.set_yscale('log')
ax1.set_title('Ensemble "Cloud" Strategy vs Buy & Hold')
ax1.legend()
ax1.grid(True, linestyle='--', alpha=0.3)

# Trend Consensus
ax2 = plt.subplot(3, 1, 2, sharex=ax1)
ax2.plot(df.index, df['trend_consensus'], color='orange', alpha=0.8, label='Trend Consensus (0-100%)')
ax2.axhline(0.5, color='black', linestyle='--', alpha=0.3)
ax2.fill_between(df.index, 0.8, 1.0, color='green', alpha=0.1, label='Strong Trend Zone')
ax2.set_ylabel('Market Alignment')
ax2.legend(loc='upper left')
ax2.grid(True, alpha=0.3)

# Leverage
ax3 = plt.subplot(3, 1, 3, sharex=ax1)
ax3.plot(df.index, df['leverage'], color='purple', linewidth=1)
ax3.fill_between(df.index, 0, df['leverage'], color='purple', alpha=0.2)
ax3.set_ylabel('Leverage')
ax3.set_title(f'Dynamic Leverage (Max {MAX_LEV}x)')
ax3.grid(True, alpha=0.3)

plt.tight_layout()
plot_path = os.path.join('/app/static', 'ensemble_results.png')
if not os.path.exists('/app/static'): os.makedirs('/app/static')
plt.savefig(plot_path)

app = Flask(__name__)
@app.route('/')
def serve_plot(): return send_file(plot_path, mimetype='image/png')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
