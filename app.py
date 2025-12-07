import ccxt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from flask import Flask, send_file
import os
import itertools

# 1. CONFIGURATION
symbol = 'BTC/USDT'
timeframe = '1d'
start_date_str = '2018-01-01 00:00:00'

# Strategy Params
SMA_FAST = 57
SMA_SLOW = 128
SL_PCT = 0.02
TP_PCT = 0.16
III_WINDOW = 35 

# --- GRID SEARCH SPACE DEFINITION (5 VARIABLES) ---

# Threshold Search Space (T_Low, T_High): 0.13 and 0.18 only
THRESH_RANGE = np.array([0.13, 0.18]) 

# Leverage Search Space (L_Low, L_Mid, L_High): 0.5, 4.5, and 2.45 only
LEV_RANGE = np.array([0.5, 4.5, 2.45]) 

# MDD Constraint
MAX_MDD_CONSTRAINT = -0.50 # Must be less than 50% drawdown

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

# 2. DATA PREP & BASE RETURNS
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

# Pre-calculate 1x Strategy Returns
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
    
    # Trend Logic
    if prev_close > prev_fast and prev_close > prev_slow:
        entry = open_p
        sl = entry * (1 - SL_PCT)
        tp = entry * (1 + TP_PCT)
        if low_p <= sl: daily_ret = -SL_PCT
        elif high_p >= tp: daily_ret = TP_PCT
        else: daily_ret = (close_p - entry) / entry
        
    elif prev_close < prev_fast and prev_close < prev_slow:
        entry = open_p
        sl = entry * (1 + SL_PCT)
        tp = entry * (1 - TP_PCT)
        if high_p >= sl: daily_ret = -SL_PCT
        elif low_p <= tp: daily_ret = TP_PCT
        else: daily_ret = (entry - close_p) / entry
        
    base_returns.append(daily_ret)

df['base_ret'] = base_returns

# 3. LEVERAGE GRID SEARCH (Vectorized for Speed)
total_iterations = len(THRESH_RANGE) * len(THRESH_RANGE) * len(LEV_RANGE)**3
print(f"Starting Exhaustive 5-Variable Grid Search ({total_iterations} total combinations)...")

base_ret_arr = np.array(base_returns)
iii_prev = df['iii'].shift(1).fillna(0).values

best_sharpe = -999
best_combo = (0.13, 0.18, 0.5, 4.5, 2.45) # T_Low, T_High, L_Low, L_Mid, L_High
best_mdd = 0

# Metrics function optimized for arrays


def get_final_metrics(equity_series):
    """Calculate total return, CAGR, max drawdown, and Sharpe ratio from equity series."""
    if len(equity_series) < 2:
        return 0, 0, 0, 0
    
    # Total return
    total_return = equity_series.iloc[-1] / equity_series.iloc[0] - 1
    
    # CAGR
    years = (equity_series.index[-1] - equity_series.index[0]).days / 365.25
    if years > 0:
        cagr = (equity_series.iloc[-1] / equity_series.iloc[0]) ** (1 / years) - 1
    else:
        cagr = 0
    
    # Max drawdown
    roll_max = equity_series.cummax()
    drawdown = (equity_series - roll_max) / roll_max
    max_dd = drawdown.min()
    
    # Sharpe ratio (assuming daily returns)
    daily_returns = equity_series.pct_change().dropna()
    if len(daily_returns) > 0:
        sharpe = np.sqrt(365) * daily_returns.mean() / daily_returns.std() if daily_returns.std() > 0 else 0
    else:
        sharpe = 0
    
    return total_return, cagr, max_dd, sharpedef calculate_sharpe_mdd(returns):
    cum_ret = np.cumprod(1 + returns)
    if cum_ret.size == 0 or cum_ret.iloc[0] == 0: return 0, 0
    
    # Sharpe
    mean_ret = np.mean(returns)
    std_ret = np.std(returns)
    sharpe = (mean_ret / std_ret) * np.sqrt(365) if std_ret > 0 else 0
    
    # MDD
    roll_max = np.maximum.accumulate(cum_ret)
    drawdown = (cum_ret - roll_max) / roll_max
    max_dd = drawdown.min()
    
    return sharpe, max_dd

# Outer loop: Thresholds (T_Low, T_High)
for t_low, t_high in itertools.product(THRESH_RANGE, repeat=2):
    # Enforce logical constraint
    if t_low >= t_high: continue 
    
    # Create Tier Mask for this specific threshold combo
    # 0 = Low Tier (III < T_Low)
    # 1 = Mid Tier (T_Low <= III < T_High)
    # 2 = High Tier (III >= T_High)
    
    # Vectorized mask creation
    tier_mask = np.full(len(df), 2, dtype=int) # Default High
    tier_mask[iii_prev < t_high] = 1 # Mid
    tier_mask[iii_prev < t_low] = 0  # Low

    # Inner loop: Leverages (L_Low, L_Mid, L_High)
    for l_low, l_mid, l_high in itertools.product(LEV_RANGE, repeat=3):
        
        # Construct leverage array using the calculated tiers
        lookup = np.array([l_low, l_mid, l_high])
        lev_arr = lookup[tier_mask]
        
        final_rets = base_ret_arr * lev_arr
        
        # Calculate Sharpe and MDD (Only analyze period where strategy is active)
        sharpe, mdd = calculate_sharpe_mdd(pd.Series(final_rets[start_idx:]))
        
        # Check against MDD constraint
        if mdd > MAX_MDD_CONSTRAINT: 
            if sharpe > best_sharpe:
                best_sharpe = sharpe
                best_combo = (round(t_low, 2), round(t_high, 2), round(l_low, 2), round(l_mid, 2), round(l_high, 2))
                best_mdd = mdd


# 4. FINAL BACKTEST WITH FIXED PARAMS
# Use the best combination from grid search
OPT_T_LOW, OPT_T_HIGH, OPT_L_LOW, OPT_L_MID, OPT_L_HIGH = best_combo

# Create tier mask for best thresholds
tier_mask_final = np.full(len(df), 2, dtype=int) # Default High
tier_mask_final[iii_prev < OPT_T_HIGH] = 1 # Mid
tier_mask_final[iii_prev < OPT_T_LOW] = 0  # Low

# Construct leverage array for best leverages
lookup_final = np.array([OPT_L_LOW, OPT_L_MID, OPT_L_HIGH])
lev_arr_final = lookup_final[tier_mask_final]
final_rets_final = base_ret_arr * lev_arr_final

# Backtest simulation for plot data
df['strategy_equity'] = 1.0
df['leverage_used'] = 0.0
equity = 1.0
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

# 5. METRICS & PLOT
plot_data = df.iloc[start_idx:].copy()
s_tot, s_cagr, s_mdd, s_sharpe = get_final_metrics(plot_data['strategy_equity'])
# Calculate buy & hold equity for comparison
plot_data['buy_hold_equity'] = plot_data['close'] / plot_data['close'].iloc[0]

# Calculate strategy metrics
s_tot, s_cagr, s_mdd, s_sharpe = get_final_metrics(plot_data['strategy_equity'])

print("\n" + "="*45)
print(f"FIXED PARAMETERS BASED ON USER INPUT (Constrained MDD < {MAX_MDD_CONSTRAINT*100:.0f}%)")
print(f"Fixed Thresholds: {OPT_T_LOW:.2f} (Low) / {OPT_T_HIGH:.2f} (High)")
print(f"Fixed Leverages: {OPT_L_LOW:.1f}x / {OPT_L_MID:.1f}x / {OPT_L_HIGH:.1f}x")
print("-" * 45)
print(f"{'Sharpe Ratio':<15} | {s_sharpe:>10.2f}")
print(f"{'Max Drawdown':<15} | {s_mdd*100:>10.1f}%")
print(f"{'CAGR':<15} | {s_cagr*100:>10.1f}%")
print("="*45 + "\n")

plt.figure(figsize=(12, 10))

ax1 = plt.subplot(3, 1, 1)
ax1.plot(plot_data.index, plot_data['strategy_equity'], label=f'Best Strategy (Sharpe: {s_sharpe:.2f})', color='blue')
ax1.plot(plot_data.index, plot_data['buy_hold_equity'], label='Buy & Hold', color='gray', alpha=0.5)
ax1.set_yscale('log')
ax1.set_title(f'Strategy with Fixed Parameters (T: {OPT_T_LOW}/{OPT_T_HIGH} | L: {OPT_L_LOW}x/{OPT_L_MID}x/{OPT_L_HIGH}x)')
ax1.legend()
ax1.grid(True, which='both', linestyle='--', alpha=0.3)

# Add Stats Box
stats = f"CAGR: {s_cagr*100:.1f}%\nMaxDD: {s_mdd*100:.1f}%\nSharpe: {s_sharpe:.2f}"
ax1.text(0.02, 0.85, stats, transform=ax1.transAxes, bbox=dict(facecolor='white', alpha=0.8))

ax2 = plt.subplot(3, 1, 2, sharex=ax1)
ax2.step(plot_data.index, plot_data['leverage_used'], where='post', color='purple', linewidth=1)
ax2.fill_between(plot_data.index, 0, plot_data['leverage_used'], step='post', color='purple', alpha=0.2)
ax2.set_title('Leverage Deployed')
ax2.set_yticks(np.unique(plot_data['leverage_used']))
ax2.set_ylabel('Leverage (x)')
ax2.grid(True, axis='x', alpha=0.3)

ax3 = plt.subplot(3, 1, 3, sharex=ax1)
# Drawdown
roll_max = plot_data['strategy_equity'].cummax()
dd = (plot_data['strategy_equity'] - roll_max) / roll_max
ax3.plot(plot_data.index, dd, color='red')
ax3.fill_between(plot_data.index, dd, 0, color='red', alpha=0.1)
ax3.axhline(MAX_MDD_CONSTRAINT, color='black', linestyle='--')
ax3.set_title('Drawdown Profile (Constrained < 50%)')
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
