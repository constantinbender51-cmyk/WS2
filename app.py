import requests
import pandas as pd
import numpy as np
import time
import sys
import io
import base64
from datetime import datetime
import matplotlib.pyplot as plt
from numba import jit
from flask import Flask, render_template, render_template_string

# -----------------------------------------------------------------------------
# 1. DATA FETCHING
# -----------------------------------------------------------------------------
def fetch_binance_data(symbol="BTCUSDT", interval="1d", start_date="2018-01-01"):
    base_url = "https://api.binance.com/api/v3/klines"
    start_ts = int(datetime.strptime(start_date, "%Y-%m-%d").timestamp() * 1000)
    
    all_data = []
    limit = 1000
    current_start = start_ts
    
    print(f"Fetching {symbol} data from {start_date}...")
    
    while True:
        params = {"symbol": symbol, "interval": interval, "startTime": current_start, "limit": limit}
        try:
            response = requests.get(base_url, params=params)
            data = response.json()
            if not isinstance(data, list) or len(data) == 0: break
            all_data.extend(data)
            current_start = data[-1][6] + 1
            if len(data) < limit: break
            time.sleep(0.05)
        except Exception as e:
            print(f"Error: {e}")
            break
            
    df = pd.DataFrame(all_data, columns=[
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "qav", "num_trades", "taker_base_vol", "taker_quote_vol", "ignore"
    ])
    
    numeric_cols = ["open", "high", "low", "close", "volume"]
    df[numeric_cols] = df[numeric_cols].astype(float)
    df["date"] = pd.to_datetime(df["open_time"], unit="ms")
    df.set_index("date", inplace=True)
    
    return df[["open", "high", "low", "close"]]

# -----------------------------------------------------------------------------
# 2. NUMBA OPTIMIZED LOGIC
# -----------------------------------------------------------------------------
@jit(nopython=True)
def calculate_dual_sma_strategy(opens, highs, lows, intraday_returns, sma1_arr, sma2_arr, x, s, r):
    """
    SMA 1: Logic + Proximity
    SMA 2: Filter
    x: Band width
    s: Stop Loss %
    r: Take Profit %
    """
    n = len(opens)
    strategy_returns = np.zeros(n)
    
    # State Flag for SMA 1: 0 = None, 1 = Just Crossed Up, -1 = Just Crossed Down
    cross_flag = 0 
    
    for i in range(1, n):
        # Skip if either SMA is NaN
        if np.isnan(sma1_arr[i]) or np.isnan(sma2_arr[i]):
            continue
            
        current_open = opens[i]
        prev_open = opens[i-1]
        
        cur_sma1 = sma1_arr[i]
        cur_sma2 = sma2_arr[i]
        
        # --- SMA 1 STATE MACHINE ---
        upper_band = cur_sma1 * (1 + x)
        lower_band = cur_sma1 * (1 - x)
        
        # 1. Determine Cross Event
        if (prev_open < cur_sma1) and (current_open > cur_sma1):
            cross_flag = 1
        elif (prev_open > cur_sma1) and (current_open < cur_sma1):
            cross_flag = -1
            
        # 2. Check Band Exit
        if (current_open > upper_band) or (current_open < lower_band):
            cross_flag = 0
            
        # 3. Determine Base Signal
        signal = 0
        if (current_open > upper_band) or ((current_open > cur_sma1) and (cross_flag == 1)):
            signal = 1
        elif (current_open < lower_band) or ((current_open < cur_sma1) and (cross_flag == -1)):
            signal = -1
            
        # --- SMA 2 FILTER ---
        if signal == 1 and current_open < cur_sma2:
            signal = 0
        elif signal == -1 and current_open > cur_sma2:
            signal = 0
        
        # 4. Calculate Return, Stop Loss & Take Profit
        if signal != 0:
            daily_ret = signal * intraday_returns[i]
            
            # Pessimistic Assumption: Check Stop Loss First
            sl_hit = False
            
            # STOP LOSS LOGIC
            if signal == 1:
                if lows[i] < current_open * (1 - s):
                    daily_ret = np.log(1 - s)
                    sl_hit = True
            elif signal == -1:
                if highs[i] > current_open * (1 + s):
                    daily_ret = np.log(1 - s)
                    sl_hit = True
            
            # TAKE PROFIT LOGIC (Only if SL not hit)
            if not sl_hit:
                if signal == 1:
                    if highs[i] > current_open * (1 + r):
                        daily_ret = np.log(1 + r)
                elif signal == -1:
                    if lows[i] < current_open * (1 - r):
                        daily_ret = np.log(1 + r)
                    
            strategy_returns[i] = daily_ret
            
    return strategy_returns

def calculate_monthly_returns(daily_returns, dates):
    df_returns = pd.DataFrame({'date': dates, 'return': daily_returns})
    df_returns.set_index('date', inplace=True)
    monthly_sum = df_returns.groupby([df_returns.index.year, df_returns.index.month])['return'].sum()
    monthly_returns_list = (np.exp(monthly_sum.values) - 1) * 100 
    return monthly_returns_list.tolist()

# -----------------------------------------------------------------------------
# 3. GRID SEARCH
# -----------------------------------------------------------------------------
def run_dual_sma_grid(df):
    print("\n--- Starting Grid Search (5-Dimensional) ---")
    
    # Adjusted steps for performance (Total combos ~60k, runs in <5s)
    sma1_periods = np.arange(10, 200, 10) # 19 steps
    sma2_periods = np.arange(20, 200, 20) # 9 steps
    x_values = np.arange(0.00, 0.051, 0.01) # 6 steps (0-5%)
    s_values = np.arange(0.02, 0.101, 0.02) # 5 steps (2-10%)
    r_values = np.arange(0.02, 0.201, 0.04) # 5 steps (2-20%)
    
    total_combos = len(sma1_periods)*len(sma2_periods)*len(x_values)*len(s_values)*len(r_values)
    print(f"Scanning {total_combos} combinations...")
    
    opens = df['open'].to_numpy()
    highs = df['high'].to_numpy()
    lows = df['low'].to_numpy()
    closes = df['close'].to_numpy()
    
    with np.errstate(divide='ignore', invalid='ignore'):
        intraday_returns = np.log(closes / opens)
    intraday_returns = np.nan_to_num(intraday_returns)
    
    benchmark_returns = np.log(df['close'] / df['close'].shift(1)).fillna(0).to_numpy()
    
    # Pre-calc SMAs
    sma_cache = {}
    needed_periods = set(np.concatenate((sma1_periods, sma2_periods)))
    for p in needed_periods:
        sma_cache[p] = df['close'].rolling(window=p).mean().shift(1).to_numpy()
    
    best_sharpe = -np.inf
    best_params = None
    best_curve = None
    
    # Heatmap: SMA1 vs SMA2 (Projecting best X, S, R)
    results_matrix = np.zeros((len(sma1_periods), len(sma2_periods)))
    
    start_time = time.time()
    
    for i, p1 in enumerate(sma1_periods):
        sma1_arr = sma_cache[p1]
        for j, p2 in enumerate(sma2_periods):
            sma2_arr = sma_cache[p2]
            
            local_best_sharpe = -np.inf
            
            for x in x_values:
                for s in s_values:
                    for r in r_values:
                        
                        strat_ret = calculate_dual_sma_strategy(
                            opens, highs, lows, intraday_returns, 
                            sma1_arr, sma2_arr, x, s, r
                        )
                        
                        # Calculate Sharpe
                        start_idx = max(p1, p2)
                        active_rets = strat_ret[start_idx:]
                        
                        if len(active_rets) < 10:
                            sharpe = 0
                        else:
                            mean_r = np.mean(active_rets)
                            std_r = np.std(active_rets)
                            sharpe = (mean_r / std_r) * np.sqrt(365) if std_r > 1e-9 else 0
                        
                        if sharpe > local_best_sharpe:
                            local_best_sharpe = sharpe
                        
                        if sharpe > best_sharpe:
                            best_sharpe = sharpe
                            best_params = (p1, p2, x, s, r)
                            best_curve = np.cumsum(strat_ret)
            
            results_matrix[i, j] = local_best_sharpe

    print(f"Optimization complete in {time.time() - start_time:.2f}s.")
    
    monthly_returns = []
    if best_curve is not None:
        daily_returns = np.diff(best_curve, prepend=0)
        leveraged_daily_returns = daily_returns * 3
        monthly_returns = calculate_monthly_returns(leveraged_daily_returns, df.index)
    
    return best_params, best_sharpe, best_curve, benchmark_returns, results_matrix, sma1_periods, sma2_periods, monthly_returns

# -----------------------------------------------------------------------------
# 4. WEB SERVER
# -----------------------------------------------------------------------------
app = Flask(__name__)

@app.route('/')
def index():
    SYMBOL = "BTCUSDT"
    df = fetch_binance_data(SYMBOL)
    
    best_params, best_sharpe, best_curve, benchmark_ret, heatmap_data, smas1, smas2, monthly_returns = run_dual_sma_grid(df)
    
    if best_params is None:
        return "No valid parameters found. Try increasing data range.", 500
    
    best_sma1, best_sma2, best_x, best_s, best_r = best_params
    
    leveraged_curve = best_curve * 3
    
    # Calc Avg 4-month return
    window_size = 120
    if len(leveraged_curve) >= window_size:
        window_returns = []
        for i in range(len(leveraged_curve) - window_size + 1):
            val = leveraged_curve[i + window_size - 1] - (leveraged_curve[i - 1] if i > 0 else 0)
            window_returns.append(val)
        avg_return_4months = np.mean(window_returns) if window_returns else 0
    else:
        avg_return_4months = leveraged_curve[-1] if len(leveraged_curve) > 0 else 0
    
    # Plotting
    fig = plt.figure(figsize=(14, 10))
    gs = fig.add_gridspec(2, 2)
    
    # Equity Curve
    ax1 = fig.add_subplot(gs[0, :])
    dates = df.index
    market_cum = np.exp(np.cumsum(benchmark_ret))
    strat_cum = np.exp(np.pad(leveraged_curve, (0,0))) 
    
    ax1.plot(dates, market_cum, label="Buy & Hold", color='gray', alpha=0.5)
    ax1.plot(dates, strat_cum, label=f"3x Strategy (TP: {best_r:.1%})", color='darkorange')
    ax1.set_title(f"Equity Curve: Dual SMA + Band + TP/SL (Log Scale)")
    ax1.set_yscale('log')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Heatmap
    ax2 = fig.add_subplot(gs[1, :])
    X, Y = np.meshgrid(smas2, smas1)
    c = ax2.pcolormesh(X, Y, results_matrix, shading='auto', cmap='magma')
    fig.colorbar(c, ax=ax2, label='Sharpe')
    ax2.plot(best_sma2, best_sma1, 'g*', markersize=15, markeredgecolor='white')
    ax2.set_title("Sharpe Landscape (SMA1 vs SMA2)")
    ax2.set_xlabel("Filter SMA")
    ax2.set_ylabel("Logic SMA")
    
    plt.tight_layout()
    
    img = io.BytesIO()
    plt.savefig(img, format='png', dpi=100)
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode('utf8')
    plt.close(fig)
    
    data = {
        'symbol': SYMBOL,
        'best_sma1': int(best_sma1),
        'best_sma2': int(best_sma2),
        'best_x': f"{best_x:.2%}",
        'best_s': f"{best_s:.2%}",
        'best_r': f"{best_r:.2%}",
        'best_sharpe': f"{best_sharpe:.4f}",
        'avg_return_4months': f"{np.exp(avg_return_4months)-1:.2%}",
        'plot_url': plot_url,
        'monthly_returns': [f"{r:.2f}%" for r in monthly_returns[-12:]] # Show last 12 months
    }
    
    return render_template('index.html', data=data)

if __name__ == "__main__":
    print("Starting web server on port 8080...")
    app.run(host='0.0.0.0', port=8080, debug=False)
