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
from flask import Flask, render_template_string

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
    n = len(opens)
    strategy_returns = np.zeros(n)
    
    # State Flag for SMA 1: 0 = None, 1 = Just Crossed Up, -1 = Just Crossed Down
    cross_flag = 0 
    
    for i in range(1, n):
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
        
        # 4. Calculate Return, SL & TP
        if signal != 0:
            daily_ret = signal * intraday_returns[i]
            sl_hit = False
            
            # STOP LOSS
            if signal == 1:
                if lows[i] < current_open * (1 - s):
                    daily_ret = np.log(1 - s)
                    sl_hit = True
            elif signal == -1:
                if highs[i] > current_open * (1 + s):
                    daily_ret = np.log(1 - s)
                    sl_hit = True
            
            # TAKE PROFIT (If SL not hit)
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
    
    # Range Settings
    sma1_periods = np.arange(10, 200, 5) # Logic
    sma2_periods = np.arange(20, 200, 5) # Filter
    x_values = np.arange(0.00, 0.051, 0.01) # Band
    s_values = np.arange(0.00, 0.101, 0.02) # Stop Loss
    r_values = np.arange(0.00, 0.201, 0.04) # Take Profit
    
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
# 4. HTML TEMPLATE
# -----------------------------------------------------------------------------
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Trading Strategy Optimizer</title>
    <style>
        body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background-color: #f4f4f9; padding: 20px; color: #333; }
        .container { max-width: 1200px; margin: 0 auto; background: white; padding: 30px; border-radius: 12px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }
        h1 { color: #2c3e50; text-align: center; margin-bottom: 30px; }
        .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; margin-bottom: 30px; }
        .card { background: #f8f9fa; padding: 15px; border-radius: 8px; border-left: 5px solid #3498db; }
        .card h3 { margin: 0 0 10px 0; font-size: 0.9em; color: #7f8c8d; text-transform: uppercase; }
        .card p { margin: 0; font-size: 1.5em; font-weight: bold; color: #2c3e50; }
        .plot-container { text-align: center; margin-bottom: 30px; }
        img { max-width: 100%; height: auto; border-radius: 8px; border: 1px solid #eee; }
        .returns-list { list-style: none; padding: 0; display: flex; flex-wrap: wrap; gap: 10px; justify-content: center; }
        .returns-list li { background: #ecf0f1; padding: 5px 10px; border-radius: 4px; font-size: 0.9em; }
        .positive { color: #27ae60; }
        .negative { color: #c0392b; }
    </style>
</head>
<body>
    <div class="container">
        <h1>Optimal Strategy Results: {{ data.symbol }}</h1>
        <div class="grid">
            <div class="card"><h3>Logic SMA</h3><p>{{ data.best_sma1 }}</p></div>
            <div class="card"><h3>Filter SMA</h3><p>{{ data.best_sma2 }}</p></div>
            <div class="card"><h3>Band Width (X)</h3><p>{{ data.best_x }}</p></div>
            <div class="card"><h3>Stop Loss (S)</h3><p>{{ data.best_s }}</p></div>
            <div class="card" style="border-left-color: #27ae60;"><h3>Take Profit (R)</h3><p>{{ data.best_r }}</p></div>
            <div class="card" style="border-left-color: #8e44ad;"><h3>Sharpe Ratio</h3><p>{{ data.best_sharpe }}</p></div>
            <div class="card" style="border-left-color: #e67e22;"><h3>Avg 4-Mo Return (3x)</h3><p>{{ data.avg_return_4months }}</p></div>
        </div>
        <div class="plot-container">
            <img src="data:image/png;base64,{{ data.plot_url }}" alt="Strategy Performance Plot">
        </div>
        <div style="text-align: center;">
            <h3>Recent Monthly Returns (3x Lev)</h3>
            <ul class="returns-list">
                {% for ret in data.monthly_returns %}
                    <li class="{{ 'positive' if '-' not in ret else 'negative' }}">{{ ret }}</li>
                {% endfor %}
            </ul>
        </div>
    </div>
</body>
</html>
"""

# -----------------------------------------------------------------------------
# 5. WEB SERVER
# -----------------------------------------------------------------------------
app = Flask(__name__)

@app.route('/')
def index():
    SYMBOL = "BTCUSDT"
    df = fetch_binance_data(SYMBOL)
    
    # Run Grid Search
    # Note: run_dual_sma_grid returns 'results_matrix' as the 5th element
    best_params, best_sharpe, best_curve, benchmark_ret, heatmap_data, smas1, smas2, monthly_returns = run_dual_sma_grid(df)
    
    if best_params is None:
        return "No valid parameters found. Try increasing data range.", 500
    
    best_sma1, best_sma2, best_x, best_s, best_r = best_params
    leveraged_curve = best_curve * 3
    
    # Avg 4-month return
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
    fig = plt.figure(figsize=(14, 12))
    gs = fig.add_gridspec(3, 2)
    
    # A: Equity Curve
    ax1 = fig.add_subplot(gs[0, :])
    dates = df.index
    market_cum = np.exp(np.cumsum(benchmark_ret))
    strat_cum = np.exp(np.pad(leveraged_curve, (0,0))) 
    
    # Calculate 365-day SMA
    sma_365 = df['close'].rolling(window=365).mean()
    
    ax1.plot(dates, market_cum, label="Buy & Hold", color='gray', alpha=0.5)
    ax1.plot(dates, strat_cum, label=f"3x Strategy (TP: {best_r:.1%})", color='darkorange')
    ax1.plot(dates, sma_365, label="365-day SMA", color='blue', alpha=0.7, linestyle='--')
    ax1.set_title(f"Equity Curve: Dual SMA + Band + TP/SL (Log Scale)")
    ax1.set_yscale('log')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # B: Price with SMA
    ax2 = fig.add_subplot(gs[1, :])
    ax2.plot(dates, df['close'], label="Close Price", color='green', alpha=0.8)
    ax2.plot(dates, sma_365, label="365-day SMA", color='red', alpha=0.7, linestyle='--')
    ax2.set_title("Price with 365-day Simple Moving Average")
    ax2.set_ylabel("Price (USD)")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # C: Heatmap
    ax3 = fig.add_subplot(gs[2, :])
    X, Y = np.meshgrid(smas2, smas1)
    
    # Use 'heatmap_data' which was returned from the function
    c = ax3.pcolormesh(X, Y, heatmap_data, shading='auto', cmap='magma')
    fig.colorbar(c, ax=ax3, label='Sharpe')
    ax3.plot(best_sma2, best_sma1, 'g*', markersize=15, markeredgecolor='white')
    ax3.set_title("Sharpe Landscape (SMA1 vs SMA2)")
    ax3.set_xlabel("Filter SMA")
    ax3.set_ylabel("Logic SMA")
    
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
        'monthly_returns': [f"{r:.2f}%" for r in monthly_returns[-12:]] 
    }
    
    return render_template_string(HTML_TEMPLATE, data=data)

if __name__ == "__main__":
    print("Starting web server on port 8080...")
    app.run(host='0.0.0.0', port=8080, debug=False)
