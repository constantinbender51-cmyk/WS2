import requests
import pandas as pd
import numpy as np
import time
from datetime import datetime
import matplotlib.pyplot as plt
from numba import jit

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
# 2. NUMBA OPTIMIZED LOGIC (Dual SMA + State Machine)
# -----------------------------------------------------------------------------
@jit(nopython=True)
def calculate_dual_sma_strategy(opens, highs, lows, intraday_returns, sma1_arr, sma2_arr, x, s):
    """
    SMA 1: Handles the 'Just Crossed' State Machine logic and Proximity bands.
    SMA 2: Acts as a hard filter (Trend Confirmation).
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
        
        # SMA 1 Values (Primary Logic)
        cur_sma1 = sma1_arr[i]
        
        # SMA 2 Value (Filter)
        cur_sma2 = sma2_arr[i]
        
        # --- SMA 1 STATE MACHINE ---
        upper_band = cur_sma1 * (1 + x)
        lower_band = cur_sma1 * (1 - x)
        
        # 1. Determine Cross Event (Crossing SMA 1)
        if (prev_open < cur_sma1) and (current_open > cur_sma1):
            cross_flag = 1
        elif (prev_open > cur_sma1) and (current_open < cur_sma1):
            cross_flag = -1
            
        # 2. Check Band Exit (Unset Flag)
        if (current_open > upper_band) or (current_open < lower_band):
            cross_flag = 0
            
        # 3. Determine Base Signal (SMA 1 Logic)
        signal = 0
        
        # Long Base: Above Band OR (Above SMA1 AND Just Crossed)
        if (current_open > upper_band) or ((current_open > cur_sma1) and (cross_flag == 1)):
            signal = 1
        # Short Base: Below Band OR (Below SMA1 AND Just Crossed)
        elif (current_open < lower_band) or ((current_open < cur_sma1) and (cross_flag == -1)):
            signal = -1
            
        # --- SMA 2 FILTER ---
        # Flatten if conflicting with SMA 2
        
        if signal == 1:
            # If Long, must be above SMA 2
            if current_open < cur_sma2:
                signal = 0
        elif signal == -1:
            # If Short, must be below SMA 2
            if current_open > cur_sma2:
                signal = 0
        
        # 4. Calculate Return & Stop Loss
        if signal != 0:
            daily_ret = signal * intraday_returns[i]
            
            # Stop Loss Check
            if signal == 1:
                if lows[i] < current_open * (1 - s):
                    daily_ret = np.log(1 - s)
            elif signal == -1:
                if highs[i] > current_open * (1 + s):
                    daily_ret = np.log(1 - s)
                    
            strategy_returns[i] = daily_ret
            
    return strategy_returns

# -----------------------------------------------------------------------------
# 3. GRID SEARCH
# -----------------------------------------------------------------------------
def run_dual_sma_grid(df):
    print("\n--- Starting Grid Search (Dual SMA: Logic + Filter) ---")
    
    # --- Parameters ---
    # Using larger steps to accommodate 4D search space without taking forever
    # SMA1 (Logic): 5 to 205, step 2
    sma1_periods = np.arange(5, 205, 2)
    # SMA2 (Filter): 10 to 370, step 2
    sma2_periods = np.arange(10, 370, 2)
    
    # X: 0% to 6%
    x_values = np.arange(0.00, 0.061, 0.01) 
    # S: 2% to 10%
    s_values = np.arange(0.02, 0.101, 0.02)
    
    print(f"Total Combinations: {len(sma1_periods) * len(sma2_periods) * len(x_values) * len(s_values)}")
    
    # Pre-convert to Numpy
    opens = df['open'].to_numpy()
    highs = df['high'].to_numpy()
    lows = df['low'].to_numpy()
    closes = df['close'].to_numpy()
    
    # Pre-calc Intraday Returns
    with np.errstate(divide='ignore', invalid='ignore'):
        intraday_returns = np.log(closes / opens)
    intraday_returns = np.nan_to_num(intraday_returns)
    
    benchmark_returns = np.log(df['close'] / df['close'].shift(1)).fillna(0).to_numpy()
    
    # Pre-calculate ALL SMAs to avoid re-rolling in loops
    print("Pre-calculating SMA matrix...")
    max_period = max(np.max(sma1_periods), np.max(sma2_periods))
    # Dictionary to store {period: numpy_array}
    sma_cache = {}
    needed_periods = set(np.concatenate((sma1_periods, sma2_periods)))
    
    for p in needed_periods:
        # Shift by 1 for lookahead prevention
        sma_cache[p] = df['close'].rolling(window=p).mean().shift(1).to_numpy()
    
    best_sharpe = -np.inf
    best_params = None
    best_curve = None
    
    # For Heatmap: We'll plot SMA1 vs SMA2 (taking best X/S for each cell)
    results_matrix = np.zeros((len(sma1_periods), len(sma2_periods)))
    
    start_time = time.time()
    
    # Loop SMA 1
    for i, p1 in enumerate(sma1_periods):
        sma1_arr = sma_cache[p1]
        
        # Loop SMA 2
        for j, p2 in enumerate(sma2_periods):
            sma2_arr = sma_cache[p2]
            
            # Optimization: If p1 == p2, the filter is redundant, but we run it anyway
            
            local_best_sharpe = -np.inf
            
            # Loop X
            for x in x_values:
                # Loop S
                for s in s_values:
                    
                    strat_ret = calculate_dual_sma_strategy(
                        opens, highs, lows, intraday_returns, 
                        sma1_arr, sma2_arr, x, s
                    )
                    
                    # Sharpe
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
                        best_params = (p1, p2, x, s)
                        best_curve = np.cumsum(strat_ret)
            
            results_matrix[i, j] = local_best_sharpe
            
        if i % 5 == 0:
            print(f"Processed SMA1 period {p1}...")

    print(f"Optimization complete in {time.time() - start_time:.2f}s.")
    return best_params, best_sharpe, best_curve, benchmark_returns, results_matrix, sma1_periods, sma2_periods

# -----------------------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    SYMBOL = "BTCUSDT"
    
    # 1. Fetch
    df = fetch_binance_data(SYMBOL)
    
    # 2. Run
    best_params, best_sharpe, best_curve, benchmark_ret, heatmap_data, smas1, smas2 = run_dual_sma_grid(df)
    
    best_sma1, best_sma2, best_x, best_s = best_params
    
    print(f"\n--- RESULTS (Dual SMA) ---")
    print(f"Best SMA 1 (Logic) : {best_sma1}")
    print(f"Best SMA 2 (Filter): {best_sma2}")
    print(f"Best Band Width X  : {best_x:.1%}")
    print(f"Best Stop Loss S   : {best_s:.1%}")
    print(f"Best Sharpe Ratio  : {best_sharpe:.4f}")
    
    # 3. Visualization
    fig = plt.figure(figsize=(14, 10))
    gs = fig.add_gridspec(2, 2)
    
    # Plot A: Equity Curve
    ax1 = fig.add_subplot(gs[0, :])
    dates = df.index
    market_cum = np.exp(np.cumsum(benchmark_ret))
    strat_cum = np.exp(np.pad(best_curve, (0,0))) 
    
    ax1.plot(dates, market_cum, label="Buy & Hold", color='gray', alpha=0.5)
    ax1.plot(dates, strat_cum, label=f"Strategy (SMA{best_sma1} & SMA{best_sma2})", color='darkorange')
    ax1.set_title(f"Equity Curve: Dual SMA + Band Latch")
    ax1.set_yscale('log')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot B: Heatmap (SMA1 vs SMA2)
    ax2 = fig.add_subplot(gs[1, :])
    X, Y = np.meshgrid(smas2, smas1)
    
    c = ax2.pcolormesh(X, Y, results_matrix, shading='auto', cmap='magma')
    fig.colorbar(c, ax=ax2, label='Best Sharpe (across X & S)')
    
    ax2.plot(best_sma2, best_sma1, 'g*', markersize=15, markeredgecolor='white', label='Optimal')
    
    ax2.set_title("Sharpe Landscape: SMA1 (Logic) vs SMA2 (Filter)")
    ax2.set_xlabel("SMA 2 (Filter)")
    ax2.set_ylabel("SMA 1 (Logic)")
    ax2.legend()
    
    plt.tight_layout()
    plt.show()
