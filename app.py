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
# 2. NUMBA OPTIMIZED LOGIC (State Machine)
# -----------------------------------------------------------------------------
@jit(nopython=True)
def calculate_strategy_numba(opens, highs, lows, intraday_returns, sma_arr, x, s):
    """
    Runs the path-dependent logic loop very fast.
    """
    n = len(opens)
    strategy_returns = np.zeros(n)
    
    # State Flag: 0 = None, 1 = Just Crossed Up, -1 = Just Crossed Down
    cross_flag = 0 
    
    # Iterate through days (start at 1 for prev comparison)
    for i in range(1, n):
        # Skip if SMA is NaN
        if np.isnan(sma_arr[i]):
            continue
            
        current_open = opens[i]
        prev_open = opens[i-1]
        current_sma = sma_arr[i]
        prev_sma = sma_arr[i-1] # Note: SMA array is already shifted in main loop? 
                                # No, let's pass raw SMA and handle shift logic here or pre-shift.
                                # To be safe/clear: logic uses TODAY's open vs YESTERDAY's SMA?
                                # Previous scripts used Shifted SMA. 
                                # Let's assume sma_arr passed in is ALREADY shifted (i.e. sma[i] is yesterday's close avg)
        
        upper_band = current_sma * (1 + x)
        lower_band = current_sma * (1 - x)
        
        # 1. Determine Cross Event (Crossing the CENTER line)
        # We compare Current Open vs Current SMA line
        # Logic: Did we flip sides relative to SMA?
        # Note: Using prev_open to detect cross
        
        # Cross UP
        if (prev_open < current_sma) and (current_open > current_sma):
            cross_flag = 1
        # Cross DOWN
        elif (prev_open > current_sma) and (current_open < current_sma):
            cross_flag = -1
            
        # 2. Check Band Exit (Unset Flag)
        # "Unset when open leaves x% band"
        # If we are outside the band (Above Upper or Below Lower), the "proximity" flag is cleared.
        if (current_open > upper_band) or (current_open < lower_band):
            cross_flag = 0
            
        # 3. Determine Signal
        signal = 0
        
        # Long Logic
        # A: Explicitly Above Band
        # B: Above SMA AND Flag is Set (Just Crossed)
        if (current_open > upper_band) or ((current_open > current_sma) and (cross_flag == 1)):
            signal = 1
            
        # Short Logic
        # A: Explicitly Below Band
        # B: Below SMA AND Flag is Set (Just Crossed)
        elif (current_open < lower_band) or ((current_open < current_sma) and (cross_flag == -1)):
            signal = -1
            
        # Else Flat (signal 0)
        
        # 4. Calculate Return & Stop Loss
        if signal != 0:
            daily_ret = signal * intraday_returns[i]
            
            # Stop Loss Check
            # Long Stop: Low < Open * (1-s)
            if signal == 1:
                if lows[i] < current_open * (1 - s):
                    daily_ret = np.log(1 - s)
            # Short Stop: High > Open * (1+s)
            elif signal == -1:
                if highs[i] > current_open * (1 + s):
                    daily_ret = np.log(1 - s)
                    
            strategy_returns[i] = daily_ret
            
    return strategy_returns

# -----------------------------------------------------------------------------
# 3. GRID SEARCH
# -----------------------------------------------------------------------------
def run_state_machine_grid(df):
    print("\n--- Starting Grid Search (State Machine Logic) ---")
    
    # Parameters
    sma_periods = np.arange(1, 366, 2)
    # X needs to be positive for "Proximity" logic to make sense (Band Width)
    # Scanning 0% to 10%
    x_values = np.arange(0.00, 0.101, 0.005) 
    s_values = np.arange(0.01, 0.11, 0.01)
    
    # Pre-convert to Numpy for Numba
    opens = df['open'].to_numpy()
    highs = df['high'].to_numpy()
    lows = df['low'].to_numpy()
    closes = df['close'].to_numpy() # Used for SMA calc only
    
    # Pre-calc Intraday Returns
    # Avoid division by zero
    with np.errstate(divide='ignore', invalid='ignore'):
        intraday_returns = np.log(closes / opens)
    intraday_returns = np.nan_to_num(intraday_returns)
    
    benchmark_returns = np.log(df['close'] / df['close'].shift(1)).fillna(0).to_numpy()
    
    best_sharpe = -np.inf
    best_params = None
    best_curve = None
    
    # For Heatmap (SMA vs X)
    results_matrix = np.zeros((len(sma_periods), len(x_values)))
    
    start_time = time.time()
    
    # Loop SMAs
    for i, period in enumerate(sma_periods):
        # Calculate SMA (shifted by 1 for lookahead prevention)
        # Numba needs a clean array, so we do rolling in Pandas first
        sma_series = df['close'].rolling(window=period).mean().shift(1)
        sma_arr = sma_series.to_numpy() # Contains NaNs at start
        
        # Loop X
        for j, x in enumerate(x_values):
            
            local_best_s_sharpe = -np.inf
            
            # Loop S
            for s in s_values:
                # Run Core Logic (Compiled)
                strat_ret = calculate_strategy_numba(opens, highs, lows, intraday_returns, sma_arr, x, s)
                
                # Sharpe Calc
                # Exclude initial NaNs (where returns are 0 due to no signal)
                # We can just ignore 0s if we assume we are in market mostly? 
                # Better: slice from 'period' onwards
                active_rets = strat_ret[period:] 
                
                if len(active_rets) < 10:
                    sharpe = 0
                else:
                    mean_r = np.mean(active_rets)
                    std_r = np.std(active_rets)
                    sharpe = (mean_r / std_r) * np.sqrt(365) if std_r > 1e-9 else 0
                
                # Update Bests
                if sharpe > local_best_s_sharpe:
                    local_best_s_sharpe = sharpe
                
                if sharpe > best_sharpe:
                    best_sharpe = sharpe
                    best_params = (period, x, s)
                    best_curve = np.cumsum(strat_ret)
            
            results_matrix[i, j] = local_best_s_sharpe

    print(f"Optimization complete in {time.time() - start_time:.2f}s.")
    return best_params, best_sharpe, best_curve, benchmark_returns, results_matrix, sma_periods, x_values

# -----------------------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    SYMBOL = "BTCUSDT"
    
    # 1. Fetch
    df = fetch_binance_data(SYMBOL)
    
    # 2. Run
    # Note: First run might be slightly slower due to Numba compilation
    best_params, best_sharpe, best_curve, benchmark_ret, heatmap_data, smas, xs = run_state_machine_grid(df)
    
    best_sma, best_x, best_s = best_params
    
    print(f"\n--- RESULTS (SMA Cross + Band Latch) ---")
    print(f"Best SMA Period : {best_sma}")
    print(f"Best Band Width X: {best_x:.1%}")
    print(f"Best Stop Loss S : {best_s:.1%}")
    print(f"Best Sharpe Ratio: {best_sharpe:.4f}")
    
    # 3. Visualization
    fig = plt.figure(figsize=(14, 10))
    gs = fig.add_gridspec(2, 2)
    
    # Plot A: Equity Curve
    ax1 = fig.add_subplot(gs[0, :])
    dates = df.index
    market_cum = np.exp(np.cumsum(benchmark_ret))
    strat_cum = np.exp(np.pad(best_curve, (0,0))) 
    
    ax1.plot(dates, market_cum, label="Buy & Hold", color='gray', alpha=0.5)
    ax1.plot(dates, strat_cum, label=f"Strategy (SMA {best_sma}, x={best_x:.1%})", color='darkorange')
    ax1.set_title(f"Equity Curve: Proximity Logic (Cross=Active, Enter=Flat)")
    ax1.set_yscale('log')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot B: Heatmap
    ax2 = fig.add_subplot(gs[1, :])
    X, Y = np.meshgrid(xs * 100, smas)
    
    c = ax2.pcolormesh(X, Y, heatmap_data, shading='auto', cmap='magma')
    fig.colorbar(c, ax=ax2, label='Sharpe Ratio')
    
    ax2.plot(best_x*100, best_sma, 'g*', markersize=15, markeredgecolor='white', label='Optimal')
    
    ax2.set_title("Sharpe Ratio Landscape")
    ax2.set_xlabel("Band Width X (%)")
    ax2.set_ylabel("SMA Period")
    ax2.legend()
    
    plt.tight_layout()
    plt.show()
