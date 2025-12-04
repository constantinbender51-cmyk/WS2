import requests
import pandas as pd
import numpy as np
import time
from datetime import datetime
import matplotlib.pyplot as plt

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
    
    # Return Open, High, Low, Close for Stop Loss logic
    return df[["open", "high", "low", "close"]]

# -----------------------------------------------------------------------------
# 2. GRID SEARCH (SINGLE SMA + STOP LOSS)
# -----------------------------------------------------------------------------
def run_single_sma_grid(df):
    """
    Optimizes:
    1. SMA Period (1-365)
    2. Entry Threshold X% (-5% to +5%)
    3. Stop Loss S% (1% to 10%)
    """
    print("\n--- Starting Grid Search (SMA + Stop Loss) ---")
    
    # --- Parameter Spaces ---
    # SMA: 1 to 365 in steps of 4 (Speeds up search slightly while maintaining granularity)
    sma_periods = np.arange(1, 366, 2)
    
    # X: -0.05 to +0.05
    x_values = np.arange(-0.05, 0.051, 0.005)
    
    # S: Stop Loss from 1% to 10%
    s_values = np.arange(0.01, 0.11, 0.01)
    
    # Base Arrays (Numpy for speed)
    closes = df['close'].to_numpy()
    opens = df['open'].to_numpy()
    highs = df['high'].to_numpy()
    lows = df['low'].to_numpy()
    
    # Standard Market Returns (Log Close-to-Close)
    market_returns = np.log(df['close'] / df['close'].shift(1)).fillna(0).to_numpy()
    
    best_sharpe = -np.inf
    best_params = None
    best_curve = None
    
    # For Heatmap, we will project the best S onto the SMA vs X plane
    results_matrix = np.zeros((len(sma_periods), len(x_values)))
    
    start_time = time.time()
    
    # 2. Loop over SMAs
    for i, period in enumerate(sma_periods):
        # Calculate shifted SMA (yesterday's SMA)
        sma = df['close'].rolling(window=period).mean().shift(1).to_numpy()
        valid_mask = ~np.isnan(sma)
        
        # 3. Loop over Entry Threshold X
        for j, x in enumerate(x_values):
            
            upper_band = sma * (1 + x)
            lower_band = sma * (1 - x)
            
            # --- Determine Raw Positions ---
            long_entry = (opens > upper_band)
            short_entry = (opens < lower_band)
            
            # Resolve overlaps (if x < 0, Long priority)
            raw_positions = np.zeros_like(opens)
            # Apply Short first
            raw_positions[short_entry] = -1
            # Apply Long (overwrites Short if overlap)
            raw_positions[long_entry] = 1
            
            # --- Inner Loop: Stop Loss S ---
            # We check which S yields best result for this specific SMA/X combo
            best_s_sharpe = -np.inf
            
            for s in s_values:
                # Copy positions to avoid modifying the base array for next S iteration
                # (Actually, we calculate returns derived from positions, so we don't need to copy positions array itself
                #  if we build the return array fresh).
                
                # 1. Calculate Base Strategy Returns
                daily_returns = raw_positions * market_returns
                
                # 2. Check Stop Loss Conditions
                # Long Stop: Low < Open * (1 - s)
                # Short Stop: High > Open * (1 + s)
                
                # Masks for where stops are hit
                # Note: We only care if we are actually IN a position
                sl_long_hit = (raw_positions == 1) & (lows < opens * (1 - s))
                sl_short_hit = (raw_positions == -1) & (highs > opens * (1 + s))
                
                # 3. Apply Penalty
                # If stop hit, return is fixed to -s% (using log(1-s) for log-return consistency)
                penalty = np.log(1 - s)
                
                # Overwrite returns where stop was hit
                daily_returns[sl_long_hit] = penalty
                daily_returns[sl_short_hit] = penalty
                
                # 4. Calculate Sharpe
                active_rets = daily_returns[valid_mask]
                if len(active_rets) == 0:
                    sharpe = 0
                else:
                    mean_r = np.mean(active_rets)
                    std_r = np.std(active_rets)
                    sharpe = (mean_r / std_r) * np.sqrt(365) if std_r > 1e-9 else 0
                
                # Check against local best (for this SMA/X cell)
                if sharpe > best_s_sharpe:
                    best_s_sharpe = sharpe
                
                # Check against global best
                if sharpe > best_sharpe:
                    best_sharpe = sharpe
                    best_params = (period, x, s)
                    best_curve = np.cumsum(daily_returns)
            
            # Store best Sharpe for this SMA/X combination (across all S) in matrix
            results_matrix[i, j] = best_s_sharpe

    print(f"Optimization complete in {time.time() - start_time:.2f}s.")
    return best_params, best_sharpe, best_curve, market_returns, results_matrix, sma_periods, x_values

# -----------------------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    SYMBOL = "BTCUSDT"
    
    # 1. Fetch
    df = fetch_binance_data(SYMBOL)
    
    # 2. Run Grid Search
    best_params, best_sharpe, best_curve, market_ret, heatmap_data, smas, xs = run_single_sma_grid(df)
    
    best_sma, best_x, best_s = best_params
    
    print(f"\n--- RESULTS ---")
    print(f"Best SMA Period : {best_sma}")
    print(f"Best Threshold X: {best_x:.1%}")
    print(f"Best Stop Loss S: {best_s:.1%}")
    print(f"Best Sharpe Ratio: {best_sharpe:.4f}")
    
    # 3. Visualization
    fig = plt.figure(figsize=(14, 10))
    gs = fig.add_gridspec(2, 2)
    
    # Plot A: Equity Curve
    ax1 = fig.add_subplot(gs[0, :])
    dates = df.index
    market_cum = np.exp(np.cumsum(market_ret))
    strat_cum = np.exp(np.pad(best_curve, (0,0))) 
    
    ax1.plot(dates, market_cum, label="Buy & Hold", color='gray', alpha=0.5)
    ax1.plot(dates, strat_cum, label=f"Best Strategy (SMA {best_sma}, x={best_x:.1%}, sl={best_s:.0%})", color='purple')
    ax1.set_title(f"Equity Curve: SMA {best_sma} Breakout with {best_s:.0%} Stop Loss")
    ax1.set_yscale('log')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot B: Heatmap (Sharpe Ratio Landscape)
    ax2 = fig.add_subplot(gs[1, :])
    X, Y = np.meshgrid(xs * 100, smas)
    
    c = ax2.pcolormesh(X, Y, heatmap_data, shading='auto', cmap='viridis')
    fig.colorbar(c, ax=ax2, label='Best Sharpe (across all S)')
    
    ax2.plot(best_x*100, best_sma, 'r*', markersize=15, markeredgecolor='white', label='Optimal')
    
    ax2.set_title("Sharpe Ratio Heatmap (Best S per cell)")
    ax2.set_xlabel("Threshold X (%)")
    ax2.set_ylabel("SMA Period")
    ax2.legend()
    
    plt.tight_layout()
    plt.show()
