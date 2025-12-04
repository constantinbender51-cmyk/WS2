import requests
import pandas as pd
import numpy as np
import time
from datetime import datetime
import matplotlib.pyplot as plt

# -----------------------------------------------------------------------------
# 1. DATA FETCHING (Binance Public API)
# -----------------------------------------------------------------------------
def fetch_binance_data(symbol="BTCUSDT", interval="1d", start_date="2018-01-01"):
    """
    Fetches OHLCV data from Binance API with pagination.
    """
    base_url = "https://api.binance.com/api/v3/klines"
    
    # Convert start_date to milliseconds timestamp
    start_ts = int(datetime.strptime(start_date, "%Y-%m-%d").timestamp() * 1000)
    
    all_data = []
    limit = 1000  # Max limit per request
    current_start = start_ts
    
    print(f"Fetching {symbol} data from {start_date}...")
    
    while True:
        params = {
            "symbol": symbol,
            "interval": interval,
            "startTime": current_start,
            "limit": limit
        }
        
        try:
            response = requests.get(base_url, params=params)
            data = response.json()
            
            # Handle empty response or errors
            if not isinstance(data, list) or len(data) == 0:
                break
                
            all_data.extend(data)
            
            # Update start time for next batch (last close time + 1ms)
            last_close_time = data[-1][6]
            current_start = last_close_time + 1
            
            # Check if we reached current time
            if len(data) < limit:
                break
                
            # Respect rate limits
            time.sleep(0.1)
            
        except Exception as e:
            print(f"Error fetching data: {e}")
            break
            
    if not all_data:
        raise ValueError("No data fetched.")

    # Convert to DataFrame
    df = pd.DataFrame(all_data, columns=[
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "qav", "num_trades", "taker_base_vol", "taker_quote_vol", "ignore"
    ])
    
    # Type conversion
    numeric_cols = ["open", "high", "low", "close", "volume"]
    df[numeric_cols] = df[numeric_cols].astype(float)
    
    # Date handling
    df["date"] = pd.to_datetime(df["open_time"], unit="ms")
    df.set_index("date", inplace=True)
    
    print(f"Successfully fetched {len(df)} candles.")
    return df[["open", "high", "low", "close", "volume"]]

# -----------------------------------------------------------------------------
# 2. STRATEGY LOGIC
# -----------------------------------------------------------------------------
def calculate_strategy(df, sma_period, x_pct):
    """
    Calculates strategy returns and stats.
    
    Logic:
    - Long if Open > SMA * (1 + x)
    - Short if Open < SMA * (1 - x)
    - Flat otherwise
    
    Note: Uses shifted SMA (yesterday's close) to avoid lookahead bias.
    """
    data = df.copy()
    
    # Calculate SMA
    data['sma'] = data['close'].rolling(window=sma_period).mean()
    
    # Shift SMA to align with 'Open' logic (decision made at Open based on prev SMA)
    data['prev_sma'] = data['sma'].shift(1)
    
    # Define Bands
    upper_band = data['prev_sma'] * (1 + x_pct)
    lower_band = data['prev_sma'] * (1 - x_pct)
    
    # Determine Signals
    # 1 = Long, -1 = Short, 0 = Flat
    conditions = [
        (data['open'] > upper_band),
        (data['open'] < lower_band)
    ]
    choices = [1, -1]
    
    # 'select' creates the position vector. Default 0 (Flat)
    data['position'] = np.select(conditions, choices, default=0)
    
    # Calculate Returns
    # Market return: (Close - Open) / Open or (Close - PrevClose) / PrevClose?
    # Simple approach: Log returns of Close-to-Close
    data['market_return'] = np.log(data['close'] / data['close'].shift(1))
    
    # Strategy return: Position * Market Return
    # We shift position by 1 because position calculated at Open acts on that day's price action
    # However, if we enter at Open, the return for that day is (Close - Open). 
    # To keep it standard vectorized: Position[t] * (Close[t]/Close[t-1] - 1) is approx correct 
    # IF we assume position is held through the close. 
    # A more precise execution model:
    # If Position=1, return = (Close - Open) / Open + (Open - PrevClose)/PrevClose (gap)? 
    # Let's stick to standard vectorized: Strategy Return = Position * Market Return.
    data['strategy_return'] = data['position'] * data['market_return']
    
    # Drop NaN (initial SMA period)
    data.dropna(inplace=True)
    
    return data

def get_performance_metrics(strategy_returns):
    """Calculates Sharpe Ratio and Total Return."""
    if len(strategy_returns) == 0:
        return -999, 0
    
    # Annualized Sharpe (assuming daily data, 365 crypto days)
    mean_return = strategy_returns.mean()
    std_return = strategy_returns.std()
    
    if std_return == 0:
        return 0, 0
        
    sharpe = (mean_return / std_return) * np.sqrt(365)
    total_return = strategy_returns.sum() # Sum of log returns = total cumulative return
    
    return sharpe, total_return

# -----------------------------------------------------------------------------
# 3. GRID SEARCH
# -----------------------------------------------------------------------------
def run_grid_search(df, sma_period=365):
    """
    Iterates over a range of X percentages to find optimal Sharpe.
    """
    print(f"\n--- Starting Grid Search (SMA {sma_period}) ---")
    
    # Grid: x from 0.0% to 15.0% in steps of 0.5%
    x_values = np.arange(0.00, 0.15, 0.005) 
    
    best_sharpe = -np.inf
    best_x = 0
    best_data = None
    results = []

    for x in x_values:
        strat_data = calculate_strategy(df, sma_period, x)
        sharpe, total_ret = get_performance_metrics(strat_data['strategy_return'])
        
        results.append((x, sharpe, total_ret))
        
        if sharpe > best_sharpe:
            best_sharpe = sharpe
            best_x = x
            best_data = strat_data
            
    print(f"Optimization Complete.")
    return best_x, best_sharpe, best_data, results

# -----------------------------------------------------------------------------
# MAIN EXECUTION
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    # Settings
    SYMBOL = "BTCUSDT"
    SMA_PERIOD = 120
    
    try:
        # 1. Get Data
        df = fetch_binance_data(SYMBOL, interval="1d", start_date="2018-01-01")
        
        if len(df) < SMA_PERIOD:
            print("Not enough data to calculate SMA.")
        else:
            # 2. Run Grid Search
            best_x, best_sharpe, best_data, all_results = run_grid_search(df, SMA_PERIOD)
            
            # 3. Output Results
            print(f"\n--- RESULTS for {SYMBOL} ---")
            print(f"Optimal X: {best_x:.2%} (SMA {SMA_PERIOD} +/- X%)")
            print(f"Best Sharpe Ratio: {best_sharpe:.4f}")
            print(f"Cumulative Log Return: {best_data['strategy_return'].sum():.4f}")
            
            # 4. Visualization
            plt.figure(figsize=(12, 8))
            
            # Plot 1: Cumulative Returns
            plt.subplot(2, 1, 1)
            cumulative_returns = np.exp(best_data['strategy_return'].cumsum())
            cumulative_market = np.exp(best_data['market_return'].cumsum())
            
            plt.plot(cumulative_returns, label=f'Strategy (x={best_x:.1%})', color='green')
            plt.plot(cumulative_market, label='Buy & Hold', color='gray', alpha=0.5)
            plt.title(f'Equity Curve: SMA {SMA_PERIOD} Band Breakout')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # Plot 2: Sharpe vs X (Grid Search Landscape)
            plt.subplot(2, 1, 2)
            xs = [r[0] * 100 for r in all_results]
            sharpes = [r[1] for r in all_results]
            
            plt.plot(xs, sharpes, marker='o', linestyle='-', markersize=4)
            plt.axvline(x=best_x*100, color='r', linestyle='--', label=f'Optimal: {best_x:.1%}')
            plt.title('Grid Search: Sharpe Ratio vs Band Threshold (x%)')
            plt.xlabel('X %')
            plt.ylabel('Sharpe Ratio')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.show()

    except Exception as e:
        print(f"An error occurred: {e}")
