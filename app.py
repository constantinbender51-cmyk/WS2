import ccxt
import pandas as pd
import numpy as np
from datetime import datetime
import time

# --- Configuration ---
SYMBOL = 'BTC/USDT'
TIMEFRAME = '1d'
START_DATE = '2018-01-01 00:00:00'
MAX_SMA = 120
RISK_FREE_RATE = 0.0  # Simplified for raw Sharpe comparison

def fetch_data(symbol, timeframe, start_str):
    """Fetches full history from Binance."""
    print(f"--- Fetching {symbol} data since {start_str} ---")
    exchange = ccxt.binance({'enableRateLimit': True})
    since = exchange.parse8601(start_str)
    all_ohlcv = []
    
    while True:
        try:
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=1000)
            if not ohlcv: break
            all_ohlcv.extend(ohlcv)
            since = ohlcv[-1][0] + 1
            if len(ohlcv) < 1000: break
        except Exception as e:
            print(f"Error: {e}")
            break

    df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    print(f"Data loaded: {len(df)} candles.")
    return df

def calculate_sharpe(returns):
    """Calculates Annualized Sharpe Ratio (assuming 365 trading days for Crypto)."""
    if np.std(returns) == 0: return 0
    return (np.mean(returns) / np.std(returns)) * np.sqrt(365)

def run_massive_analysis(df):
    print(f"\n--- Starting Massive Analysis (Max SMA: {MAX_SMA}) ---")
    
    # 1. Pre-calculate Returns and Price Array
    # 'returns' is the daily percent change of the asset
    market_returns = df['close'].pct_change().fillna(0).to_numpy()
    close_prices = df['close'].to_numpy()
    
    # 2. Pre-calculate ALL SMAs into a Matrix (Time x 120)
    # This avoids re-calculating rolling means inside loops
    print("Pre-calculating SMA Matrix...")
    sma_matrix = np.zeros((len(df), MAX_SMA))
    
    # Fill matrix: Column 0 is SMA_1, Column 119 is SMA_120
    for i in range(MAX_SMA):
        period = i + 1
        sma_matrix[:, i] = df['close'].rolling(window=period).mean().fillna(0).to_numpy()

    results = []

    # --- STRATEGY TYPE 1: Price vs SMA ---
    # Long if Price > SMA, Short if Price < SMA
    print("Backtesting Type 1: Price vs SMA (1-120)...")
    
    # Vectorized comparison: Price (T,1) vs SMA_Matrix (T, 120)
    # signal_raw: 1 if Price > SMA, -1 if Price < SMA
    # We use broadcasting: close_prices[:, None] makes it a column vector to compare against matrix
    price_signal_raw = np.where(close_prices[:, None] > sma_matrix, 1, -1)
    
    # Shift signals by 1 day (Trade today based on Yesterday's signal)
    # We roll the array forward and set the first row to 0
    price_signals = np.roll(price_signal_raw, 1, axis=0)
    price_signals[0, :] = 0
    
    # Calculate Strategy Returns: Signal * Market_Returns
    # Market Returns (T,) needs to be broadcast to (T, 120)
    type1_returns = price_signals * market_returns[:, None]
    
    # Compute Sharpes for all 120 strategies at once
    # mean and std along axis 0 (time)
    means = np.mean(type1_returns, axis=0)
    stds = np.std(type1_returns, axis=0)
    sharpes = np.divide(means, stds, out=np.zeros_like(means), where=stds!=0) * np.sqrt(365)
    
    for i in range(MAX_SMA):
        results.append({
            'Strategy': f"Price vs SMA({i+1})",
            'Sharpe': sharpes[i]
        })

    # --- STRATEGY TYPE 2: SMA vs SMA Crossover ---
    # SMA1 > SMA2 -> Long, SMA1 < SMA2 -> Short
    # Combinations: 120 * 119 = ~14,280 pairs
    print("Backtesting Type 2: SMA vs SMA Crossovers (This may take a moment)...")
    
    # We iterate through all unique pairs to save memory, 
    # though full matrix broadcasting (120x120xT) would be too RAM heavy (~1GB+ for 2000 days).
    # A loop over 120 outer SMAs is efficient enough.
    
    for fast_idx in range(MAX_SMA):
        # We want to compare SMA(fast_idx+1) against ALL other SMAs
        # fast_sma is a vector (T,)
        fast_sma = sma_matrix[:, fast_idx]
        
        # Compare fast_sma against the entire sma_matrix
        # Result is (T, 120) signals for this specific fast_sma
        cross_signals_raw = np.where(fast_sma[:, None] > sma_matrix, 1, -1)
        
        # Shift
        cross_signals = np.roll(cross_signals_raw, 1, axis=0)
        cross_signals[0, :] = 0
        
        # Returns
        strat_returns = cross_signals * market_returns[:, None]
        
        # Stats
        c_means = np.mean(strat_returns, axis=0)
        c_stds = np.std(strat_returns, axis=0)
        c_sharpes = np.divide(c_means, c_stds, out=np.zeros_like(c_means), where=c_stds!=0) * np.sqrt(365)
        
        for slow_idx in range(MAX_SMA):
            if fast_idx == slow_idx: continue # Skip comparing SMA to itself
            
            # fast_idx + 1 is the period
            p1 = fast_idx + 1
            p2 = slow_idx + 1
            
            results.append({
                'Strategy': f"SMA({p1}) vs SMA({p2})",
                'Sharpe': c_sharpes[slow_idx]
            })

    # --- Final Ranking ---
    print("\nSorting results...")
    df_results = pd.DataFrame(results)
    df_results = df_results.sort_values(by='Sharpe', ascending=False).reset_index(drop=True)
    
    print("\n" + "="*50)
    print(f"TOP 25 STRATEGIES (out of {len(df_results)} tested)")
    print("="*50)
    print(df_results.head(25).to_string())
    print("="*50)

if __name__ == '__main__':
    df = fetch_data(SYMBOL, TIMEFRAME, START_DATE)
    run_massive_analysis(df)
