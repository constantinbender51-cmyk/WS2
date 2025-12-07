import ccxt
import pandas as pd
import numpy as np
from datetime import datetime
import time

# --- Configuration ---
SYMBOL = 'BTC/USDT'
TIMEFRAME = '1d'
START_DATE = '2018-01-01 00:00:00'
MAX_SMA = 365 
MIN_SHARPE = 1.0  # Threshold for a "winning" strategy

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

def get_sharpe(returns_array):
    """Vectorized Sharpe calculation."""
    # axis=0 is time
    means = np.mean(returns_array, axis=0)
    stds = np.std(returns_array, axis=0)
    # Avoid division by zero
    return np.divide(means, stds, out=np.zeros_like(means), where=stds!=0) * np.sqrt(365)

def run_meta_optimization(df):
    print(f"\n--- Starting Meta-Strategy Optimization (k-sweep) ---")
    start_time = time.time()
    
    # 1. Setup Data
    market_returns = df['close'].pct_change().fillna(0).to_numpy()
    close_prices = df['close'].to_numpy()
    n_days = len(market_returns)
    
    # 2. Pre-calculate SMA Matrix
    print(f"Pre-calculating SMA Matrix (1 to {MAX_SMA})...")
    sma_matrix = np.zeros((n_days, MAX_SMA))
    for i in range(MAX_SMA):
        sma_matrix[:, i] = df['close'].rolling(window=i+1).mean().fillna(0).to_numpy()

    # We will collect ALL signals here. 
    # To save memory, we'll process in batches or just be careful with types.
    # ~130k strategies * 2000 days * 1 byte = ~260MB. This fits in RAM easily as int8.
    
    print("Generating signals for all strategies...")
    
    # --- A. Price vs SMA Signals ---
    # Price > SMA = 1 (Long), Price < SMA = -1 (Short)
    # Shape: (T, 365)
    pv_sma_raw = np.where(close_prices[:, None] > sma_matrix, 1, -1).astype(np.int8)
    # Shift signals (Trade today based on yesterday)
    pv_sma_signals = np.roll(pv_sma_raw, 1, axis=0)
    pv_sma_signals[0, :] = 0
    
    # --- B. SMA vs SMA Signals ---
    # We need to flatten the SMA-SMA pairs into a 2D matrix (T, N_pairs)
    # or process them to find winners immediately. 
    # To keep code simple and vectorizable, let's generate returns, filter, and keep only winning signals.
    
    winning_signals_list = []
    
    # 1. Check Price vs SMA Performance
    print("Evaluating Price vs SMA strategies...")
    pv_returns = pv_sma_signals * market_returns[:, None]
    pv_sharpes = get_sharpe(pv_returns)
    
    # Filter
    pv_winners_mask = pv_sharpes > MIN_SHARPE
    if np.any(pv_winners_mask):
        print(f"  -> Found {np.sum(pv_winners_mask)} Price-SMA strategies with Sharpe > {MIN_SHARPE}")
        winning_signals_list.append(pv_sma_signals[:, pv_winners_mask])
    
    # 2. Check SMA vs SMA Performance
    print("Evaluating SMA vs SMA strategies (iterative batches)...")
    
    # We iterate fast_idx and compare against all slow_idx at once
    total_sma_winners = 0
    
    for fast_idx in range(MAX_SMA):
        # Broadcasting: (T, 1) vs (T, MAX_SMA) -> (T, MAX_SMA)
        fast_sma_vec = sma_matrix[:, fast_idx][:, None]
        
        # Raw signals for this batch
        batch_raw = np.where(fast_sma_vec > sma_matrix, 1, -1).astype(np.int8)
        
        # Shift
        batch_signals = np.roll(batch_raw, 1, axis=0)
        batch_signals[0, :] = 0
        
        # Returns
        batch_returns = batch_signals * market_returns[:, None]
        
        # Sharpe
        batch_sharpes = get_sharpe(batch_returns)
        
        # Filter:
        # We must exclude the diagonal (where fast_idx == slow_idx) and likely the inverse pairs 
        # (SMA_A vs SMA_B is inverse of SMA_B vs SMA_A).
        # However, purely checking Sharpe > 1 handles logic automatically (inverse strat has negative Sharpe).
        # We only need to ensure we don't compare a SMA to itself (Sharpe=0 or NaN).
        
        # Create mask
        mask = batch_sharpes > MIN_SHARPE
        
        if np.any(mask):
            count = np.sum(mask)
            total_sma_winners += count
            winning_signals_list.append(batch_signals[:, mask])

    print(f"  -> Found {total_sma_winners} SMA-SMA strategies with Sharpe > {MIN_SHARPE}")

    if not winning_signals_list:
        print("No strategies found with Sharpe > 1. Cannot optimize k.")
        return

    # 3. Consolidate Winning Signals
    print("Consolidating consensus matrix...")
    # Concatenate along columns (axis 1)
    # Result: (Time, N_Winners)
    all_winners = np.hstack(winning_signals_list)
    n_winners = all_winners.shape[1]
    print(f"Total Winning Strategies: {n_winners}")
    
    # 4. Compute Daily Votes
    # Count how many +1s and -1s exist per row
    # (T,) arrays
    votes_long = (all_winners == 1).sum(axis=1)
    votes_short = (all_winners == -1).sum(axis=1)
    
    # 5. Sweep k
    print("\n--- Optimization Results: Confirmation Entry (k% Threshold) ---")
    print(f"{'k (Threshold)':<15} {'Sharpe':<10} {'Ann. Return %':<15} {'Trades/Year':<15}")
    print("-" * 60)
    
    best_k = 0
    best_sharpe = -999
    
    # Range 0.0 to 1.0 inclusive, step 0.05
    k_values = np.arange(0.0, 1.01, 0.05)
    
    for k in k_values:
        # Determine threshold count based on k and n_winners
        # Note: k is a percentage of TOTAL winners
        required_votes = k * n_winners
        
        # Vectorized Signal Generation
        # Logic: 
        # If long_votes >= threshold -> 1
        # Elif short_votes >= threshold -> -1
        # Else -> 0
        
        # Note: If k is low (e.g. 0.1), both conditions could be true.
        # Standard convention: If both trigger, they cancel out or one takes precedence.
        # We'll assume Flat if conflicted, or Long-Short cancellation. 
        # Let's use simple logic: If Longs > Shorts and Longs > Threshold -> 1, etc.
        # But prompt says: "If k% winners indicate long, go long."
        # We will prioritize Long if Long threshold met, Short if Short threshold met.
        # If both met? (Unlikely for high k, possible for low k). We'll set to 0 (conflict).
        
        final_signal = np.zeros(n_days, dtype=np.int8)
        
        long_cond = votes_long >= required_votes
        short_cond = votes_short >= required_votes
        
        # Apply Longs
        final_signal[long_cond] = 1
        # Apply Shorts (overwrite if short cond is met? Or handle conflict?)
        # Let's handle conflict: if both are true, set to 0.
        conflict = long_cond & short_cond
        final_signal[short_cond] = -1
        final_signal[conflict] = 0
        
        # Calculate Returns
        strat_returns = final_signal * market_returns
        
        # Metrics
        ann_sharpe = get_sharpe(strat_returns) # Returns scalar
        ann_return = np.mean(strat_returns) * 365 * 100
        
        # Trades per year (roughly: count changes in signal / years)
        # Just approximate non-zero days for now or actual flips
        # Let's just count days in market for simplicity or turnover
        days_in_market = np.sum(final_signal != 0)
        
        print(f"{k*100:5.0f}%          {ann_sharpe:6.3f}     {ann_return:8.1f}%       {days_in_market:>5} days")
        
        if ann_sharpe > best_sharpe:
            best_sharpe = ann_sharpe
            best_k = k

    print("-" * 60)
    print(f"Optimal k: {best_k*100:.0f}% with Sharpe: {best_sharpe:.3f}")

if __name__ == '__main__':
    df = fetch_data(SYMBOL, TIMEFRAME, START_DATE)
    run_meta_optimization(df)
