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
MIN_SHARPE = 0.5  # Lowered to capture more strategies for the consensus

def fetch_data(symbol, timeframe, start_str):
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
    """Vectorized Sharpe calculation (Annualized)."""
    means = np.mean(returns_array, axis=0)
    stds = np.std(returns_array, axis=0)
    return np.divide(means, stds, out=np.zeros_like(means), where=stds!=0) * np.sqrt(365)

def diagnose_specific_pair(df, fast=40, slow=120):
    """Deep dive into the specific strategy mentioned by the user."""
    print(f"\n" + "="*60)
    print(f"DIAGNOSIS: SMA {fast} vs SMA {slow}")
    print("="*60)
    
    prices = df['close']
    returns = prices.pct_change().fillna(0)
    
    sma_f = prices.rolling(window=fast).mean()
    sma_s = prices.rolling(window=slow).mean()
    
    # 1. Raw Signal (1 where Fast > Slow, else -1)
    raw_signal = np.where(sma_f > sma_s, 1, -1)
    
    # 2. Shift (Trade tomorrow based on today)
    signal = np.roll(raw_signal, 1)
    signal[0] = 0
    
    # --- Mode A: Long / Short ---
    returns_ls = signal * returns
    sharpe_ls = (returns_ls.mean() / returns_ls.std()) * np.sqrt(365)
    ann_ret_ls = returns_ls.mean() * 365 * 100
    
    # --- Mode B: Long / Flat ---
    # Only take the long signals (where signal == 1), otherwise 0
    signal_lo = np.where(signal == 1, 1, 0)
    returns_lo = signal_lo * returns
    sharpe_lo = (returns_lo.mean() / returns_lo.std()) * np.sqrt(365)
    ann_ret_lo = returns_lo.mean() * 365 * 100
    
    # --- Buy & Hold ---
    sharpe_bh = (returns.mean() / returns.std()) * np.sqrt(365)
    
    print(f"{'Metric':<20} {'Long/Short':<15} {'Long/Flat':<15} {'Buy & Hold':<15}")
    print("-" * 65)
    print(f"{'Sharpe Ratio':<20} {sharpe_ls:<15.3f} {sharpe_lo:<15.3f} {sharpe_bh:<15.3f}")
    print(f"{'Ann. Return %':<20} {ann_ret_ls:<15.1f} {ann_ret_lo:<15.1f} {100*returns.mean()*365:<15.1f}")
    print("="*60 + "\n")

def run_meta_optimization(df):
    print(f"--- Starting Meta-Strategy Optimization ---")
    
    # 1. Setup
    market_returns = df['close'].pct_change().fillna(0).to_numpy()
    close_prices = df['close'].to_numpy()
    n_days = len(market_returns)
    
    # 2. Pre-calculate SMA Matrix
    print(f"Pre-calculating SMA Matrix (1 to {MAX_SMA})...")
    sma_matrix = np.zeros((n_days, MAX_SMA))
    for i in range(MAX_SMA):
        sma_matrix[:, i] = df['close'].rolling(window=i+1).mean().fillna(0).to_numpy()

    winning_signals_list = []
    
    # --- A. Price vs SMA ---
    print("Scanning Price vs SMA strategies...")
    pv_raw = np.where(close_prices[:, None] > sma_matrix, 1, -1).astype(np.int8)
    pv_signals = np.roll(pv_raw, 1, axis=0)
    pv_signals[0, :] = 0
    
    # Check Performance (Long/Short)
    pv_returns = pv_signals * market_returns[:, None]
    pv_sharpes = get_sharpe(pv_returns)
    
    mask = pv_sharpes > MIN_SHARPE
    if np.any(mask):
        winning_signals_list.append(pv_signals[:, mask])
        print(f"  -> Found {np.sum(mask)} Price-SMA strategies > {MIN_SHARPE} Sharpe")

    # --- B. SMA vs SMA ---
    print("Scanning SMA vs SMA strategies...")
    count_sma_winners = 0
    
    for fast_idx in range(MAX_SMA):
        fast_vec = sma_matrix[:, fast_idx][:, None]
        # Compare fast vs ALL slow
        raw = np.where(fast_vec > sma_matrix, 1, -1).astype(np.int8)
        
        # Shift
        sigs = np.roll(raw, 1, axis=0)
        sigs[0, :] = 0
        
        # Perf
        rets = sigs * market_returns[:, None]
        sharpes = get_sharpe(rets)
        
        # Filter (exclude diagonal and poor performers)
        # Note: We exclude cases where sharpes are NaN or Inf
        valid_mask = np.isfinite(sharpes) & (sharpes > MIN_SHARPE) & (np.arange(MAX_SMA) != fast_idx)
        
        if np.any(valid_mask):
            count_sma_winners += np.sum(valid_mask)
            winning_signals_list.append(sigs[:, valid_mask])
            
    print(f"  -> Found {count_sma_winners} SMA-SMA strategies > {MIN_SHARPE} Sharpe")
    
    if not winning_signals_list:
        print("No strategies met the criteria.")
        return

    # 3. Consensus
    all_winners = np.hstack(winning_signals_list)
    n_winners = all_winners.shape[1]
    
    # Votes
    votes_long = (all_winners == 1).sum(axis=1)
    votes_short = (all_winners == -1).sum(axis=1)
    
    print("\n--- Consensus Optimization (k-sweep) ---")
    print(f"{'k%':<10} {'Sharpe':<10} {'Ann. Ret':<12} {'Trades':<8}")
    print("-" * 45)
    
    best_sharpe = -999
    best_k = 0
    
    for k in np.arange(0.0, 1.01, 0.05):
        threshold = k * n_winners
        
        final_sig = np.zeros(n_days, dtype=np.int8)
        
        # If winners vote > threshold, take position
        is_long = votes_long >= threshold
        is_short = votes_short >= threshold
        
        final_sig[is_long] = 1
        final_sig[is_short] = -1
        # Conflict resolution: Flat
        final_sig[is_long & is_short] = 0
        
        strat_rets = final_sig * market_returns
        s = get_sharpe(strat_rets.reshape(-1, 1))[0] # get_sharpe expects 2d usually or handles 1d?
                                                     # Our get_sharpe does axis=0. It works on 1D too if properly shaped or not.
                                                     # Actually get_sharpe(strat_rets) where strat_rets is 1D returns scalar.
        
        if np.isnan(s): s = 0
        
        trades = np.sum(np.abs(np.diff(final_sig)))
        
        print(f"{k*100:3.0f}%       {s:6.3f}     {np.mean(strat_rets)*365*100:6.1f}%    {trades:4}")
        
        if s > best_sharpe:
            best_sharpe = s
            best_k = k

    print("-" * 45)
    print(f"Optimal Consensus k: {best_k*100:.0f}% | Sharpe: {best_sharpe:.3f}")

if __name__ == '__main__':
    df = fetch_data(SYMBOL, TIMEFRAME, START_DATE)
    diagnose_specific_pair(df, 40, 120)
    run_meta_optimization(df)
