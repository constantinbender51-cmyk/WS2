import ccxt
import pandas as pd
import numpy as np
import time

# --- Configuration ---
SYMBOL = 'BTC/USDT'
TIMEFRAME = '1d'
START_DATE = '2018-01-01 00:00:00'
MAX_SMA = 365
TOP_N_SURVIVORS = 50  # Number of best strategies to carry over to the next level
RISK_FREE_RATE = 0.0

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

def get_sharpe_vectorized(returns_matrix):
    """
    Calculates Annualized Sharpe Ratio for a matrix of returns (Time x N_Strategies).
    """
    means = np.mean(returns_matrix, axis=0)
    stds = np.std(returns_matrix, axis=0)
    
    with np.errstate(divide='ignore', invalid='ignore'):
        sharpes = np.divide(means, stds) * np.sqrt(365)
    
    sharpes[np.isnan(sharpes)] = 0
    return sharpes

def run_greedy_search(df):
    start_time = time.time()
    
    # 1. Prepare Data
    prices = df['close'].to_numpy()
    market_returns = df['close'].pct_change().fillna(0).to_numpy() # (T,)
    n_days = len(market_returns)
    
    print("Pre-calculating SMA Boolean Matrices...")
    # sma_values: (T, 365)
    sma_values = np.zeros((n_days, MAX_SMA))
    for i in range(MAX_SMA):
        sma_values[:, i] = df['close'].rolling(window=i+1).mean().fillna(0).to_numpy()
        
    # Pre-calculate boolean conditions for speed
    # long_conds[t, i] is True if Price[t] > SMA_i[t]
    long_conds = prices[:, None] > sma_values
    short_conds = prices[:, None] < sma_values
    
    all_results = [] # To store dicts of {indices: [1, 2], sharpe: 1.5}
    
    # --- LEVEL 1: Single SMAs ---
    print("\n--- Level 1: Single SMAs ---")
    # Signal: 1 if L, -1 if S, 0 else
    l1_sigs = long_conds.astype(np.int8) - short_conds.astype(np.int8)
    
    # Shift signals (Trade tomorrow based on today)
    l1_sigs = np.roll(l1_sigs, 1, axis=0)
    l1_sigs[0, :] = 0
    
    l1_rets = l1_sigs * market_returns[:, None]
    l1_sharpes = get_sharpe_vectorized(l1_rets)
    
    level_1_results = []
    for i in range(MAX_SMA):
        s = l1_sharpes[i]
        level_1_results.append({'indices': [i], 'sharpe': s, 'level': 1})
    
    # Sort and Keep Best
    level_1_results.sort(key=lambda x: x['sharpe'], reverse=True)
    all_results.extend(level_1_results[:25]) # Keep top 25 for final report
    print(f"Best L1: SMA {level_1_results[0]['indices'][0]+1} (Sharpe: {level_1_results[0]['sharpe']:.3f})")

    # --- LEVEL 2: All Pairs (Brute Force) ---
    print("\n--- Level 2: All Pairs (Brute Force) ---")
    # We loop through SMA 'i' and compare against all 'j'
    level_2_results = []
    
    # Optimization: To avoid N^2 loop in Python, we vectorize the inner loop.
    for i in range(MAX_SMA):
        # Base condition for SMA i
        base_L = long_conds[:, i] # (T,)
        base_S = short_conds[:, i] # (T,)
        
        # Combine with ALL other SMAs (j)
        # Broadcasting: (T, 1) & (T, 365) -> (T, 365)
        combined_L = base_L[:, None] & long_conds
        combined_S = base_S[:, None] & short_conds
        
        # Signal logic: Long if Both > Price, Short if Both < Price
        sigs = combined_L.astype(np.int8) - combined_S.astype(np.int8)
        
        # Shift
        sigs = np.roll(sigs, 1, axis=0)
        sigs[0, :] = 0
        
        # Returns
        rets = sigs * market_returns[:, None]
        sharpes = get_sharpe_vectorized(rets)
        
        # Store results where j > i (to avoid duplicates and self-pairs)
        # We can iterate the numpy array result directly
        for j in range(i + 1, MAX_SMA):
            s = sharpes[j]
            if s > 0: # Filter out garbage
                level_2_results.append({'indices': [i, j], 'sharpe': s, 'level': 2})
                
    level_2_results.sort(key=lambda x: x['sharpe'], reverse=True)
    all_results.extend(level_2_results[:25])
    
    best_l2 = level_2_results[0]
    p1, p2 = [x+1 for x in best_l2['indices']]
    print(f"Best L2: SMA {p1} & {p2} (Sharpe: {best_l2['sharpe']:.3f})")
    
    # Check User's Specific 120 & 40 Case
    # indices are 1-based in prompt, 0-based in array
    u_idx = sorted([39, 119]) 
    found_user = next((r for r in level_2_results if sorted(r['indices']) == u_idx), None)
    if found_user:
        print(f"Specific Check (SMA 40 & 120): Sharpe {found_user['sharpe']:.3f}")

    # --- LEVEL 3 to 5: Greedy Extension ---
    # Start with top survivors from previous level and add 1 SMA
    current_survivors = level_2_results[:TOP_N_SURVIVORS]
    
    for level in range(3, 6):
        print(f"\n--- Level {level}: Greedy Search ---")
        next_gen_results = []
        
        for strat in current_survivors:
            base_indices = strat['indices']
            
            # Construct Base Signals (Logic: AND across all indices)
            # We calculate this once per survivor
            base_L = np.ones(n_days, dtype=bool)
            base_S = np.ones(n_days, dtype=bool)
            
            for idx in base_indices:
                base_L &= long_conds[:, idx]
                base_S &= short_conds[:, idx]
            
            # Try adding every possible remaining SMA
            # Vectorized test against all columns
            combined_L = base_L[:, None] & long_conds
            combined_S = base_S[:, None] & short_conds
            
            sigs = combined_L.astype(np.int8) - combined_S.astype(np.int8)
            sigs = np.roll(sigs, 1, axis=0)
            sigs[0, :] = 0
            
            rets = sigs * market_returns[:, None]
            sharpes = get_sharpe_vectorized(rets)
            
            # Scan results
            for k in range(MAX_SMA):
                if k not in base_indices:
                    new_indices = base_indices + [k]
                    # Sort indices for consistent ID
                    new_indices.sort()
                    next_gen_results.append({
                        'indices': new_indices,
                        'sharpe': sharpes[k],
                        'level': level
                    })
        
        # Deduplicate results (since [A, B] + C is same as [A, C] + B)
        # Use tuple of indices as key
        unique_results = {}
        for r in next_gen_results:
            key = tuple(r['indices'])
            if key not in unique_results:
                unique_results[key] = r
            else:
                # theoretical duplicate should have same sharpe
                pass
        
        next_gen_results = list(unique_results.values())
        next_gen_results.sort(key=lambda x: x['sharpe'], reverse=True)
        
        best = next_gen_results[0]
        smas_str = ", ".join([str(x+1) for x in best['indices']])
        print(f"Best L{level}: SMAs [{smas_str}] (Sharpe: {best['sharpe']:.3f})")
        
        all_results.extend(next_gen_results[:25])
        current_survivors = next_gen_results[:TOP_N_SURVIVORS]

    # --- FINAL RANKING ---
    print("\n" + "="*60)
    print("TOP 25 CONFIRMATION STRATEGIES (ALL LEVELS)")
    print("="*60)
    all_results.sort(key=lambda x: x['sharpe'], reverse=True)
    
    # Deduplicate global list just in case
    seen = set()
    unique_final = []
    for r in all_results:
        k = tuple(r['indices'])
        if k not in seen:
            seen.add(k)
            unique_final.append(r)
            
    top_25 = unique_final[:25]
    for i, r in enumerate(top_25):
        smas = [str(x+1) for x in r['indices']]
        print(f"{i+1:2d}. Level {r['level']} | SMAs: {', '.join(smas):<20} | Sharpe: {r['sharpe']:.3f}")

    # --- CONSENSUS SCORE (0-5) ---
    print("\n" + "="*60)
    print("CONSENSUS SCORE ANALYSIS (Using Best Strategy from Levels 1-5)")
    print("="*60)
    
    # 1. Identify the champion for each level (1 to 5)
    # We re-find them to ensure we have exactly one per level
    champions = []
    for lvl in range(1, 6):
        champ = next((r for r in unique_final if r['level'] == lvl), None)
        if champ:
            champions.append(champ)
            
    print(f"Consensus built from {len(champions)} strategies (Best L1, Best L2, etc.)")
    
    # 2. Generate Signals for these champions
    # signals_matrix: (T, N_Champs)
    signals_list = []
    
    for champ in champions:
        # Reconstruct signal
        base_L = np.ones(n_days, dtype=bool)
        base_S = np.ones(n_days, dtype=bool)
        for idx in champ['indices']:
            base_L &= long_conds[:, idx]
            base_S &= short_conds[:, idx]
            
        sig = base_L.astype(np.int8) - base_S.astype(np.int8)
        sig = np.roll(sig, 1)
        sig[0] = 0
        signals_list.append(sig)
        
    signals_matrix = np.column_stack(signals_list) # (T, 5)
    
    # 3. Compute Score (Sum of signals)
    # Range: -5 to +5 (if 5 strategies found)
    consensus_score = np.sum(signals_matrix, axis=1)
    
    # 4. Evaluate Trading the Score
    # We test thresholds: Go Long if Score >= X, Short if Score <= -X
    print(f"\n{'Threshold':<10} {'Condition':<20} {'Sharpe':<10} {'Ann. Ret':<10} {'Trades'}")
    print("-" * 65)
    
    max_score = len(champions)
    # Test thresholds 1 to 5
    for thresh in range(1, max_score + 1):
        # Long if Score >= thresh
        # Short if Score <= -thresh
        final_sig = np.zeros(n_days)
        final_sig[consensus_score >= thresh] = 1
        final_sig[consensus_score <= -thresh] = -1
        
        rets = final_sig * market_returns
        sharpe = (np.mean(rets) / np.std(rets)) * np.sqrt(365) if np.std(rets) > 0 else 0
        ann_ret = np.mean(rets) * 365 * 100
        trades = np.sum(np.abs(np.diff(final_sig)))
        
        print(f"{thresh:<10} Score >= {thresh}/{max_score}      {sharpe:<10.3f} {ann_ret:6.1f}%    {trades:.0f}")

if __name__ == '__main__':
    df = fetch_data(SYMBOL, TIMEFRAME, START_DATE)
    run_greedy_search(df)
