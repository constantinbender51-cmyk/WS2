import os
import sys
import json
import time
import random
import urllib.request
import pandas as pd
from datetime import datetime
from collections import Counter, defaultdict

# --- CONFIGURATION ---
SYMBOL = "ETHUSDT"
TARGET_INTERVAL = "60m" # Using the key from 26's TIMEFRAMES logic (60m = 1h)
START_DATE = "2020-01-01"
BASE_INTERVAL = "15m"   # Download base data like in 26

# Timeframe mapping (for resampling)
TIMEFRAMES = {
    "15m": None,
    "30m": "30min",
    "60m": "1h",
    "240m": "4h",
    "1d": "1D"
}

# --- 1. DATA FETCHING (Logic from 26) ---

def get_binance_data(symbol, start_str=START_DATE):
    """Fetches raw 15m data to be resampled, ensuring consistency."""
    print(f"\n[{symbol}] Fetching raw {BASE_INTERVAL} data from {start_str}...")
    start_ts = int(datetime.strptime(start_str, "%Y-%m-%d").timestamp() * 1000)
    end_ts = int(time.time() * 1000)
    all_candles = []
    current_start = start_ts
    
    while current_start < end_ts:
        url = f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval={BASE_INTERVAL}&startTime={current_start}&limit=1000"
        try:
            with urllib.request.urlopen(url) as response:
                data = json.loads(response.read().decode())
                if not data: break
                # Extract close price (index 4) and close time (index 6)
                batch = [(int(c[6]), float(c[4])) for c in data]
                all_candles.extend(batch)
                current_start = data[-1][6] + 1
                
                # Simple progress
                sys.stdout.write(f"\rDownloaded {len(all_candles)} raw candles...")
                sys.stdout.flush()
        except Exception as e:
            print(f"Error fetching: {e}")
            break
            
    print(f"\n[{symbol}] Total raw candles: {len(all_candles)}")
    return all_candles

def resample_prices(raw_data, target_freq):
    """Resamples raw data to target frequency using Pandas."""
    if target_freq is None: return [x[1] for x in raw_data]
    
    df = pd.DataFrame(raw_data, columns=['timestamp', 'price'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    
    # Resample using 'last' (close price logic)
    resampled = df['price'].resample(target_freq).last().dropna()
    return resampled.tolist()

# --- 2. CORE STRATEGY LOGIC (Updated to 26's Fixed Logic) ---

def get_bucket(price, bucket_size):
    if price >= 0: return (int(price) // bucket_size) + 1
    else: return (int(price + 1) // bucket_size) - 1

def train_models(train_buckets, seq_len):
    """
    Train models using the fixed indexing from 26.
    """
    abs_map = defaultdict(Counter)
    der_map = defaultdict(Counter)
    
    for i in range(len(train_buckets) - seq_len):
        # 1. Absolute Sequence
        a_seq = tuple(train_buckets[i : i + seq_len])
        a_succ = train_buckets[i + seq_len]
        abs_map[a_seq][a_succ] += 1
        
        # 2. Derivative Sequence (FIXED LOGIC)
        # We start at i+1 to perform internal diffs only (buckets[i+1] - buckets[i])
        d_seq = tuple(train_buckets[j] - train_buckets[j-1] for j in range(i + 1, i + seq_len))
        
        d_succ = train_buckets[i + seq_len] - train_buckets[i + seq_len - 1]
        der_map[d_seq][d_succ] += 1
            
    return abs_map, der_map

def get_prediction(model_type, abs_map, der_map, a_seq, d_seq, last_val, all_vals, all_changes):
    if model_type == "Absolute":
        if a_seq in abs_map: return abs_map[a_seq].most_common(1)[0][0]
        return random.choice(all_vals)
        
    elif model_type == "Derivative":
        if d_seq in der_map: pred_change = der_map[d_seq].most_common(1)[0][0]
        else: pred_change = random.choice(all_changes)
        return last_val + pred_change
        
    elif model_type == "Combined":
        abs_cand = abs_map.get(a_seq, Counter())
        der_cand = der_map.get(d_seq, Counter())
        
        poss = set(abs_cand.keys())
        for c in der_cand.keys(): poss.add(last_val + c)
        
        if not poss: return random.choice(all_vals)
        
        best, max_s = None, -1
        for v in poss:
            # Add votes from absolute expert + votes from derivative expert
            s = abs_cand[v] + der_cand[v - last_val]
            if s > max_s: max_s, best = s, v
        return best
    return last_val

def evaluate_parameters(prices, bucket_size, seq_len):
    """
    Evaluates config using the fixed indexing logic.
    """
    buckets = [get_bucket(p, bucket_size) for p in prices]
    split_idx = int(len(buckets) * 0.7)
    train = buckets[:split_idx]
    test = buckets[split_idx:]
    
    if len(train) < seq_len + 10: return 0, "None", 0
    
    all_vals = list(set(train))
    all_changes = list(set(train[j] - train[j-1] for j in range(1, len(train))))
    abs_map, der_map = train_models(train, seq_len)
    
    stats = {"Absolute": [0,0], "Derivative": [0,0], "Combined": [0,0]}
    
    for i in range(len(test) - seq_len):
        curr = split_idx + i
        a_seq = tuple(buckets[curr : curr+seq_len])
        
        # FIX: range(curr + 1, ...) ensures we don't look back before the window
        d_seq = tuple(buckets[j] - buckets[j-1] for j in range(curr + 1, curr+seq_len))
        
        last_val = a_seq[-1]
        actual_val = buckets[curr+seq_len]
        act_diff = actual_val - last_val
        
        p_abs = get_prediction("Absolute", abs_map, der_map, a_seq, d_seq, last_val, all_vals, all_changes)
        p_der = get_prediction("Derivative", abs_map, der_map, a_seq, d_seq, last_val, all_vals, all_changes)
        p_comb = get_prediction("Combined", abs_map, der_map, a_seq, d_seq, last_val, all_vals, all_changes)
        
        for name, pred in [("Absolute", p_abs), ("Derivative", p_der), ("Combined", p_comb)]:
            p_diff = pred - last_val
            if p_diff != 0 and act_diff != 0:
                stats[name][1] += 1
                if (p_diff > 0 and act_diff > 0) or (p_diff < 0 and act_diff < 0):
                    stats[name][0] += 1
                    
    best_acc, best_name, best_trades = -1, "None", 0
    for name, (corr, tot) in stats.items():
        if tot > 0:
            acc = (corr/tot)*100
            if acc > best_acc: best_acc, best_name, best_trades = acc, name, tot
            
    return best_acc, best_name, best_trades

def run_portfolio_analysis(prices, top_configs):
    """
    Runs the portfolio analysis (from 27) but using the FIXED sequence logic (from 26).
    """
    print(f"\n--- Running Portfolio Analysis (Union of Top {len(top_configs)}) ---")
    
    models = []
    for config in top_configs:
        b_size = config['bucket']
        s_len = config['seq_len']
        
        buckets = [get_bucket(p, b_size) for p in prices]
        split_idx = int(len(buckets) * 0.7)
        train_buckets = buckets[:split_idx]
        
        # Train
        abs_map, der_map = train_models(train_buckets, s_len)
        
        models.append({
            "config": config,
            "buckets": buckets,
            "split_idx": split_idx,
            "abs_map": abs_map,
            "der_map": der_map,
            "all_vals": list(set(train_buckets)),
            "all_changes": list(set(train_buckets[j] - train_buckets[j-1] for j in range(1, len(train_buckets))))
        })

    start_test_idx = models[0]['split_idx']
    max_seq_len = max(m['config']['seq_len'] for m in models)
    total_test_len = len(models[0]['buckets']) - start_test_idx - max_seq_len
    
    unique_correct = 0
    unique_total = 0
    
    print(f"Scanning {total_test_len} time steps for combined signals...")
    
    for i in range(total_test_len):
        curr_raw_idx = start_test_idx + i
        active_directions = []
        
        for model in models:
            c = model['config']
            seq_len = c['seq_len']
            buckets = model['buckets']
            
            # Sequence Generation (FIXED)
            a_seq = tuple(buckets[curr_raw_idx : curr_raw_idx + seq_len])
            d_seq = tuple(buckets[j] - buckets[j-1] for j in range(curr_raw_idx + 1, curr_raw_idx + seq_len))
            
            last_val = a_seq[-1]
            actual_val = buckets[curr_raw_idx + seq_len]
            
            diff = actual_val - last_val
            model_actual_dir = 1 if diff > 0 else -1 if diff < 0 else 0
            
            pred_val = get_prediction(c['model'], model['abs_map'], model['der_map'], 
                                      a_seq, d_seq, last_val, model['all_vals'], model['all_changes'])
            
            pred_diff = pred_val - last_val
            
            if pred_diff != 0:
                direction = 1 if pred_diff > 0 else -1
                is_correct = (direction == model_actual_dir)
                is_flat = (model_actual_dir == 0)
                
                active_directions.append({
                    "dir": direction,
                    "is_correct": is_correct,
                    "is_flat": is_flat
                })

        # --- Aggregate Signals ---
        if not active_directions: continue
            
        dirs = [x['dir'] for x in active_directions]
        
        # Conflict Check
        if (1 in dirs) and (-1 in dirs): continue
            
        # Directional Accuracy Check
        any_correct = any(x['is_correct'] for x in active_directions)
        any_wrong_direction = any((not x['is_correct'] and not x['is_flat']) for x in active_directions)
        all_flat = all(x['is_flat'] for x in active_directions)
        
        if not all_flat:
            unique_total += 1
            if any_correct and not any_wrong_direction:
                unique_correct += 1

    return unique_correct, unique_total

def run_grid_search():
    # 1. Fetch Raw Data
    raw_candles = get_binance_data(SYMBOL)
    if not raw_candles: return

    # 2. Resample (Using logic from 26)
    target_pd_freq = TIMEFRAMES.get(TARGET_INTERVAL, "1h")
    print(f"Resampling to {TARGET_INTERVAL} ({target_pd_freq})...")
    prices = resample_prices(raw_candles, target_pd_freq)
    
    if len(prices) < 500:
        print("Not enough data after resampling.")
        return

    # 3. Dynamic Grid Setup
    avg_price = sum(prices) / len(prices)
    base_step = max(1, int(avg_price * 0.001)) # 0.1% base step
    # Generates 10 bucket sizes scaling up
    bucket_sizes = [base_step * i for i in range(1, 11)] 
    seq_lengths = [3, 4, 5, 6]
    
    results = []
    
    print(f"\n--- Starting Grid Search ({len(bucket_sizes) * len(seq_lengths)} combinations) ---")
    print(f"{'Bucket':<8} | {'SeqLen':<8} | {'Best Model':<12} | {'Dir Acc %':<10} | {'Trades':<8}")
    print("-" * 60)

    for b_size in bucket_sizes:
        for s_len in seq_lengths:
            accuracy, model_name, trades = evaluate_parameters(prices, b_size, s_len)
            if trades > 20: 
                results.append({
                    "bucket": b_size,
                    "seq_len": s_len,
                    "model": model_name,
                    "accuracy": accuracy,
                    "trades": trades
                })
                print(f"{b_size:<8} | {s_len:<8} | {model_name:<12} | {accuracy:<10.2f} | {trades:<8}")

    # Top 3
    results.sort(key=lambda x: x['accuracy'], reverse=True)
    top_3 = results[:3]
    
    print("\n" + "="*40)
    print(" TOP 3 CONFIGURATIONS ")
    print("="*40)
    for i, res in enumerate(top_3):
        print(f"#{i+1}: Bucket {res['bucket']}, Len {res['seq_len']} ({res['model']}) -> {res['accuracy']:.2f}% ({res['trades']} trades)")

    # Combined Analysis
    if len(top_3) > 0:
        u_correct, u_total = run_portfolio_analysis(prices, top_3)
        print("\n" + "="*40)
        print(" FINAL PORTFOLIO RESULT (Union of Top 3) ")
        print("="*40)
        if u_total > 0:
            print(f"Combined Unique Trades: {u_total}")
            print(f"Combined Accuracy:      {(u_correct/u_total)*100:.2f}%")
        else:
            print("No unique trades found.")

if __name__ == "__main__":
    run_grid_search()