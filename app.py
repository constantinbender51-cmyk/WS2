import random
import time
import sys
import json
import urllib.request
from collections import Counter, defaultdict
from datetime import datetime

# --- CONFIGURATION ---
SYMBOL = "ETHUSDT"
INTERVAL = "1h"
START_DATE = "2020-01-01" 

def get_binance_data(symbol=SYMBOL, interval=INTERVAL, start_str=START_DATE):
    """Fetches historical kline data from Binance public API."""
    print(f"\n--- Fetching {symbol} {interval} data from Binance (Once) ---")
    
    start_ts = int(datetime.strptime(start_str, "%Y-%m-%d").timestamp() * 1000)
    end_ts = int(time.time() * 1000)
    
    all_prices = []
    current_start = start_ts
    total_time = end_ts - start_ts
    
    while current_start < end_ts:
        url = f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval={interval}&startTime={current_start}&limit=1000"
        try:
            with urllib.request.urlopen(url) as response:
                data = json.loads(response.read().decode())
                if not data:
                    break
                
                batch_prices = [float(candle[4]) for candle in data]
                all_prices.extend(batch_prices)
                current_start = data[-1][6] + 1
                
                progress = (current_start - start_ts) / total_time
                sys.stdout.write(f"\rDownload Progress: [{int(progress*20)*'#'}{(20-int(progress*20))*'-'}] {len(all_prices)} candles")
                sys.stdout.flush()
                
        except Exception as e:
            print(f"\nError fetching data: {e}")
            break
            
    print(f"\nTotal data points fetched: {len(all_prices)}")
    return all_prices

# --- BUCKET LOGIC ---

def get_bucket(price, bucket_size):
    """Converts price to bucket index based on dynamic size."""
    if bucket_size <= 0: bucket_size = 1e-9
    
    if price >= 0:
        return int(price // bucket_size)
    else:
        return int(price // bucket_size) - 1

def calculate_bucket_size(prices, bucket_count):
    """Calculates bucket size based on total range and target count."""
    if not prices: return 1.0
    min_p = min(prices)
    max_p = max(prices)
    price_range = max_p - min_p
    
    if bucket_count <= 0: return 1.0
    
    size = price_range / bucket_count
    return size if size > 0 else 0.01

# --- CORE LOGIC ---

def train_models(train_buckets, seq_len):
    """Helper to train models and return the maps."""
    abs_map = defaultdict(Counter)
    der_map = defaultdict(Counter)
    
    for i in range(len(train_buckets) - seq_len):
        a_seq = tuple(train_buckets[i : i + seq_len])
        a_succ = train_buckets[i + seq_len]
        abs_map[a_seq][a_succ] += 1
        
        if i > 0:
            d_seq = tuple(train_buckets[j] - train_buckets[j-1] for j in range(i, i + seq_len))
            d_succ = train_buckets[i + seq_len] - train_buckets[i + seq_len - 1]
            der_map[d_seq][d_succ] += 1
            
    return abs_map, der_map

def get_prediction(model_type, abs_map, der_map, a_seq, d_seq, last_val):
    """
    Returns the predicted next VALUE (Bucket index).
    Returns None if the model abstains (no pattern found or contradiction).
    """
    
    if model_type == "Absolute":
        if a_seq in abs_map:
            return abs_map[a_seq].most_common(1)[0][0]
        return None # Abstain
        
    elif model_type == "Derivative":
        if d_seq in der_map:
            pred_change = der_map[d_seq].most_common(1)[0][0]
            return last_val + pred_change
        return None # Abstain
        
    elif model_type == "Combined":
        # Recursively get individual predictions
        p_abs = get_prediction("Absolute", abs_map, der_map, a_seq, d_seq, last_val)
        p_der = get_prediction("Derivative", abs_map, der_map, a_seq, d_seq, last_val)
        
        # Case 1: Neither predicts
        if p_abs is None and p_der is None:
            return None
            
        # Case 2: Only Derivative predicts
        if p_abs is None and p_der is not None:
            return p_der
            
        # Case 3: Only Absolute predicts
        if p_abs is not None and p_der is None:
            return p_abs
            
        # Case 4: Both predict - Check Contradiction
        diff_abs = p_abs - last_val
        diff_der = p_der - last_val
        
        # Directions: 1 (Up), -1 (Down), 0 (Flat)
        dir_abs = 1 if diff_abs > 0 else (-1 if diff_abs < 0 else 0)
        dir_der = 1 if diff_der > 0 else (-1 if diff_der < 0 else 0)
        
        # Contradiction check: One says Up (1), one says Down (-1)
        # We allow (1 and 0) or (-1 and 0) as non-contradictory
        if (dir_abs == 1 and dir_der == -1) or (dir_abs == -1 and dir_der == 1):
            return None # Contradiction -> Abstain
            
        # If no contradiction, use Absolute as requested
        return p_abs
    
    return None

def evaluate_parameters(prices, bucket_size, seq_len):
    """
    Runs analysis. Returns stats for ALL 3 models.
    """
    buckets = [get_bucket(p, bucket_size) for p in prices]
    split_idx = int(len(buckets) * 0.7)
    train_buckets = buckets[:split_idx]
    test_buckets = buckets[split_idx:]
    
    abs_map, der_map = train_models(train_buckets, seq_len)

    # Stats: [correct_count, valid_total_count]
    stats = {"Absolute": [0, 0], "Derivative": [0, 0], "Combined": [0, 0]}
    
    total_samples = len(test_buckets) - seq_len
    
    for i in range(total_samples):
        curr_idx = split_idx + i
        a_seq = tuple(buckets[curr_idx : curr_idx + seq_len])
        d_seq = tuple(buckets[j] - buckets[j-1] for j in range(curr_idx, curr_idx + seq_len))
        last_val = a_seq[-1]
        actual_val = buckets[curr_idx + seq_len]
        actual_diff = actual_val - last_val

        # Get predictions (No randomness, allows None)
        p_abs = get_prediction("Absolute", abs_map, der_map, a_seq, d_seq, last_val)
        p_der = get_prediction("Derivative", abs_map, der_map, a_seq, d_seq, last_val)
        p_comb = get_prediction("Combined", abs_map, der_map, a_seq, d_seq, last_val)

        for name, pred in [("Absolute", p_abs), ("Derivative", p_der), ("Combined", p_comb)]:
            # If model abstained (None), skip this trade
            if pred is None:
                continue
                
            pred_diff = pred - last_val
            
            # Only count if the model predicts a move (not 0) and reality moved (not 0)
            # (Standard directional accuracy metric)
            if pred_diff != 0 and actual_diff != 0:
                stats[name][1] += 1
                if (pred_diff > 0 and actual_diff > 0) or (pred_diff < 0 and actual_diff < 0):
                    stats[name][0] += 1

    # Return best
    best_acc = -1
    best_model = "None"
    best_trades = 0
    for name, (correct, total) in stats.items():
        if total > 0:
            acc = (correct / total) * 100
            if acc > best_acc:
                best_acc = acc
                best_model = name
                best_trades = total
    
    return best_acc, best_model, best_trades

def run_portfolio_analysis(prices, top_configs):
    """
    Runs the 'Union' strategy on the top configurations.
    """
    print(f"\n--- Running Portfolio Analysis (Union of Top {len(top_configs)}) ---")
    
    models = []
    for config in top_configs:
        b_size = config['bucket_size'] 
        s_len = config['seq_len']
        
        buckets = [get_bucket(p, b_size) for p in prices]
        split_idx = int(len(buckets) * 0.7)
        train_buckets = buckets[:split_idx]
        
        abs_map, der_map = train_models(train_buckets, s_len)
        
        models.append({
            "config": config,
            "buckets": buckets,
            "split_idx": split_idx,
            "abs_map": abs_map,
            "der_map": der_map
        })

    if not models:
        return 0, 0

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
            
            a_seq = tuple(buckets[curr_raw_idx : curr_raw_idx + seq_len])
            d_seq = tuple(buckets[j] - buckets[j-1] for j in range(curr_raw_idx, curr_raw_idx + seq_len))
            last_val = a_seq[-1]
            actual_val = buckets[curr_raw_idx + seq_len]
            
            diff = actual_val - last_val
            model_actual_dir = 1 if diff > 0 else (-1 if diff < 0 else 0)
            
            # Predict
            pred_val = get_prediction(c['model'], model['abs_map'], model['der_map'], 
                                      a_seq, d_seq, last_val)
            
            # Skip if Abstain (None)
            if pred_val is None:
                continue
            
            pred_diff = pred_val - last_val
            
            if pred_diff != 0:
                direction = 1 if pred_diff > 0 else -1
                active_directions.append({
                    "dir": direction,
                    "is_correct": (direction == model_actual_dir),
                    "is_flat": (model_actual_dir == 0)
                })

        # --- Aggregate Signals ---
        if not active_directions:
            continue
            
        dirs = [x['dir'] for x in active_directions]
        
        # Conflict Check (Portfolio level)
        has_up = 1 in dirs
        has_down = -1 in dirs
        
        if has_up and has_down:
            continue
            
        any_correct = any(x['is_correct'] for x in active_directions)
        any_wrong_direction = any((not x['is_correct'] and not x['is_flat']) for x in active_directions)
        all_flat = all(x['is_flat'] for x in active_directions)
        
        if not all_flat:
            unique_total += 1
            if any_correct and not any_wrong_direction:
                unique_correct += 1

    return unique_correct, unique_total

def run_grid_search():
    prices = get_binance_data()
    if len(prices) < 500: return

    # Adaptive Grid Search
    bucket_counts = [10, 25, 50, 75, 100, 150, 200, 300]
    seq_lengths = [3, 4, 5, 6, 8]
    
    results = []
    
    print(f"\n--- Starting Grid Search ({len(bucket_counts) * len(seq_lengths)} combinations) ---")
    print(f"{'Count':<8} | {'Size':<8} | {'SeqLen':<8} | {'Best Model':<12} | {'Dir Acc %':<10} | {'Trades':<8}")
    print("-" * 75)

    for b_count in bucket_counts:
        b_size = calculate_bucket_size(prices, b_count)
        
        for s_len in seq_lengths:
            accuracy, model_name, trades = evaluate_parameters(prices, b_size, s_len)
            
            if trades > 20: 
                results.append({
                    "bucket_count": b_count,
                    "bucket_size": b_size,
                    "seq_len": s_len,
                    "model": model_name,
                    "accuracy": accuracy,
                    "trades": trades
                })
                print(f"{b_count:<8} | {b_size:<8.4f} | {s_len:<8} | {model_name:<12} | {accuracy:<10.2f} | {trades:<8}")

    results.sort(key=lambda x: x['accuracy'], reverse=True)
    top_3 = results[:3]
    
    print("\n" + "="*50)
    print(" TOP 3 CONFIGURATIONS ")
    print("="*50)
    for i, res in enumerate(top_3):
        print(f"#{i+1}: Count {res['bucket_count']} (Size {res['bucket_size']:.4f}), Len {res['seq_len']} ({res['model']}) -> {res['accuracy']:.2f}% ({res['trades']} trades)")

    if len(top_3) > 0:
        u_correct, u_total = run_portfolio_analysis(prices, top_3)
        print("\n" + "="*50)
        print(" FINAL PORTFOLIO RESULT (Union of Top 3) ")
        print("="*50)
        if u_total > 0:
            print(f"Combined Unique Trades: {u_total}")
            print(f"Combined Accuracy:      {(u_correct/u_total)*100:.2f}%")
        else:
            print("No unique trades found (Models might be conflicting or abstaining).")

if __name__ == "__main__":
    run_grid_search()
