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

# --- UPDATED BUCKET LOGIC (From strategies_1_2.py) ---

def get_bucket(price, bucket_size):
    """Converts price to bucket index based on dynamic size."""
    # Safety for extremely small sizes to prevent division by zero
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
        
        # UPDATED: Derive sequence strictly from a_seq (no lookback)
        # Length becomes seq_len - 1
        if seq_len > 1:
            d_seq = tuple(a_seq[k] - a_seq[k-1] for k in range(1, len(a_seq)))
            d_succ = train_buckets[i + seq_len] - train_buckets[i + seq_len - 1]
            der_map[d_seq][d_succ] += 1
            
    return abs_map, der_map

def get_prediction(model_type, abs_map, der_map, a_seq, d_seq, last_val, all_vals, all_changes):
    """Helper to get a single prediction based on model type."""
    
    if model_type == "Absolute":
        if a_seq in abs_map:
            return abs_map[a_seq].most_common(1)[0][0]
        return random.choice(all_vals)
        
    elif model_type == "Derivative":
        if d_seq in der_map:
            pred_change = der_map[d_seq].most_common(1)[0][0]
        else:
            pred_change = random.choice(all_changes)
        return last_val + pred_change
        
    elif model_type == "Combined":
        abs_candidates = abs_map.get(a_seq, Counter())
        der_candidates = der_map.get(d_seq, Counter())
        
        possible_next_vals = set(abs_candidates.keys())
        for change in der_candidates.keys():
            possible_next_vals.add(last_val + change)
            
        if not possible_next_vals:
            return random.choice(all_vals)
        
        best_val = None
        max_combined_score = -1
        for val in possible_next_vals:
            implied_change = val - last_val
            score = abs_candidates[val] + der_candidates[implied_change]
            if score > max_combined_score:
                max_combined_score = score
                best_val = val
        return best_val
    
    return last_val

def evaluate_parameters(prices, bucket_size, seq_len):
    """
    Runs analysis. Returns stats for ALL 3 models so we can pick the best.
    """
    buckets = [get_bucket(p, bucket_size) for p in prices]
    split_idx = int(len(buckets) * 0.7)
    train_buckets = buckets[:split_idx]
    test_buckets = buckets[split_idx:]
    
    all_train_values = list(set(train_buckets))
    all_train_changes = list(set(train_buckets[j] - train_buckets[j-1] for j in range(1, len(train_buckets))))

    abs_map, der_map = train_models(train_buckets, seq_len)

    # Stats: [correct_count, valid_total_count]
    stats = {"Absolute": [0, 0], "Derivative": [0, 0], "Combined": [0, 0]}
    
    total_samples = len(test_buckets) - seq_len
    
    for i in range(total_samples):
        curr_idx = split_idx + i
        a_seq = tuple(buckets[curr_idx : curr_idx + seq_len])
        
        # UPDATED: Derive d_seq from a_seq (Length = seq_len - 1)
        if seq_len > 1:
            d_seq = tuple(a_seq[k] - a_seq[k-1] for k in range(1, len(a_seq)))
        else:
            d_seq = ()

        last_val = a_seq[-1]
        actual_val = buckets[curr_idx + seq_len]
        actual_diff = actual_val - last_val

        # Get predictions
        p_abs = get_prediction("Absolute", abs_map, der_map, a_seq, d_seq, last_val, all_train_values, all_train_changes)
        p_der = get_prediction("Derivative", abs_map, der_map, a_seq, d_seq, last_val, all_train_values, all_train_changes)
        p_comb = get_prediction("Combined", abs_map, der_map, a_seq, d_seq, last_val, all_train_values, all_train_changes)

        for name, pred in [("Absolute", p_abs), ("Derivative", p_der), ("Combined", p_comb)]:
            pred_diff = pred - last_val
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
    Runs the 'Union' strategy on the top 3 configurations.
    """
    print(f"\n--- Running Portfolio Analysis (Union of Top {len(top_configs)}) ---")
    
    # 1. Train all top models
    models = []
    for config in top_configs:
        # Use the stored bucket size (derived from count)
        b_size = config['bucket_size'] 
        s_len = config['seq_len']
        m_type = config['model']
        
        # Re-process buckets for this specific config
        buckets = [get_bucket(p, b_size) for p in prices]
        split_idx = int(len(buckets) * 0.7)
        train_buckets = buckets[:split_idx]
        
        # Train
        abs_map, der_map = train_models(train_buckets, s_len)
        
        # Store everything needed to predict
        models.append({
            "config": config,
            "buckets": buckets, # Full bucket list for this config
            "split_idx": split_idx,
            "abs_map": abs_map,
            "der_map": der_map,
            "all_vals": list(set(train_buckets)),
            "all_changes": list(set(train_buckets[j] - train_buckets[j-1] for j in range(1, len(train_buckets))))
        })

    if not models:
        return 0, 0

    # 2. Iterate through TEST time (aligned by raw price index)
    start_test_idx = models[0]['split_idx']
    max_seq_len = max(m['config']['seq_len'] for m in models)
    
    # Effective range
    total_test_len = len(models[0]['buckets']) - start_test_idx - max_seq_len
    
    unique_correct = 0
    unique_total = 0
    
    print(f"Scanning {total_test_len} time steps for combined signals...")
    
    for i in range(total_test_len):
        # Raw index in the test set
        curr_raw_idx = start_test_idx + i
        
        active_directions = [] # Stores 1 (Up) or -1 (Down)
        
        # Check each model
        for model in models:
            c = model['config']
            seq_len = c['seq_len']
            buckets = model['buckets']
            
            a_seq = tuple(buckets[curr_raw_idx : curr_raw_idx + seq_len])
            
            # UPDATED: Derive d_seq from a_seq (Length = seq_len - 1)
            if seq_len > 1:
                d_seq = tuple(a_seq[k] - a_seq[k-1] for k in range(1, len(a_seq)))
            else:
                d_seq = ()
                
            last_val = a_seq[-1]
            actual_val = buckets[curr_raw_idx + seq_len]
            
            # Determine actual move for this specific bucket size
            diff = actual_val - last_val
            if diff > 0: model_actual_dir = 1
            elif diff < 0: model_actual_dir = -1
            else: model_actual_dir = 0
            
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
        if not active_directions:
            continue
            
        dirs = [x['dir'] for x in active_directions]
        
        # Conflict Check
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

    # --- UPDATED GRID SEARCH LOGIC ---
    # Testing buckets from 10 to 250 in steps of 10
    bucket_counts = list(range(10, 251, 10))
    seq_lengths = [3, 4, 5, 6, 8]
    
    results = []
    
    print(f"\n--- Starting Grid Search ({len(bucket_counts) * len(seq_lengths)} combinations) ---")
    print(f"{'Count':<8} | {'Size':<8} | {'SeqLen':<8} | {'Best Model':<12} | {'Dir Acc %':<10} | {'Trades':<8}")
    print("-" * 75)

    for b_count in bucket_counts:
        # Calculate size dynamically based on this count target
        b_size = calculate_bucket_size(prices, b_count)
        
        for s_len in seq_lengths:
            accuracy, model_name, trades = evaluate_parameters(prices, b_size, s_len)
            
            # Filter low trade count
            if trades > 20: 
                results.append({
                    "bucket_count": b_count, # Store count for ref
                    "bucket_size": b_size,   # Store size for usage
                    "seq_len": s_len,
                    "model": model_name,
                    "accuracy": accuracy,
                    "trades": trades
                })
                print(f"{b_count:<8} | {b_size:<8.4f} | {s_len:<8} | {model_name:<12} | {accuracy:<10.2f} | {trades:<8}")

    # Top 3
    results.sort(key=lambda x: x['accuracy'], reverse=True)
    top_3 = results[:3]
    
    print("\n" + "="*50)
    print(" TOP 3 CONFIGURATIONS ")
    print("="*50)
    for i, res in enumerate(top_3):
        print(f"#{i+1}: Count {res['bucket_count']} (Size {res['bucket_size']:.4f}), Len {res['seq_len']} ({res['model']}) -> {res['accuracy']:.2f}% ({res['trades']} trades)")

    # Combined Analysis
    if len(top_3) > 0:
        u_correct, u_total = run_portfolio_analysis(prices, top_3)
        print("\n" + "="*50)
        print(" FINAL PORTFOLIO RESULT (Union of Top 3) ")
        print("="*50)
        if u_total > 0:
            print(f"Combined Unique Trades: {u_total}")
            print(f"Combined Accuracy:      {(u_correct/u_total)*100:.2f}%")
        else:
            print("No unique trades found (Models might be conflicting or silence each other).")

if __name__ == "__main__":
    run_grid_search()