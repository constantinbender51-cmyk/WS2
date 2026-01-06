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

def get_bucket(price, bucket_size):
    """Converts price to variable bucket size."""
    if price >= 0:
        return (int(price) // bucket_size) + 1
    else:
        return (int(price + 1) // bucket_size) - 1

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

def get_prediction(model_type, abs_map, der_map, a_seq, d_seq, last_val, all_vals, all_changes):
    """
    Helper to get a single prediction based on model type.
    Returns: (prediction_value, is_unseen_boolean)
    """
    is_unseen = False
    prediction = last_val

    if model_type == "Absolute":
        if a_seq in abs_map:
            prediction = abs_map[a_seq].most_common(1)[0][0]
        else:
            prediction = random.choice(all_vals)
            is_unseen = True
        
    elif model_type == "Derivative":
        if d_seq in der_map:
            pred_change = der_map[d_seq].most_common(1)[0][0]
            prediction = last_val + pred_change
        else:
            pred_change = random.choice(all_changes)
            prediction = last_val + pred_change
            is_unseen = True
        
    elif model_type == "Combined":
        abs_candidates = abs_map.get(a_seq, Counter())
        der_candidates = der_map.get(d_seq, Counter())
        
        possible_next_vals = set(abs_candidates.keys())
        for change in der_candidates.keys():
            possible_next_vals.add(last_val + change)
            
        if not possible_next_vals:
            # If neither model has a clue, it's truly unseen/random
            prediction = random.choice(all_vals)
            is_unseen = True
        else:
            best_val = None
            max_combined_score = -1
            for val in possible_next_vals:
                implied_change = val - last_val
                score = abs_candidates[val] + der_candidates[implied_change]
                if score > max_combined_score:
                    max_combined_score = score
                    best_val = val
            prediction = best_val
    
    return prediction, is_unseen

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

    # Stats: [correct_count, valid_total_count, unseen_count]
    stats = {"Absolute": [0, 0, 0], "Derivative": [0, 0, 0], "Combined": [0, 0, 0]}
    
    total_samples = len(test_buckets) - seq_len
    
    for i in range(total_samples):
        curr_idx = split_idx + i
        a_seq = tuple(buckets[curr_idx : curr_idx + seq_len])
        d_seq = tuple(buckets[j] - buckets[j-1] for j in range(curr_idx, curr_idx + seq_len))
        last_val = a_seq[-1]
        actual_val = buckets[curr_idx + seq_len]
        actual_diff = actual_val - last_val

        # Get predictions
        p_abs, u_abs = get_prediction("Absolute", abs_map, der_map, a_seq, d_seq, last_val, all_train_values, all_train_changes)
        p_der, u_der = get_prediction("Derivative", abs_map, der_map, a_seq, d_seq, last_val, all_train_values, all_train_changes)
        p_comb, u_comb = get_prediction("Combined", abs_map, der_map, a_seq, d_seq, last_val, all_train_values, all_train_changes)

        # Update stats
        # List of tuples: (ModelName, Prediction, IsUnseen)
        results = [
            ("Absolute", p_abs, u_abs),
            ("Derivative", p_der, u_der),
            ("Combined", p_comb, u_comb)
        ]

        for name, pred, unseen in results:
            if unseen:
                stats[name][2] += 1
            
            pred_diff = pred - last_val
            if pred_diff != 0 and actual_diff != 0:
                stats[name][1] += 1
                if (pred_diff > 0 and actual_diff > 0) or (pred_diff < 0 and actual_diff < 0):
                    stats[name][0] += 1

    # Return best
    best_acc = -1
    best_model = "None"
    best_trades = 0
    best_unseen_pct = 0.0

    for name, (correct, total, unseen_cnt) in stats.items():
        if total > 0:
            acc = (correct / total) * 100
            if acc > best_acc:
                best_acc = acc
                best_model = name
                best_trades = total
                best_unseen_pct = (unseen_cnt / total_samples) * 100
    
    return best_acc, best_model, best_trades, best_unseen_pct

def run_portfolio_analysis(prices, top_configs):
    """
    Runs the 'Union' strategy on the top 3 configurations.
    """
    print(f"\n--- Running Portfolio Analysis (Union of Top {len(top_configs)}) ---")
    
    # 1. Train all top models
    models = []
    for config in top_configs:
        b_size = config['bucket']
        s_len = config['seq_len']
        
        # Re-process buckets for this specific config
        buckets = [get_bucket(p, b_size) for p in prices]
        split_idx = int(len(buckets) * 0.7)
        train_buckets = buckets[:split_idx]
        
        # Train
        abs_map, der_map = train_models(train_buckets, s_len)
        
        # Store everything needed to predict
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
            
            a_seq = tuple(buckets[curr_raw_idx : curr_raw_idx + seq_len])
            d_seq = tuple(buckets[j] - buckets[j-1] for j in range(curr_raw_idx, curr_raw_idx + seq_len))
            last_val = a_seq[-1]
            actual_val = buckets[curr_raw_idx + seq_len]
            
            diff = actual_val - last_val
            model_actual_dir = 1 if diff > 0 else (-1 if diff < 0 else 0)
            
            # Use updated get_prediction (unpack tuple)
            pred_val, _ = get_prediction(c['model'], model['abs_map'], model['der_map'], 
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

    # Grid
    bucket_sizes = list(range(100, 9, -10))
    seq_lengths = [3, 4, 5, 6]
    
    results = []
    
    print(f"\n--- Starting Grid Search ({len(bucket_sizes) * len(seq_lengths)} combinations) ---")
    print(f"{'Bucket':<8} | {'SeqLen':<8} | {'Best Model':<12} | {'Dir Acc %':<10} | {'Unseen %':<10} | {'Trades':<8}")
    print("-" * 75)

    for b_size in bucket_sizes:
        for s_len in seq_lengths:
            accuracy, model_name, trades, unseen_pct = evaluate_parameters(prices, b_size, s_len)
            
            # Only consider if trades > 30 to avoid small sample size noise
            if trades > 30: 
                results.append({
                    "bucket": b_size,
                    "seq_len": s_len,
                    "model": model_name,
                    "accuracy": accuracy,
                    "trades": trades,
                    "unseen_pct": unseen_pct
                })
                print(f"{b_size:<8} | {s_len:<8} | {model_name:<12} | {accuracy:<10.2f} | {unseen_pct:<10.2f} | {trades:<8}")

    # Top 3
    results.sort(key=lambda x: x['accuracy'], reverse=True)
    top_3 = results[:3]
    
    print("\n" + "="*50)
    print(" TOP 3 CONFIGURATIONS ")
    print("="*50)
    for i, res in enumerate(top_3):
        print(f"#{i+1}: Bucket {res['bucket']}, Len {res['seq_len']} ({res['model']}) -> Acc: {res['accuracy']:.2f}% | Unseen: {res['unseen_pct']:.2f}% | Trades: {res['trades']}")

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
            print("No unique trades found.")

if __name__ == "__main__":
    run_grid_search()