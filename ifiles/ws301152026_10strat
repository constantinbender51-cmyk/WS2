import os
import sys
import json
import time
import base64
import requests
import random
import urllib.request
from datetime import datetime
from collections import Counter, defaultdict
import pandas as pd

# --- CONFIGURATION ---
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

GITHUB_PAT = os.getenv("PAT")
REPO_OWNER = "constantinbender51-cmyk"
REPO_NAME = "Models"
GITHUB_API_URL = f"https://api.github.com/repos/{REPO_OWNER}/{REPO_NAME}/contents/"

ASSETS = [
    "BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT", 
    "ADAUSDT", "DOGEUSDT", "AVAXUSDT", "DOTUSDT", "LINKUSDT"
]

# Base timeframe download
BASE_INTERVAL = "15m" 
START_DATE = "2020-01-01"
END_DATE = "2026-01-01"

DATA_DIR = "data"  # Directory to store cached OHLC data

# Resampling Targets
TIMEFRAMES = {
    "15m": None, # Base
    "30m": "30min",
    "60m": "1h",
    "240m": "4h",
    "1d": "1D"
}

# --- 1. DATA FETCHING & CACHING ---

def get_binance_data(symbol, start_str=START_DATE, end_str=END_DATE):
    """
    Fetches 15m historical kline data from Binance.
    Checks local 'data/' folder first. If missing, downloads and saves.
    """
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)

    filename = f"{DATA_DIR}/{symbol}_{BASE_INTERVAL}_{start_str}_{end_str}.json"

    # 1. Try Load from Cache
    if os.path.exists(filename):
        print(f"\n[{symbol}] Found cached data at {filename}. Loading...")
        try:
            with open(filename, 'r') as f:
                all_candles = json.load(f)
            print(f"[{symbol}] Loaded {len(all_candles)} candles from cache.")
            return all_candles
        except Exception as e:
            print(f"Error loading cache: {e}. Redownloading...")

    # 2. Download from API if no cache
    print(f"\n[{symbol}] Fetching raw {BASE_INTERVAL} data from {start_str} to {end_str}...")
    
    start_ts = int(datetime.strptime(start_str, "%Y-%m-%d").timestamp() * 1000)
    end_ts = int(datetime.strptime(end_str, "%Y-%m-%d").timestamp() * 1000)
    
    all_candles = []
    current_start = start_ts
    
    while current_start < end_ts:
        url = f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval={BASE_INTERVAL}&startTime={current_start}&endTime={end_ts}&limit=1000"
        try:
            with urllib.request.urlopen(url) as response:
                data = json.loads(response.read().decode())
                if not data:
                    break
                
                # Store (Close Time, Close Price) - minimize size
                batch = [(int(c[6]), float(c[4])) for c in data]
                all_candles.extend(batch)
                
                last_time = data[-1][6]
                if last_time >= end_ts - 1:
                    break
                current_start = last_time + 1
                
        except Exception as e:
            print(f"Error fetching: {e}")
            break
            
    print(f"[{symbol}] Downloaded {len(all_candles)} raw candles.")

    # 3. Save to Cache
    if all_candles:
        try:
            with open(filename, 'w') as f:
                json.dump(all_candles, f)
            print(f"[{symbol}] Saved data to {filename}")
        except Exception as e:
            print(f"Error saving cache: {e}")

    return all_candles

def resample_prices(raw_data, target_freq):
    """Resamples (Close_Time, Price) tuples to target frequency."""
    if target_freq is None:
        return [x[1] for x in raw_data]
    
    df = pd.DataFrame(raw_data, columns=['timestamp', 'price'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    
    resampled = df['price'].resample(target_freq).last().dropna()
    return resampled.tolist()

# --- 2. CORE STRATEGY & SERIALIZATION ---

def get_bucket(price, bucket_size):
    if bucket_size <= 0: bucket_size = 1e-9
    return int(price // bucket_size)

def calculate_bucket_size(prices, bucket_count):
    min_p, max_p = min(prices), max(prices)
    price_range = max_p - min_p
    if bucket_count <= 0: return 1.0
    size = price_range / bucket_count
    return size if size > 0 else 0.01

def train_models(train_buckets, seq_len):
    """
    Trains the Probabilistic Models (Markov Chains) on the provided sequence.
    """
    abs_map = defaultdict(Counter)
    der_map = defaultdict(Counter)
    
    # We iterate up to len - 1 because we need a 'next' value for the last sequence
    # However, standard practice is range(len - seq_len).
    # train_buckets[i+seq_len] is the target.
    
    for i in range(len(train_buckets) - seq_len):
        a_seq = tuple(train_buckets[i : i + seq_len])
        a_succ = train_buckets[i + seq_len]
        abs_map[a_seq][a_succ] += 1
        
        if seq_len > 1:
            d_seq = tuple(a_seq[k] - a_seq[k-1] for k in range(1, len(a_seq)))
            d_succ = train_buckets[i + seq_len] - train_buckets[i + seq_len - 1]
            der_map[d_seq][d_succ] += 1
            
    return abs_map, der_map

def serialize_map(model_map):
    """
    Converts tuple keys (1, 2, 3) to string keys "1|2|3" for JSON storage.
    """
    serialized = {}
    for seq_tuple, counter_obj in model_map.items():
        key_str = "|".join(map(str, seq_tuple))
        serialized[key_str] = dict(counter_obj)
    return serialized

def prepare_deployment_models(prices, top_configs):
    """
    Prepares the final JSON payload. 
    Since we are now training on the full dataset during optimization, 
    this just replicates that training for serialization.
    """
    serialized_models = []
    
    for config in top_configs:
        b_count = config['bucket_count']
        b_size = calculate_bucket_size(prices, b_count)
        s_len = config['seq_len']
        
        # 1. Bucketize full history
        buckets = [get_bucket(p, b_size) for p in prices]
        
        # 2. Train on full history
        abs_map, der_map = train_models(buckets, s_len)
        
        # 3. Serialize
        model_entry = {
            "config": config,
            "trained_parameters": {
                "bucket_size": b_size, 
                "seq_len": s_len,
                "abs_map": serialize_map(abs_map),
                "der_map": serialize_map(der_map),
                "all_vals": list(set(buckets)),
                "all_changes": list(set(buckets[j] - buckets[j-1] for j in range(1, len(buckets))))
            }
        }
        serialized_models.append(model_entry)
        
    return serialized_models

# --- PREDICTION LOGIC ---

def get_prediction(model_type, abs_map, der_map, a_seq, d_seq, last_val, all_vals, all_changes):
    """
    Determines the predicted value based on the trained maps.
    """
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
            # Score is sum of frequencies from both models
            s = abs_cand[v] + der_cand[v - last_val]
            if s > max_s: max_s, best = s, v
        return best
    return last_val

def evaluate_config(prices, bucket_size, seq_len):
    """
    Optimized for FULL DATASET training and testing.
    No train/test split. Checks in-sample accuracy.
    """
    buckets = [get_bucket(p, bucket_size) for p in prices]
    
    if len(buckets) < seq_len + 10: return -1, "None", 0
    
    # 1. Train on the ENTIRE dataset
    abs_map, der_map = train_models(buckets, seq_len)
    
    # Pre-calculate domain lists
    all_vals = list(set(buckets))
    if not all_vals: all_vals = [0]
    all_changes = list(set(buckets[j] - buckets[j-1] for j in range(1, len(buckets))))
    if not all_changes: all_changes = [0]
    
    stats = {"Absolute": [0,0], "Derivative": [0,0], "Combined": [0,0]}
    
    # 2. Evaluate on the ENTIRE dataset (self-consistency check)
    # We start iterating from `seq_len` to predict the next value.
    total_samples = len(buckets) - seq_len
    
    for i in range(total_samples):
        # Current index for the start of the sequence
        curr = i 
        a_seq = tuple(buckets[curr : curr+seq_len])
        
        if seq_len > 1:
            d_seq = tuple(a_seq[k] - a_seq[k-1] for k in range(1, len(a_seq)))
        else:
            d_seq = ()
            
        last = a_seq[-1]
        act = buckets[curr+seq_len] # The actual value that happened
        act_diff = act - last
        
        p_abs = get_prediction("Absolute", abs_map, der_map, a_seq, d_seq, last, all_vals, all_changes)
        p_der = get_prediction("Derivative", abs_map, der_map, a_seq, d_seq, last, all_vals, all_changes)
        p_comb = get_prediction("Combined", abs_map, der_map, a_seq, d_seq, last, all_vals, all_changes)
        
        for name, pred in [("Absolute", p_abs), ("Derivative", p_der), ("Combined", p_comb)]:
            p_diff = pred - last
            # Only count trade if model predicts a move AND price actually moved
            if p_diff != 0 and act_diff != 0:
                stats[name][1] += 1 # Total trades
                # Check direction (Sign matching)
                if (p_diff > 0 and act_diff > 0) or (p_diff < 0 and act_diff < 0):
                    stats[name][0] += 1 # Correct trades
                    
    best_acc, best_name, best_trades = -1, "None", 0
    for name, (corr, tot) in stats.items():
        if tot > 0:
            acc = (corr/tot)*100
            if acc > best_acc: best_acc, best_name, best_trades = acc, name, tot
            
    return best_acc, best_name, best_trades

def run_portfolio_analysis(prices, top_configs):
    """
    Runs the 'Committee' (Portfolio) analysis on the FULL dataset.
    """
    models = []
    
    # 1. Train all selected top configs on full data
    for config in top_configs:
        b_count = config['bucket_count']
        b_size = calculate_bucket_size(prices, b_count)
        s_len = config['seq_len']
        
        buckets = [get_bucket(p, b_size) for p in prices]
        
        abs_map, der_map = train_models(buckets, s_len)
        
        t_vals = list(set(buckets))
        t_changes = list(set(buckets[j] - buckets[j-1] for j in range(1, len(buckets))))
        
        models.append({
            "config": config,
            "buckets": buckets,
            "seq_len": s_len,
            "abs_map": abs_map,
            "der_map": der_map,
            "all_vals": t_vals if t_vals else [0],
            "all_changes": t_changes if t_changes else [0]
        })

    if not models: return 0, 0
    
    max_seq_len = max(m['seq_len'] for m in models)
    # Testing range: From max_seq_len to end
    total_test_len = len(models[0]['buckets']) - max_seq_len
    
    unique_correct = 0
    unique_total = 0
    
    for i in range(total_test_len):
        curr_raw_idx = i 
        active_directions = [] 
        
        for model in models:
            c = model['config']
            seq_len = model['seq_len']
            buckets = model['buckets']
            
            # Adjust index for this model's specific sequence length if needed
            # (Note: we use curr_raw_idx relative to the start of valid data)
            # Actually, simplify: we are iterating through time.
            # Time T is the target.
            # We need inputs [T-seq_len : T]
            
            target_idx = curr_raw_idx + max_seq_len
            start_seq_idx = target_idx - seq_len
            
            a_seq = tuple(buckets[start_seq_idx : target_idx])
            
            if seq_len > 1:
                d_seq = tuple(a_seq[k] - a_seq[k-1] for k in range(1, len(a_seq)))
            else:
                d_seq = ()
                
            last_val = a_seq[-1]
            actual_val = buckets[target_idx]
            
            diff = actual_val - last_val
            model_actual_dir = 1 if diff > 0 else (-1 if diff < 0 else 0)
            
            pred_val = get_prediction(c['model_type'], model['abs_map'], model['der_map'], 
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

        if not active_directions: continue
            
        dirs = [x['dir'] for x in active_directions]
        up_votes = dirs.count(1)
        down_votes = dirs.count(-1)
        
        final_dir = 0
        if up_votes > down_votes: final_dir = 1
        elif down_votes > up_votes: final_dir = -1
        else: continue
        
        winning_voters = [x for x in active_directions if x['dir'] == final_dir]
        if all(x['is_flat'] for x in winning_voters): continue 
            
        unique_total += 1
        # If the consensus direction was correct (based on ANY of the winning voters being correct)
        # Simplified: If final_dir matches actual direction (calculated from price change, not bucket change ideally, but bucket change is proxy)
        # Here we check if the voters were marked correct.
        if any(x['is_correct'] for x in winning_voters):
            unique_correct += 1

    return unique_correct, unique_total

def find_optimal_strategy(prices):
    """
    Sweeps parameters to find the best configuration on the FULL dataset.
    """
    bucket_counts = list(range(10, 251, 10))
    seq_lengths = [3, 4, 5, 6, 8]
    
    results = []
    
    for b_count in bucket_counts:
        b_size = calculate_bucket_size(prices, b_count)
        for s in seq_lengths:
            acc, mod, tr = evaluate_config(prices, b_size, s)
            # Threshold: Must have at least 20 trades to be statistically interesting
            if tr > 20: 
                results.append({
                    "bucket_count": b_count, 
                    "bucket_size": b_size,   
                    "seq_len": s,
                    "model_type": mod,
                    "accuracy": acc,
                    "trades": tr
                })
    
    # Sort by accuracy descending
    results.sort(key=lambda x: x['accuracy'], reverse=True)
    return results[:5]

# --- 3. GITHUB UPLOAD ---

def upload_to_github(file_path, content):
    if not GITHUB_PAT:
        print("Error: No PAT found in .env")
        return

    url = GITHUB_API_URL + file_path
    headers = {
        "Authorization": f"Bearer {GITHUB_PAT}",
        "Accept": "application/vnd.github.v3+json"
    }
    
    json_content = json.dumps(content, indent=2)
    b64_content = base64.b64encode(json_content.encode("utf-8")).decode("utf-8")
    
    data = {
        "message": f"Update optimized model for {file_path}",
        "content": b64_content
    }
    
    check_resp = requests.get(url, headers=headers)
    if check_resp.status_code == 200:
        data["sha"] = check_resp.json()["sha"]
        
    resp = requests.put(url, headers=headers, data=json.dumps(data))
    if resp.status_code in [200, 201]:
        print(f"Success: Uploaded {file_path}")
    else:
        print(f"Failed to upload {file_path}: {resp.text}")

# --- MAIN LOOP ---

def main():
    if not GITHUB_PAT:
        print("WARNING: 'PAT' not found. GitHub upload will fail.")

    for asset in ASSETS:
        # 1. Get Base Data (Cached)
        raw_data = get_binance_data(asset)
        if not raw_data: continue
        
        for tf_name, tf_pandas in TIMEFRAMES.items():
            print(f"Processing {asset} [{tf_name}]...")
            
            # 2. Resample
            prices = resample_prices(raw_data, tf_pandas)
            if len(prices) < 200:
                print(f"Not enough data for {tf_name}. Skipping.")
                continue
                
            # 3. Optimize (FULL DATASET)
            # Finds the configuration that best fits the entire history
            top_configs = find_optimal_strategy(prices)
            
            if not top_configs:
                print(f"No valid strategy found for {asset} {tf_name}")
                continue
            
            # 4. Verify Combined Accuracy (FULL DATASET)
            u_correct, u_total = run_portfolio_analysis(prices, top_configs)
            u_acc = (u_correct / u_total * 100) if u_total > 0 else 0
            
            print(f"--> {asset} {tf_name} In-Sample Accuracy: {u_acc:.2f}% ({u_total} trades)")

            # 5. Serialize Models
            serialized_strategies = prepare_deployment_models(prices, top_configs)

            # 6. Prepare Payload
            final_payload = {
                "asset": asset,
                "timeframe": tf_name,
                "timestamp": datetime.now().isoformat(),
                "data_points": len(prices),
                "optimization_type": "full_dataset_fit",
                "combined_accuracy": u_acc, 
                "combined_trades": u_total,
                "strategy_union": serialized_strategies 
            }
            
            # 7. Upload
            filename = f"{asset}_{tf_name}.json"
            upload_to_github(filename, final_payload)
            
    print("\n--- FULL DATASET OPTIMIZATION & UPLOAD COMPLETE ---")

if __name__ == "__main__":
    main()
