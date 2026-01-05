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

# Resampling Targets
TIMEFRAMES = {
    "15m": None,
    "30m": "30min",
    "60m": "1h",
    "240m": "4h",
    "1d": "1D"
}

# --- 1. DATA FETCHING ---

def get_binance_data(symbol, start_str=START_DATE):
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
                batch = [(int(c[6]), float(c[4])) for c in data]
                all_candles.extend(batch)
                current_start = data[-1][6] + 1
        except Exception as e:
            print(f"Error fetching: {e}")
            break
            
    print(f"[{symbol}] Downloaded {len(all_candles)} raw candles.")
    return all_candles

def resample_prices(raw_data, target_freq):
    if target_freq is None: return [x[1] for x in raw_data]
    df = pd.DataFrame(raw_data, columns=['timestamp', 'price'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    resampled = df['price'].resample(target_freq).last().dropna()
    return resampled.tolist()

# --- 2. CORE STRATEGY LOGIC (FIXED) ---

def get_bucket(price, bucket_size):
    if price >= 0: return (int(price) // bucket_size) + 1
    else: return (int(price + 1) // bucket_size) - 1

def train_models(train_buckets, seq_len):
    abs_map = defaultdict(Counter)
    der_map = defaultdict(Counter)
    
    for i in range(len(train_buckets) - seq_len):
        # 1. Absolute Sequence (Length = seq_len)
        a_seq = tuple(train_buckets[i : i + seq_len])
        a_succ = train_buckets[i + seq_len]
        abs_map[a_seq][a_succ] += 1
        
        # 2. Derivative Sequence (Length = seq_len - 1)
        # FIX: We start at i+1 to perform internal diffs only (buckets[i+1] - buckets[i])
        # This removes the dependency on "lookback" data outside the sequence window.
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
        der_cand = der_map.get(d_seq, Counter()) # d_seq is now len 4 for a seq of 5
        
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

def evaluate_config(prices, bucket_size, seq_len):
    buckets = [get_bucket(p, bucket_size) for p in prices]
    split_idx = int(len(buckets) * 0.7)
    train = buckets[:split_idx]
    test = buckets[split_idx:]
    
    if len(train) < seq_len + 10: return -1, "None", 0
    
    all_vals = list(set(train))
    all_changes = list(set(train[j] - train[j-1] for j in range(1, len(train))))
    abs_map, der_map = train_models(train, seq_len)
    
    stats = {"Absolute": [0,0], "Derivative": [0,0], "Combined": [0,0]}
    
    for i in range(len(test) - seq_len):
        curr = split_idx + i
        a_seq = tuple(buckets[curr : curr+seq_len])
        
        # FIX: Same logic here. Use range(curr + 1, ...) for internal diffs only
        d_seq = tuple(buckets[j] - buckets[j-1] for j in range(curr + 1, curr+seq_len))
        
        last, act = a_seq[-1], buckets[curr+seq_len]
        act_diff = act - last
        
        p_abs = get_prediction("Absolute", abs_map, der_map, a_seq, d_seq, last, all_vals, all_changes)
        p_der = get_prediction("Derivative", abs_map, der_map, a_seq, d_seq, last, all_vals, all_changes)
        p_comb = get_prediction("Combined", abs_map, der_map, a_seq, d_seq, last, all_vals, all_changes)
        
        for name, pred in [("Absolute", p_abs), ("Derivative", p_der), ("Combined", p_comb)]:
            p_diff = pred - last
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

def find_optimal_strategy(prices):
    avg_price = sum(prices) / len(prices)
    base_step = max(1, int(avg_price * 0.001)) 
    bucket_sizes = [base_step * i for i in range(1, 11)] 
    seq_lengths = [3, 4, 5, 6]
    
    results = []
    for b in bucket_sizes:
        for s in seq_lengths:
            acc, mod, tr = evaluate_config(prices, b, s)
            if tr > 20: 
                results.append({
                    "bucket_size": b,
                    "seq_len": s,
                    "model_type": mod,
                    "accuracy": acc,
                    "trades": tr
                })
    
    results.sort(key=lambda x: x['accuracy'], reverse=True)
    return results[:3]

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
        "message": f"Update model for {file_path}",
        "content": b64_content
    }
    
    check_resp = requests.get(url, headers=headers)
    if check_resp.status_code == 200:
        data["sha"] = check_resp.json()["sha"]
        print(f"Updating existing file: {file_path}")
    else:
        print(f"Creating new file: {file_path}")
        
    resp = requests.put(url, headers=headers, data=json.dumps(data))
    if resp.status_code in [200, 201]:
        print(f"Success: Uploaded {file_path}")
    else:
        print(f"Failed to upload {file_path}: {resp.text}")

def main():
    if not GITHUB_PAT:
        print("WARNING: 'PAT' not found in environment. GitHub upload will fail.")

    for asset in ASSETS:
        raw_data = get_binance_data(asset)
        if not raw_data: continue
        
        for tf_name, tf_pandas in TIMEFRAMES.items():
            print(f"Processing {asset} [{tf_name}]...")
            prices = resample_prices(raw_data, tf_pandas)
            if len(prices) < 200: continue
            
            top_configs = find_optimal_strategy(prices)
            if not top_configs: continue
                
            final_payload = {
                "asset": asset,
                "timeframe": tf_name,
                "timestamp": datetime.now().isoformat(),
                "data_points": len(prices),
                "strategy_union": top_configs 
            }
            
            filename = f"{asset}_{tf_name}.json"
            upload_to_github(filename, final_payload)
            
    print("\n--- OPTIMIZATION & UPLOAD COMPLETE ---")

if __name__ == "__main__":
    main()