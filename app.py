import random
import time
import sys
import json
import urllib.request
import os
import base64
import requests
from collections import Counter, defaultdict
from datetime import datetime

# --- CONFIGURATION ---
INTERVALS = ["1h", "30m", "15m"]
SYMBOL = "ETHUSDT"
# GitHub Config
GITHUB_REPO = "constantinbender51-cmyk/Models"
GITHUB_FILE_PATH = "signals.txt"
GITHUB_BRANCH = "main"

# Get PAT from Environment Variable
GITHUB_TOKEN = os.getenv("PAT")

if not GITHUB_TOKEN:
    print("WARNING: 'PAT' environment variable not found. GitHub upload will fail.")

# --- BINANCE DATA FETCHING ---

def get_binance_data(symbol, interval):
    """Fetches historical kline data from Binance public API."""
    # We fetch a shorter history for 15m/30m to keep processing fast in the loop
    limit_candles = 1500 
    
    print(f"\n--- Fetching {symbol} {interval} data ---")
    url = f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval={interval}&limit={limit_candles}"
    
    try:
        with urllib.request.urlopen(url) as response:
            data = json.loads(response.read().decode())
            if not data: return []
            # Close price is index 4
            prices = [float(candle[4]) for candle in data]
            return prices
    except Exception as e:
        print(f"Error fetching Binance data: {e}")
        return []

# --- GITHUB INTEGRATION ---

def push_to_github(new_content):
    """Appends content to a file in GitHub, creating it if it doesn't exist."""
    if not GITHUB_TOKEN:
        print("Skipping GitHub upload: No Token.")
        return

    url = f"https://api.github.com/repos/{GITHUB_REPO}/contents/{GITHUB_FILE_PATH}"
    headers = {
        "Authorization": f"token {GITHUB_TOKEN}",
        "Accept": "application/vnd.github.v3+json"
    }

    try:
        # 1. Try to get existing file
        response = requests.get(url, headers=headers)
        
        sha = None
        existing_content = ""

        if response.status_code == 200:
            file_data = response.json()
            sha = file_data['sha']
            existing_content = base64.b64decode(file_data['content']).decode('utf-8')
        elif response.status_code == 404:
            print("File not found on GitHub. Creating new one.")
        else:
            print(f"GitHub Error: {response.status_code} - {response.text}")
            return

        # 2. Append new content
        full_content = existing_content + new_content
        encoded_content = base64.b64encode(full_content.encode('utf-8')).decode('utf-8')

        # 3. specific payload
        data = {
            "message": f"Update signals {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            "content": encoded_content,
            "branch": GITHUB_BRANCH
        }
        if sha:
            data["sha"] = sha

        # 4. Push update
        put_response = requests.put(url, headers=headers, json=data)
        
        if put_response.status_code in [200, 201]:
            print("Successfully uploaded signals to GitHub.")
        else:
            print(f"Failed to upload: {put_response.status_code} - {put_response.text}")

    except Exception as e:
        print(f"GitHub Exception: {e}")

# --- MODEL LOGIC ---

def get_bucket(price, bucket_size):
    if bucket_size <= 0: bucket_size = 1e-9
    return int(price // bucket_size) if price >= 0 else int(price // bucket_size) - 1

def calculate_bucket_size(prices, bucket_count):
    if not prices: return 1.0
    price_range = max(prices) - min(prices)
    if bucket_count <= 0: return 1.0
    size = price_range / bucket_count
    return size if size > 0 else 0.01

def train_models(train_buckets, seq_len):
    abs_map = defaultdict(Counter)
    der_map = defaultdict(Counter)
    
    for i in range(len(train_buckets) - seq_len):
        a_seq = tuple(train_buckets[i : i + seq_len])
        a_succ = train_buckets[i + seq_len]
        abs_map[a_seq][a_succ] += 1
        
        if seq_len > 1:
            d_seq = tuple(a_seq[k] - a_seq[k-1] for k in range(1, len(a_seq)))
            d_succ = train_buckets[i + seq_len] - train_buckets[i + seq_len - 1]
            der_map[d_seq][d_succ] += 1
            
    return abs_map, der_map

def get_prediction(model_type, abs_map, der_map, a_seq, d_seq, last_val, all_vals, all_changes):
    if model_type == "Absolute":
        if a_seq in abs_map: return abs_map[a_seq].most_common(1)[0][0]
        return random.choice(all_vals)
    elif model_type == "Derivative":
        if d_seq in der_map:
            change = der_map[d_seq].most_common(1)[0][0]
            return last_val + change
        return last_val + random.choice(all_changes)
    elif model_type == "Combined":
        abs_c = abs_map.get(a_seq, Counter())
        der_c = der_map.get(d_seq, Counter())
        possible = set(abs_c.keys())
        for ch in der_c.keys(): possible.add(last_val + ch)
        
        if not possible: return random.choice(all_vals)
        
        best, max_s = None, -1
        for val in possible:
            s = abs_c[val] + der_c[val - last_val]
            if s > max_s: max_s, best = s, val
        return best
    return last_val

def evaluate_parameters(prices, bucket_size, seq_len):
    buckets = [get_bucket(p, bucket_size) for p in prices]
    # Use last 30% for validation
    split_idx = int(len(buckets) * 0.7)
    train_buckets = buckets[:split_idx]
    test_buckets = buckets[split_idx:]
    
    abs_map, der_map = train_models(train_buckets, seq_len)
    all_vals = list(set(train_buckets))
    all_chg = list(set(train_buckets[j] - train_buckets[j-1] for j in range(1, len(train_buckets))))

    stats = {"Absolute": [0, 0], "Derivative": [0, 0], "Combined": [0, 0]}
    
    for i in range(len(test_buckets) - seq_len):
        curr_idx = split_idx + i
        a_seq = tuple(buckets[curr_idx : curr_idx + seq_len])
        d_seq = tuple(a_seq[k] - a_seq[k-1] for k in range(1, len(a_seq))) if seq_len > 1 else ()
        
        last_val = a_seq[-1]
        actual_val = buckets[curr_idx + seq_len]
        actual_diff = actual_val - last_val

        for m_type in stats:
            pred = get_prediction(m_type, abs_map, der_map, a_seq, d_seq, last_val, all_vals, all_chg)
            pred_diff = pred - last_val
            
            if pred_diff != 0 and actual_diff != 0:
                stats[m_type][1] += 1
                if (pred_diff > 0 and actual_diff > 0) or (pred_diff < 0 and actual_diff < 0):
                    stats[m_type][0] += 1
                    
    best_acc, best_model, best_trades = -1, "None", 0
    for m, (cor, tot) in stats.items():
        if tot > 0:
            acc = (cor/tot)*100
            if acc > best_acc: best_acc, best_model, best_trades = acc, m, tot
            
    return best_acc, best_model, best_trades

# --- LIVE PREDICTION LOGIC ---

def generate_live_signal(prices, top_configs):
    """
    Takes top configs, retrains on FULL data, and votes on the NEXT move.
    """
    votes = []
    
    for config in top_configs:
        b_size = config['bucket_size']
        s_len = config['seq_len']
        m_type = config['model']
        
        # 1. Retrain on ALL data
        buckets = [get_bucket(p, b_size) for p in prices]
        abs_map, der_map = train_models(buckets, s_len) # Note: Trained on everything
        
        all_vals = list(set(buckets))
        all_chg = list(set(buckets[j] - buckets[j-1] for j in range(1, len(buckets))))
        
        # 2. Get the very last sequence
        if len(buckets) < s_len: continue
        
        last_seq = tuple(buckets[-s_len:])
        last_d_seq = tuple(last_seq[k] - last_seq[k-1] for k in range(1, len(last_seq))) if s_len > 1 else ()
        last_val = last_seq[-1]
        
        # 3. Predict NEXT bucket
        pred_val = get_prediction(m_type, abs_map, der_map, last_seq, last_d_seq, last_val, all_vals, all_chg)
        
        diff = pred_val - last_val
        if diff > 0: votes.append("BUY")
        elif diff < 0: votes.append("SELL")
        else: votes.append("NEUTRAL")
        
    # Majority Vote
    if not votes: return "NEUTRAL", 0.0
    
    counts = Counter(votes)
    winner, count = counts.most_common(1)[0]
    
    # Calculate confidence (percentage of top models agreeing)
    confidence = (count / len(votes)) * 100
    
    return winner, confidence

def analyze_interval(interval):
    prices = get_binance_data(SYMBOL, interval)
    if len(prices) < 200:
        return f"[{interval}] Not enough data."

    # 1. Grid Search
    bucket_counts = list(range(20, 201, 20)) # Reduced range for speed
    seq_lengths = [3, 4, 5, 6, 8]
    results = []

    for b_count in bucket_counts:
        b_size = calculate_bucket_size(prices, b_count)
        for s_len in seq_lengths:
            acc, model, trades = evaluate_parameters(prices, b_size, s_len)
            if trades > 15: # Min trade filter
                results.append({
                    "bucket_count": b_count,
                    "bucket_size": b_size,
                    "seq_len": s_len,
                    "model": model,
                    "accuracy": acc,
                    "trades": trades
                })
    
    results.sort(key=lambda x: x['accuracy'], reverse=True)
    top_5 = results[:5]
    
    if not top_5:
        return f"[{interval}] No valid models found."
        
    # 2. Generate Live Signal
    signal, conf = generate_live_signal(prices, top_5)
    current_price = prices[-1]
    
    log_entry = (
        f"TIME: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC | "
        f"PAIR: {SYMBOL} | INT: {interval} | "
        f"PRICE: {current_price} | "
        f"SIGNAL: {signal} (Conf: {conf:.1f}%) | "
        f"Top Model Acc: {top_5[0]['accuracy']:.2f}%"
    )
    print(log_entry)
    return log_entry

# --- MAIN LOOP ---

def main():
    print(f"Starting Continuous AI Trader for {SYMBOL}...")
    print(f"Target Repo: {GITHUB_REPO}/{GITHUB_FILE_PATH}")
    
    while True:
        cycle_start = time.time()
        batch_log = "\n" + "-"*60 + "\n"
        
        # Analyze all intervals
        for interval in INTERVALS:
            try:
                result = analyze_interval(interval)
                batch_log += result + "\n"
            except Exception as e:
                err = f"[{interval}] Error: {e}"
                print(err)
                batch_log += err + "\n"
                
        # Push to GitHub
        print("Pushing batch results to GitHub...")
        push_to_github(batch_log)
        
        # Sleep logic (Wait 15 mins)
        # We assume 15m is the fastest cycle. 
        # Ideally, you'd align this with the clock (e.g., :00, :15, :30, :45)
        # But a simple sleep works for this scope.
        sleep_sec = 900 
        print(f"Sleeping for {sleep_sec/60} minutes...")
        time.sleep(sleep_sec)

if __name__ == "__main__":
    main()
