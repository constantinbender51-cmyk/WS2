import random
import time
import sys
import json
import urllib.request
from collections import Counter, defaultdict
from datetime import datetime

# =========================================
# 1. CONFIGURATION
# =========================================

# --- Exchange Settings ---
SYMBOL = "ETHUSDT"
INTERVAL = "1h"
START_DATE = "2020-01-01" 

# --- Validation Settings ---
VAL_MONTHS = 1           # Number of months to hold back for validation
HOURS_PER_MONTH = 720    # Approx candles per month

# --- Grid Search Ranges ---
BUCKET_COUNTS = range(10, 251, 10)  # 10 to 250
SEQ_LENGTHS = [3, 4, 5, 6, 8]
MIN_TRADES = 20          # Min trades to consider a strategy valid during training

# =========================================
# 2. DATA UTILITIES
# =========================================

def get_binance_data(symbol=SYMBOL, interval=INTERVAL, start_str=START_DATE):
    """Fetches historical kline data from Binance public API."""
    print(f"\n--- Fetching {symbol} {interval} data from Binance ---")
    
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

# =========================================
# 3. STRATEGY LOGIC
# =========================================

def get_bucket(price, bucket_size):
    if bucket_size <= 0: bucket_size = 1e-9
    return int(price // bucket_size)

def calculate_bucket_size(prices, bucket_count):
    if not prices: return 1.0
    min_p, max_p = min(prices), max(prices)
    price_range = max_p - min_p
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

def get_single_prediction(mode, abs_map, der_map, a_seq, d_seq, last_val):
    """Helper to get raw predictions for individual components."""
    if mode == "Absolute":
        if a_seq in abs_map:
            return abs_map[a_seq].most_common(1)[0][0]
    elif mode == "Derivative":
        if d_seq in der_map:
            pred_change = der_map[d_seq].most_common(1)[0][0]
            return last_val + pred_change
    return None

def get_prediction(model_type, abs_map, der_map, a_seq, d_seq, last_val):
    """
    Returns the predicted next VALUE.
    Returns None if pattern is unknown or (in Combined mode) if signals conflict.
    """
    
    # 1. Handle Single Models
    if model_type == "Absolute":
        return get_single_prediction("Absolute", abs_map, der_map, a_seq, d_seq, last_val)
        
    elif model_type == "Derivative":
        return get_single_prediction("Derivative", abs_map, der_map, a_seq, d_seq, last_val)
        
    # 2. Handle Combined (Voting Logic)
    elif model_type == "Combined":
        # Get individual predictions
        pred_abs = get_single_prediction("Absolute", abs_map, der_map, a_seq, d_seq, last_val)
        pred_der = get_single_prediction("Derivative", abs_map, der_map, a_seq, d_seq, last_val)
        
        # Determine Directions
        dir_abs = 0
        if pred_abs is not None:
            if pred_abs > last_val: dir_abs = 1
            elif pred_abs < last_val: dir_abs = -1
            
        dir_der = 0
        if pred_der is not None:
            if pred_der > last_val: dir_der = 1
            elif pred_der < last_val: dir_der = -1
            
        # VOTE LOGIC
        
        # Case A: Both Abstain
        if dir_abs == 0 and dir_der == 0:
            return None
            
        # Case B: Conflict (One UP, One DOWN) -> ABSTAIN
        if dir_abs != 0 and dir_der != 0 and dir_abs != dir_der:
            return None
            
        # Case C: Consensus or Solo Signal
        # If both trigger (and agree), we prefer Absolute target as requested.
        # If only Abs triggers, use Abs.
        if dir_abs != 0:
            return pred_abs
            
        # If only Der triggers, use Der.
        if dir_der != 0:
            return pred_der
            
    return None

def test_strategy(train_prices, test_prices, bucket_count, seq_len, model_type):
    # 1. Setup Buckets (Size derived from TRAIN to keep consistency)
    bucket_size = calculate_bucket_size(train_prices, bucket_count)
    
    train_buckets = [get_bucket(p, bucket_size) for p in train_prices]
    test_buckets = [get_bucket(p, bucket_size) for p in test_prices]
    
    # 2. Train
    abs_map, der_map = train_models(train_buckets, seq_len)
    
    # 3. Test
    correct = 0
    total_trades = 0
    
    loop_range = len(test_buckets) - seq_len
    
    for i in range(loop_range):
        curr_slice = test_buckets[i : i + seq_len + 1]
        a_seq = tuple(curr_slice[:-1])
        actual_val = curr_slice[-1]
        last_val = a_seq[-1]
        
        if seq_len > 1:
            d_seq = tuple(a_seq[k] - a_seq[k-1] for k in range(1, len(a_seq)))
        else:
            d_seq = ()
            
        # Predict
        pred_val = get_prediction(model_type, abs_map, der_map, a_seq, d_seq, last_val)
        
        if pred_val is None:
            continue # ABSTAIN
            
        # Check Direction
        pred_diff = pred_val - last_val
        actual_diff = actual_val - last_val
        
        if pred_diff != 0 and actual_diff != 0:
            total_trades += 1
            if (pred_diff > 0 and actual_diff > 0) or (pred_diff < 0 and actual_diff < 0):
                correct += 1
                
    accuracy = (correct / total_trades * 100) if total_trades > 0 else 0.0
    return accuracy, total_trades

# =========================================
# 4. MAIN EXECUTION
# =========================================

def run_analysis():
    # 1. Load Data
    prices = get_binance_data()
    if len(prices) < 1000:
        print("Not enough data.")
        return

    # 2. Split Data (Training vs Validation)
    val_len = VAL_MONTHS * HOURS_PER_MONTH
    split_idx = len(prices) - val_len
    
    train_prices = prices[:split_idx]
    val_prices = prices[split_idx:]
    
    print(f"\n=== DATA SPLIT CONFIGURATION ===")
    print(f"Total Candles : {len(prices)}")
    print(f"Training Set  : {len(train_prices)} candles")
    print(f"Validation Set: {len(val_prices)} candles (Last {VAL_MONTHS} Month(s))")
    print("--------------------------------")

    # 3. Grid Search on TRAINING Data
    print(f"\n--- Running Grid Search on TRAINING Set ---")
    results = []
    
    combinations = len(list(BUCKET_COUNTS)) * len(SEQ_LENGTHS) * 3
    counter = 0
    
    for b_count in BUCKET_COUNTS:
        for s_len in SEQ_LENGTHS:
            for m_type in ["Absolute", "Derivative", "Combined"]:
                counter += 1
                if counter % 50 == 0:
                    sys.stdout.write(f"\rScanning config {counter}/{combinations}...")
                    sys.stdout.flush()
                
                # In-Sample Test
                acc, trades = test_strategy(train_prices, train_prices, b_count, s_len, m_type)
                
                if trades >= MIN_TRADES:
                    # NEW METRIC: Accuracy * Trades
                    score = (acc / 100) * trades 
                    results.append({
                        "b_count": b_count,
                        "s_len": s_len,
                        "model": m_type,
                        "train_acc": acc,
                        "train_trades": trades,
                        "score": score
                    })

    print(f"\nGrid Search Complete. Found {len(results)} valid configurations.")

    # 4. Select Top Strategies by SCORE (Acc * Trades)
    results.sort(key=lambda x: x['score'], reverse=True)
    top_5 = results[:5]
    
    # 5. Validate Top Strategies
    print(f"\n=== TOP 5 STRATEGIES (Sorted by Score = Acc * Trades) ===")
    print(f"{'#':<3} | {'Config':<25} | {'Score':<8} | {'Train Acc':<10} | {'Train Trd':<10} | {'VAL ACC':<10} | {'VAL TRD':<8}")
    print("-" * 100)
    
    for i, res in enumerate(top_5):
        # Run Validation Test (Train on Train, Test on Val)
        val_acc, val_trades = test_strategy(train_prices, val_prices, 
                                            res['b_count'], res['s_len'], res['model'])
        
        config_str = f"B={res['b_count']} L={res['s_len']} ({res['model'][0:3]})"
        score_val = res['score']
        
        print(f"{i+1:<3} | {config_str:<25} | {score_val:<8.1f} | {res['train_acc']:.1f}%     | {res['train_trades']:<10} | {val_acc:.1f}%     | {val_trades:<8}")

    print("-" * 100)

if __name__ == "__main__":
    run_analysis()
