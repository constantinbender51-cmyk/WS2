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
VAL_MONTHS = 3          # Number of months to hold back for validation
HOURS_PER_MONTH = 720    # Approx candles per month

# --- Grid Search Ranges ---
BUCKET_COUNTS = range(10, 111, 10)  # 10 to 250
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
    if mode == "Absolute":
        if a_seq in abs_map:
            return abs_map[a_seq].most_common(1)[0][0]
    elif mode == "Derivative":
        if d_seq in der_map:
            pred_change = der_map[d_seq].most_common(1)[0][0]
            return last_val + pred_change
    return None

def get_prediction(model_type, abs_map, der_map, a_seq, d_seq, last_val):
    """Returns the predicted next VALUE or None (Abstain)."""
    
    if model_type == "Absolute":
        return get_single_prediction("Absolute", abs_map, der_map, a_seq, d_seq, last_val)
        
    elif model_type == "Derivative":
        return get_single_prediction("Derivative", abs_map, der_map, a_seq, d_seq, last_val)
        
    elif model_type == "Combined":
        # Vote Logic for internal strategy
        pred_abs = get_single_prediction("Absolute", abs_map, der_map, a_seq, d_seq, last_val)
        pred_der = get_single_prediction("Derivative", abs_map, der_map, a_seq, d_seq, last_val)
        
        dir_abs = 0
        if pred_abs is not None:
            dir_abs = 1 if pred_abs > last_val else -1 if pred_abs < last_val else 0
            
        dir_der = 0
        if pred_der is not None:
            dir_der = 1 if pred_der > last_val else -1 if pred_der < last_val else 0
            
        if dir_abs == 0 and dir_der == 0: return None
        if dir_abs != 0 and dir_der != 0 and dir_abs != dir_der: return None # Conflict
        
        if dir_abs != 0: return pred_abs
        if dir_der != 0: return pred_der
            
    return None

def test_strategy(train_prices, test_prices, bucket_count, seq_len, model_type):
    # Size strictly from TRAIN
    bucket_size = calculate_bucket_size(train_prices, bucket_count)
    
    train_buckets = [get_bucket(p, bucket_size) for p in train_prices]
    test_buckets = [get_bucket(p, bucket_size) for p in test_prices]
    
    abs_map, der_map = train_models(train_buckets, seq_len)
    
    correct = 0
    total_trades = 0
    
    loop_range = len(test_buckets) - seq_len
    
    for i in range(loop_range):
        curr_slice = test_buckets[i : i + seq_len + 1]
        a_seq = tuple(curr_slice[:-1])
        actual_val = curr_slice[-1]
        last_val = a_seq[-1]
        
        d_seq = tuple(a_seq[k] - a_seq[k-1] for k in range(1, len(a_seq))) if seq_len > 1 else ()
            
        pred_val = get_prediction(model_type, abs_map, der_map, a_seq, d_seq, last_val)
        
        if pred_val is None: continue 
            
        pred_diff = pred_val - last_val
        actual_diff = actual_val - last_val
        
        if pred_diff != 0 and actual_diff != 0:
            total_trades += 1
            if (pred_diff > 0 and actual_diff > 0) or (pred_diff < 0 and actual_diff < 0):
                correct += 1
                
    accuracy = (correct / total_trades * 100) if total_trades > 0 else 0.0
    return accuracy, total_trades

# =========================================
# 4. FINAL ENSEMBLE LOGIC
# =========================================

def run_final_ensemble(train_prices, val_prices, top_configs):
    print(f"\n=== FINAL COMBINED MODEL (Union of Top {len(top_configs)}) ===")
    
    # 1. Train all top models
    models = []
    for cfg in top_configs:
        b_size = calculate_bucket_size(train_prices, cfg['b_count'])
        t_buckets = [get_bucket(p, b_size) for p in train_prices]
        v_buckets = [get_bucket(p, b_size) for p in val_prices] # Pre-bucketize Validation
        
        abs_map, der_map = train_models(t_buckets, cfg['s_len'])
        
        models.append({
            "cfg": cfg,
            "b_size": b_size,
            "val_buckets": v_buckets,
            "abs_map": abs_map,
            "der_map": der_map
        })
        
    # 2. Iterate Validation Set
    # We loop through time indices of the Validation Set.
    # Each model has a different seq_len, so we must respect the longest one 
    # to ensure all models *could* have voted (even if some have shorter memory).
    max_seq = max(m['cfg']['s_len'] for m in models)
    validation_len = len(val_prices)
    
    correct = 0
    total_trades = 0
    abstains = 0
    conflicts = 0
    
    # Loop through validation candles
    for i in range(max_seq, validation_len - 1):
        
        active_signals = []
        
        # Poll each model
        for model in models:
            s_len = model['cfg']['s_len']
            v_bkts = model['val_buckets']
            
            # The sequence ends at current index i. Target is i+1.
            # Sequence: [i - s_len + 1 ... i]
            
            # Extract sequence
            curr_slice = v_bkts[i - s_len + 1 : i + 1]
            a_seq = tuple(curr_slice)
            last_val = a_seq[-1]
            
            d_seq = tuple(a_seq[k] - a_seq[k-1] for k in range(1, len(a_seq))) if s_len > 1 else ()
            
            pred_val = get_prediction(model['cfg']['model'], model['abs_map'], model['der_map'], 
                                      a_seq, d_seq, last_val)
            
            if pred_val is not None:
                # Check Direction
                p_diff = pred_val - last_val
                if p_diff != 0:
                    direction = 1 if p_diff > 0 else -1
                    active_signals.append({
                        "dir": direction,
                        "b_count": model['cfg']['b_count'], # Lower is better
                        "pred_val": pred_val,
                        "last_val": last_val,
                        "actual_val": v_bkts[i+1] # The reality for THIS bucket size
                    })

        # --- ENSEMBLE LOGIC ---
        
        if not active_signals:
            abstains += 1
            continue
            
        # Check Conflict
        directions = {x['dir'] for x in active_signals}
        if len(directions) > 1:
            conflicts += 1
            continue # ABSTAIN (Conflict)
            
        # If we are here, all signals agree.
        # Select strategy with LOWEST BUCKET COUNT.
        active_signals.sort(key=lambda x: x['b_count']) # Ascending sort
        winner = active_signals[0]
        
        # Verify result
        w_pred_diff = winner['pred_val'] - winner['last_val']
        w_act_diff = winner['actual_val'] - winner['last_val']
        
        if w_act_diff != 0: # If price actually moved
            total_trades += 1
            if (w_pred_diff > 0 and w_act_diff > 0) or (w_pred_diff < 0 and w_act_diff < 0):
                correct += 1

    acc = (correct / total_trades * 100) if total_trades > 0 else 0
    print(f"Final Validation Result:")
    print(f" > Accuracy : {acc:.2f}%")
    print(f" > Trades   : {total_trades}")
    print(f" > Conflicts: {conflicts} (Avoided due to disagreement)")
    print(f" > Abstains : {abstains} (No models found pattern)")
    print("-" * 60)

# =========================================
# 5. MAIN EXECUTION
# =========================================

def run_analysis():
    # 1. Load Data
    prices = get_binance_data()
    if len(prices) < 1000:
        print("Not enough data.")
        return

    # 2. Split Data
    val_len = VAL_MONTHS * HOURS_PER_MONTH
    split_idx = len(prices) - val_len
    
    train_prices = prices[:split_idx]
    val_prices = prices[split_idx:]
    
    print(f"\n=== DATA SPLIT CONFIGURATION ===")
    print(f"Total Candles : {len(prices)}")
    print(f"Training Set  : {len(train_prices)} candles")
    print(f"Validation Set: {len(val_prices)} candles (Last {VAL_MONTHS} Month(s))")
    print("--------------------------------")

    # 3. Grid Search (Training)
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
                
                acc, trades = test_strategy(train_prices, train_prices, b_count, s_len, m_type)
                
                if trades >= MIN_TRADES:
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

    # 4. Select Top Strategies
    results.sort(key=lambda x: x['score'], reverse=True)
    top_5 = results[:5]
    
    # 5. Show Individual Performance
    print(f"\n=== TOP 5 STRATEGIES (Individual Performance) ===")
    print(f"{'#':<3} | {'Config':<25} | {'Score':<8} | {'Train Acc':<10} | {'VAL ACC':<10} | {'VAL TRD':<8}")
    print("-" * 80)
    
    for i, res in enumerate(top_5):
        val_acc, val_trades = test_strategy(train_prices, val_prices, 
                                            res['b_count'], res['s_len'], res['model'])
        config_str = f"B={res['b_count']} L={res['s_len']} ({res['model'][0:3]})"
        print(f"{i+1:<3} | {config_str:<25} | {res['score']:<8.1f} | {res['train_acc']:.1f}%     | {val_acc:.1f}%     | {val_trades:<8}")

    # 6. Run Final Combined Model
    run_final_ensemble(train_prices, val_prices, top_5)

if __name__ == "__main__":
    run_analysis()
