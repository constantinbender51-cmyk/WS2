import random
import time
import sys
import json
import os
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

# --- Data Split Settings ---
VAL_MONTHS = 6           # Final Holdout (The real test)
PRE_VAL_MONTHS = 6      # Pre-Validation (Used for scoring/selection)
HOURS_PER_MONTH = 720    # Approx candles per month

# --- Grid Search Ranges ---
BUCKET_COUNTS = range(10, 131, 10)  # 10 to 250
SEQ_LENGTHS = [3, 4, 5, 6, 8]
MIN_TRADES = 20          # Min trades to consider a strategy valid during training

# =========================================
# 2. DATA UTILITIES
# =========================================
def get_binance_data(symbol=SYMBOL, interval=INTERVAL, start_str=START_DATE):
    """Fetches data from Binance or loads from local cache if available."""
    
    # Define file path
    data_dir = "/app/data"
    file_name = f"{symbol}_{interval}_{start_str}.json"
    full_path = os.path.join(data_dir, file_name)

    # 1. Check if local cache exists
    if os.path.exists(full_path):
        print(f"--- Loading {symbol} {interval} data from local cache ---")
        with open(full_path, 'r') as f:
            return json.load(f)

    # 2. Fetch if not present
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

    # 3. Save to local cache
    if all_prices:
        try:
            if not os.path.exists(data_dir):
                os.makedirs(data_dir)
            with open(full_path, 'w') as f:
                json.dump(all_prices, f)
            print(f"Data cached to {full_path}")
        except Exception as e:
            print(f"Warning: Could not save cache: {e}")

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
    """
    Builds the probability maps from the training buckets.
    """
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
    """
    Standardizes the evaluation process.
    - Uses bucket_size derived from TRAIN prices (Simulating real deployment)
    - Trains on TRAIN prices
    - Tests on TEST prices (which can be Train, PreVal, or Val)
    """
    # 1. Setup Buckets (Size strictly from TRAIN to avoid look-ahead bias)
    bucket_size = calculate_bucket_size(train_prices, bucket_count)
    
    train_buckets = [get_bucket(p, bucket_size) for p in train_prices]
    test_buckets = [get_bucket(p, bucket_size) for p in test_prices]
    
    # 2. Train Model
    abs_map, der_map = train_models(train_buckets, seq_len)
    
    # 3. Test Model
    correct = 0
    total_trades = 0
    abstains = 0
    
    # We loop through the test set. 
    loop_range = len(test_buckets) - seq_len
    
    for i in range(loop_range):
        curr_slice = test_buckets[i : i + seq_len + 1]
        a_seq = tuple(curr_slice[:-1])
        actual_val = curr_slice[-1]
        last_val = a_seq[-1]
        
        d_seq = tuple(a_seq[k] - a_seq[k-1] for k in range(1, len(a_seq))) if seq_len > 1 else ()
            
        pred_val = get_prediction(model_type, abs_map, der_map, a_seq, d_seq, last_val)
        
        if pred_val is None: 
            abstains += 1
            continue 
            
        pred_diff = pred_val - last_val
        actual_diff = actual_val - last_val
        
        if pred_diff != 0 and actual_diff != 0:
            total_trades += 1
            if (pred_diff > 0 and actual_diff > 0) or (pred_diff < 0 and actual_diff < 0):
                correct += 1
                
    accuracy = (correct / total_trades * 100) if total_trades > 0 else 0.0
    return accuracy, total_trades, abstains

# =========================================
# 4. FINAL ENSEMBLE LOGIC
# =========================================

def run_final_ensemble(train_prices, val_prices, top_configs):
    print(f"\n=== FINAL COMBINED MODEL (Union of Top {len(top_configs)}) ===")
    
    models = []
    for cfg in top_configs:
        b_size = calculate_bucket_size(train_prices, cfg['b_count'])
        t_buckets = [get_bucket(p, b_size) for p in train_prices]
        v_buckets = [get_bucket(p, b_size) for p in val_prices] 
        
        abs_map, der_map = train_models(t_buckets, cfg['s_len'])
        
        models.append({
            "cfg": cfg,
            "val_buckets": v_buckets,
            "abs_map": abs_map,
            "der_map": der_map
        })
        
    max_seq = max(m['cfg']['s_len'] for m in models)
    validation_len = len(val_prices)
    
    correct = 0
    total_trades = 0
    abstains = 0
    conflicts = 0
    
    for i in range(max_seq, validation_len - 1):
        active_signals = []
        
        for model in models:
            s_len = model['cfg']['s_len']
            v_bkts = model['val_buckets']
            
            # Lookback window for this specific model
            curr_slice = v_bkts[i - s_len + 1 : i + 1]
            a_seq = tuple(curr_slice)
            last_val = a_seq[-1]
            
            d_seq = tuple(a_seq[k] - a_seq[k-1] for k in range(1, len(a_seq))) if s_len > 1 else ()
            
            pred_val = get_prediction(model['cfg']['model'], model['abs_map'], model['der_map'], 
                                      a_seq, d_seq, last_val)
            
            if pred_val is not None:
                p_diff = pred_val - last_val
                if p_diff != 0:
                    direction = 1 if p_diff > 0 else -1
                    active_signals.append({
                        "dir": direction,
                        "b_count": model['cfg']['b_count'], 
                        "pred_val": pred_val,
                        "last_val": last_val,
                        "actual_val": v_bkts[i+1] 
                    })

        if not active_signals:
            abstains += 1
            continue
            
        directions = {x['dir'] for x in active_signals}
        if len(directions) > 1:
            conflicts += 1
            continue 
            
        active_signals.sort(key=lambda x: x['b_count'])
        winner = active_signals[0]
        
        w_pred_diff = winner['pred_val'] - winner['last_val']
        w_act_diff = winner['actual_val'] - winner['last_val']
        
        if w_act_diff != 0: 
            total_trades += 1
            if (w_pred_diff > 0 and w_act_diff > 0) or (w_pred_diff < 0 and w_act_diff < 0):
                correct += 1

    acc = (correct / total_trades * 100) if total_trades > 0 else 0
    print(f"Final Validation Result:")
    print(f" > Accuracy : {acc:.2f}%")
    print(f" > Trades   : {total_trades}")
    print(f" > Conflicts: {conflicts}")
    print(f" > Abstains : {abstains}")
    print("-" * 60)

# =========================================
# 5. MAIN EXECUTION
# =========================================

def run_analysis():
    prices = get_binance_data()
    if len(prices) < 2000:
        print("Not enough data.")
        return

    # --- 3-Way Split ---
    val_points = VAL_MONTHS * HOURS_PER_MONTH
    preval_points = PRE_VAL_MONTHS * HOURS_PER_MONTH
    
    end_idx = len(prices)
    val_start_idx = end_idx - val_points
    preval_start_idx = val_start_idx - preval_points
    
    train_prices = prices[:preval_start_idx]     
    preval_prices = prices[preval_start_idx:val_start_idx] 
    val_prices = prices[val_start_idx:]          
    
    print(f"\n=== DATA SPLIT CONFIGURATION ===")
    print(f"Total Candles : {len(prices)}")
    print(f"1. Training Set  : {len(train_prices)} candles")
    print(f"2. Pre-Validation: {len(preval_prices)} candles")
    print(f"3. Validation Set: {len(val_prices)} candles")

    print(f"\n--- Running Grid Search ---")
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
                
                # 1. Performance on TRAIN
                t_acc, t_trades, t_abst = test_strategy(train_prices, train_prices, b_count, s_len, m_type)
                
                # 2. Performance on PRE-VAL
                p_acc, p_trades, p_abst = test_strategy(train_prices, preval_prices, b_count, s_len, m_type)
                
                # Filter by minimum trades on train to ensure statistical significance
                if t_trades >= MIN_TRADES and p_trades > 0:
                    
                    # --- NEW COST FUNCTION ---
                    # (PreValAcc % / 100 - 0.5) * PreValTrades
                    # This measures "Excess Accuracy" weighted by volume.
                    score = ((p_acc / 100.0) - 0.55) * p_trades
                    
                    results.append({
                        "b_count": b_count,
                        "s_len": s_len,
                        "model": m_type,
                        "score": score,
                        "stats": {
                            "train": (t_acc, t_trades, t_abst),
                            "preval": (p_acc, p_trades, p_abst)
                        }
                    })

    print(f"\nGrid Search Complete. Found {len(results)} valid configurations.")

    # Select Top Strategies (Sorting by the new score)
    results.sort(key=lambda x: x['score'], reverse=True)
    top_5 = results[:5]
    
    print(f"\n=== TOP 5 STRATEGIES (Sorted by (Acc-0.5)*Trades) ===")
    print(f"{'#':<3} | {'Config':<22} | {'Score':<6} | {'TRAIN Acc':<9} | {'PRE-VAL Acc':<11} {'Trds':<5}")
    print("-" * 85)
    
    for i, res in enumerate(top_5):
        t_acc, t_trd, _ = res['stats']['train']
        p_acc, p_trd, _ = res['stats']['preval']
        config_str = f"B={res['b_count']} L={res['s_len']} ({res['model'][0:3]})"
        print(f"{i+1:<3} | {config_str:<22} | {res['score']:<6.2f} | {t_acc:.1f}%      | {p_acc:.1f}%       {p_trd:<5}")

    run_final_ensemble(train_prices, val_prices, top_5)

if __name__ == "__main__":
    run_analysis()
