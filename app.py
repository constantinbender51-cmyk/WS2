import ccxt
import pandas as pd
import numpy as np
import time
import os
import sys
from sklearn.ensemble import RandomForestRegressor

# =========================================
# 1. CONFIGURATION
# =========================================

# --- Exchange & Data Settings ---
SYMBOL = 'ETH/USDT'
TIMEFRAME = '15m'
START_DATE = '2020-01-01 00:00:00'
END_DATE = '2026-01-01 00:00:00'

# --- Grid Search Ranges ---
# N: Number of Buckets to divide the Training Range into
N_BUCKETS_LIST = [16, 32, 64, 128]

# B: Lookback candles for lag features
B_START = 2
B_END = 10

# --- Filtering ---
MIN_ACCURACY_THRESHOLD = 0.60  # 60%

# --- Data Storage ---
DATA_DIR = '/app/data/'
FILE_NAME = 'ethusdt_15m_2020_2026.csv'
FILE_PATH = os.path.join(DATA_DIR, FILE_NAME)

# --- Model Settings ---
RF_ESTIMATORS = 30 
RF_MAX_DEPTH = 8
RANDOM_STATE = 42

# --- Output Settings ---
PRINT_DELAY = 0.01

# =========================================
# 2. HELPER FUNCTIONS
# =========================================

def slow_print(text, delay=PRINT_DELAY):
    print(text)
    sys.stdout.flush()
    time.sleep(delay)

def get_ohlcv_data(symbol, timeframe, start_str, end_str):
    if os.path.exists(FILE_PATH):
        slow_print(f"[CACHE] Loading data from {FILE_PATH}...")
        df = pd.read_csv(FILE_PATH)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)
        return df

    slow_print(f"[API] Fetching new data from Binance...")
    exchange = ccxt.binance()
    start_ts = exchange.parse8601(start_str)
    end_ts = exchange.parse8601(end_str)
    
    data = []
    current_ts = start_ts
    
    while current_ts < end_ts:
        try:
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since=current_ts, limit=1000)
            if not ohlcv:
                break
            current_ts = ohlcv[-1][0] + 1
            data += ohlcv
            time.sleep(exchange.rateLimit / 1000)
            if current_ts >= exchange.milliseconds():
                break
        except Exception as e:
            slow_print(f"[ERROR] {str(e)}")
            break

    df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    
    mask = (df.index >= start_str) & (df.index <= end_str)
    df = df.loc[mask]
    
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
    
    df.to_csv(FILE_PATH)
    return df

def prepare_features_and_target(df_input, step_size, b):
    """
    df_input: Full dataset (or slice)
    step_size: Calculated externally based on Training Range
    b: Number of lag features
    """
    df = df_input.copy()
    
    # 1. Bucketize Price (Discretization)
    # Using floor division to get bucket index
    df['bucket_index'] = np.floor(df['close'] / step_size).astype(int) + 1

    # 2. Log Returns (Features)
    df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
    
    # 3. Target: Derivative of Bucket Index
    df['target_derivative'] = df['bucket_index'].shift(-1) - df['bucket_index']
    
    # 4. Create Lag Features
    feature_cols = []
    for i in range(1, b + 1):
        col_name = f'lag_{i}'
        df[col_name] = df['log_returns'].shift(i)
        feature_cols.append(col_name)
    
    df.dropna(inplace=True)
    return df, feature_cols

def calculate_metrics_and_score(y_true, y_pred_raw):
    # Round predictions to nearest bucket step
    y_pred = np.round(y_pred_raw)
    
    results = pd.DataFrame({'actual': y_true, 'pred': y_pred})
    
    # Filter for directional predictions
    directional_preds = results[results['pred'] != 0]
    total_trades = len(directional_preds)
    
    if total_trades == 0:
        return 0.0, 0.0, 0, -1.0 

    # 1. Flat Outcome Ratio
    flat_outcomes = directional_preds[directional_preds['actual'] == 0]
    flat_ratio = len(flat_outcomes) / total_trades

    # 2. Non-Flat Accuracy
    valid_scoring_moves = directional_preds[directional_preds['actual'] != 0]
    if len(valid_scoring_moves) == 0:
        accuracy = 0.0
    else:
        correct_direction = np.sign(valid_scoring_moves['pred']) == np.sign(valid_scoring_moves['actual'])
        accuracy = correct_direction.sum() / len(valid_scoring_moves)

    # 3. CUSTOM SCORE
    # Score = (Accuracy - 0.5) * Trades * (1 - Flat_Ratio)
    score = (accuracy - 0.50) * total_trades * (1 - flat_ratio)

    return accuracy, flat_ratio, total_trades, score

# =========================================
# 3. MAIN EXECUTION
# =========================================

def main():
    slow_print("--- STARTING GRID SEARCH (BUCKET LOGIC) ---")
    
    # 1. Load Data
    try:
        df_raw = get_ohlcv_data(SYMBOL, TIMEFRAME, START_DATE, END_DATE)
    except Exception as e:
        slow_print(f"[CRITICAL] {e}")
        return

    # 2. Determine Split Points
    n_total = len(df_raw)
    train_end_idx = int(n_total * 0.60)
    
    # 3. Calculate Training Range (For Step Size)
    # CRITICAL: Only use training data to determine step size to avoid leakage
    train_slice = df_raw.iloc[:train_end_idx]
    train_min = train_slice['close'].min()
    train_max = train_slice['close'].max()
    train_range = train_max - train_min
    
    slow_print(f"[INFO] Train Data Range: {train_min:.2f} to {train_max:.2f} (Diff: {train_range:.2f})")

    # 4. Prepare Grid
    b_values = list(range(B_START, B_END + 1))
    total_combinations = len(N_BUCKETS_LIST) * len(b_values)
    
    slow_print(f"[INFO] Grid: N_Buckets({len(N_BUCKETS_LIST)}) * B({len(b_values)}) = {total_combinations} combinations.")
    slow_print(f"[INFO] FILTER: Discard if Accuracy < {MIN_ACCURACY_THRESHOLD:.0%}")
    slow_print("-" * 40)

    best_score = -float('inf')
    best_params = {}
    best_metrics = {}
    
    counter = 0
    
    # --- LOOP ---
    for n in N_BUCKETS_LIST:
        # Calculate Step Size for this N
        step_size = train_range / n
        
        for b in b_values:
            counter += 1
            
            # Prepare Data with dynamic Step Size
            df_processed, feature_cols = prepare_features_and_target(df_raw, step_size, b)
            
            X_feat = df_processed[feature_cols]
            y = df_processed['target_derivative']
            
            n_proc = len(df_processed)
            t_end = int(n_proc * 0.60)
            v_end = int(n_proc * 0.80)
            
            X_train = X_feat.iloc[:t_end]
            y_train = y.iloc[:t_end]
            X_val = X_feat.iloc[t_end:v_end]
            y_val = y.iloc[t_end:v_end]
            
            # Train Random Forest
            rf = RandomForestRegressor(n_estimators=RF_ESTIMATORS, 
                                       max_depth=RF_MAX_DEPTH, 
                                       n_jobs=-1, 
                                       random_state=RANDOM_STATE)
            rf.fit(X_train, y_train)
            
            # Validate
            preds = rf.predict(X_val)
            acc, flat_ratio, trades, score = calculate_metrics_and_score(y_val, preds)
            
            # --- FILTER ---
            if acc < MIN_ACCURACY_THRESHOLD:
                continue
            
            # Output valid candidate
            if counter % 5 == 0 or score > best_score:
                print(f"[{counter}/{total_combinations}] N={n} (Step:{step_size:.2f}), B={b} | Score: {score:.2f} | Acc: {acc:.1%} | Trades: {trades} | Flat: {flat_ratio:.1%}")

            if score > best_score:
                best_score = score
                best_params = {'n': n, 'step_size': step_size, 'b': b}
                best_metrics = {'acc': acc, 'trades': trades, 'flat': flat_ratio}

    slow_print("-" * 40)
    slow_print("--- GRID SEARCH COMPLETE ---")
    
    if not best_params:
        slow_print("[WARNING] No parameters met the 60% accuracy threshold.")
        return

    slow_print(f"BEST VALIDATION PARAMETERS FOUND:")
    slow_print(f"N (Buckets)    : {best_params['n']}")
    slow_print(f"Step Size      : {best_params['step_size']:.4f}")
    slow_print(f"B (Candles)    : {best_params['b']}")
    slow_print(f"Validation Score : {best_score:.4f}")
    slow_print(f" > Accuracy      : {best_metrics['acc']:.2%}")
    slow_print(f" > Trades        : {best_metrics['trades']}")
    slow_print(f" > Flat Ratio    : {best_metrics['flat']:.2%}")
    
    slow_print("-" * 40)
    slow_print("[INFO] Test set results hidden as requested.")

if __name__ == "__main__":
    main()
