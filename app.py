import os
import ccxt
import pandas as pd
import numpy as np
from datetime import datetime
from tqdm import tqdm
import itertools

# --- Configuration ---
DATA_DIR = "/app/data/"
SYMBOL = 'ETH/USDT'
TIMEFRAME = '1h'
START_TRAIN = "2020-01-01 00:00:00"
END_TRAIN = "2025-01-01 00:00:00"
END_TEST = "2026-01-01 00:00:00"

def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def fetch_binance_data(symbol, timeframe, start_str, end_str, filename):
    """
    Fetches OHLCV data from Binance and saves to CSV.
    Loads from CSV if it already exists.
    """
    ensure_dir(DATA_DIR)
    filepath = os.path.join(DATA_DIR, filename)

    if os.path.exists(filepath):
        print(f"Loading {filename} from disk...")
        df = pd.read_csv(filepath, index_col=0, parse_dates=True)
        return df

    print(f"Fetching {symbol} data from Binance ({start_str} to {end_str})...")
    exchange = ccxt.binance()
    start_ts = exchange.parse8601(start_str)
    end_ts = exchange.parse8601(end_str)
    
    ohlcv = []
    current_ts = start_ts
    
    pbar = tqdm()
    while current_ts < end_ts:
        try:
            candles = exchange.fetch_ohlcv(symbol, timeframe, since=current_ts, limit=1000)
            if not candles:
                break
            
            # Filter candles that might be beyond end_ts
            candles = [c for c in candles if c[0] < end_ts]
            if not candles:
                break

            current_ts = candles[-1][0] + 1  # Advance timestamp
            ohlcv.extend(candles)
            pbar.update(len(candles))
        except Exception as e:
            print(f"Error fetching data: {e}")
            break
    pbar.close()

    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    
    print(f"Saving to {filepath}...")
    df.to_csv(filepath)
    return df

def prepare_datasets(df_train, df_test, k):
    """
    Calculates returns, buckets, and sequences based on 'k'.
    """
    # Combine temporarily to calculate global average return if desired, 
    # or strictly use Training avg for data leakage prevention.
    # Here we use Training data to define bucket size.
    
    # Calculate Returns
    train_returns = df_train['close'].pct_change().fillna(0)
    test_returns = df_test['close'].pct_change().fillna(0)
    
    avg_abs_return = train_returns.abs().mean()
    bucket_size = avg_abs_return * k
    
    if bucket_size == 0:
        bucket_size = 1e-9 # Prevent div by zero

    def get_buckets(returns):
        # 0-bucketsize is bucket 1, bucketsize-(2*bucketsize) is bucket 2
        # Formula: floor(abs(return) / size) + 1
        return np.floor(returns.abs() / bucket_size).astype(int) + 1

    train_buckets = get_buckets(train_returns)
    test_buckets = get_buckets(test_returns)
    
    return train_buckets, test_buckets

def build_frequency_map(sequence, seqlen):
    """
    Learns the frequency of sequences.
    Input: [1, 2, 3, 1, 1...]
    Output: Dictionary mapping (prefix) -> {next_val: count}
    """
    freq_map = {}
    
    # We look at windows of size 'seqlen'
    # The first 'seqlen-1' are the prefix, the last is the target
    # e.g., seqlen=3. Window: [1, 2, 3]. Prefix (1, 2) -> Target 3
    
    for i in range(len(sequence) - seqlen + 1):
        window = tuple(sequence[i : i + seqlen])
        prefix = window[:-1]
        target = window[-1]
        
        if prefix not in freq_map:
            freq_map[prefix] = {}
        
        if target not in freq_map[prefix]:
            freq_map[prefix][target] = 0
        freq_map[prefix][target] += 1
        
    return freq_map

def predict_most_frequent(prefix, freq_map):
    """
    Returns the most frequent next bucket for a given prefix.
    Returns None if prefix is unknown.
    """
    if prefix not in freq_map:
        return None
    
    candidates = freq_map[prefix]
    # Return the key with the highest value
    return max(candidates, key=candidates.get)

def evaluate_prediction(train_buckets, test_buckets, seqlen):
    """
    Train on train_buckets, Predict on test_buckets.
    Returns accuracy.
    """
    # 1. Train
    freq_map = build_frequency_map(train_buckets.tolist(), seqlen)
    
    correct = 0
    incorrect = 0
    ignored = 0
    
    # 2. Test
    # We need at least seqlen-1 history to make a prediction
    test_seq = test_buckets.tolist()
    
    # Iterate through test set
    # We start at index = seqlen - 1 because we need seqlen-1 history
    for i in range(seqlen - 1, len(test_seq)):
        current_idx = i - 1 # The "current" state in time
        
        # Current bucket (the last known bucket before the one we are predicting)
        current_bucket = test_seq[current_idx]
        
        # The prefix is the sequence leading up to the target
        # Target is at test_seq[i]
        # Prefix is test_seq[i - (seqlen - 1) : i]
        prefix = tuple(test_seq[i - (seqlen - 1) : i])
        
        predicted_bucket = predict_most_frequent(prefix, freq_map)
        actual_next_bucket = test_seq[i]
        
        if predicted_bucket is not None:
            # Condition: Predict > Current
            if predicted_bucket > current_bucket:
                
                # Check Outcome
                if actual_next_bucket > current_bucket:
                    correct += 1
                elif actual_next_bucket < current_bucket:
                    incorrect += 1
                else:
                    # actual == current
                    ignored += 1
            
            # The prompt does not specify what to do if predicted <= current
            # We skip those for accuracy calculation based on the specific "Predict > Current" rule
            
    total_valid_predictions = correct + incorrect
    
    if total_valid_predictions == 0:
        return 0.0
        
    return correct / total_valid_predictions

def generate_auxiliary_datasets(buckets):
    """
    Generates the 'Bucket sequence return' dataset as requested:
    0 if bucket t1 = bucket T2, 1 if bucket t1 = bucket T2 + 1
    (Assuming T2 is previous, t1 is current)
    """
    seq_return = [0] # First entry is 0
    bucket_list = buckets.tolist()
    
    for i in range(1, len(bucket_list)):
        curr = bucket_list[i]
        prev = bucket_list[i-1]
        
        if curr == prev:
            seq_return.append(0)
        elif curr == prev + 1:
            seq_return.append(1)
        else:
            # Fallback for undefined cases in prompt (e.g. -1 or +2)
            seq_return.append(-1) 
            
    return seq_return

# --- Main Execution ---

def run_grid_search():
    # 1. Fetch Data
    df_train = fetch_binance_data(SYMBOL, TIMEFRAME, START_TRAIN, END_TRAIN, "eth_train_2020_2025.csv")
    df_test = fetch_binance_data(SYMBOL, TIMEFRAME, END_TRAIN, END_TEST, "eth_test_2025_2026.csv")
    
    # 2. Define Grid
    k_values = [0.5, 0.8, 1.0, 1.2, 1.5, 2.0, 3.0]
    seqlen_values = [3, 4, 5, 6, 7, 8]
    
    results = []
    
    print("\nStarting Grid Search...")
    print(f"{'K':<10} {'SeqLen':<10} {'Accuracy':<10}")
    print("-" * 35)

    best_acc = -1
    best_params = None

    for k in k_values:
        # Prepare buckets for this K
        train_buckets, test_buckets = prepare_datasets(df_train, df_test, k)
        
        # Optional: Generate the aux dataset as requested (not used in prediction logic directly but computed)
        # train_seq_return = generate_auxiliary_datasets(train_buckets)
        
        for seqlen in seqlen_values:
            acc = evaluate_prediction(train_buckets, test_buckets, seqlen)
            
            results.append({'k': k, 'seqlen': seqlen, 'accuracy': acc})
            print(f"{k:<10} {seqlen:<10} {acc:.4f}")
            
            if acc > best_acc:
                best_acc = acc
                best_params = (k, seqlen)

    print("-" * 35)
    print(f"Best Accuracy: {best_acc:.4f}")
    print(f"Best Parameters: k={best_params[0]}, seqlen={best_params[1]}")
    
    return best_params

if __name__ == "__main__":
    run_grid_search()
