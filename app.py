import os
import ccxt
import pandas as pd
import numpy as np
from datetime import datetime
from tqdm import tqdm

# --- Configuration ---
DATA_DIR = "/app/data/"
SYMBOL = 'ETH/USDT'
TIMEFRAME = '1h'
START_TRAIN = "2020-01-01 00:00:00"
END_TRAIN = "2025-01-01 00:00:00"
END_TEST = "2026-01-01 00:00:00"
MIN_CORRECT = 50  # Threshold for valid strategies

def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def fetch_binance_data(symbol, timeframe, start_str, end_str, filename):
    """
    Fetches OHLCV data from Binance and saves to CSV.
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
            
            candles = [c for c in candles if c[0] < end_ts]
            if not candles:
                break

            current_ts = candles[-1][0] + 1
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
    Calculates buckets based on training volatility and k.
    """
    train_returns = df_train['close'].pct_change().fillna(0)
    test_returns = df_test['close'].pct_change().fillna(0)
    
    avg_abs_return = train_returns.abs().mean()
    bucket_size = avg_abs_return * k
    
    if bucket_size == 0:
        bucket_size = 1e-9 

    def get_buckets(returns):
        return np.floor(returns.abs() / bucket_size).astype(int) + 1

    train_buckets = get_buckets(train_returns)
    test_buckets = get_buckets(test_returns)
    
    return train_buckets, test_buckets

def build_frequency_map(sequence, seqlen):
    """
    Maps sequences (prefixes) to the frequency of the next bucket.
    """
    freq_map = {}
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
    if prefix not in freq_map:
        return None
    candidates = freq_map[prefix]
    return max(candidates, key=candidates.get)

def evaluate_prediction(train_buckets, test_buckets, seqlen):
    """
    Returns (accuracy, correct_count).
    """
    freq_map = build_frequency_map(train_buckets.tolist(), seqlen)
    
    correct = 0
    incorrect = 0
    ignored = 0
    
    test_seq = test_buckets.tolist()
    
    for i in range(seqlen - 1, len(test_seq)):
        current_idx = i - 1 
        current_bucket = test_seq[current_idx]
        
        prefix = tuple(test_seq[i - (seqlen - 1) : i])
        
        predicted_bucket = predict_most_frequent(prefix, freq_map)
        actual_next_bucket = test_seq[i]
        
        if predicted_bucket is not None:
            # Only evaluate if Prediction > Current
            if predicted_bucket > current_bucket:
                if actual_next_bucket > current_bucket:
                    correct += 1
                elif actual_next_bucket < current_bucket:
                    incorrect += 1
                else:
                    ignored += 1
            
    total_valid_predictions = correct + incorrect
    
    if total_valid_predictions == 0:
        return 0.0, 0
        
    accuracy = correct / total_valid_predictions
    return accuracy, correct

def run_grid_search():
    df_train = fetch_binance_data(SYMBOL, TIMEFRAME, START_TRAIN, END_TRAIN, "eth_train_2020_2025.csv")
    df_test = fetch_binance_data(SYMBOL, TIMEFRAME, END_TRAIN, END_TEST, "eth_test_2025_2026.csv")
    
    k_values = [0.5, 0.8, 1.0, 1.2, 1.5, 2.0, 3.0]
    seqlen_values = [3, 4, 5, 6, 7, 8]
    
    print("\nStarting Grid Search...")
    print(f"Skipping strategies with < {MIN_CORRECT} correct predictions.\n")
    print(f"{'K':<10} {'SeqLen':<10} {'Accuracy':<10} {'Correct':<10} {'Status':<10}")
    print("-" * 55)

    best_acc = -1
    best_params = None
    best_correct_count = 0

    for k in k_values:
        train_buckets, test_buckets = prepare_datasets(df_train, df_test, k)
        
        for seqlen in seqlen_values:
            acc, correct_count = evaluate_prediction(train_buckets, test_buckets, seqlen)
            
            # --- Logic Filter ---
            if correct_count < MIN_CORRECT:
                status = "SKIPPED"
                # We print it to show it was calculated, but marked as skipped
                print(f"{k:<10} {seqlen:<10} {acc:.4f}     {correct_count:<10} {status}")
                continue
            else:
                status = "VALID"
                print(f"{k:<10} {seqlen:<10} {acc:.4f}     {correct_count:<10} {status}")

            # Only update best if VALID
            if acc > best_acc:
                best_acc = acc
                best_correct_count = correct_count
                best_params = (k, seqlen)

    print("-" * 55)
    
    if best_params:
        print(f"Best Valid Configuration:")
        print(f"K: {best_params[0]}")
        print(f"SeqLen: {best_params[1]}")
        print(f"Accuracy: {best_acc:.4f}")
        print(f"Total Correct Predictions: {best_correct_count}")
    else:
        print("No strategy met the minimum correct prediction threshold.")

if __name__ == "__main__":
    run_grid_search()
