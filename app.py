import os
import ccxt
import pandas as pd
import numpy as np
from datetime import datetime
from tqdm import tqdm

# --- Configuration ---
# Data Storage
DATA_DIR = "/app/data/"
TRAIN_FILENAME = "eth_train_30m_2020_2025.csv"  # Updated filename
TEST_FILENAME = "eth_test_30m_2025_2026.csv"    # Updated filename

# Exchange / Symbol Settings
SYMBOL = 'ETH/USDT'
TIMEFRAME = '30m'  # Changed to 30m to allow offset resampling

# Date Ranges
START_TRAIN = "2020-01-01 00:00:00"
END_TRAIN = "2025-01-01 00:00:00"
END_TEST = "2026-01-01 00:00:00"

# Grid Search Parameters
K_VALUES = [0.05, 0.1, 0.2, 0.4, 0.5, 0.8]
SEQLEN_VALUES = [3, 4, 5, 6, 7, 8]

# Qualification Thresholds
MIN_CORRECT_PREDICTIONS = 100  # Increased threshold since we have 2x data


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

    print(f"Fetching {symbol} {timeframe} data ({start_str} to {end_str})...")
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

def resample_dataset(df_30m, offset_minutes=0):
    """
    Resamples 30m data into 1h candles with a specific offset.
    offset=0  -> Candles start at 00:00, 01:00 (Standard)
    offset=30 -> Candles start at 00:30, 01:30 (Offset)
    """
    agg_dict = {
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }
    
    offset_str = f'{offset_minutes}min'
    
    # Resample 30m -> 1h
    # closed='left', label='left' ensures the candle at 00:30 aggregates 00:30 and 01:00 data points
    df_resampled = df_30m.resample('1h', offset=offset_str, closed='left', label='left').agg(agg_dict)
    
    # Drop any NaN rows (incomplete periods at the end)
    df_resampled.dropna(inplace=True)
    
    return df_resampled

def prepare_buckets(train_df_list, test_df_list, k):
    """
    Calculates buckets based on training volatility of BOTH offsets combined.
    train_df_list: [train_df_00, train_df_30]
    test_df_list:  [test_df_00, test_df_30]
    """
    
    # 1. Calculate Returns for all training data (both offsets) to find a global bucket size
    all_train_returns = []
    for df in train_df_list:
        all_train_returns.append(df['close'].pct_change().fillna(0))
    
    combined_train_returns = pd.concat(all_train_returns)
    avg_abs_return = combined_train_returns.abs().mean()
    
    bucket_size = avg_abs_return * k
    if bucket_size == 0: bucket_size = 1e-9 

    def get_buckets(df):
        returns = df['close'].pct_change().fillna(0)
        return np.floor(returns.abs() / bucket_size).astype(int) + 1

    # Generate buckets for all datasets
    train_buckets_list = [get_buckets(df) for df in train_df_list]
    test_buckets_list = [get_buckets(df) for df in test_df_list]
    
    return train_buckets_list, test_buckets_list

def build_frequency_map(sequences_list, seqlen):
    """
    Maps sequences (prefixes) to the frequency of the next bucket.
    Accepts a LIST of bucket sequences (one for each offset) and builds a shared map.
    """
    freq_map = {}
    
    for sequence in sequences_list:
        seq_array = sequence.tolist()
        for i in range(len(seq_array) - seqlen + 1):
            window = tuple(seq_array[i : i + seqlen])
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

def evaluate_prediction(test_buckets_list, freq_map, seqlen):
    """
    Runs evaluation on multiple test streams (00 and 30) using the shared freq_map.
    Returns aggregated stats.
    """
    total_correct = 0
    total_incorrect = 0
    total_ignored = 0
    
    # Iterate over both test datasets (Dataset A and Dataset B)
    for test_buckets in test_buckets_list:
        test_seq = test_buckets.tolist()
        
        # Start iterating where we have enough history for the prefix
        for i in range(seqlen - 1, len(test_seq)):
            current_idx = i - 1 
            current_bucket = test_seq[current_idx]
            
            prefix = tuple(test_seq[i - (seqlen - 1) : i])
            
            predicted_bucket = predict_most_frequent(prefix, freq_map)
            actual_next_bucket = test_seq[i]
            
            if predicted_bucket is not None:
                # Prediction logic: Only bet if Predicted > Current
                if predicted_bucket > current_bucket:
                    if actual_next_bucket > current_bucket:
                        total_correct += 1
                    elif actual_next_bucket < current_bucket:
                        total_incorrect += 1
                    else:
                        total_ignored += 1
            
    total_valid_predictions = total_correct + total_incorrect
    
    if total_valid_predictions == 0:
        return 0.0, 0
        
    accuracy = total_correct / total_valid_predictions
    return accuracy, total_correct

def run_grid_search():
    # 1. Fetch Raw 30m Data
    df_train_raw = fetch_binance_data(SYMBOL, TIMEFRAME, START_TRAIN, END_TRAIN, TRAIN_FILENAME)
    df_test_raw = fetch_binance_data(SYMBOL, TIMEFRAME, END_TRAIN, END_TEST, TEST_FILENAME)
    
    # 2. Resample into two universes: offset 00 and offset 30
    print("\nResampling data to 1h offset streams...")
    
    # Train Sets
    train_df_00 = resample_dataset(df_train_raw, offset_minutes=0)
    train_df_30 = resample_dataset(df_train_raw, offset_minutes=30)
    
    # Test Sets
    test_df_00 = resample_dataset(df_test_raw, offset_minutes=0)
    test_df_30 = resample_dataset(df_test_raw, offset_minutes=30)

    train_dfs = [train_df_00, train_df_30]
    test_dfs = [test_df_00, test_df_30]

    print("\nStarting Dual-Stream Grid Search...")
    print(f"{'K':<10} {'SeqLen':<10} {'Accuracy':<10} {'Correct':<10}")
    print("-" * 45)

    best_acc = -1
    best_params = None
    best_correct_count = 0

    for k in K_VALUES:
        # Prepare buckets for all streams using a unified bucket size
        train_buckets_list, test_buckets_list = prepare_buckets(train_dfs, test_dfs, k)
        
        for seqlen in SEQLEN_VALUES:
            # Train Shared Frequency Map (Data Augmentation)
            freq_map = build_frequency_map(train_buckets_list, seqlen)
            
            # Evaluate on both test streams combined
            acc, correct_count = evaluate_prediction(test_buckets_list, freq_map, seqlen)
            
            # Print result
            print(f"{k:<10} {seqlen:<10} {acc:.4f}     {correct_count:<10}")
            
            if correct_count >= MIN_CORRECT_PREDICTIONS:
                if acc > best_acc:
                    best_acc = acc
                    best_correct_count = correct_count
                    best_params = (k, seqlen)

    print("-" * 45)
    
    if best_params:
        print(f"Best Configuration (Combined Streams):")
        print(f"K: {best_params[0]}")
        print(f"SeqLen: {best_params[1]}")
        print(f"Accuracy: {best_acc:.4f}")
        print(f"Total Correct Predictions: {best_correct_count}")
    else:
        print(f"No configuration met the threshold of {MIN_CORRECT_PREDICTIONS} correct predictions.")

if __name__ == "__main__":
    run_grid_search()
