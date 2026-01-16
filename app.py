
import numpy as np
import pandas as pd
import ccxt
from datetime import datetime, timedelta
from collections import defaultdict, Counter
from itertools import product
import warnings
warnings.filterwarnings('ignore')

def fetch_binance_data(symbol='ETH/USDT', timeframe='30m', start_date='2020-01-01', end_date='2026-01-01'):
    """Fetch OHLC data from Binance"""
    print(f"Fetching {symbol} {timeframe} data from {start_date} to {end_date}...")
    
    exchange = ccxt.binance({
        'enableRateLimit': True,
    })
    
    start_ts = exchange.parse8601(f'{start_date}T00:00:00Z')
    end_ts = exchange.parse8601(f'{end_date}T00:00:00Z')
    
    all_candles = []
    current_ts = start_ts
    
    while current_ts < end_ts:
        try:
            candles = exchange.fetch_ohlcv(symbol, timeframe, current_ts, limit=1000)
            if not candles:
                break
            
            all_candles.extend(candles)
            current_ts = candles[-1][0] + 1
            
            print(f"Fetched {len(all_candles)} candles...", end='\r')
            
            if candles[-1][0] >= end_ts:
                break
                
        except Exception as e:
            print(f"\nError fetching data: {e}")
            break
    
    df = pd.DataFrame(all_candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df = df[df['timestamp'] < end_ts]
    
    print(f"\nFetched {len(df)} candles total")
    return df

def create_datasets(df):
    """Create original and base datasets"""
    original = df.copy()
    base = df.iloc[1:].reset_index(drop=True)  # Start at candle 2 (index 1)
    
    print(f"Original dataset: {len(original)} candles")
    print(f"Base dataset: {len(base)} candles (starts at candle 2)")
    
    return original, base

def resample_to_1h(df):
    """Resample 30m data to 1h"""
    df_copy = df.copy()
    df_copy.set_index('timestamp', inplace=True)
    
    resampled = df_copy.resample('1H').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }).dropna()
    
    resampled.reset_index(inplace=True)
    print(f"Resampled to 1H: {len(resampled)} candles")
    
    return resampled

def calculate_buckets(df, k):
    """Calculate bucket assignments for each candle"""
    returns = df['close'].pct_change().dropna()
    avg_abs_return = returns.abs().mean()
    bucket_size = avg_abs_return * k
    
    print(f"Average |return|: {avg_abs_return:.6f}")
    print(f"Bucket size (k={k}): {bucket_size:.6f}")
    
    # Calculate buckets for each candle
    buckets = []
    for ret in returns:
        if ret >= 0:
            bucket = int(ret / bucket_size) + 1
        else:
            bucket = -int(abs(ret) / bucket_size) - 1
        buckets.append(bucket)
    
    return buckets, bucket_size

def create_sequences(buckets):
    """Create bucket sequence and derivative sequence"""
    bucket_seq = buckets
    derivative_seq = [0]  # First element is 0
    
    for i in range(1, len(bucket_seq)):
        derivative_seq.append(bucket_seq[i] - bucket_seq[i-1])
    
    return bucket_seq, derivative_seq

def get_subsequence_frequencies(sequence, seq_len):
    """Get frequency counts for all subsequences of given length"""
    freq = Counter()
    
    for i in range(len(sequence) - seq_len + 1):
        subseq = tuple(sequence[i:i+seq_len])
        freq[subseq] += 1
    
    return freq

def predict_completion(incomplete_subseq, freq_dict, sequence_type='bucket'):
    """Predict the next element based on most frequent completion"""
    incomplete_tuple = tuple(incomplete_subseq)
    
    # Find all subsequences that start with the incomplete subsequence
    matching = {}
    for subseq, count in freq_dict.items():
        if len(subseq) > len(incomplete_tuple) and subseq[:len(incomplete_tuple)] == incomplete_tuple:
            next_val = subseq[len(incomplete_tuple)]
            matching[next_val] = matching.get(next_val, 0) + count
    
    if not matching:
        return None
    
    # Return the most probable next value
    return max(matching.items(), key=lambda x: x[1])[0]

def evaluate_predictions(train_bucket_seq, train_deriv_seq, val_bucket_seq, val_deriv_seq, k, seq_len):
    """Evaluate prediction accuracy on validation set"""
    
    # Build frequency dictionaries from training data
    bucket_freq = get_subsequence_frequencies(train_bucket_seq, seq_len + 1)
    deriv_freq = get_subsequence_frequencies(train_deriv_seq, seq_len + 1)
    
    correct_bucket = 0
    correct_deriv = 0
    total_predictions = 0
    
    # Test on all incomplete subsequences in validation set
    for i in range(len(val_bucket_seq) - seq_len):
        incomplete_bucket = val_bucket_seq[i:i+seq_len]
        incomplete_deriv = val_deriv_seq[i:i+seq_len]
        
        actual_bucket_next = val_bucket_seq[i+seq_len]
        actual_deriv_next = val_deriv_seq[i+seq_len]
        
        # Predict using bucket sequence
        pred_bucket = predict_completion(incomplete_bucket, bucket_freq, 'bucket')
        
        # Predict using derivative sequence
        pred_deriv = predict_completion(incomplete_deriv, deriv_freq, 'derivative')
        
        if pred_bucket is not None:
            total_predictions += 1
            if pred_bucket == actual_bucket_next:
                correct_bucket += 1
        
        if pred_deriv is not None:
            if pred_deriv == actual_deriv_next:
                correct_deriv += 1
    
    if total_predictions == 0:
        return 0.0, 0.0, 0
    
    accuracy_bucket = correct_bucket / total_predictions
    accuracy_deriv = correct_deriv / total_predictions
    
    return accuracy_bucket, accuracy_deriv, total_predictions

def grid_search(train_original, train_base, val_original, val_base, k_values, seq_len_values):
    """Perform grid search over k and seq_len parameters"""
    print("\n" + "="*80)
    print("STARTING GRID SEARCH")
    print("="*80)
    
    results = []
    
    for dataset_type in ['original', 'base']:
        print(f"\n--- Testing {dataset_type.upper()} dataset ---")
        
        train_df = train_original if dataset_type == 'original' else train_base
        val_df = val_original if dataset_type == 'original' else val_base
        
        # Resample to 1h
        train_1h = resample_to_1h(train_df)
        val_1h = resample_to_1h(val_df)
        
        for k in k_values:
            print(f"\nTesting k={k}")
            
            # Calculate buckets for training and validation
            train_buckets, bucket_size = calculate_buckets(train_1h, k)
            val_buckets, _ = calculate_buckets(val_1h, k)
            
            # Create sequences
            train_bucket_seq, train_deriv_seq = create_sequences(train_buckets)
            val_bucket_seq, val_deriv_seq = create_sequences(val_buckets)
            
            for seq_len in seq_len_values:
                if seq_len >= len(train_bucket_seq):
                    continue
                
                # Evaluate predictions
                acc_bucket, acc_deriv, total_preds = evaluate_predictions(
                    train_bucket_seq, train_deriv_seq,
                    val_bucket_seq, val_deriv_seq,
                    k, seq_len
                )
                
                # Combined accuracy (average of both)
                combined_acc = (acc_bucket + acc_deriv) / 2
                
                results.append({
                    'dataset': dataset_type,
                    'k': k,
                    'seq_len': seq_len,
                    'bucket_accuracy': acc_bucket,
                    'derivative_accuracy': acc_deriv,
                    'combined_accuracy': combined_acc,
                    'total_predictions': total_preds,
                    'bucket_size': bucket_size
                })
                
                print(f"  seq_len={seq_len}: Bucket Acc={acc_bucket:.4f}, Deriv Acc={acc_deriv:.4f}, "
                      f"Combined={combined_acc:.4f}, Predictions={total_preds}")
    
    return results

def main():
    # 1. Fetch data
    df = fetch_binance_data(symbol='ETH/USDT', timeframe='30m', 
                           start_date='2020-01-01', end_date='2026-01-17')
    
    # Split into training (2020-2025) and validation (2025-2026)
    train_df = df[df['timestamp'] < '2025-01-01'].copy()
    val_df = df[df['timestamp'] >= '2025-01-01'].copy()
    
    print(f"\nTraining period: {train_df['timestamp'].min()} to {train_df['timestamp'].max()}")
    print(f"Validation period: {val_df['timestamp'].min()} to {val_df['timestamp'].max()}")
    print(f"Training samples: {len(train_df)}, Validation samples: {len(val_df)}")
    
    # 2. Create original and base datasets
    train_original, train_base = create_datasets(train_df)
    val_original, val_base = create_datasets(val_df)
    
    # 3. Grid search parameters
    k_values = [0.5, 1.0, 1.5, 2.0, 3.0, 5.0]
    seq_len_values = [3, 5, 7, 10, 15, 20]
    
    # 4. Perform grid search
    results = grid_search(train_original, train_base, val_original, val_base, 
                         k_values, seq_len_values)
    
    # 5. Find and print winning configuration
    results_df = pd.DataFrame(results)
    best_result = results_df.loc[results_df['combined_accuracy'].idxmax()]
    
    print("\n" + "="*80)
    print("WINNING CONFIGURATION")
    print("="*80)
    print(f"Dataset: {best_result['dataset']}")
    print(f"k: {best_result['k']}")
    print(f"Sequence Length: {best_result['seq_len']}")
    print(f"Bucket Size: {best_result['bucket_size']:.6f}")
    print(f"Bucket Sequence Accuracy: {best_result['bucket_accuracy']:.4f}")
    print(f"Derivative Sequence Accuracy: {best_result['derivative_accuracy']:.4f}")
    print(f"Combined Accuracy: {best_result['combined_accuracy']:.4f}")
    print(f"Total Predictions: {best_result['total_predictions']}")
    print("="*80)
    
    # Print top 10 configurations
    print("\nTop 10 Configurations:")
    print(results_df.nlargest(10, 'combined_accuracy')[['dataset', 'k', 'seq_len', 
                                                          'bucket_accuracy', 'derivative_accuracy',
                                                          'combined_accuracy', 'total_predictions']])
    
    return results_df

if __name__ == "__main__":
    results = main()