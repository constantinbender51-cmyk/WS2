import os
import ccxt
import pandas as pd
import numpy as np
import pickle
import requests
import base64
import json
from datetime import datetime
from tqdm import tqdm
from dotenv import load_dotenv

# --- Load Secrets ---
load_dotenv()
GITHUB_TOKEN = os.getenv("GITHUB_PAT")

# --- Configuration ---
GITHUB_OWNER = "constantinbender51-cmyk"
GITHUB_REPO = "model-2"
GITHUB_BRANCH = "main"

# The "Big 3" Assets
SYMBOLS = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT']
TIMEFRAME = '15m'

# Date Ranges
START_TRAIN = "2020-01-01 00:00:00"
END_TRAIN = "2025-01-01 00:00:00"
END_TEST = "2026-01-01 00:00:00"

# Grid Search Parameters
# k now represents the multiplier for the threshold of a "directional move"
K_VALUES = [0.1, 0.2, 0.5, 0.8, 1.0] 
SEQLEN_VALUES = [3, 4, 5, 6, 7, 8]
MIN_CORRECT_PREDICTIONS = 100 

DATA_DIR = "/app/data/"

def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def fetch_binance_data(symbol, timeframe, start_str, end_str):
    ensure_dir(DATA_DIR)
    safe_symbol = symbol.replace('/', '_')
    filename = f"{safe_symbol}_{timeframe}_{start_str[:4]}_{end_str[:4]}.csv"
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
            if not candles: break
            candles = [c for c in candles if c[0] < end_ts]
            if not candles: break
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
    df.to_csv(filepath)
    return df

def resample_dataset(df_15m, offset_minutes=0):
    agg_dict = {'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'}
    offset_str = f'{offset_minutes}min'
    df_resampled = df_15m.resample('1h', offset=offset_str, closed='left', label='left').agg(agg_dict)
    df_resampled.dropna(inplace=True)
    return df_resampled

def prepare_buckets(train_df_list, test_df_list, k):
    """
    DIRECTIONAL LOGIC:
    Buckets are now signed integers.
    Bucket 0 = Flat (within +/- bucket_size)
    Bucket +1 = Up 1 unit, Bucket -1 = Down 1 unit
    """
    all_train_returns = []
    for df in train_df_list:
        all_train_returns.append(df['close'].pct_change().fillna(0))
    
    combined_train_returns = pd.concat(all_train_returns)
    # Average absolute move serves as the baseline unit
    avg_abs_return = combined_train_returns.abs().mean()
    
    bucket_size = avg_abs_return * k
    if bucket_size == 0: bucket_size = 1e-9 

    def get_buckets(df):
        returns = df['close'].pct_change().fillna(0)
        # Signed quantization: int(return / size)
        # e.g. return 0.5, size 0.2 -> 2 (Up)
        # e.g. return -0.5, size 0.2 -> -2 (Down)
        return (returns / bucket_size).astype(int)

    train_buckets_list = [get_buckets(df) for df in train_df_list]
    test_buckets_list = [get_buckets(df) for df in test_df_list]
    
    return train_buckets_list, test_buckets_list, bucket_size

def build_frequency_map(sequences_list, seqlen):
    freq_map = {}
    for sequence in sequences_list:
        seq_array = sequence.tolist()
        for i in range(len(seq_array) - seqlen + 1):
            window = tuple(seq_array[i : i + seqlen])
            prefix = window[:-1]
            target = window[-1]
            if prefix not in freq_map: freq_map[prefix] = {}
            if target not in freq_map[prefix]: freq_map[prefix][target] = 0
            freq_map[prefix][target] += 1
    return freq_map

def predict_most_frequent(prefix, freq_map):
    if prefix not in freq_map: return None
    candidates = freq_map[prefix]
    return max(candidates, key=candidates.get)

def evaluate_prediction(test_buckets_list, freq_map, seqlen):
    total_correct = 0
    total_incorrect = 0
    
    for test_buckets in test_buckets_list:
        test_seq = test_buckets.tolist()
        for i in range(seqlen - 1, len(test_seq)):
            # Predict the current step based on history
            prefix = tuple(test_seq[i - (seqlen - 1) : i])
            predicted_bucket = predict_most_frequent(prefix, freq_map)
            actual_next_bucket = test_seq[i]
            
            # LOGIC: DIRECTIONAL ACCURACY
            if predicted_bucket is not None:
                # Ignore Flat actuals (0)
                if actual_next_bucket == 0:
                    continue
                
                # Check Direction Matches
                pred_sign = np.sign(predicted_bucket)
                actual_sign = np.sign(actual_next_bucket)
                
                # We also ignore if model predicts 0 (Neutral)
                if pred_sign == 0:
                    continue

                if pred_sign == actual_sign:
                    total_correct += 1
                else:
                    total_incorrect += 1
                    
    total_valid = total_correct + total_incorrect
    if total_valid == 0: return 0.0, 0
    return total_correct / total_valid, total_correct

def upload_to_github(file_path, repo_owner, repo_name, token, branch="main"):
    if not token:
        print("Error: GITHUB_PAT not found in .env")
        return
    file_name = os.path.basename(file_path)
    print(f"Uploading {file_name}...")
    api_url = f"https://api.github.com/repos/{repo_owner}/{repo_name}/contents/{file_name}"
    headers = {"Authorization": f"token {token}", "Accept": "application/vnd.github.v3+json"}
    
    with open(file_path, "rb") as f:
        content = f.read()
    content_b64 = base64.b64encode(content).decode("utf-8")

    sha = None
    r_check = requests.get(api_url, headers=headers, params={"ref": branch})
    if r_check.status_code == 200:
        sha = r_check.json().get("sha")

    data = {
        "message": f"Update model {file_name} {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "content": content_b64,
        "branch": branch
    }
    if sha: data["sha"] = sha
    
    requests.put(api_url, headers=headers, data=json.dumps(data))
    print(f"✅ Uploaded {file_name}")

def run_multi_asset_search():
    print(f"Starting Multi-Asset Grid Search for: {SYMBOLS}")
    
    for symbol in SYMBOLS:
        print(f"\n{'='*60}")
        print(f"PROCESSING ASSET: {symbol}")
        print(f"{'='*60}")
        
        # Fetch Data
        df_train_raw = fetch_binance_data(symbol, TIMEFRAME, START_TRAIN, END_TRAIN)
        df_test_raw = fetch_binance_data(symbol, TIMEFRAME, END_TRAIN, END_TEST)
        
        # Resample
        offsets = [0, 15, 30, 45]
        train_dfs = [resample_dataset(df_train_raw, offset) for offset in offsets]
        test_dfs = [resample_dataset(df_test_raw, offset) for offset in offsets]
        
        best_acc = -1
        best_model_data = None
        
        # Grid Search
        print(f"{'K':<10} {'SeqLen':<10} {'Accuracy':<10} {'Correct':<10}")
        print("-" * 45)
        
        for k in K_VALUES:
            train_buckets_list, test_buckets_list, bucket_size = prepare_buckets(train_dfs, test_dfs, k)
            for seqlen in SEQLEN_VALUES:
                freq_map = build_frequency_map(train_buckets_list, seqlen)
                acc, correct_count = evaluate_prediction(test_buckets_list, freq_map, seqlen)
                print(f"{k:<10} {seqlen:<10} {acc:.4f}     {correct_count:<10}")
                
                if correct_count >= MIN_CORRECT_PREDICTIONS and acc > best_acc:
                    best_acc = acc
                    best_model_data = {
                        "weights": freq_map,
                        "config": {
                            "symbol": symbol,
                            "k": k,
                            "seqlen": seqlen,
                            "bucket_size": bucket_size,
                            "type": "directional"
                        },
                        "metrics": {"accuracy": acc, "correct": correct_count}
                    }

        # Save and Upload for THIS symbol
        if best_model_data:
            ticker = symbol.split('/')[0] # BTC, ETH, SOL
            filename = f"model_{ticker}.pkl"
            
            print(f"\n🏆 Best {symbol} Model: Acc {best_acc:.4f} | K {best_model_data['config']['k']}")
            
            with open(filename, "wb") as f:
                pickle.dump(best_model_data, f)
            
            upload_to_github(filename, GITHUB_OWNER, GITHUB_REPO, GITHUB_TOKEN, GITHUB_BRANCH)
        else:
            print(f"❌ No valid model found for {symbol}")

if __name__ == "__main__":
    run_multi_asset_search()
