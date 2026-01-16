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
# GitHub Settings
GITHUB_OWNER = "constantinbender51-cmyk"
GITHUB_REPO = "model-2"
GITHUB_BRANCH = "main"
MODEL_FILENAME = "best_model.pkl"

# Data Storage
DATA_DIR = "/app/data/"
TRAIN_FILENAME = "eth_train_15m_2020_2025.csv"
TEST_FILENAME = "eth_test_15m_2025_2026.csv"

# Exchange / Symbol Settings
SYMBOL = 'ETH/USDT'
TIMEFRAME = '15m'

# Date Ranges
START_TRAIN = "2020-01-01 00:00:00"
END_TRAIN = "2025-01-01 00:00:00"
END_TEST = "2026-01-01 00:00:00"

# Grid Search Parameters
K_VALUES = [0.05, 0.1, 0.2, 0.4, 0.5, 0.8]
SEQLEN_VALUES = [3, 4, 5, 6, 7, 8]

# Qualification Thresholds
MIN_CORRECT_PREDICTIONS = 200  

def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def fetch_binance_data(symbol, timeframe, start_str, end_str, filename):
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
    print(f"Saving to {filepath}...")
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
    Returns bucket lists AND the calculated bucket_size (float).
    """
    all_train_returns = []
    for df in train_df_list:
        all_train_returns.append(df['close'].pct_change().fillna(0))
    
    combined_train_returns = pd.concat(all_train_returns)
    avg_abs_return = combined_train_returns.abs().mean()
    
    # Freeze the bucket size for this specific model configuration
    bucket_size = avg_abs_return * k
    if bucket_size == 0: bucket_size = 1e-9 

    def get_buckets(df):
        returns = df['close'].pct_change().fillna(0)
        return np.floor(returns.abs() / bucket_size).astype(int) + 1

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
            current_idx = i - 1 
            current_bucket = test_seq[current_idx]
            prefix = tuple(test_seq[i - (seqlen - 1) : i])
            predicted_bucket = predict_most_frequent(prefix, freq_map)
            actual_next_bucket = test_seq[i]
            
            if predicted_bucket is not None:
                if predicted_bucket > current_bucket:
                    if actual_next_bucket > current_bucket:
                        total_correct += 1
                    elif actual_next_bucket < current_bucket:
                        total_incorrect += 1
            
    total_valid = total_correct + total_incorrect
    if total_valid == 0: return 0.0, 0
    return total_correct / total_valid, total_correct

def upload_to_github(file_path, repo_owner, repo_name, token, branch="main"):
    """
    Uploads a file to GitHub, updating it if it already exists (using SHA).
    """
    if not token:
        print("Error: GITHUB_PAT not found in .env")
        return

    print(f"Uploading {file_path} to GitHub...")
    file_name = os.path.basename(file_path)
    api_url = f"https://api.github.com/repos/{repo_owner}/{repo_name}/contents/{file_name}"
    
    headers = {
        "Authorization": f"token {token}",
        "Accept": "application/vnd.github.v3+json"
    }

    # 1. Read and encode file
    with open(file_path, "rb") as f:
        content = f.read()
    content_b64 = base64.b64encode(content).decode("utf-8")

    # 2. Check if file exists to get SHA (needed for update)
    sha = None
    r_check = requests.get(api_url, headers=headers, params={"ref": branch})
    if r_check.status_code == 200:
        sha = r_check.json().get("sha")
        print("File exists on GitHub, performing update...")

    # 3. Prepare Payload
    data = {
        "message": f"Update model weights {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "content": content_b64,
        "branch": branch
    }
    if sha:
        data["sha"] = sha

    # 4. Upload
    r = requests.put(api_url, headers=headers, data=json.dumps(data))
    if r.status_code in [200, 201]:
        print(f"✅ Success! Model uploaded to https://github.com/{repo_owner}/{repo_name}/blob/{branch}/{file_name}")
    else:
        print(f"❌ Failed to upload. Status: {r.status_code}, Response: {r.text}")

def run_grid_search():
    df_train_raw = fetch_binance_data(SYMBOL, TIMEFRAME, START_TRAIN, END_TRAIN, TRAIN_FILENAME)
    df_test_raw = fetch_binance_data(SYMBOL, TIMEFRAME, END_TRAIN, END_TEST, TEST_FILENAME)
    
    print("\nResampling data to 4x 1h offset streams (00, 15, 30, 45)...")
    offsets = [0, 15, 30, 45]
    train_dfs = [resample_dataset(df_train_raw, offset) for offset in offsets]
    test_dfs = [resample_dataset(df_test_raw, offset) for offset in offsets]

    print("\nStarting Grid Search...")
    print(f"{'K':<10} {'SeqLen':<10} {'Accuracy':<10} {'Correct':<10}")
    print("-" * 45)

    best_acc = -1
    best_model_data = None

    for k in K_VALUES:
        # We now capture bucket_size to save it with the model
        train_buckets_list, test_buckets_list, bucket_size = prepare_buckets(train_dfs, test_dfs, k)
        
        for seqlen in SEQLEN_VALUES:
            freq_map = build_frequency_map(train_buckets_list, seqlen)
            acc, correct_count = evaluate_prediction(test_buckets_list, freq_map, seqlen)
            
            print(f"{k:<10} {seqlen:<10} {acc:.4f}     {correct_count:<10}")
            
            if correct_count >= MIN_CORRECT_PREDICTIONS:
                if acc > best_acc:
                    best_acc = acc
                    # Save everything needed for inference
                    best_model_data = {
                        "weights": freq_map,  # The core "model"
                        "config": {
                            "k": k,
                            "seqlen": seqlen,
                            "bucket_size": bucket_size,
                            "symbol": SYMBOL,
                            "timeframe_source": TIMEFRAME,
                            "offset_strategy": "multi-offset-1h"
                        },
                        "metrics": {
                            "accuracy": acc,
                            "correct_predictions": correct_count
                        }
                    }

    print("-" * 45)
    
    if best_model_data:
        print(f"Best Found: Acc: {best_model_data['metrics']['accuracy']:.4f} | K: {best_model_data['config']['k']}")
        
        # Save locally
        with open(MODEL_FILENAME, "wb") as f:
            pickle.dump(best_model_data, f)
        print(f"Model saved locally to {MODEL_FILENAME}")

        # Upload to GitHub
        upload_to_github(MODEL_FILENAME, GITHUB_OWNER, GITHUB_REPO, GITHUB_TOKEN, GITHUB_BRANCH)
        
    else:
        print(f"No configuration met the threshold of {MIN_CORRECT_PREDICTIONS} correct predictions.")

if __name__ == "__main__":
    run_grid_search()
