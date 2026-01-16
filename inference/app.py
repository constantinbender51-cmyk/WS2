import os
import ccxt
import pandas as pd
import numpy as np
import pickle
import requests
from datetime import datetime, timedelta

# --- Configuration ---
MODEL_PATH = "best_model.pkl"
GITHUB_URL = "https://raw.githubusercontent.com/constantinbender51-cmyk/model-2/main/best_model.pkl"

def load_model():
    """
    Loads the model from disk. If missing, attempts to download from GitHub.
    """
    if not os.path.exists(MODEL_PATH):
        print(f"Model not found locally. Downloading from {GITHUB_URL}...")
        try:
            r = requests.get(GITHUB_URL)
            r.raise_for_status()
            with open(MODEL_PATH, "wb") as f:
                f.write(r.content)
            print("Model downloaded successfully.")
        except Exception as e:
            raise Exception(f"Could not load model: {e}")

    with open(MODEL_PATH, "rb") as f:
        model_data = pickle.load(f)
    
    return model_data

def fetch_recent_data(symbol, timeframe, lookback_days=2):
    """
    Fetches enough recent 15m data to construct the required hourly sequence.
    """
    exchange = ccxt.binance()
    # Fetch slightly more than needed to ensure we can resample correctly
    since = exchange.milliseconds() - (lookback_days * 24 * 60 * 60 * 1000)
    
    candles = exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=1000)
    
    df = pd.DataFrame(candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    return df

def get_current_offset_data(df_15m, target_seqlen):
    """
    Resamples 15m data into 1h candles based on the *current* time's alignment.
    If it's 10:15, it builds 1h candles ending at 10:15 (09:15-10:15).
    """
    # 1. Determine the offset based on the last available timestamp
    last_time = df_15m.index[-1]
    current_minute = last_time.minute
    
    # We support 0, 15, 30, 45. 
    # If the data ends at 10:14, we can't close the 10:15 candle yet.
    # We assume the script is run right after a 15m candle close.
    
    offset_str = f'{current_minute}min'
    print(f"Live Inference: Detected Offset {offset_str}")

    # 2. Resample
    # We use the same logic as training: closed='left', label='left'
    agg_dict = {'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'}
    
    # We resample the whole buffer, then take the tail
    df_resampled = df_15m.resample('1h', offset=offset_str, closed='left', label='left').agg(agg_dict)
    df_resampled.dropna(inplace=True)
    
    # We need exactly `seqlen` candles (the sequence to predict FROM)
    # The training logic used a sequence of length `seqlen` where the last item was the target.
    # For inference, we need a sequence of length `seqlen - 1` (the prefix) to predict the NEXT one.
    # However, to be safe, let's grab the last `seqlen` rows to visualize context.
    
    return df_resampled

def tokens_from_prices(df, bucket_size):
    """
    Converts prices -> returns -> buckets using the MODEL'S bucket_size.
    """
    returns = df['close'].pct_change().fillna(0)
    # IMPORTANT: Use the loaded bucket_size, do not recalculate!
    buckets = np.floor(returns.abs() / bucket_size).astype(int) + 1
    return buckets

def predict_next(prefix, freq_map):
    if prefix not in freq_map:
        return None, None
    
    candidates = freq_map[prefix]
    # Find the most frequent next bucket
    predicted_bucket = max(candidates, key=candidates.get)
    confidence = candidates[predicted_bucket] / sum(candidates.values())
    
    return predicted_bucket, confidence

def main():
    # 1. Load the Brain
    try:
        model = load_model()
    except Exception as e:
        print(str(e))
        return

    config = model['config']
    weights = model['weights']
    
    print(f"Model Loaded.")
    print(f"Config: K={config['k']}, SeqLen={config['seqlen']}, BucketSize={config['bucket_size']:.6f}")
    
    # 2. Get Live Data
    print("Fetching live data...")
    df_15m = fetch_recent_data(config['symbol'], "15m")
    
    # 3. Resample to current Offset
    # We need a sequence length of (SeqLen - 1) to form a Prefix
    required_history = config['seqlen'] + 5 # grab a few extra for safety
    df_hourly = get_current_offset_data(df_15m, required_history)
    
    if len(df_hourly) < config['seqlen']:
        print("Not enough history to make a prediction yet.")
        return

    # 4. Tokenize
    # We must calculate returns, so we need at least 1 extra row of price history
    buckets_series = tokens_from_prices(df_hourly, config['bucket_size'])
    
    # Extract the Prefix (The last SeqLen-1 buckets)
    # If SeqLen is 5, we need the last 4 buckets to predict the 5th.
    prefix_len = config['seqlen'] - 1
    current_prefix = tuple(buckets_series.iloc[-prefix_len:].tolist())
    current_prices = df_hourly.iloc[-prefix_len:]
    
    # 5. Predict
    last_bucket = current_prefix[-1]
    prediction, confidence = predict_next(current_prefix, weights)
    
    print("\n--- INFERENCE REPORT ---")
    print(f"Time: {df_hourly.index[-1]} (Close of last candle)")
    print(f"Current Sequence: {current_prefix}")
    print(f"Current Bucket:   {last_bucket}")
    
    if prediction is None:
        print("Result: NO SIGNAL (Pattern not seen in training)")
    else:
        print(f"Predicted Next:   {prediction}")
        print(f"Confidence:       {confidence:.2%}")
        
        # 6. Interpret Signal
        if prediction > last_bucket:
            print(">> SIGNAL: VOLATILITY EXPANSION (Long/Short depending on trend)")
            print(">> Action: The model expects the next candle to be LARGER than the current one.")
        elif prediction < last_bucket:
            print(">> SIGNAL: CONTRACTION")
            print(">> Action: Expect smaller movement.")
        else:
            print(">> SIGNAL: NEUTRAL")

if __name__ == "__main__":
    main()
