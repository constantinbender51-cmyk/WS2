import sys
import os
import ccxt
import pandas as pd
import numpy as np
import pickle
import requests
from datetime import datetime, timedelta

# --- Usage ---
# Run: python inference.py BTC
# Run: python inference.py SOL
# Run: python inference.py (Defaults to ETH)

TICKER = sys.argv[1] if len(sys.argv) > 1 else "ETH"
MODEL_FILENAME = f"model_{TICKER}.pkl"
GITHUB_URL = f"https://raw.githubusercontent.com/constantinbender51-cmyk/model-2/main/{MODEL_FILENAME}"

def load_model():
    if not os.path.exists(MODEL_FILENAME):
        print(f"Downloading {MODEL_FILENAME} from GitHub...")
        try:
            r = requests.get(GITHUB_URL)
            r.raise_for_status()
            with open(MODEL_FILENAME, "wb") as f:
                f.write(r.content)
        except Exception as e:
            print(f"Could not download model for {TICKER}: {e}")
            sys.exit(1)

    with open(MODEL_FILENAME, "rb") as f:
        return pickle.load(f)

def fetch_recent_data(symbol):
    exchange = ccxt.binance()
    # Fetch data
    since = exchange.milliseconds() - (5 * 24 * 60 * 60 * 1000)
    candles = exchange.fetch_ohlcv(symbol, '15m', since=since, limit=1000)
    
    df = pd.DataFrame(candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)

    # --- SAFETY CHECK: DROP STUB CANDLE ---
    now = datetime.utcnow()
    current_minute = now.minute
    minute_floor = (current_minute // 15) * 15
    current_open_candle_time = now.replace(minute=minute_floor, second=0, microsecond=0)

    if not df.empty and df.index[-1] >= current_open_candle_time:
        df = df.iloc[:-1]

    return df

def get_current_offset_data(df_15m, target_seqlen):
    last_time = df_15m.index[-1]
    offset_str = f'{last_time.minute}min'
    agg = {'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'}
    df_resampled = df_15m.resample('1h', offset=offset_str, closed='left', label='left').agg(agg).dropna()
    return df_resampled

def main():
    print(f"--- 🚀 INFERENCE: {TICKER}/USDT ---")
    
    # 1. Load Model
    model = load_model()
    cfg = model['config']
    weights = model['weights']
    
    print(f"Model Config: K={cfg['k']} | SeqLen={cfg['seqlen']} | BucketSize={cfg['bucket_size']:.6f}")
    
    # 2. Fetch Data
    symbol = cfg['symbol'] # e.g. 'BTC/USDT'
    df_15m = fetch_recent_data(symbol)
    df_hourly = get_current_offset_data(df_15m, cfg['seqlen'])
    
    if len(df_hourly) < cfg['seqlen']:
        print("Not enough history.")
        return

    # 3. Tokenize (DIRECTIONAL LOGIC)
    returns = df_hourly['close'].pct_change().fillna(0)
    # Signed integers: int(return / size)
    buckets = (returns / cfg['bucket_size']).astype(int).tolist()
    
    # 4. Predict
    prefix_len = cfg['seqlen'] - 1
    current_prefix = tuple(buckets[-prefix_len:])
    
    if current_prefix in weights:
        candidates = weights[current_prefix]
        prediction = max(candidates, key=candidates.get)
        confidence = candidates[prediction] / sum(candidates.values())
        
        print(f"\nSequence: {current_prefix}")
        print(f"Predicted Bucket: {prediction} (Conf: {confidence:.1%})")
        
        # Output Signals for user parsing
        if prediction > 0:
            print(f">> SIGNAL {TICKER}: 1 (LONG)")
        elif prediction < 0:
            print(f">> SIGNAL {TICKER}: -1 (SHORT)")
        else:
            print(f">> SIGNAL {TICKER}: 0 (NEUTRAL)")
            
    else:
        print(f"\nSequence {current_prefix} not seen in training. No prediction.")

if __name__ == "__main__":
    main()
