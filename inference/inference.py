import sys
import os
import ccxt
import pandas as pd
import numpy as np
import pickle
import requests
from datetime import datetime, timedelta

TICKER = sys.argv[1] if len(sys.argv) > 1 else "ETH"
MODEL_FILENAME = f"model_{TICKER}.pkl"
GITHUB_URL = f"https://raw.githubusercontent.com/constantinbender51-cmyk/model-2/main/{MODEL_FILENAME}"

def load_model():
    if not os.path.exists(MODEL_FILENAME):
        try:
            r = requests.get(GITHUB_URL)
            r.raise_for_status()
            with open(MODEL_FILENAME, "wb") as f:
                f.write(r.content)
        except Exception as e:
            sys.exit(1)

    with open(MODEL_FILENAME, "rb") as f:
        return pickle.load(f)

def fetch_recent_data(symbol):
    exchange = ccxt.binance()
    since = exchange.milliseconds() - (5 * 24 * 60 * 60 * 1000)
    candles = exchange.fetch_ohlcv(symbol, '15m', since=since, limit=1000)
    
    df = pd.DataFrame(candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)

    now = datetime.utcnow()
    minute_floor = (now.minute // 15) * 15
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
    print(f"--- 🚀 DIRECTIONAL INFERENCE: {TICKER}/USDT ---")
    
    model = load_model()
    cfg = model['config']
    weights = model['weights']
    
    symbol = cfg['symbol']
    df_15m = fetch_recent_data(symbol)
    df_hourly = get_current_offset_data(df_15m, cfg['seqlen'])
    
    if len(df_hourly) < cfg['seqlen']:
        print("Not enough history.")
        return

    # UPDATED: Use raw returns (no .abs()) for directional bucketing
    returns = df_hourly['close'].pct_change().fillna(0)
    buckets = np.floor(returns / cfg['bucket_size']).astype(int).tolist()
    
    prefix_len = cfg['seqlen'] - 1
    current_prefix = tuple(buckets[-prefix_len:])
    
    if current_prefix in weights:
        candidates = weights[current_prefix]
        prediction = max(candidates, key=candidates.get)
        confidence = candidates[prediction] / sum(candidates.values())
        
        print(f"\nSequence: {current_prefix}")
        print(f"PREDICTION BUCKET: {prediction} (Conf: {confidence:.1%})")
        
        # UPDATED: Directional Logic
        if prediction > 0:
            print(">> SIGNAL: BULLISH (Expected Price Up)")
        elif prediction < 0:
            print(">> SIGNAL: BEARISH (Expected Price Down)")
        else:
            print(">> SIGNAL: NEUTRAL (No significant movement)")
    else:
        print(f"\nSequence {current_prefix} not seen in training.")

if __name__ == "__main__":
    main()
