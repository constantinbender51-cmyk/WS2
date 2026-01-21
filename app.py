import time
import json
import csv
import os
import requests
from datetime import datetime, timezone, timedelta

# --- Configuration ---
PREDICTION_URL = "https://workspace-production-9fae.up.railway.app/predictions"
ACTIVE_TRADES_FILE = "active_trades.json"
HISTORY_FILE = "trade_history.csv"

# Timeframe in minutes and their mapping to array indices
TIMEFRAMES = {
    0: {"label": "15m", "minutes": 15},
    1: {"label": "30m", "minutes": 30},
    2: {"label": "1h",  "minutes": 60},
    3: {"label": "4h",  "minutes": 240},
    4: {"label": "1d",  "minutes": 1440}
}

def load_active_trades():
    if os.path.exists(ACTIVE_TRADES_FILE):
        try:
            with open(ACTIVE_TRADES_FILE, 'r') as f:
                return json.load(f)
        except json.JSONDecodeError:
            return []
    return []

def save_active_trades(trades):
    with open(ACTIVE_TRADES_FILE, 'w') as f:
        json.dump(trades, f, indent=4)

def append_to_history(trade):
    file_exists = os.path.exists(HISTORY_FILE)
    with open(HISTORY_FILE, 'a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["Symbol", "Timeframe", "Direction", "OpenTime", "CloseTime", "OpenPrice", "ClosePrice", "PnL", "PnL_Percent"])
        
        # PnL Calculation: (New - Old) / Old
        pnl_val = (trade['close_price'] - trade['open_price']) / trade['open_price']
        
        # Invert PnL if signal was Short (assuming -1 indicates short, 1 indicates long)
        if trade['signal'] < 0:
            pnl_val = pnl_val * -1

        writer.writerow([
            trade['symbol'],
            trade['timeframe'],
            trade['signal'],
            datetime.fromtimestamp(trade['start_ts'], timezone.utc).isoformat(),
            datetime.fromtimestamp(trade['end_ts'], timezone.utc).isoformat(),
            trade['open_price'],
            trade['close_price'],
            round(pnl_val, 8),
            f"{round(pnl_val * 100, 4)}%"
        ])
    print(f"[HISTORY] Saved {trade['symbol']} {trade['timeframe']} | PnL: {round(pnl_val*100, 4)}%")

def get_binance_ohlc(symbol, interval, start_time_ms):
    """
    Fetches specific candle data from Binance.
    """
    url = "https://api.binance.com/api/v3/klines"
    params = {
        "symbol": symbol,
        "interval": interval,
        "startTime": start_time_ms,
        "limit": 1
    }
    try:
        r = requests.get(url, params=params)
        r.raise_for_status()
        data = r.json()
        if data:
            # Binance response: [Open Time, Open, High, Low, Close, Volume, Close Time, ...]
            return float(data[0][1]), float(data[0][4]) # Return Open, Close
    except Exception as e:
        print(f"[ERROR] Binance fetch failed for {symbol}: {e}")
    return None, None

def get_candle_times(current_time_dt, timeframe_minutes):
    """
    Calculates the start and end timestamp of the 'current' candle based on the timeframe.
    """
    # Align to 00:00
    total_minutes = current_time_dt.hour * 60 + current_time_dt.minute
    remainder = total_minutes % timeframe_minutes
    
    # Start of the current candle
    start_dt = current_time_dt - timedelta(minutes=remainder, seconds=current_time_dt.second, microseconds=current_time_dt.microsecond)
    
    # End of the current candle
    end_dt = start_dt + timedelta(minutes=timeframe_minutes)
    
    return int(start_dt.timestamp() * 1000), int(end_dt.timestamp() * 1000)

def process_expiries():
    """
    Checks active trades to see if their candle has closed.
    """
    active_trades = load_active_trades()
    remaining_trades = []
    current_ms = int(time.time() * 1000)
    
    # Allow a small buffer (e.g. 10 seconds) to ensure Binance has the data
    buffer_ms = 10000 

    for trade in active_trades:
        if current_ms > (trade['end_ts'] + buffer_ms):
            # Trade has expired, fetch result
            print(f"[EXPIRY] Processing expired trade: {trade['symbol']} {trade['timeframe']}")
            
            open_price, close_price = get_binance_ohlc(
                trade['symbol'], 
                trade['binance_interval'], 
                trade['start_ts']
            )
            
            if open_price is not None and close_price is not None:
                trade['open_price'] = open_price
                trade['close_price'] = close_price
                append_to_history(trade)
            else:
                print(f"[WARN] Could not fetch data for {trade['symbol']}, keeping in active list.")
                remaining_trades.append(trade)
        else:
            remaining_trades.append(trade)
            
    save_active_trades(remaining_trades)

def fetch_signals():
    print(f"[FETCH] Retrieveing signals at {datetime.now().strftime('%H:%M:%S')}")
    try:
        r = requests.get(PREDICTION_URL)
        r.raise_for_status()
        data = r.json()
        
        active_trades = load_active_trades()
        current_dt = datetime.now(timezone.utc)
        
        new_trades_count = 0

        for symbol, info in data.items():
            comp = info.get('comp', [])
            
            for idx, signal in enumerate(comp):
                if signal != 0: # Active Signal
                    tf_info = TIMEFRAMES[idx]
                    start_ms, end_ms = get_candle_times(current_dt, tf_info['minutes'])
                    
                    # Unique ID for the trade to prevent duplicates
                    trade_id = f"{symbol}_{tf_info['label']}_{start_ms}"
                    
                    # Check if already active
                    if any(t['id'] == trade_id for t in active_trades):
                        continue
                        
                    print(f"[SIGNAL] Found {symbol} {tf_info['label']} (Val: {signal})")
                    
                    # Store trade metadata (Prices fetched at expiry)
                    new_trade = {
                        "id": trade_id,
                        "symbol": symbol,
                        "signal": signal,
                        "timeframe": tf_info['label'],
                        "binance_interval": tf_info['label'], # Binance uses same format (15m, 1h)
                        "start_ts": start_ms,
                        "end_ts": end_ms
                    }
                    active_trades.append(new_trade)
                    new_trades_count += 1
        
        save_active_trades(active_trades)
        print(f"[UPDATE] Added {new_trades_count} new trades.")
        
    except Exception as e:
        print(f"[ERROR] Failed to fetch predictions: {e}")

def get_seconds_until_next_run():
    """
    Calculates seconds until the next XX:06, XX:21, XX:36, XX:51.
    """
    now = datetime.now()
    minutes = now.minute
    
    # Target minute markers
    targets = [6, 21, 36, 51]
    
    # Find next target
    next_target = None
    for t in targets:
        if minutes < t:
            next_target = t
            break
            
    if next_target is None:
        # Wrap around to next hour's first target (06)
        wait_minutes = (60 - minutes) + targets[0]
    else:
        wait_minutes = next_target - minutes
        
    target_time = now + timedelta(minutes=wait_minutes)
    target_time = target_time.replace(second=0, microsecond=0)
    
    seconds_wait = (target_time - now).total_seconds()
    return seconds_wait

def main():
    print("--- Crypto Signal Bot Started ---")
    print("Syncing to 6 minutes after every 15m interval (XX:06, XX:21, XX:36, XX:51)...")
    
    while True:
        # 1. Process any pending expiries first (check regardless of schedule)
        process_expiries()
        
        # 2. Wait for schedule
        sleep_sec = get_seconds_until_next_run()
        print(f"[WAIT] Sleeping for {int(sleep_sec)} seconds...")
        time.sleep(sleep_sec)
        
        # 3. Double check synchronization (ensure we didn't wake up slightly early)
        time.sleep(1) 
        
        # 4. Run Cycle
        fetch_signals()
        
        # Small sleep to prevent double execution within the same second
        time.sleep(5)

if __name__ == "__main__":
    main()
