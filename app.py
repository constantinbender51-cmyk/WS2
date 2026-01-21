import time
import json
import csv
import os
import base64
import requests
from datetime import datetime, timezone, timedelta

# --- Configuration ---
PREDICTION_URL = "https://workspace-production-9fae.up.railway.app/predictions"
ACTIVE_TRADES_FILE = "active_trades.json"
HISTORY_FILE = "trade_history.csv"

# GitHub Configuration
GITHUB_REPO = "constantinbender51-cmyk/Models"
GITHUB_FILE_PATH = "data/logs/trade_history.csv"
GITHUB_BRANCH = "main"
GITHUB_TOKEN = os.environ.get("PAT")  # Reads from environment variable

# Timeframe mapping
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

def upload_to_github():
    """
    Reads the local CSV and pushes it to GitHub.
    Handles both creating a new file and updating an existing one (using SHA).
    """
    if not GITHUB_TOKEN:
        print("[ERROR] GitHub 'PAT' environment variable not found.")
        return

    if not os.path.exists(HISTORY_FILE):
        return

    print(f"[GITHUB] Uploading {HISTORY_FILE}...")
    
    url = f"https://api.github.com/repos/{GITHUB_REPO}/contents/{GITHUB_FILE_PATH}"
    headers = {
        "Authorization": f"token {GITHUB_TOKEN}",
        "Accept": "application/vnd.github.v3+json"
    }

    # 1. Read local file content
    with open(HISTORY_FILE, "rb") as f:
        content = f.read()
    b64_content = base64.b64encode(content).decode("utf-8")

    # 2. Check if file exists on GitHub to get SHA (needed for updates)
    sha = None
    try:
        r_check = requests.get(url, headers=headers, params={"ref": GITHUB_BRANCH})
        if r_check.status_code == 200:
            sha = r_check.json().get("sha")
    except Exception as e:
        print(f"[GITHUB] Error checking file existence: {e}")

    # 3. Prepare Payload
    payload = {
        "message": f"Update trade history {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "content": b64_content,
        "branch": GITHUB_BRANCH
    }
    if sha:
        payload["sha"] = sha

    # 4. Upload
    try:
        r_put = requests.put(url, headers=headers, json=payload)
        if r_put.status_code in [200, 201]:
            print("[GITHUB] Upload successful.")
        else:
            print(f"[GITHUB] Upload failed: {r_put.status_code} {r_put.text}")
    except Exception as e:
        print(f"[GITHUB] Exception during upload: {e}")

def append_to_history(trade):
    file_exists = os.path.exists(HISTORY_FILE)
    
    # Calculate PnL
    pnl_val = (trade['close_price'] - trade['open_price']) / trade['open_price']
    if trade['signal'] < 0: # Short logic
        pnl_val = pnl_val * -1

    # Write to CSV
    with open(HISTORY_FILE, 'a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["Symbol", "Timeframe", "Direction", "OpenTime", "CloseTime", "OpenPrice", "ClosePrice", "PnL", "PnL_Percent"])

        writer.writerow([
            trade['symbol'],
            trade['timeframe'],
            trade['signal'],
            datetime.fromtimestamp(trade['start_ts'] / 1000, timezone.utc).isoformat(),
            datetime.fromtimestamp(trade['end_ts'] / 1000, timezone.utc).isoformat(),
            trade['open_price'],
            trade['close_price'],
            round(pnl_val, 8),
            f"{round(pnl_val * 100, 4)}%"
        ])
    
    print(f"[HISTORY] Saved {trade['symbol']} {trade['timeframe']} | PnL: {round(pnl_val*100, 4)}%")
    
    # Trigger GitHub Upload after saving
    upload_to_github()

def get_binance_ohlc(symbol, interval, start_time_ms):
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
            return float(data[0][1]), float(data[0][4]) # Open, Close
    except Exception as e:
        print(f"[ERROR] Binance fetch failed for {symbol}: {e}")
    return None, None

def get_candle_times(current_time_dt, timeframe_minutes):
    total_minutes = current_time_dt.hour * 60 + current_time_dt.minute
    remainder = total_minutes % timeframe_minutes
    start_dt = current_time_dt - timedelta(minutes=remainder, seconds=current_time_dt.second, microseconds=current_time_dt.microsecond)
    end_dt = start_dt + timedelta(minutes=timeframe_minutes)
    return int(start_dt.timestamp() * 1000), int(end_dt.timestamp() * 1000)

def process_expiries():
    active_trades = load_active_trades()
    remaining_trades = []
    current_ms = int(time.time() * 1000)
    buffer_ms = 10000 

    for trade in active_trades:
        if current_ms > (trade['end_ts'] + buffer_ms):
            print(f"[EXPIRY] Processing {trade['symbol']} {trade['timeframe']}")
            open_price, close_price = get_binance_ohlc(trade['symbol'], trade['binance_interval'], trade['start_ts'])
            
            if open_price is not None and close_price is not None:
                trade['open_price'] = open_price
                trade['close_price'] = close_price
                append_to_history(trade)
            else:
                print(f"[WARN] No data for {trade['symbol']}, keeping active.")
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
                if signal != 0:
                    tf_info = TIMEFRAMES[idx]
                    start_ms, end_ms = get_candle_times(current_dt, tf_info['minutes'])
                    trade_id = f"{symbol}_{tf_info['label']}_{start_ms}"
                    
                    if any(t['id'] == trade_id for t in active_trades):
                        continue
                        
                    print(f"[SIGNAL] Found {symbol} {tf_info['label']} ({signal})")
                    active_trades.append({
                        "id": trade_id,
                        "symbol": symbol,
                        "signal": signal,
                        "timeframe": tf_info['label'],
                        "binance_interval": tf_info['label'],
                        "start_ts": start_ms,
                        "end_ts": end_ms
                    })
                    new_trades_count += 1
        
        save_active_trades(active_trades)
        print(f"[UPDATE] Added {new_trades_count} new trades.")
        
    except Exception as e:
        print(f"[ERROR] Failed to fetch predictions: {e}")

def get_seconds_until_next_run():
    now = datetime.now()
    minutes = now.minute
    targets = [6, 21, 36, 51]
    
    next_target = None
    for t in targets:
        if minutes < t:
            next_target = t
            break
            
    if next_target is None:
        wait_minutes = (60 - minutes) + targets[0]
    else:
        wait_minutes = next_target - minutes
        
    target_time = now + timedelta(minutes=wait_minutes)
    target_time = target_time.replace(second=0, microsecond=0)
    return (target_time - now).total_seconds()

def main():
    print("--- Crypto Signal Bot with GitHub Sync ---")
    
    while True:
        process_expiries()
        sleep_sec = get_seconds_until_next_run()
        print(f"[WAIT] Sleeping {int(sleep_sec)}s...")
        time.sleep(sleep_sec)
        time.sleep(1) 
        fetch_signals()
        time.sleep(5)

if __name__ == "__main__":
    main()
