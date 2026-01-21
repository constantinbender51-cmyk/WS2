import time
import json
import csv
import os
import base64
import requests
from datetime import datetime, timezone, timedelta

# --- Configuration ---
# Existing Bot Config
PREDICTION_URL = "https://workspace-production-9fae.up.railway.app/predictions"
ACTIVE_TRADES_FILE = "active_trades.json"
HISTORY_FILE = "trade_history.csv"

# New ETH Bot Config
ETH_URL = "https://web-production-73e1d.up.railway.app/"
ETH_STATE_FILE = "eth_state.json"
ETH_HISTORY_FILE = "eth_history.csv"

# GitHub Configuration
GITHUB_REPO = "constantinbender51-cmyk/Models"
GITHUB_FILE_PATH_MAIN = "data/logs/trade_history.csv"
GITHUB_FILE_PATH_ETH = "data/logs/eth_history.csv"
GITHUB_BRANCH = "main"
GITHUB_TOKEN = os.environ.get("PAT")

# Timeframe mapping
TIMEFRAMES = {
    0: {"label": "15m", "minutes": 15},
    1: {"label": "30m", "minutes": 30},
    2: {"label": "1h",  "minutes": 60},
    3: {"label": "4h",  "minutes": 240},
    4: {"label": "1d",  "minutes": 1440}
}

# --- Helpers ---

def get_binance_price(symbol, timestamp_ms=None):
    """
    Fetches price from Binance. 
    If timestamp_ms is provided, fetches historical open price closest to time.
    If None, fetches current price.
    """
    url = "https://api.binance.com/api/v3/klines"
    if timestamp_ms is None:
        # Get current price via ticker if no time specified (faster)
        try:
            r = requests.get(f"https://api.binance.com/api/v3/ticker/price?symbol={symbol}")
            r.raise_for_status()
            return float(r.json()["price"])
        except:
            return None

    # Historical
    params = {
        "symbol": symbol,
        "interval": "1m",
        "startTime": timestamp_ms,
        "limit": 1
    }
    try:
        r = requests.get(url, params=params)
        r.raise_for_status()
        data = r.json()
        if data:
            return float(data[0][1]) # Open price of that minute
    except Exception as e:
        print(f"[ERROR] Binance fetch failed for {symbol}: {e}")
    return None

def upload_to_github(local_file, remote_path):
    """
    Generic uploader for both history files.
    """
    if not GITHUB_TOKEN:
        print("[ERROR] GitHub 'PAT' environment variable not found.")
        return

    if not os.path.exists(local_file):
        return

    print(f"[GITHUB] Uploading {local_file} to {remote_path}...")
    
    url = f"https://api.github.com/repos/{GITHUB_REPO}/contents/{remote_path}"
    headers = {
        "Authorization": f"token {GITHUB_TOKEN}",
        "Accept": "application/vnd.github.v3+json"
    }

    try:
        with open(local_file, "rb") as f:
            content = f.read()
        b64_content = base64.b64encode(content).decode("utf-8")

        sha = None
        try:
            r_check = requests.get(url, headers=headers, params={"ref": GITHUB_BRANCH})
            if r_check.status_code == 200:
                sha = r_check.json().get("sha")
        except:
            pass

        payload = {
            "message": f"Update {local_file} {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            "content": b64_content,
            "branch": GITHUB_BRANCH
        }
        if sha:
            payload["sha"] = sha

        r_put = requests.put(url, headers=headers, json=payload)
        if r_put.status_code in [200, 201]:
            print(f"[GITHUB] Upload of {local_file} successful.")
        else:
            print(f"[GITHUB] Upload failed: {r_put.status_code} {r_put.text}")
    except Exception as e:
        print(f"[GITHUB] Exception during upload: {e}")

# --- Legacy/Main Bot Functions ---

def load_active_trades():
    if os.path.exists(ACTIVE_TRADES_FILE):
        try:
            with open(ACTIVE_TRADES_FILE, 'r') as f:
                return json.load(f)
        except:
            return []
    return []

def save_active_trades(trades):
    with open(ACTIVE_TRADES_FILE, 'w') as f:
        json.dump(trades, f, indent=4)

def append_to_history(trade):
    file_exists = os.path.exists(HISTORY_FILE)
    
    pnl_val = (trade['close_price'] - trade['open_price']) / trade['open_price']
    if trade['signal'] < 0:
        pnl_val = pnl_val * -1

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
    upload_to_github(HISTORY_FILE, GITHUB_FILE_PATH_MAIN)

def process_expiries():
    active_trades = load_active_trades()
    remaining_trades = []
    current_ms = int(time.time() * 1000)
    buffer_ms = 10000 

    changed = False
    for trade in active_trades:
        if current_ms > (trade['end_ts'] + buffer_ms):
            print(f"[EXPIRY] Processing {trade['symbol']} {trade['timeframe']}")
            # Reuse get_binance_price logic
            open_price = get_binance_price(trade['symbol'], trade['start_ts'])
            close_price = get_binance_price(trade['symbol'], current_ms) # Approx close at current time
            
            # Fallback if get_binance_price used the generic one which returns single float
            if open_price is not None and close_price is not None:
                trade['open_price'] = open_price
                trade['close_price'] = close_price
                append_to_history(trade)
                changed = True
            else:
                print(f"[WARN] No data for {trade['symbol']}, keeping active.")
                remaining_trades.append(trade)
        else:
            remaining_trades.append(trade)
            
    if changed or len(remaining_trades) != len(active_trades):
        save_active_trades(remaining_trades)

def fetch_signals():
    print(f"[FETCH] Main Signals at {datetime.now().strftime('%H:%M:%S')}")
    try:
        r = requests.get(PREDICTION_URL)
        r.raise_for_status()
        data = r.json()
        
        active_trades = load_active_trades()
        current_dt = datetime.now(timezone.utc)
        new_trades_count = 0

        # Helper for candle times
        def get_candle_times(curr_dt, mins):
            total = curr_dt.hour * 60 + curr_dt.minute
            rem = total % mins
            s_dt = curr_dt - timedelta(minutes=rem, seconds=curr_dt.second, microseconds=curr_dt.microsecond)
            e_dt = s_dt + timedelta(minutes=mins)
            return int(s_dt.timestamp() * 1000), int(e_dt.timestamp() * 1000)

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
                        "start_ts": start_ms,
                        "end_ts": end_ms
                    })
                    new_trades_count += 1
        
        save_active_trades(active_trades)
        print(f"[UPDATE] Added {new_trades_count} new main trades.")
        
    except Exception as e:
        print(f"[ERROR] Failed to fetch predictions: {e}")

# --- New ETH Bot Functions ---

def load_eth_state():
    if os.path.exists(ETH_STATE_FILE):
        try:
            with open(ETH_STATE_FILE, 'r') as f:
                return json.load(f)
        except:
            pass
    return {
        "last_net_sum": None,
        "wins": 0,
        "losses": 0,
        "active_trades": []
    }

def save_eth_state(state):
    with open(ETH_STATE_FILE, 'w') as f:
        json.dump(state, f, indent=4)

def run_eth_cycle():
    print(f"[ETH] Running cycle at {datetime.now().strftime('%H:%M:%S')}")
    state = load_eth_state()
    active_trades = state.get("active_trades", [])
    
    # 1. Process Expiries (Wait 24h from update date)
    now = datetime.now(timezone.utc)
    remaining_trades = []
    trades_closed = False
    
    for trade in active_trades:
        # Parse stored ISO time
        trade_start_dt = datetime.fromisoformat(trade['start_time'])
        # Ensure timezone awareness if lost
        if trade_start_dt.tzinfo is None:
            trade_start_dt = trade_start_dt.replace(tzinfo=timezone.utc)
            
        expiry_dt = trade_start_dt + timedelta(hours=24)
        
        if now >= expiry_dt:
            print(f"[ETH] Closing trade from {trade['start_time']}")
            # Get close price at expiry time
            close_price = get_binance_price("ETHUSDT", int(expiry_dt.timestamp() * 1000))
            
            if close_price and trade['open_price']:
                pnl = (close_price - trade['open_price']) / trade['open_price']
                if trade['direction'] == -1:
                    pnl = -pnl
                
                # Check noise threshold (1%)
                is_noise = abs(pnl) < 0.01
                
                if not is_noise:
                    if pnl > 0:
                        state['wins'] += 1
                        res = "WIN"
                    else:
                        state['losses'] += 1
                        res = "LOSS"
                else:
                    res = "NOISE"

                # Log to CSV
                file_exists = os.path.exists(ETH_HISTORY_FILE)
                with open(ETH_HISTORY_FILE, 'a', newline='') as f:
                    writer = csv.writer(f)
                    if not file_exists:
                        writer.writerow(["Symbol", "Direction", "OpenTime", "CloseTime", "OpenPrice", "ClosePrice", "PnL", "Result"])
                    writer.writerow([
                        "ETHUSDT", trade['direction'], trade['start_time'], expiry_dt.isoformat(),
                        trade['open_price'], close_price, round(pnl, 5), res
                    ])
                trades_closed = True
                print(f"[ETH] Result: {res} | PnL: {round(pnl*100, 2)}%")
            else:
                # Retry next loop if data fetch failed
                print("[ETH] Failed to fetch close price, retrying later.")
                remaining_trades.append(trade)
        else:
            remaining_trades.append(trade)
    
    state['active_trades'] = remaining_trades

    # 2. Fetch New Signal
    try:
        r = requests.get(ETH_URL)
        r.raise_for_status()
        data = r.json()
        
        # Parse data
        current_net_sum = data.get("netSum")
        last_update_str = data.get("lastUpdate") # "2026-01-21 06:32:02.190657"
        
        # Parse timestamp string to datetime object (assume UTC or local? Usually UTC in APIs, but format lacks Z)
        # We will treat it as UTC for consistency
        try:
            update_dt = datetime.strptime(last_update_str, "%Y-%m-%d %H:%M:%S.%f")
        except ValueError:
            update_dt = datetime.strptime(last_update_str, "%Y-%m-%d %H:%M:%S")
        update_dt = update_dt.replace(tzinfo=timezone.utc)

        prev_net_sum = state.get("last_net_sum")
        
        # Only trade if we have history
        if prev_net_sum is not None:
            diff = current_net_sum - prev_net_sum
            direction = 0
            if diff > 0: direction = 1
            elif diff < 0: direction = -1
            
            if direction != 0:
                # Fetch Open Price for the exact time of lastUpdate
                open_price = get_binance_price("ETHUSDT", int(update_dt.timestamp() * 1000))
                
                if open_price:
                    new_trade = {
                        "direction": direction,
                        "open_price": open_price,
                        "start_time": update_dt.isoformat(),
                        "net_sum_diff": diff
                    }
                    
                    # Add to active trades
                    state['active_trades'].append(new_trade)
                    
                    # Enforce Max 1440 limit
                    if len(state['active_trades']) > 1440:
                        state['active_trades'].pop(0) # Remove oldest
                        
                    print(f"[ETH] New Trade: Dir {direction} @ {open_price}")
        
        # Update state
        state['last_net_sum'] = current_net_sum
        
    except Exception as e:
        print(f"[ETH] Error fetching/processing: {e}")

    # 3. Print Accuracy & Save
    total = state['wins'] + state['losses']
    acc = (state['wins'] / total) if total > 0 else 0.0
    print(f"[ETH] Accuracy: {round(acc*100, 2)}% ({state['wins']}/{total})")
    
    save_eth_state(state)
    
    if trades_closed:
        upload_to_github(ETH_HISTORY_FILE, GITHUB_FILE_PATH_ETH)

# --- Scheduler / Main ---

def main():
    print("--- Crypto Signal Bot (Dual Mode) ---")
    print(f"Main Config: {PREDICTION_URL}")
    print(f"ETH Config: {ETH_URL}")
    
    # Timing state
    last_eth_run_minute = -1
    last_main_run_minute = -1
    
    while True:
        now = datetime.now()
        
        # --- Task 1: ETH Bot (Every minute at :30s) ---
        if now.second == 30 and now.minute != last_eth_run_minute:
            run_eth_cycle()
            last_eth_run_minute = now.minute
            
        # --- Task 2: Main Bot (Minutes 6, 21, 36, 51 at :00s approx) ---
        # We allow a small window (e.g., first 5 seconds of the minute) to catch it
        if now.minute in [6, 21, 36, 51] and now.minute != last_main_run_minute:
            process_expiries() # Check expiries for main bot
            fetch_signals()    # Fetch new signals
            last_main_run_minute = now.minute
            
        # --- Check Main Expiries more frequently? ---
        # The original code checked expiries every loop. 
        # We can check main expiries every minute (e.g. at :10s) to be safe
        if now.second == 10 and now.minute % 5 == 0: 
             process_expiries()

        time.sleep(1)

if __name__ == "__main__":
    main()
