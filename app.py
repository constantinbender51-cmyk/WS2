import time
import json
import csv
import os
import base64
import io
import requests
from datetime import datetime, timezone, timedelta

# --- Configuration ---
# Existing Bot Config
PREDICTION_URL = "https://workspace-production-9fae.up.railway.app/predictions"
ACTIVE_TRADES_FILE = "active_trades.json"  # Local fallback
HISTORY_FILE = "trade_history.csv"         # Local fallback

# New ETH Bot Config
ETH_URL = "https://web-production-73e1d.up.railway.app/"
ETH_STATE_FILE = "eth_state.json"          # Local fallback
ETH_HISTORY_FILE = "eth_history.csv"       # Local fallback

# GitHub Configuration
GITHUB_REPO = "constantinbender51-cmyk/Models"
GITHUB_BRANCH = "main"
GITHUB_TOKEN = os.environ.get("PAT")

# Remote Paths (Where files live on GitHub)
GITHUB_PATH_HISTORY_MAIN = "data/logs/trade_history.csv"
GITHUB_PATH_HISTORY_ETH = "data/logs/eth_history.csv"
GITHUB_PATH_STATE_MAIN = "data/states/active_trades.json"
GITHUB_PATH_STATE_ETH = "data/states/eth_state.json"

# Timeframe mapping
TIMEFRAMES = {
    0: {"label": "15m", "minutes": 15},
    1: {"label": "30m", "minutes": 30},
    2: {"label": "1h",  "minutes": 60},
    3: {"label": "4h",  "minutes": 240},
    4: {"label": "1d",  "minutes": 1440}
}

# --- GitHub Helpers ---

def get_github_file(remote_path):
    """
    Fetches file content and SHA from GitHub.
    Returns: (content_string, sha_string) or (None, None) if not found.
    """
    if not GITHUB_TOKEN:
        print("[ERROR] GitHub 'PAT' environment variable not found.")
        return None, None

    url = f"https://api.github.com/repos/{GITHUB_REPO}/contents/{remote_path}"
    headers = {
        "Authorization": f"token {GITHUB_TOKEN}",
        "Accept": "application/vnd.github.v3+json"
    }
    try:
        r = requests.get(url, headers=headers, params={"ref": GITHUB_BRANCH})
        if r.status_code == 200:
            data = r.json()
            content = base64.b64decode(data['content']).decode('utf-8')
            return content, data['sha']
        elif r.status_code == 404:
            return None, None
        else:
            print(f"[GITHUB] Fetch failed for {remote_path}: {r.status_code}")
    except Exception as e:
        print(f"[GITHUB] Exception fetching {remote_path}: {e}")
    return None, None

def update_github_file(remote_path, content, message, sha=None):
    """
    Creates or updates a file on GitHub.
    """
    if not GITHUB_TOKEN:
        return

    url = f"https://api.github.com/repos/{GITHUB_REPO}/contents/{remote_path}"
    headers = {
        "Authorization": f"token {GITHUB_TOKEN}",
        "Accept": "application/vnd.github.v3+json"
    }
    
    encoded_content = base64.b64encode(content.encode("utf-8")).decode("utf-8")
    
    payload = {
        "message": message,
        "content": encoded_content,
        "branch": GITHUB_BRANCH
    }
    if sha:
        payload["sha"] = sha

    try:
        r = requests.put(url, headers=headers, json=payload)
        if r.status_code in [200, 201]:
            print(f"[GITHUB] Successfully saved to {remote_path}")
        else:
            print(f"[GITHUB] Save failed {remote_path}: {r.status_code} {r.text}")
    except Exception as e:
        print(f"[GITHUB] Exception saving {remote_path}: {e}")

def append_to_github_csv(remote_path, row_data, header):
    """
    Downloads existing CSV, appends row, and re-uploads.
    Prevents history overwrite.
    """
    print(f"[LOGGING] Updating history at {remote_path}...")
    existing_content, sha = get_github_file(remote_path)
    
    output = io.StringIO()
    writer = csv.writer(output)
    
    # If file is new, add headers
    if not existing_content:
        writer.writerow(header)
        existing_content = ""
    
    writer.writerow(row_data)
    new_csv_chunk = output.getvalue()
    
    # Ensure clean newline handling
    final_content = existing_content
    if final_content and not final_content.endswith('\n'):
        final_content += '\n'
    final_content += new_csv_chunk
    
    update_github_file(
        remote_path, 
        final_content, 
        f"Log trade {datetime.now().strftime('%Y-%m-%d %H:%M')}", 
        sha
    )

def save_json_state_to_github(remote_path, state_data):
    """Saves state JSON to GitHub to persist across restarts."""
    _, sha = get_github_file(remote_path)
    json_str = json.dumps(state_data, indent=4)
    update_github_file(
        remote_path, 
        json_str, 
        f"Update state {datetime.now().strftime('%H:%M')}", 
        sha
    )

def load_json_state_from_github(remote_path, default_val):
    """Loads state JSON from GitHub."""
    content, _ = get_github_file(remote_path)
    if content:
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            print(f"[WARN] Corrupt JSON in {remote_path}, using default.")
    return default_val

# --- Market Data Helpers ---

def get_binance_price(symbol, timestamp_ms=None):
    """
    Fetches price from Binance. 
    """
    url = "https://api.binance.com/api/v3/klines"
    if timestamp_ms is None:
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

# --- Main Bot Functions ---

def load_active_trades():
    # Try GitHub first for persistence
    print("[INIT] Loading Main Bot state...")
    return load_json_state_from_github(GITHUB_PATH_STATE_MAIN, [])

def save_active_trades(trades):
    # Save to GitHub for persistence
    save_json_state_to_github(GITHUB_PATH_STATE_MAIN, trades)

def process_expiries():
    active_trades = load_active_trades()
    if not active_trades: 
        return

    remaining_trades = []
    current_ms = int(time.time() * 1000)
    buffer_ms = 10000 
    changed = False

    for trade in active_trades:
        if current_ms > (trade['end_ts'] + buffer_ms):
            print(f"[EXPIRY] Processing {trade['symbol']} {trade['timeframe']}")
            
            # Fetch prices
            open_price = get_binance_price(trade['symbol'], trade['start_ts'])
            close_price = get_binance_price(trade['symbol'], current_ms)
            
            if open_price is not None and close_price is not None:
                # Calculate PnL
                trade['open_price'] = open_price
                trade['close_price'] = close_price
                
                pnl_val = (close_price - open_price) / open_price
                if trade['signal'] < 0:
                    pnl_val = pnl_val * -1
                
                # Log to GitHub
                row_data = [
                    trade['symbol'],
                    trade['timeframe'],
                    trade['signal'],
                    datetime.fromtimestamp(trade['start_ts'] / 1000, timezone.utc).isoformat(),
                    datetime.fromtimestamp(trade['end_ts'] / 1000, timezone.utc).isoformat(),
                    trade['open_price'],
                    trade['close_price'],
                    round(pnl_val, 8),
                    f"{round(pnl_val * 100, 4)}%"
                ]
                headers = ["Symbol", "Timeframe", "Direction", "OpenTime", "CloseTime", "OpenPrice", "ClosePrice", "PnL", "PnL_Percent"]
                
                append_to_github_csv(GITHUB_PATH_HISTORY_MAIN, row_data, headers)
                print(f"[HISTORY] Saved {trade['symbol']} {trade['timeframe']} | PnL: {round(pnl_val*100, 4)}%")
                
                changed = True
            else:
                print(f"[WARN] No data for {trade['symbol']}, keeping active.")
                remaining_trades.append(trade)
        else:
            remaining_trades.append(trade)
            
    if changed:
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
        
        if new_trades_count > 0:
            save_active_trades(active_trades)
            print(f"[UPDATE] Added {new_trades_count} new main trades.")
        
    except Exception as e:
        print(f"[ERROR] Failed to fetch predictions: {e}")

# --- New ETH Bot Functions ---

def load_eth_state():
    print("[INIT] Loading ETH Bot state...")
    default_state = {
        "last_net_sum": None,
        "wins": 0,
        "losses": 0,
        "active_trades": []
    }
    return load_json_state_from_github(GITHUB_PATH_STATE_ETH, default_state)

def save_eth_state(state):
    save_json_state_to_github(GITHUB_PATH_STATE_ETH, state)

def run_eth_cycle():
    print(f"[ETH] Running cycle at {datetime.now().strftime('%H:%M:%S')}")
    state = load_eth_state()
    active_trades = state.get("active_trades", [])
    
    # 1. Process Expiries
    now = datetime.now(timezone.utc)
    remaining_trades = []
    state_changed = False
    
    for trade in active_trades:
        trade_start_dt = datetime.fromisoformat(trade['start_time'])
        if trade_start_dt.tzinfo is None:
            trade_start_dt = trade_start_dt.replace(tzinfo=timezone.utc)
            
        expiry_dt = trade_start_dt + timedelta(hours=24)
        
        if now >= expiry_dt:
            print(f"[ETH] Closing trade from {trade['start_time']}")
            close_price = get_binance_price("ETHUSDT", int(expiry_dt.timestamp() * 1000))
            
            if close_price and trade['open_price']:
                pnl = (close_price - trade['open_price']) / trade['open_price']
                if trade['direction'] == -1:
                    pnl = -pnl
                
                is_noise = abs(pnl) < 0.01
                res = "NOISE"
                if not is_noise:
                    if pnl > 0:
                        state['wins'] += 1
                        res = "WIN"
                    else:
                        state['losses'] += 1
                        res = "LOSS"

                # Log to GitHub
                row_data = [
                    "ETHUSDT", 
                    trade['direction'], 
                    trade['start_time'], 
                    expiry_dt.isoformat(),
                    trade['open_price'], 
                    close_price, 
                    round(pnl, 5), 
                    res
                ]
                headers = ["Symbol", "Direction", "OpenTime", "CloseTime", "OpenPrice", "ClosePrice", "PnL", "Result"]
                
                append_to_github_csv(GITHUB_PATH_HISTORY_ETH, row_data, headers)
                print(f"[ETH] Result: {res} | PnL: {round(pnl*100, 2)}%")
                state_changed = True
            else:
                print("[ETH] Failed to fetch close price, retrying later.")
                remaining_trades.append(trade)
        else:
            remaining_trades.append(trade)
    
    if len(active_trades) != len(remaining_trades):
        state['active_trades'] = remaining_trades
        state_changed = True

    # 2. Fetch New Signal
    try:
        r = requests.get(ETH_URL)
        r.raise_for_status()
        data = r.json()
        
        current_net_sum = data.get("netSum")
        last_update_str = data.get("lastUpdate")
        
        try:
            update_dt = datetime.strptime(last_update_str, "%Y-%m-%d %H:%M:%S.%f")
        except ValueError:
            update_dt = datetime.strptime(last_update_str, "%Y-%m-%d %H:%M:%S")
        update_dt = update_dt.replace(tzinfo=timezone.utc)

        prev_net_sum = state.get("last_net_sum")
        
        if prev_net_sum is not None and prev_net_sum != current_net_sum:
            diff = current_net_sum - prev_net_sum
            direction = 0
            if diff > 0: direction = 1
            elif diff < 0: direction = -1
            
            if direction != 0:
                open_price = get_binance_price("ETHUSDT", int(update_dt.timestamp() * 1000))
                
                if open_price:
                    new_trade = {
                        "direction": direction,
                        "open_price": open_price,
                        "start_time": update_dt.isoformat(),
                        "net_sum_diff": diff
                    }
                    state['active_trades'].append(new_trade)
                    if len(state['active_trades']) > 1440:
                        state['active_trades'].pop(0)
                        
                    print(f"[ETH] New Trade: Dir {direction} @ {open_price}")
                    state_changed = True
        
        if state['last_net_sum'] != current_net_sum:
            state['last_net_sum'] = current_net_sum
            state_changed = True
            
    except Exception as e:
        print(f"[ETH] Error fetching/processing: {e}")

    # 3. Print Accuracy & Save if changed
    total = state['wins'] + state['losses']
    acc = (state['wins'] / total) if total > 0 else 0.0
    print(f"[ETH] Accuracy: {round(acc*100, 2)}% ({state['wins']}/{total})")
    
    if state_changed:
        save_eth_state(state)

# --- Scheduler / Main ---

def main():
    print("--- Crypto Signal Bot (Dual Mode - GitHub Persisted) ---")
    
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
        if now.minute in [6, 21, 36, 51] and now.minute != last_main_run_minute:
            process_expiries()
            fetch_signals()
            last_main_run_minute = now.minute
            
        # --- Main Expiries Frequent Check (Every 5 mins) ---
        if now.second == 10 and now.minute % 5 == 0: 
             process_expiries()

        time.sleep(1)

if __name__ == "__main__":
    main()
