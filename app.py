import requests
import pandas as pd
import numpy as np
import io
import time
import threading
import datetime
import os
from collections import defaultdict, Counter
from flask import Flask, render_template_string, jsonify
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import base64

# --- Configuration ---
DATA_DIR = "/app/data"
BINANCE_URL = "https://api.binance.com/api/v3/klines"
PORT = 8080
SECTIONS = 100
SEQUENCE_LEN = 8  # 7 input + 1 target

# List of assets based on your request
ASSETS = [
    "BTCUSDT", "ETHUSDT", "XRPUSDT", "SOLUSDT", "DOGEUSDT", 
    "ADAUSDT", "BCHUSDT", "LINKUSDT", "XLMUSDT", "SUIUSDT", 
    "AVAXUSDT", "LTCUSDT", "HBARUSDT", "SHIBUSDT", "TONUSDT"
]

app = Flask(__name__)

# --- Global State ---
# All state is now dictionaries keyed by Symbol
models = {} # Stores { "BTCUSDT": { sequences, min, max, bin_size, ready } }
cached_dfs = {}
live_outcomes = [] # Global list of all finished trades
current_predictions = {} # { "BTCUSDT": prediction_obj, ... }

system_status = {
    "loading": True,
    "last_check_time": "Never",
    "next_check_time": "Calculating...",
    "errors": []
}

# --- Data Management ---

def download_binance_history(symbol):
    print(f"[{symbol}] Downloading historical data (2020-2026)...")
    
    start_ts = int(datetime.datetime(2020, 1, 1).timestamp() * 1000)
    end_ts = int(datetime.datetime(2026, 1, 1).timestamp() * 1000)
    
    all_candles = []
    current_ts = start_ts
    
    while current_ts < end_ts:
        try:
            params = {
                'symbol': symbol,
                'interval': '15m',
                'limit': 1000,
                'startTime': current_ts,
                'endTime': end_ts
            }
            r = requests.get(BINANCE_URL, params=params)
            r.raise_for_status()
            data = r.json()
            
            if not data:
                break
                
            all_candles.extend(data)
            last_close_time = data[-1][6]
            current_ts = last_close_time + 1
            
            # Avoid rate limits
            time.sleep(0.05) 
            
        except Exception as e:
            print(f"[{symbol}] Error downloading chunk: {e}")
            time.sleep(2)
            
    print(f"[{symbol}] Download complete. Total candles: {len(all_candles)}")
    
    parsed_data = []
    for c in all_candles:
        parsed_data.append({
            "timestamp": c[0],
            "close": float(c[4])
        })
        
    df = pd.DataFrame(parsed_data)
    return df

def fetch_data(symbol):
    global cached_dfs
    if symbol in cached_dfs: return cached_dfs[symbol]

    if not os.path.exists(DATA_DIR): os.makedirs(DATA_DIR)
    
    file_path = os.path.join(DATA_DIR, f"{symbol.lower()}_15m.csv")
        
    if os.path.exists(file_path):
        print(f"[{symbol}] Loading data from file...")
        try:
            df = pd.read_csv(file_path)
            df = df.sort_values('timestamp').reset_index(drop=True)
            cached_dfs[symbol] = df 
            return df
        except Exception as e:
            print(f"[{symbol}] Error reading file: {e}. Redownloading...")
    
    df = download_binance_history(symbol)
    if not df.empty:
        print(f"[{symbol}] Saving data to file...")
        df.to_csv(file_path, index=False)
        cached_dfs[symbol] = df 
    return df

def get_bin(price, min_p, bin_size):
    if bin_size == 0: return 0
    b = int((price - min_p) / bin_size)
    # Clamp bin to 0-(SECTIONS-1) to handle new highs/lows outside historic range
    return max(0, min(b, SECTIONS - 1))

def build_models_worker():
    global models, system_status
    try:
        for symbol in ASSETS:
            df = fetch_data(symbol)
            if df.empty:
                print(f"[{symbol}] No data found. Skipping.")
                continue

            print(f"[{symbol}] Building model...")
            min_price = df['close'].min()
            max_price = df['close'].max()
            
            # Dynamic price range logic (Removed hardcoded clamps for BTC)
            price_range = max_price - min_price
            if price_range == 0: price_range = min_price * 0.01 # Fallback
            bin_size = price_range / SECTIONS

            df['section'] = df['close'].apply(lambda x: get_bin(x, min_price, bin_size))
            sections = df['section'].values
            
            sequences = defaultdict(Counter)
            for i in range(len(sections) - SEQUENCE_LEN + 1):
                seq = tuple(sections[i : i + SEQUENCE_LEN])
                input_seq = seq[:-1]
                target = seq[-1]
                sequences[input_seq][target] += 1
                
            models[symbol] = {
                "sequences": sequences,
                "min_price": min_price,
                "max_price": max_price,
                "bin_size": bin_size,
                "ready": True,
                "last_price": df['close'].iloc[-1]
            }
            print(f"[{symbol}] Model ready. Range: {min_price:.4f}-{max_price:.4f}.")
            
    except Exception as e:
        print(f"Fatal error during initialization: {e}")
        system_status["errors"].append(str(e))
    finally:
        system_status["loading"] = False

def fetch_live_candles(symbol, limit=20):
    try:
        params = {'symbol': symbol, 'interval': '15m', 'limit': limit}
        r = requests.get(BINANCE_URL, params=params)
        r.raise_for_status()
        data = r.json()
        closes = []
        for candle in data:
            closes.append({
                "timestamp": candle[0],
                "close": float(candle[4])
            })
        return closes
    except Exception as e:
        print(f"[{symbol}] Fetch Error: {e}")
        return []

# --- Live Bot Loop ---

def live_prediction_loop():
    global current_predictions, live_outcomes, system_status
    
    while system_status["loading"]:
        time.sleep(5)

    print("Multi-Asset Live prediction bot started.")
    
    while True:
        now = datetime.datetime.utcnow()
        
        # Calculate time to next 15m candle close
        next_quarter = (now.minute // 15 + 1) * 15
        next_hour = now.hour
        if next_quarter >= 60:
            next_quarter = 0
            next_hour = (next_hour + 1) % 24
            
        target_time = now.replace(hour=next_hour, minute=next_quarter, second=5, microsecond=0)
        seconds_to_wait = (target_time - now).total_seconds()
        
        if seconds_to_wait < 0: seconds_to_wait += 900

        system_status["last_check_time"] = now.strftime('%H:%M:%S UTC')
        system_status["next_check_time"] = target_time.strftime('%H:%M:%S UTC')
        
        print(f"Waiting {seconds_to_wait:.1f}s until {system_status['next_check_time']}...")
        time.sleep(seconds_to_wait)
        
        # --- EXECUTE FOR ALL ASSETS ---
        print(f"Executing Live Logic for {len(ASSETS)} assets...")
        
        for symbol in ASSETS:
            if symbol not in models or not models[symbol]["ready"]:
                continue

            # Fetch Data
            candles = fetch_live_candles(symbol, limit=10)
            if len(candles) < 8: continue
            
            last_closed_candle = candles[-2]
            current_close = last_closed_candle['close']
            models[symbol]["last_price"] = current_close # Update dashboard tracker
            
            m = models[symbol]

            # 1. Resolve Previous Prediction for this Symbol
            if symbol in current_predictions:
                prev = current_predictions[symbol]
                entry_price = prev['entry_price']
                direction = prev['direction']
                
                pnl_percent = 0.0
                if direction == "UP":
                    pnl_percent = ((current_close - entry_price) / entry_price) * 100
                elif direction == "DOWN":
                    pnl_percent = ((entry_price - current_close) / entry_price) * 100
                    
                outcome = {
                    "symbol": symbol,
                    "timestamp": last_closed_candle['timestamp'],
                    "entry_price": entry_price,
                    "exit_price": current_close,
                    "direction": direction,
                    "pnl_percent": round(pnl_percent, 4),
                    "inputs": prev['inputs']
                }
                live_outcomes.append(outcome)
                del current_predictions[symbol]

            # 2. Make New Prediction
            input_objs = candles[-8:-1] 
            input_prices = [c['close'] for c in input_objs]
            input_sections = tuple([get_bin(p, m['min_price'], m['bin_size']) for p in input_prices])
            current_section = input_sections[-1]
            
            if input_sections in m['sequences']:
                predicted_section = m['sequences'][input_sections].most_common(1)[0][0]
                
                direction = "FLAT"
                if predicted_section > current_section:
                    direction = "UP"
                elif predicted_section < current_section:
                    direction = "DOWN"
                
                if direction != "FLAT":
                    current_predictions[symbol] = {
                        "symbol": symbol,
                        "timestamp": datetime.datetime.utcnow().timestamp() * 1000,
                        "entry_price": input_prices[-1],
                        "current_section": current_section,
                        "predicted_section": predicted_section,
                        "direction": direction,
                        "inputs": {
                            "prices": input_prices,
                            "sections": list(input_sections)
                        }
                    }
            
            # Small delay to be polite to Binance API
            time.sleep(0.1)

# --- Threads ---
init_thread = threading.Thread(target=build_models_worker, daemon=True)
init_thread.start()

live_thread = threading.Thread(target=live_prediction_loop, daemon=True)
live_thread.start()

# --- Backtesting Logic ---

def run_backtest(symbol):
    if symbol not in models: return [], 0, 0
    
    df = fetch_data(symbol)
    if df.empty: return [], 0, 0

    m = models[symbol]
    sequences = m["sequences"]
    
    df_calc = df.copy() 
    df_calc['section'] = df_calc['close'].apply(lambda x: get_bin(x, m["min_price"], m["bin_size"]))
    
    split_idx = int(len(df_calc) * 0.85) # Reduced backtest size for speed
    test_df = df_calc.iloc[split_idx:].copy()
    
    results = []
    total_pnl = 0.0
    correct_dir = 0
    total_preds = 0
    
    test_closes = test_df['close'].values
    test_sections = test_df['section'].values
    test_timestamps = test_df['timestamp'].values
    
    for i in range(len(test_sections) - SEQUENCE_LEN + 1):
        input_seq = tuple(int(x) for x in test_sections[i : i + SEQUENCE_LEN - 1])
        actual_next_sec = int(test_sections[i + SEQUENCE_LEN - 1])
        actual_price = test_closes[i + SEQUENCE_LEN - 1]
        
        current_price = test_closes[i + SEQUENCE_LEN - 2]
        current_sec = input_seq[-1]
        
        if input_seq in sequences:
            pred_next_sec = int(sequences[input_seq].most_common(1)[0][0])
            if pred_next_sec == current_sec: continue 

            total_preds += 1
            direction = "UP" if pred_next_sec > current_sec else "DOWN"
            
            trade_pnl_pct = 0.0
            is_correct = False
            
            if direction == "UP":
                trade_pnl_pct = (actual_price - current_price) / current_price * 100
                if actual_next_sec > current_sec: is_correct = True
            else:
                trade_pnl_pct = (current_price - actual_price) / current_price * 100
                if actual_next_sec < current_sec: is_correct = True
            
            if is_correct: correct_dir += 1
            total_pnl += trade_pnl_pct
            
            results.append({
                "timestamp": test_timestamps[i + SEQUENCE_LEN - 1],
                "pnl": trade_pnl_pct
            })
            
    acc = (correct_dir / total_preds * 100) if total_preds else 0
    return results, total_pnl, acc

# --- Routes ---

@app.route('/api/status')
def api_status():
    return jsonify(system_status)

@app.route('/')
def dashboard():
    if system_status["loading"]:
        return "<h1>System Initializing...</h1><p>Downloading and processing data for 15 assets. Please refresh in 30 seconds.</p>"

    # 1. Gather Portfolio Stats
    portfolio_summary = []
    total_backtest_trades = 0
    
    for symbol in ASSETS:
        res, pnl, acc = run_backtest(symbol)
        total_backtest_trades += len(res)
        
        # Active Prediction
        active = current_predictions.get(symbol, None)
        active_dir = active['direction'] if active else "-"
        
        portfolio_summary.append({
            "symbol": symbol,
            "last_price": models[symbol].get("last_price", 0),
            "backtest_pnl": round(pnl, 2),
            "backtest_acc": round(acc, 2),
            "trades": len(res),
            "active_pred": active_dir,
            "active_entry": active['entry_price'] if active else ""
        })

    now_utc = datetime.datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')
    
    html = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Multi-Asset Sequence Predictor</title>
        <meta http-equiv="refresh" content="30">
        <style>
            body { font-family: 'Segoe UI', monospace; padding: 20px; background: #f4f4f9; color: #333; }
            h2, h3 { border-bottom: 2px solid #ddd; padding-bottom: 10px; margin-top: 30px;}
            .container { max-width: 1400px; margin: 0 auto; background: white; padding: 20px; box-shadow: 0 0 10px rgba(0,0,0,0.1); }
            table { border-collapse: collapse; width: 100%; font-size: 13px; margin-bottom: 20px; }
            th, td { border: 1px solid #eee; padding: 8px; text-align: left; }
            th { background-color: #f8f9fa; font-weight: 600; }
            .up { color: #27ae60; font-weight: bold; }
            .down { color: #c0392b; font-weight: bold; }
            .symbol { font-weight: bold; color: #2c3e50; }
            
            .header-stats { display: flex; gap: 20px; margin-bottom: 20px; background: #e8f5e9; padding: 15px; border-radius: 5px;}
            .stat-box { flex: 1; }
            .stat-label { font-size: 11px; text-transform: uppercase; color: #555; }
            .stat-val { font-size: 16px; font-weight: bold; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Crypto Sequence Predictor (15 Assets)</h1>
            
            <div class="header-stats">
                <div class="stat-box">
                    <div class="stat-label">Server Time (UTC)</div>
                    <div class="stat-val">{{ now_utc }}</div>
                </div>
                <div class="stat-box">
                    <div class="stat-label">Next Prediction Cycle</div>
                    <div class="stat-val">{{ status.next_check_time }}</div>
                </div>
                <div class="stat-box">
                    <div class="stat-label">Total Backtest Trades</div>
                    <div class="stat-val">{{ total_bt_trades }}</div>
                </div>
            </div>

            <h3>Asset Overview & Performance</h3>
            <table>
                <thead>
                    <tr>
                        <th>Asset</th>
                        <th>Current Price</th>
                        <th>Active Signal</th>
                        <th>Entry Price</th>
                        <th>Backtest Acc %</th>
                        <th>Backtest PnL %</th>
                        <th>Total BT Trades</th>
                    </tr>
                </thead>
                <tbody>
                    {% for row in summary %}
                    <tr>
                        <td class="symbol">{{ row.symbol }}</td>
                        <td>{{ row.last_price }}</td>
                        <td class="{{ 'up' if row.active_pred == 'UP' else 'down' if row.active_pred == 'DOWN' else '' }}">
                            {{ row.active_pred }}
                        </td>
                        <td>{{ row.active_entry }}</td>
                        <td>{{ row.backtest_acc }}%</td>
                        <td class="{{ 'up' if row.backtest_pnl > 0 else 'down' }}">{{ row.backtest_pnl }}%</td>
                        <td>{{ row.trades }}</td>
                    </tr>
                    {% endfor %}
                </tbody>
            </table>

            <h3>Live Trade Log (All Assets)</h3>
            <table>
                <thead>
                    <tr>
                        <th>Time (UTC)</th>
                        <th>Symbol</th>
                        <th>Direction</th>
                        <th>Entry</th>
                        <th>Exit</th>
                        <th>PnL %</th>
                    </tr>
                </thead>
                <tbody>
                    {% for row in outcomes|reverse %}
                    <tr>
                        <td>{{ row.timestamp | datetime }}</td>
                        <td class="symbol">{{ row.symbol }}</td>
                        <td class="{{ 'up' if row.direction == 'UP' else 'down' }}">{{ row.direction }}</td>
                        <td>{{ row.entry_price }}</td>
                        <td>{{ row.exit_price }}</td>
                        <td class="{{ 'up' if row.pnl_percent > 0 else 'down' }}">
                            {{ row.pnl_percent }}%
                        </td>
                    </tr>
                    {% else %}
                    <tr><td colspan="6">No live trades recorded yet.</td></tr>
                    {% endfor %}
                </tbody>
            </table>
        </div>
    </body>
    </html>
    """
    
    app.jinja_env.filters['datetime'] = lambda x: datetime.datetime.fromtimestamp(x/1000).strftime('%Y-%m-%d %H:%M:%S') if isinstance(x, (int, float)) else x

    return render_template_string(html, 
                                  status=system_status,
                                  summary=portfolio_summary,
                                  total_bt_trades=total_backtest_trades,
                                  outcomes=live_outcomes,
                                  now_utc=now_utc)

if __name__ == "__main__":
    print(f"Starting Multi-Asset Server on port {PORT}...")
    app.run(host='0.0.0.0', port=PORT)
