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

ASSETS = [
    "BTCUSDT", "ETHUSDT", "XRPUSDT", "SOLUSDT", "DOGEUSDT", 
    "ADAUSDT", "BCHUSDT", "LINKUSDT", "XLMUSDT", "SUIUSDT", 
    "AVAXUSDT", "LTCUSDT", "HBARUSDT", "SHIBUSDT", "TONUSDT"
]

# Hardcoded clamps to enforce consistent bin sizes (Change #3)
PRICE_CLAMPS = {
    "BTCUSDT": (10000.0, 200000.0),
    "ETHUSDT": (500.0, 10000.0),
    "XRPUSDT": (0.1, 5.0),
    "SOLUSDT": (5.0, 500.0),
    "DOGEUSDT": (0.01, 1.0),
    "ADAUSDT": (0.1, 5.0),
    "BCHUSDT": (50.0, 2000.0),
    "LINKUSDT": (2.0, 100.0),
    "XLMUSDT": (0.01, 2.0),
    "SUIUSDT": (0.1, 10.0),
    "AVAXUSDT": (5.0, 200.0),
    "LTCUSDT": (20.0, 500.0),
    "HBARUSDT": (0.01, 2.0),
    "SHIBUSDT": (0.000001, 0.0001),
    "TONUSDT": (0.5, 20.0)
}

app = Flask(__name__)

# --- Global State ---
models = {} 
cached_dfs = {}
live_outcomes = [] 
current_predictions = {} 

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
            if not data: break
            all_candles.extend(data)
            last_close_time = data[-1][6]
            current_ts = last_close_time + 1
            time.sleep(0.05) 
        except Exception as e:
            print(f"[{symbol}] Error downloading chunk: {e}")
            time.sleep(2)
            
    parsed_data = []
    for c in all_candles:
        parsed_data.append({"timestamp": c[0], "close": float(c[4])})
    return pd.DataFrame(parsed_data)

def fetch_data(symbol):
    global cached_dfs
    if symbol in cached_dfs: return cached_dfs[symbol]

    if not os.path.exists(DATA_DIR): os.makedirs(DATA_DIR)
    file_path = os.path.join(DATA_DIR, f"{symbol.lower()}_15m.csv")
        
    if os.path.exists(file_path):
        try:
            df = pd.read_csv(file_path)
            df = df.sort_values('timestamp').reset_index(drop=True)
            cached_dfs[symbol] = df 
            return df
        except: pass
    
    df = download_binance_history(symbol)
    if not df.empty:
        df.to_csv(file_path, index=False)
        cached_dfs[symbol] = df 
    return df

def get_bin(price, min_p, bin_size):
    if bin_size == 0: return 0
    b = int((price - min_p) / bin_size)
    return max(0, min(b, SECTIONS - 1))

def train_model(sections):
    """Helper to build sequences from a list of sections"""
    sequences = defaultdict(Counter)
    for i in range(len(sections) - SEQUENCE_LEN + 1):
        seq = tuple(sections[i : i + SEQUENCE_LEN])
        input_seq = seq[:-1]
        target = seq[-1]
        sequences[input_seq][target] += 1
    return sequences

def build_models_worker():
    global models, system_status
    try:
        for symbol in ASSETS:
            df = fetch_data(symbol)
            if df.empty: continue

            # Use Hardcoded Clamps (Change #3)
            if symbol in PRICE_CLAMPS:
                min_price, max_price = PRICE_CLAMPS[symbol]
            else:
                min_price = df['close'].min()
                max_price = df['close'].max()
            
            price_range = max_price - min_price
            if price_range == 0: price_range = min_price * 0.01
            bin_size = price_range / SECTIONS

            df['section'] = df['close'].apply(lambda x: get_bin(x, min_price, bin_size))
            
            # For LIVE TRADING, we use the FULL dataset
            sequences = train_model(df['section'].values)
                
            models[symbol] = {
                "sequences": sequences,
                "min_price": min_price,
                "max_price": max_price,
                "bin_size": bin_size,
                "ready": True,
                "last_price": df['close'].iloc[-1]
            }
            print(f"[{symbol}] Model ready. Range: {min_price}-{max_price}")
            
    except Exception as e:
        system_status["errors"].append(str(e))
    finally:
        system_status["loading"] = False

def fetch_live_candles(symbol, limit=20):
    try:
        params = {'symbol': symbol, 'interval': '15m', 'limit': limit}
        r = requests.get(BINANCE_URL, params=params)
        r.raise_for_status()
        data = r.json()
        return [{"timestamp": c[0], "close": float(c[4])} for c in data]
    except: return []

# --- Live Bot Loop ---

def live_prediction_loop():
    global current_predictions, live_outcomes, system_status
    while system_status["loading"]: time.sleep(5)
    print("Multi-Asset Live prediction bot started.")
    
    while True:
        now = datetime.datetime.utcnow()
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
        time.sleep(seconds_to_wait)
        
        for symbol in ASSETS:
            if symbol not in models or not models[symbol]["ready"]: continue
            candles = fetch_live_candles(symbol, limit=10)
            if len(candles) < 8: continue
            
            last_closed_candle = candles[-2]
            current_close = last_closed_candle['close']
            models[symbol]["last_price"] = current_close
            m = models[symbol]

            # Resolve Previous
            if symbol in current_predictions:
                prev = current_predictions[symbol]
                direction = prev['direction']
                pnl = 0.0
                if direction == "UP": pnl = ((current_close - prev['entry_price']) / prev['entry_price']) * 100
                elif direction == "DOWN": pnl = ((prev['entry_price'] - current_close) / prev['entry_price']) * 100
                    
                live_outcomes.append({
                    "symbol": symbol,
                    "timestamp": last_closed_candle['timestamp'],
                    "direction": direction,
                    "pnl_percent": round(pnl, 4)
                })
                del current_predictions[symbol]

            # Make New
            input_objs = candles[-8:-1] 
            input_prices = [c['close'] for c in input_objs]
            input_sections = tuple([get_bin(p, m['min_price'], m['bin_size']) for p in input_prices])
            current_section = input_sections[-1]
            
            if input_sections in m['sequences']:
                predicted_section = m['sequences'][input_sections].most_common(1)[0][0]
                direction = "FLAT"
                if predicted_section > current_section: direction = "UP"
                elif predicted_section < current_section: direction = "DOWN"
                
                if direction != "FLAT":
                    current_predictions[symbol] = {
                        "symbol": symbol,
                        "entry_price": input_prices[-1],
                        "direction": direction,
                        "inputs": list(input_sections)
                    }
            time.sleep(0.1)

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
    
    # 1. Calculate sections using global clamps
    df_calc = df.copy() 
    df_calc['section'] = df_calc['close'].apply(lambda x: get_bin(x, m["min_price"], m["bin_size"]))
    
    # 2. Strict Train/Test Split (Change #4)
    # Use 70% for training, 30% for testing (Change #2)
    split_idx = int(len(df_calc) * 0.70)
    
    train_df = df_calc.iloc[:split_idx]
    test_df = df_calc.iloc[split_idx:].copy()
    
    # 3. Re-train model ONLY on training data (Fixing Data Leakage)
    train_sequences = train_model(train_df['section'].values)
    
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
        
        # Use the CLEAN training sequences
        if input_seq in train_sequences:
            pred_next_sec = int(train_sequences[input_seq].most_common(1)[0][0])
            
            if pred_next_sec == current_sec: continue 
            
            # Ignore Flat Outcomes (Change #1)
            # This logic mimics the single-asset script
            if actual_next_sec == current_sec: continue 

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
            results.append({"pnl": trade_pnl_pct})
            
    acc = (correct_dir / total_preds * 100) if total_preds else 0
    return results, total_pnl, acc

@app.route('/api/status')
def api_status(): return jsonify(system_status)

@app.route('/')
def dashboard():
    if system_status["loading"]: return "<h1>Initializing...</h1>"
    
    portfolio_summary = []
    total_backtest_trades = 0
    
    for symbol in ASSETS:
        res, pnl, acc = run_backtest(symbol)
        total_backtest_trades += len(res)
        active = current_predictions.get(symbol, None)
        
        portfolio_summary.append({
            "symbol": symbol,
            "last_price": models[symbol].get("last_price", 0),
            "backtest_pnl": round(pnl, 2),
            "backtest_acc": round(acc, 2),
            "trades": len(res),
            "active_pred": active['direction'] if active else "-",
            "active_entry": active['entry_price'] if active else ""
        })

    html = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Multi-Asset Predictor (Fixed)</title>
        <meta http-equiv="refresh" content="30">
        <style>
            body { font-family: sans-serif; padding: 20px; background: #f4f4f9; }
            table { border-collapse: collapse; width: 100%; margin-bottom: 20px; }
            th, td { border: 1px solid #ddd; padding: 8px; }
            th { background: #eee; }
            .up { color: green; font-weight: bold; }
            .down { color: red; font-weight: bold; }
        </style>
    </head>
    <body>
        <h2>Multi-Asset Predictor (Leakage Fixed & Clamped)</h2>
        <p><strong>Total Backtest Trades (70% Train / 30% Test):</strong> {{ total_bt_trades }}</p>
        
        <h3>Portfolio Performance</h3>
        <table>
            <thead>
                <tr><th>Asset</th><th>Price</th><th>Active</th><th>Entry</th><th>Acc %</th><th>PnL %</th><th>Trades</th></tr>
            </thead>
            <tbody>
                {% for row in summary %}
                <tr>
                    <td>{{ row.symbol }}</td><td>{{ row.last_price }}</td>
                    <td class="{{ 'up' if row.active_pred == 'UP' else 'down' if row.active_pred == 'DOWN' else '' }}">{{ row.active_pred }}</td>
                    <td>{{ row.active_entry }}</td>
                    <td>{{ row.backtest_acc }}%</td>
                    <td class="{{ 'up' if row.backtest_pnl > 0 else 'down' }}">{{ row.backtest_pnl }}%</td>
                    <td>{{ row.trades }}</td>
                </tr>
                {% endfor %}
            </tbody>
        </table>
        
        <h3>Live Trades</h3>
        <table>
            <thead><tr><th>Symbol</th><th>Direction</th><th>PnL %</th></tr></thead>
            <tbody>
                {% for row in outcomes|reverse %}
                <tr>
                    <td>{{ row.symbol }}</td>
                    <td class="{{ 'up' if row.direction == 'UP' else 'down' }}">{{ row.direction }}</td>
                    <td class="{{ 'up' if row.pnl_percent > 0 else 'down' }}">{{ row.pnl_percent }}%</td>
                </tr>
                {% endfor %}
            </tbody>
        </table>
    </body>
    </html>
    """
    return render_template_string(html, 
                                  system_status=system_status,
                                  summary=portfolio_summary,
                                  total_bt_trades=total_backtest_trades,
                                  outcomes=live_outcomes)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=PORT)
