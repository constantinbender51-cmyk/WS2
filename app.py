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

# Hardcoded range clamps to prevent outlier logic
ASSET_CONFIG = {
    "BTCUSDT": {"min": 10000, "max": 200000},
    "ETHUSDT": {"min": 500, "max": 10000},
    "XRPUSDT": {"min": 0.1, "max": 5.0},
    "SOLUSDT": {"min": 1.0, "max": 500},
    "DOGEUSDT": {"min": 0.01, "max": 2.0},
    "ADAUSDT": {"min": 0.1, "max": 5.0},
    "BCHUSDT": {"min": 50, "max": 2000},
    "LINKUSDT": {"min": 1.0, "max": 100},
    "XLMUSDT": {"min": 0.01, "max": 2.0},
    "SUIUSDT": {"min": 0.1, "max": 10.0},
    "AVAXUSDT": {"min": 1.0, "max": 200},
    "LTCUSDT": {"min": 20, "max": 500},
    "HBARUSDT": {"min": 0.01, "max": 2.0},
    "SHIBUSDT": {"min": 0.000001, "max": 0.0001},
    "TONUSDT": {"min": 0.5, "max": 20.0}
}

app = Flask(__name__)

# --- Global State ---
models = {} # Stores { "BTCUSDT": { sequences, min, max, bin_size, ready } }
cached_dfs = {}
live_outcomes = [] 
current_predictions = {} 

# Stores pre-calculated backtest results for dashboard display
portfolio_stats = []
recent_7_day_stats = [] # Stores 7-day specific backtest results
total_backtest_trades = 0

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

def fetch_7_day_data(symbol):
    """
    Fetches the last ~7 days of data (approx 700 candles) separately
    from the main historical fetch.
    """
    print(f"[{symbol}] Fetching separate 7-day dataset...")
    try:
        # 7 days * 24h * 4 (15m) = 672 candles. We request 750 to be safe.
        params = {'symbol': symbol, 'interval': '15m', 'limit': 750}
        r = requests.get(BINANCE_URL, params=params)
        r.raise_for_status()
        data = r.json()
        
        parsed = []
        for c in data:
            parsed.append({
                "timestamp": c[0],
                "close": float(c[4])
            })
        return pd.DataFrame(parsed)
    except Exception as e:
        print(f"[{symbol}] 7-Day Fetch Error: {e}")
        return pd.DataFrame()

def get_bin(price, min_p, bin_size):
    if bin_size == 0: return 0
    b = int((price - min_p) / bin_size)
    return max(0, min(b, SECTIONS - 1))

def calculate_backtest(df, sequences, min_p, bin_size):
    """
    Run backtest on a specific dataframe slice using provided sequences
    """
    results = []
    total_pnl = 0.0
    correct_dir = 0
    total_preds = 0
    
    if df.empty or 'section' not in df.columns:
        return results, total_pnl, 0
    
    test_closes = df['close'].values
    test_sections = df['section'].values
    test_timestamps = df['timestamp'].values
    
    for i in range(len(test_sections) - SEQUENCE_LEN + 1):
        input_seq = tuple(int(x) for x in test_sections[i : i + SEQUENCE_LEN - 1])
        actual_next_sec = int(test_sections[i + SEQUENCE_LEN - 1])
        actual_price = test_closes[i + SEQUENCE_LEN - 1]
        
        current_price = test_closes[i + SEQUENCE_LEN - 2]
        current_sec = input_seq[-1]
        
        if input_seq in sequences:
            pred_next_sec = int(sequences[input_seq].most_common(1)[0][0])
            
            # --- IGNORE FLAT OUTCOMES & PREDICTIONS ---
            if pred_next_sec == current_sec: continue 
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
            
            # Capture inputs used for this prediction
            input_prices = test_closes[i : i + SEQUENCE_LEN - 1]

            results.append({
                "timestamp": test_timestamps[i + SEQUENCE_LEN - 1],
                "pnl": trade_pnl_pct,
                "direction": direction,
                "entry": current_price,
                "inputs": list(input_prices)
            })
            
    acc = (correct_dir / total_preds * 100) if total_preds else 0
    return results, total_pnl, acc

def build_models_worker():
    global models, system_status, portfolio_stats, total_backtest_trades, recent_7_day_stats
    try:
        temp_stats = []
        temp_total_trades = 0
        temp_7_day_stats = []
        
        # 1. Build Base Models and Run Full Backtest
        for symbol in ASSETS:
            df = fetch_data(symbol)
            if df.empty:
                print(f"[{symbol}] No data found. Skipping.")
                continue

            print(f"[{symbol}] Building model...")
            
            # Apply Hardcoded Clamps
            raw_min = df['close'].min()
            raw_max = df['close'].max()
            
            config = ASSET_CONFIG.get(symbol, {"min": raw_min, "max": raw_max})
            min_price = config["min"]
            max_price = config["max"]
            
            price_range = max_price - min_price
            bin_size = price_range / SECTIONS

            # Apply binning to whole dataset
            df['section'] = df['close'].apply(lambda x: get_bin(x, min_price, bin_size))
            
            # Split Data 70/30
            split_idx = int(len(df) * 0.950)
            train_df = df.iloc[:split_idx].copy()
            test_df = df.iloc[split_idx:].copy()

            # Build Model Sequences ONLY on Training Data
            train_sections = train_df['section'].values
            sequences = defaultdict(Counter)
            for i in range(len(train_sections) - SEQUENCE_LEN + 1):
                seq = tuple(train_sections[i : i + SEQUENCE_LEN])
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
            
            # Run Backtest immediately on Test Data (30%)
            print(f"[{symbol}] Running full backtest...")
            res, pnl, acc = calculate_backtest(test_df, sequences, min_price, bin_size)
            
            last_bt_trade = res[-1] if len(res) > 0 else None
            
            temp_total_trades += len(res)
            temp_stats.append({
                "symbol": symbol,
                "backtest_pnl": pnl,
                "backtest_acc": acc,
                "trades": len(res),
                "last_bt_trade": last_bt_trade
            })
            
            # 2. SEPARATE 7-DAY BACKTEST
            # Fetch fresh separate data for the last 7 days
            df_7d = fetch_7_day_data(symbol)
            if not df_7d.empty:
                # Apply same binning logic using the model params
                df_7d['section'] = df_7d['close'].apply(lambda x: get_bin(x, min_price, bin_size))
                
                # Run backtest on this specific slice
                res_7d, pnl_7d, acc_7d = calculate_backtest(df_7d, sequences, min_price, bin_size)
                
                temp_7_day_stats.append({
                    "symbol": symbol,
                    "pnl": pnl_7d,
                    "acc": acc_7d,
                    "count": len(res_7d),
                    "last_trade_input": res_7d[-1]['inputs'] if len(res_7d) > 0 else []
                })
            time.sleep(0.5) # Slight delay to respect API limits

        # Update global cache
        portfolio_stats = temp_stats
        total_backtest_trades = temp_total_trades
        recent_7_day_stats = temp_7_day_stats
            
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
            
            m = models[symbol]

            # Fetch Data
            candles = fetch_live_candles(symbol, limit=10)
            if len(candles) < 8: continue
            
            last_closed_candle = candles[-2]
            current_close = last_closed_candle['close']
            models[symbol]["last_price"] = current_close # Update dashboard tracker
            
            # 1. Resolve Previous Prediction for this Symbol
            if symbol in current_predictions:
                prev = current_predictions[symbol]
                entry_price = prev['entry_price']
                direction = prev['direction']
                
                # Retrieve bounds from the prediction if available (or calc approx)
                pred_min = prev.get('predicted_min', 0)
                pred_max = prev.get('predicted_max', 0)
                
                pnl_percent = 0.0
                if direction == "UP":
                    pnl_percent = ((current_close - entry_price) / entry_price) * 100
                elif direction == "DOWN":
                    pnl_percent = ((entry_price - current_close) / entry_price) * 100
                
                # --- WIN/LOSS/DRAW LOGIC UPDATE ---
                # A loss smaller than the "section size" is considered a DRAW (noise).
                # Calculate the percentage size of 1 bin
                bin_size = m['bin_size']
                step_pct = (bin_size / entry_price) * 100
                
                result_status = "DRAW"
                
                if pnl_percent > 0:
                    result_status = "WIN"
                elif pnl_percent < 0:
                    # If loss is significant (greater than 1 bin size), it's a loss
                    if abs(pnl_percent) > step_pct:
                        result_status = "LOSS"
                    else:
                        # Loss is within the noise floor of the section logic
                        result_status = "DRAW"
                    
                outcome = {
                    "symbol": symbol,
                    "timestamp": last_closed_candle['timestamp'],
                    "entry_price": entry_price,
                    "exit_price": current_close,
                    "direction": direction,
                    "pnl_percent": round(pnl_percent, 4),
                    "result": result_status,
                    "inputs": prev['inputs'],
                    "lower_bound": pred_min,
                    "upper_bound": pred_max
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
                    # Calculate bounds for the target section
                    pred_min_price = m['min_price'] + (predicted_section * m['bin_size'])
                    pred_max_price = pred_min_price + m['bin_size']
                    
                    current_predictions[symbol] = {
                        "symbol": symbol,
                        "timestamp": datetime.datetime.utcnow().timestamp() * 1000,
                        "entry_price": input_prices[-1],
                        "current_section": current_section,
                        "predicted_section": predicted_section,
                        "predicted_min": pred_min_price,
                        "predicted_max": pred_max_price,
                        "direction": direction,
                        "inputs": {
                            "prices": input_prices,
                            "sections": list(input_sections)
                        }
                    }
            
            time.sleep(0.1)

# --- Threads ---
init_thread = threading.Thread(target=build_models_worker, daemon=True)
init_thread.start()

live_thread = threading.Thread(target=live_prediction_loop, daemon=True)
live_thread.start()

# --- Routes ---
# --- Routes ---

@app.route('/api/status')
def api_status():
    return jsonify(system_status)

@app.route('/api/signals')
def api_signals():
    """Returns the currently active live predictions."""
    return jsonify(current_predictions.copy())

@app.route('/')
def dashboard():
    if system_status["loading"]:
        return "<h1>System Initializing...</h1><p>Downloading and processing data for 15 assets. Please refresh in 30 seconds.</p>"

    # Merge cached portfolio stats with current active prediction info
    display_summary = []
    
    for stat in portfolio_stats:
        symbol = stat['symbol']
        active = current_predictions.get(symbol, None)
        active_dir = active['direction'] if active else "-"
        
        # Get active inputs if available
        active_inputs = active['inputs']['prices'] if active else []
        
        display_summary.append({
            "symbol": symbol,
            "last_price": models[symbol].get("last_price", 0),
            "backtest_pnl": round(stat['backtest_pnl'], 2),
            "backtest_acc": round(stat['backtest_acc'], 2),
            "trades": stat['trades'],
            "active_pred": active_dir,
            "active_entry": active['entry_price'] if active else "",
            "active_inputs": active_inputs,
            "last_bt_trade": stat.get('last_bt_trade')
        })

    now_utc = datetime.datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')
    
    html = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Multi-Asset Sequence Predictor</title>
        <style>
            body { font-family: 'Segoe UI', monospace; padding: 20px; background: #f4f4f9; color: #333; }
            h2, h3 { border-bottom: 2px solid #ddd; padding-bottom: 10px; margin-top: 30px;}
            .container { max-width: 1500px; margin: 0 auto; background: white; padding: 20px; box-shadow: 0 0 10px rgba(0,0,0,0.1); }
            table { border-collapse: collapse; width: 100%; font-size: 13px; margin-bottom: 20px; }
            th, td { border: 1px solid #eee; padding: 8px; text-align: left; vertical-align: top; }
            th { background-color: #f8f9fa; font-weight: 600; }
            .up { color: #27ae60; font-weight: bold; }
            .down { color: #c0392b; font-weight: bold; }
            .symbol { font-weight: bold; color: #2c3e50; }
            .small-text { font-size: 10px; color: #666; font-family: monospace; display: block; margin-top: 4px; }
            
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
                        <th>Active Entry</th>
                        <th>Active Inputs (Last 7)</th>
                        <th>Backtest Acc %</th>
                        <th>Backtest PnL %</th>
                        <th>BT Trades</th>
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
                        <td class="small-text" style="max-width: 250px; word-wrap: break-word;">
                            {{ row.active_inputs | join(', ') }}
                        </td>
                        <td>{{ row.backtest_acc }}%</td>
                        <td class="{{ 'up' if row.backtest_pnl > 0 else 'down' }}">{{ row.backtest_pnl }}%</td>
                        <td>{{ row.trades }}</td>
                    </tr>
                    {% endfor %}
                </tbody>
            </table>

            <h3>Last Backtest Signals (Diagnostics)</h3>
            <table>
                <thead>
                    <tr>
                        <th>Time (UTC)</th>
                        <th>Symbol</th>
                        <th>Direction</th>
                        <th>Entry Price</th>
                        <th>PnL Outcome</th>
                        <th>Input Sequence Prices (Last 7)</th>
                    </tr>
                </thead>
                <tbody>
                    {% for row in summary %}
                        {% if row.last_bt_trade %}
                        <tr>
                            <td>{{ row.last_bt_trade.timestamp | datetime }}</td>
                            <td class="symbol">{{ row.symbol }}</td>
                            <td class="{{ 'up' if row.last_bt_trade.direction == 'UP' else 'down' }}">
                                {{ row.last_bt_trade.direction }}
                            </td>
                            <td>{{ row.last_bt_trade.entry }}</td>
                            <td class="{{ 'up' if row.last_bt_trade.pnl > 0 else 'down' }}">
                                {{ "%.2f"|format(row.last_bt_trade.pnl) }}%
                            </td>
                            <td class="small-text" style="max-width: 400px; word-wrap: break-word;">
                                {{ row.last_bt_trade.inputs | join(', ') }}
                            </td>
                        </tr>
                        {% endif %}
                    {% endfor %}
                </tbody>
            </table>
            
            <h3>Last 7 Days Performance (Separate Fetch)</h3>
            <table>
                <thead>
                    <tr>
                        <th>Symbol</th>
                        <th>7-Day PnL %</th>
                        <th>7-Day Accuracy %</th>
                        <th>Trades (7d)</th>
                        <th>Last Input Seq (7d)</th>
                    </tr>
                </thead>
                <tbody>
                    {% for row in recent_stats %}
                    <tr>
                        <td class="symbol">{{ row.symbol }}</td>
                        <td class="{{ 'up' if row.pnl > 0 else 'down' }}">
                            {{ "%.2f"|format(row.pnl) }}%
                        </td>
                        <td>{{ "%.1f"|format(row.acc) }}%</td>
                        <td>{{ row.count }}</td>
                        <td class="small-text" style="max-width: 400px; word-wrap: break-word;">
                             {{ row.last_trade_input | join(', ') }}
                        </td>
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
                        <th>Result</th>
                        <th>Boundaries (Min-Max)</th>
                        <th>Input Prices</th>
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
                        <td class="{{ 'up' if row.result == 'WIN' else 'down' if row.result == 'LOSS' else '' }}">
                            {{ row.result }}
                        </td>
                        <td class="small-text">
                            {{ "%.2f"|format(row.lower_bound) }} - {{ "%.2f"|format(row.upper_bound) }}
                        </td>
                        <td class="small-text" style="max-width: 400px; word-wrap: break-word;">
                             {{ row.inputs.prices | join(', ') }}
                        </td>
                    </tr>
                    {% else %}
                    <tr><td colspan="9">No live trades recorded yet.</td></tr>
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
                                  summary=display_summary,
                                  recent_stats=recent_7_day_stats,
                                  total_bt_trades=total_backtest_trades,
                                  outcomes=live_outcomes,
                                  now_utc=now_utc)

if __name__ == "__main__":
    print(f"Starting Multi-Asset Server on port {PORT}...")
    app.run(host='0.0.0.0', port=PORT)
