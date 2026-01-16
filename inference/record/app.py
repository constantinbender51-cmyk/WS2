import time
import requests
import csv
import os
import re
import threading
from datetime import datetime, timedelta
from http.server import HTTPServer, BaseHTTPRequestHandler

# --- CONFIGURATION ---
SIGNAL_URL = "http://signalsexec.up.railway.app"
CSV_FILE = "pnl_log.csv"
SERVER_PORT = 8081  # Port to serve the CSV (different from scheduler)
TICKERS = ["BTC", "ETH", "SOL"]

# Map internal tickers to Binance US Tether pairs
PRICE_MAPPING = {
    "BTC": "BTCUSDT",
    "ETH": "ETHUSDT",
    "SOL": "SOLUSDT"
}

# Global state
previous_state = {}

class CSVRequestHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        """Serve the CSV file content and a summary."""
        if self.path == '/csv':
            # Serve raw CSV for downloading/parsing
            self._serve_file()
        else:
            # Serve a dashboard text view
            self._serve_dashboard()

    def _serve_file(self):
        if os.path.isfile(CSV_FILE):
            self.send_response(200)
            self.send_header("Content-type", "text/csv")
            self.send_header("Content-Disposition", f"attachment; filename={CSV_FILE}")
            self.end_headers()
            with open(CSV_FILE, 'rb') as f:
                self.wfile.write(f.read())
        else:
            self.send_error(404, "CSV file not created yet.")

    def _serve_dashboard(self):
        self.send_response(200)
        self.send_header("Content-type", "text/plain")
        self.end_headers()

        response_lines = []
        response_lines.append(f"--- PnL RECORDER DASHBOARD ({datetime.now().strftime('%H:%M:%S')}) ---")
        
        # Calculate stats on the fly
        stats = self._calculate_stats()
        response_lines.append(f"Total Trades: {stats['trades']}")
        response_lines.append(f"Total PnL:    {stats['pnl']:.4f}")
        response_lines.append(f"Accuracy:     {stats['accuracy']:.2f}%")
        response_lines.append("\n--- RECENT LOGS ---")
        
        # Read last 10 lines of CSV
        if os.path.isfile(CSV_FILE):
            with open(CSV_FILE, 'r') as f:
                lines = f.readlines()
                # Header
                response_lines.append(lines[0].strip()) 
                # Last 10 lines
                for line in lines[-10:]:
                    if line != lines[0]: # Avoid repeating header if file is short
                        response_lines.append(line.strip())
        else:
            response_lines.append("No data recorded yet.")

        response_lines.append("\n(Go to /csv to download full raw data)")
        self.wfile.write("\n".join(response_lines).encode())

    def _calculate_stats(self):
        total_pnl = 0.0
        wins = 0
        total_trades = 0
        
        if os.path.isfile(CSV_FILE):
            with open(CSV_FILE, mode='r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    total_pnl += float(row['pnl'])
                    total_trades += 1
                    if float(row['is_win']) > 0:
                        wins += 1
        
        accuracy = (wins / total_trades * 100) if total_trades > 0 else 0
        return {"pnl": total_pnl, "accuracy": accuracy, "trades": total_trades}

def start_web_server():
    """Starts the web server in a background thread."""
    server_address = ('', SERVER_PORT)
    httpd = HTTPServer(server_address, CSVRequestHandler)
    print(f"🌍 Recorder Server running at http://localhost:{SERVER_PORT}")
    httpd.serve_forever()

def get_seconds_until_next_interval(interval_minutes=15):
    now = datetime.now()
    minutes_to_add = interval_minutes - (now.minute % interval_minutes)
    target_time = now + timedelta(minutes=minutes_to_add)
    target_time = target_time.replace(second=0, microsecond=0)
    seconds_remaining = (target_time - now).total_seconds()
    return seconds_remaining + 5

def fetch_signals():
    try:
        resp = requests.get(SIGNAL_URL, timeout=10)
        resp.raise_for_status()
        text = resp.text
        signals = {}
        for ticker in TICKERS:
            match = re.search(rf"{ticker}:\s+(-?\d+)", text)
            signals[ticker] = int(match.group(1)) if match else 0
        return signals
    except Exception as e:
        print(f"❌ Error fetching signals: {e}")
        return None

def fetch_current_prices():
    prices = {}
    base_url = "https://api.binance.com/api/v3/ticker/price"
    try:
        for ticker in TICKERS:
            symbol = PRICE_MAPPING[ticker]
            resp = requests.get(f"{base_url}?symbol={symbol}", timeout=10)
            data = resp.json()
            prices[ticker] = float(data['price'])
        return prices
    except Exception as e:
        print(f"❌ Error fetching prices: {e}")
        return None

def update_csv(record):
    file_exists = os.path.isfile(CSV_FILE)
    fieldnames = ["timestamp", "ticker", "signal_start", "price_start", "price_end", "pct_return", "pnl", "is_win"]
    with open(CSV_FILE, mode='a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(record)

def run_record_loop():
    global previous_state
    
    # Start Server
    server_thread = threading.Thread(target=start_web_server, daemon=True)
    server_thread.start()
    
    print(f"--- 📉 Accuracy Recorder Started ---")
    print("Waiting for the next 15-minute mark...")

    while True:
        sleep_sec = get_seconds_until_next_interval(15)
        time.sleep(sleep_sec)
        
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print(f"⏰ Recording Snapshot: {timestamp}")

        current_signals = fetch_signals()
        current_prices = fetch_current_prices()

        if not current_signals or not current_prices:
            print("⚠️ Data fetch failed. Skipping this interval.")
            continue

        for ticker in TICKERS:
            curr_price = current_prices[ticker]
            curr_signal = current_signals[ticker]

            if ticker in previous_state:
                prev = previous_state[ticker]
                price_return = (curr_price - prev['price']) / prev['price']
                pnl = prev['signal'] * price_return
                is_win = 1 if pnl > 0 else 0
                
                record = {
                    "timestamp": timestamp,
                    "ticker": ticker,
                    "signal_start": prev['signal'],
                    "price_start": prev['price'],
                    "price_end": curr_price,
                    "pct_return": round(price_return, 6),
                    "pnl": round(pnl, 6),
                    "is_win": is_win
                }
                update_csv(record)
                print(f"   {ticker}: PnL {pnl:.6f}")

            previous_state[ticker] = {"price": curr_price, "signal": curr_signal}

if __name__ == "__main__":
    try:
        run_record_loop()
    except KeyboardInterrupt:
        print("\n🛑 Recorder stopped.")
  
