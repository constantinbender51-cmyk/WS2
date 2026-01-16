import time
import subprocess
import sys
import threading
import re
from datetime import datetime, timedelta
from http.server import HTTPServer, BaseHTTPRequestHandler

# --- CONFIGURATION ---
TICKERS = ["BTC", "ETH", "SOL"]
INFERENCE_SCRIPT = "inference.py"
SERVER_PORT = 8080
SIGNAL_DURATION_MINUTES = 60  # Signals expire after 1 hour (inference candle size)

# --- GLOBAL STATE (Thread-Safe) ---
# Structure: { "BTC": [ {'value': 1, 'expiry': datetime}, ... ], "ETH": ... }
active_signals = {t: [] for t in TICKERS}
lock = threading.Lock()

class SignalRequestHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        """Handle GET requests to serve the signal dashboard."""
        self.send_response(200)
        self.send_header("Content-type", "text/plain")
        self.end_headers()

        response_lines = []
        response_lines.append(f"--- ACTIVE SIGNALS ({datetime.now().strftime('%H:%M:%S')}) ---")
        
        with lock:
            self._prune_expired_signals()
            for ticker in TICKERS:
                # Calculate sum of active signals
                total_score = sum(s['value'] for s in active_signals[ticker])
                # Format: "BTC: 2  (Active: 1, 1, 0)"
                details = ", ".join([str(s['value']) for s in active_signals[ticker]])
                response_lines.append(f"{ticker}: {total_score}\t[Components: {details if details else 'None'}]")
        
        response_text = "\n".join(response_lines)
        self.wfile.write(response_text.encode())

    def _prune_expired_signals(self):
        """Remove signals that have passed their expiry time."""
        now = datetime.now()
        for ticker in TICKERS:
            # Keep only signals where expiry > now
            active_signals[ticker] = [s for s in active_signals[ticker] if s['expiry'] > now]

def start_web_server():
    """Starts the web server in a background thread."""
    server_address = ('', SERVER_PORT)
    httpd = HTTPServer(server_address, SignalRequestHandler)
    print(f"🌍 Web Server running at http://localhost:{SERVER_PORT}")
    httpd.serve_forever()

def get_seconds_until_next_interval(interval_minutes=15):
    now = datetime.now()
    minutes_to_add = interval_minutes - (now.minute % interval_minutes)
    target_time = now + timedelta(minutes=minutes_to_add)
    target_time = target_time.replace(second=0, microsecond=0)
    seconds_remaining = (target_time - now).total_seconds()
    if seconds_remaining < 0:
        seconds_remaining += (interval_minutes * 60)
    return seconds_remaining

def parse_inference_output(output_str):
    """Parses stdout from inference.py to find the signal."""
    # Look for the specific print statements in your inference.py
    if "SIGNAL: BUY" in output_str:
        return 1
    elif "SIGNAL: SELL" in output_str:
        return -1
    elif "SIGNAL: X" in output_str:
        return 0
    return 0 # Default if unknown or incomplete

def run_batch_inference():
    print(f"\n{'='*40}")
    print(f"⏰ Triggering Batch Inference: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*40}")

    for ticker in TICKERS:
        try:
            print(f"running {ticker}...", end=" ")
            
            # Run inference and capture output
            result = subprocess.run(
                [sys.executable, INFERENCE_SCRIPT, ticker], 
                capture_output=True, 
                text=True,
                check=True
            )
            
            # Print output to console for debugging (optional)
            # print(result.stdout)

            # Parse signal
            signal_val = parse_inference_output(result.stdout)
            
            # Update Global State
            expiry_time = datetime.now() + timedelta(minutes=SIGNAL_DURATION_MINUTES)
            
            with lock:
                active_signals[ticker].append({
                    'value': signal_val,
                    'expiry': expiry_time
                })
                
                # Prune old immediately to keep list clean
                now = datetime.now()
                active_signals[ticker] = [s for s in active_signals[ticker] if s['expiry'] > now]

            print(f"-> Signal: {signal_val} (Active Sum: {sum(s['value'] for s in active_signals[ticker])})")

        except subprocess.CalledProcessError as e:
            print(f"\n❌ Error running {ticker}: {e}")
            print(f"Stderr: {e.stderr}")
        except Exception as e:
            print(f"\n❌ Unexpected error: {e}")

def main():
    print(f"--- 🗓️ Scheduler Started ---")
    
    # 1. Start Web Server in a daemon thread
    server_thread = threading.Thread(target=start_web_server, daemon=True)
    server_thread.start()

    # 2. Scheduler Loop
    print(f"Tracking tickers: {TICKERS}")
    print("Waiting for the next 15-minute mark...")

    while True:
        sleep_seconds = get_seconds_until_next_interval(15)
        # Sleep until next mark (+1s buffer)
        time.sleep(sleep_seconds + 1)
        run_batch_inference()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n🛑 Scheduler stopped manually.")
