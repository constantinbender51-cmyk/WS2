import os
import time
import datetime
import http.server
import socketserver
import pandas as pd
import requests
import matplotlib.pyplot as plt
import numpy as np

# Configuration
DATA_DIR = "/app/data/"
FILE_PATH = os.path.join(DATA_DIR, "eth_1h_2020_2026.csv")
IMAGE_PATH = os.path.join(DATA_DIR, "plot.png")
SYMBOL = "ETHUSDT"
INTERVAL = "1h"
START_DATE = "2020-01-01"
# Current date logic to handle "2026" requirement safely
END_DATE = "2026-01-01" 
PORT = 8000

# Ensure data directory exists
os.makedirs(DATA_DIR, exist_ok=True)

def fetch_binance_data(symbol, interval, start_str, end_str):
    """
    Fetches historical kline data from Binance with pagination.
    """
    api_url = "https://api.binance.com/api/v3/klines"
    start_ts = int(pd.Timestamp(start_str).timestamp() * 1000)
    end_ts = int(pd.Timestamp(end_str).timestamp() * 1000)
    
    data = []
    current_start = start_ts
    
    print(f"Fetching {symbol} data from {start_str} to {end_str}...")
    
    while current_start < end_ts:
        params = {
            'symbol': symbol,
            'interval': interval,
            'startTime': current_start,
            'limit': 1000
        }
        
        try:
            response = requests.get(api_url, params=params)
            response.raise_for_status()
            klines = response.json()
            
            if not klines:
                break
                
            data.extend(klines)
            
            # Update start time to the close time of the last candle + 1ms
            last_close_time = klines[-1][6]
            current_start = last_close_time + 1
            
            # Rate limit handling (polite pause)
            time.sleep(0.1)
            
            # Progress indicator
            current_date = datetime.datetime.fromtimestamp(current_start / 1000)
            print(f"Fetched up to {current_date}", end='\r')
            
            # Break if we reached beyond current time (Binance won't give future data)
            if current_start > int(time.time() * 1000):
                break
                
        except Exception as e:
            print(f"\nError fetching data: {e}")
            break
            
    print("\nData fetch complete.")
    
    # Binance Columns: Open Time, Open, High, Low, Close, Volume, ...
    df = pd.DataFrame(data, columns=[
        'open_time', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'quote_asset_volume', 'trades', 
        'taker_buy_base', 'taker_buy_quote', 'ignore'
    ])
    
    # Convert types
    numeric_cols = ['open', 'high', 'low', 'close', 'volume']
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, axis=1)
    df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
    
    return df[['open_time', 'open', 'high', 'low', 'close', 'volume']]

def get_data():
    """
    Loads data from disk if available, otherwise fetches it.
    """
    if os.path.exists(FILE_PATH):
        print(f"Loading data from {FILE_PATH}...")
        df = pd.read_csv(FILE_PATH)
        df['open_time'] = pd.to_datetime(df['open_time'])
        return df
    else:
        df = fetch_binance_data(SYMBOL, INTERVAL, START_DATE, END_DATE)
        print(f"Saving data to {FILE_PATH}...")
        df.to_csv(FILE_PATH, index=False)
        return df

def process_and_plot(df):
    """
    Computes derivative (price change), bins it, and saves the plot.
    """
    # 3. Compute the Derivative (Percentage Change: (Close - PrevClose) / PrevClose)
    # Note: We use Close vs Prev Close for standard candle-to-candle return
    df['pct_change'] = df['close'].pct_change() * 100
    df = df.dropna()

    # 4. Round to lowest 0.5 step (Floor to 0.5)
    # Logic: 0.4 -> 0.0, 0.9 -> 0.5, -0.4 -> -0.5, -0.9 -> -1.0
    # Formula: floor(x * 2) / 2
    df['binned_change'] = np.floor(df['pct_change'] * 2) / 2

    # Group by the bins to get frequency
    distribution = df['binned_change'].value_counts().sort_index()

    # Filter extreme outliers for better plotting (optional, keeps plot readable)
    # Keeping mostly within -10% to +10% usually covers 99.9% of hourly moves
    plot_data = distribution.loc[-10:10] 

    # Plotting
    plt.figure(figsize=(12, 6))
    plt.bar(plot_data.index, plot_data.values, width=0.4, align='edge', color='skyblue', edgecolor='black')
    
    plt.title(f'{SYMBOL} Hourly Price Change Distribution ({START_DATE} to {END_DATE})')
    plt.xlabel('Price Change % (Rounded down to 0.5 steps)')
    plt.ylabel('Frequency (Count)')
    plt.grid(axis='y', alpha=0.5)
    plt.xticks(np.arange(-10, 10.5, 1))
    
    print(f"Saving plot to {IMAGE_PATH}...")
    plt.savefig(IMAGE_PATH)
    plt.close()

class ImageHandler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/':
            self.send_response(200)
            self.send_header("Content-type", "text/html")
            self.end_headers()
            html = f"""
            <html>
                <head><title>ETH Data Analysis</title></head>
                <body style="font-family: sans-serif; text-align: center; padding: 20px;">
                    <h1>ETH/USDT Hourly Price Change Distribution</h1>
                    <p>Data from {START_DATE} to {END_DATE}</p>
                    <img src="/plot.png" alt="Price Distribution Plot" style="max-width: 100%; border: 1px solid #ddd;">
                    <p><small>Generated at {datetime.datetime.now()}</small></p>
                </body>
            </html>
            """
            self.wfile.write(html.encode('utf-8'))
        elif self.path == '/plot.png':
            if os.path.exists(IMAGE_PATH):
                self.send_response(200)
                self.send_header("Content-type", "image/png")
                self.end_headers()
                with open(IMAGE_PATH, 'rb') as f:
                    self.wfile.write(f.read())
            else:
                self.send_error(404, "Plot not found")
        else:
            self.send_error(404, "File not found")

def run_server():
    with socketserver.TCPServer(("", PORT), ImageHandler) as httpd:
        print(f"Serving at http://localhost:{PORT}")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nServer stopped.")

if __name__ == "__main__":
    # 1 & 2. Fetch and Load
    df = get_data()
    
    # 3 & 4. Compute and Plot
    process_and_plot(df)
    
    # 2. Simple Web Server
    run_server()
