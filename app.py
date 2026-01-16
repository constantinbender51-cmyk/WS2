import os
import time
import datetime
import http.server
import socketserver
import pandas as pd
import requests
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import t  # Changed from norm to t (Student's t-distribution)

# Configuration
DATA_DIR = "/app/data/"
FILE_PATH = os.path.join(DATA_DIR, "eth_1h_2020_2026.csv")
IMAGE_PATH = os.path.join(DATA_DIR, "plot.png")
SYMBOL = "ETHUSDT"
INTERVAL = "1h"
START_DATE = "2020-01-01"
END_DATE = "2026-01-01"
PORT = 8000
STEP_SIZE = 0.1

# Ensure data directory exists
os.makedirs(DATA_DIR, exist_ok=True)

def fetch_binance_data(symbol, interval, start_str, end_str):
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
            last_close_time = klines[-1][6]
            current_start = last_close_time + 1
            time.sleep(0.1)
            
            current_date = datetime.datetime.fromtimestamp(current_start / 1000)
            print(f"Fetched up to {current_date}", end='\r')
            
            if current_start > int(time.time() * 1000):
                break
                
        except Exception as e:
            print(f"\nError fetching data: {e}")
            break
            
    print("\nData fetch complete.")
    
    df = pd.DataFrame(data, columns=[
        'open_time', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'quote_asset_volume', 'trades', 
        'taker_buy_base', 'taker_buy_quote', 'ignore'
    ])
    
    numeric_cols = ['open', 'high', 'low', 'close', 'volume']
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, axis=1)
    df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
    
    return df[['open_time', 'open', 'high', 'low', 'close', 'volume']]

def get_data():
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
    # 1. Compute Derivative
    df['pct_change'] = df['close'].pct_change() * 100
    df = df.dropna()

    # 2. Binning (Symmetric Truncation to 0.1 steps)
    inv_step = int(1 / STEP_SIZE) 
    df['binned_change'] = np.trunc(df['pct_change'] * inv_step) / inv_step

    # 3. Fit Student's t-distribution
    # We fit on the RAW data to get the true shape parameters (df, loc, scale)
    # df (degrees of freedom) controls the "fatness" of the tails. 
    # Lower df = fatter tails/sharper peak.
    param = t.fit(df['pct_change'])
    df_fit, loc_fit, scale_fit = param

    # Prepare distribution for plotting
    distribution = df['binned_change'].value_counts().sort_index()
    plot_data = distribution.loc[-5:5] 

    # 4. Generate t-distribution Curve
    x_range = np.linspace(-5, 5, 1000)
    pdf = t.pdf(x_range, df_fit, loc_fit, scale_fit)
    
    # Scale PDF
    scaling_factor = len(df) * STEP_SIZE 
    y_curve = pdf * scaling_factor

    # 5. Plotting
    plt.figure(figsize=(12, 6))
    
    # Histogram
    plt.bar(plot_data.index, plot_data.values, width=STEP_SIZE*0.8, align='center', 
            color='skyblue', edgecolor='black', linewidth=0.5, label='Actual Data')
    
    # t-distribution Line
    plt.plot(x_range, y_curve, 'r-', linewidth=2, label=f"Student's t-dist (df={df_fit:.2f})")
    
    # Stats Text Box
    textstr = '\n'.join((
        r'$\mu \approx %.4f$' % (loc_fit, ),
        r'$\sigma \approx %.4f$' % (scale_fit, ),
        r'$d.o.f.=%.2f$' % (df_fit, )))
    props = dict(boxstyle='round', facecolor='white', alpha=0.8)
    plt.gca().text(0.95, 0.95, textstr, transform=plt.gca().transAxes, fontsize=12,
            verticalalignment='top', horizontalalignment='right', bbox=props)

    plt.title(f'{SYMBOL} Hourly Returns vs Student\'s t-Distribution ({START_DATE} to {END_DATE})')
    plt.xlabel(f'Price Change % (Rounded to lowest absolute {STEP_SIZE} step)')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    plt.xticks(np.arange(-5, 5.5, 0.5))
    
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
                    <img src="/plot.png" alt="Price Distribution Plot" style="max-width: 100%; border: 1px solid #ddd;">
                    <p><i>Plot generated at {datetime.datetime.now()}</i></p>
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
    df = get_data()
    process_and_plot(df)
    run_server()
