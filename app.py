import io
import requests
import pandas as pd
import matplotlib
# 'Agg' backend prevents matplotlib from opening a GUI window on the server
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
from flask import Flask, send_file

app = Flask(__name__)

def fetch_binance_data():
    """Fetches the last 30 days of 1h BTCUSDT data from Binance (720 candles)."""
    url = "https://api.binance.com/api/v3/klines"
    params = {
        "symbol": "BTCUSDT",
        "interval": "1h",
        "limit": 720  # 30 days * 24 hours
    }
    response = requests.get(url, params=params)
    data = response.json()
    
    cols = ['OpenTime', 'Open', 'High', 'Low', 'Close', 'Volume', 
            'CloseTime', 'QuoteVolume', 'NumTrades', 
            'TakerBuyBaseVol', 'TakerBuyQuoteVol', 'Ignore']
    
    df = pd.DataFrame(data, columns=cols)
    
    # Convert Binance timestamp (milliseconds) to actual Datetime objects
    df['Datetime'] = pd.to_datetime(df['OpenTime'], unit='ms')
    
    # Convert string prices to floats
    df['Close'] = df['Close'].astype(float)
    
    return df

@app.route('/')
def serve_plot():
    # 1. Fetch Data
    df = fetch_binance_data()
    
    # 2. Create Plot (Price vs Standard Time)
    fig, ax = plt.subplots(figsize=(14, 8), facecolor='#0d1117')
    ax.set_facecolor('#0d1117')
    
    # Plot standard chronological line
    ax.plot(
        df['Datetime'], 
        df['Close'], 
        color='#00ffcc',      # Neon cyan line
        linewidth=1.5,        
        alpha=0.8,
        zorder=2
    )
    
    # Optional: fill the area under the curve slightly for aesthetics
    ax.fill_between(df['Datetime'], df['Close'], df['Close'].min(), color='#00ffcc', alpha=0.05)
    
    # 3. Styling
    ax.set_title("BTCUSDT Baseline: Standard Price vs Standard Time (1h, 30 Days)", color='white', fontsize=16, pad=20)
    ax.set_xlabel("Chronological Time (UTC)", color='white', fontsize=12)
    ax.set_ylabel("Price (USDT)", color='white', fontsize=12)
    
    ax.tick_params(colors='white')
    ax.grid(True, linestyle='-', alpha=0.1, color='white')
    
    # Rotate date labels so they don't overlap
    plt.xticks(rotation=45)
    
    # Remove harsh borders
    for spine in ax.spines.values():
        spine.set_edgecolor('#30363d')
    
    # 4. Save to memory buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', dpi=150)
    plt.close(fig)
    buf.seek(0)
    
    # 5. Serve the image
    return send_file(buf, mimetype='image/png')

if __name__ == '__main__':
    print("Serving Baseline Plot on http://0.0.0.0:5000")
    app.run(host='0.0.0.0', port=5000)