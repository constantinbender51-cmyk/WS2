import asyncio
import aiohttp
import pandas as pd
import numpy as np
from flask import Flask, render_template_string
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io
import base64
from datetime import datetime, timedelta
import requests

app = Flask(__name__)

# Binance API endpoints
BASE_URL = "https://api.binance.com/api/v3/klines"
SYMBOLS = {
    'BTCUSDT': 'BTC',
    'ETHUSDT': 'ETH', 
    'ADAUSDT': 'Cardano'
}

async def fetch_klines(session, symbol, interval='1m', limit=50):
    """Fetch klines data from Binance asynchronously"""
    url = f"{BASE_URL}?symbol={symbol}&interval={interval}&limit={limit}"
    try:
        async with session.get(url) as response:
            if response.status == 200:
                data = await response.json()
                return symbol, data
            else:
                print(f"Error fetching {symbol}: {response.status}")
                return symbol, None
    except Exception as e:
        print(f"Exception fetching {symbol}: {e}")
        return symbol, None

async def fetch_all_prices():
    """Fetch prices for all symbols asynchronously"""
    async with aiohttp.ClientSession() as session:
        tasks = [fetch_klines(session, symbol) for symbol in SYMBOLS.keys()]
        results = await asyncio.gather(*tasks)
        return dict(results)

def process_price_data(data):
    """Process the raw klines data and calculate relative price changes"""
    processed_data = {}
    timestamps = []
    
    for symbol, klines in data.items():
        if klines is None:
            continue
            
        prices = []
        for kline in klines:
            # Use the closing price
            close_price = float(kline[4])
            prices.append(close_price)
            
            # Get timestamps from BTC data (all should have same timestamps)
            if symbol == 'BTCUSDT':
                timestamp = datetime.fromtimestamp(kline[0] / 1000)
                timestamps.append(timestamp)
        
        # Calculate relative price changes (percentage from first price)
        if prices:
            first_price = prices[0]
            relative_prices = [(price / first_price - 1) * 100 for price in prices]
            asset_name = SYMBOLS[symbol]
            processed_data[asset_name] = {
                'prices': prices,
                'relative_changes': relative_prices
            }
    
    return processed_data, timestamps

def calculate_covariances(processed_data):
    """Calculate covariance of each asset with BTC"""
    covariances = {}
    
    if 'BTC' not in processed_data:
        return covariances
    
    btc_relative = processed_data['BTC']['relative_changes']
    
    for asset, data in processed_data.items():
        if asset == 'BTC':
            continue
            
        asset_relative = data['relative_changes']
        
        if len(asset_relative) == len(btc_relative):
            covariance = np.cov(asset_relative, btc_relative)[0][1]
            covariances[asset] = covariance
    
    return covariances

def create_plot(processed_data, timestamps, covariances):
    """Create matplotlib plot and return as base64 encoded image"""
    plt.figure(figsize=(12, 8))
    
    # Plot relative price changes
    for asset, data in processed_data.items():
        plt.plot(timestamps, data['relative_changes'], label=asset, linewidth=2)
    
    plt.title('Relative Price Changes (Last 50 Minutes)', fontsize=16, fontweight='bold')
    plt.xlabel('Time', fontsize=12)
    plt.ylabel('Relative Change (%)', fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # Convert plot to base64 string
    img = io.BytesIO()
    plt.savefig(img, format='png', dpi=100, bbox_inches='tight')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()
    plt.close()
    
    return plot_url

def get_current_prices(processed_data):
    """Get current prices for display"""
    current_prices = {}
    for asset, data in processed_data.items():
        current_prices[asset] = data['prices'][-1] if data['prices'] else 0
    return current_prices

@app.route('/')
def index():
    """Main route that displays the dashboard"""
    try:
        # Fetch and process data
        data = asyncio.run(fetch_all_prices())
        processed_data, timestamps = process_price_data(data)
        
        if not processed_data:
            return "Error: Could not fetch price data"
        
        # Calculate covariances
        covariances = calculate_covariances(processed_data)
        
        # Create plot
        plot_url = create_plot(processed_data, timestamps, covariances)
        
        # Get current prices
        current_prices = get_current_prices(processed_data)
        
        # HTML template
        html_template = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Crypto Price Dashboard</title>
            <style>
                body { 
                    font-family: Arial, sans-serif; 
                    margin: 20px; 
                    background-color: #f5f5f5;
                }
                .container { 
                    max-width: 1200px; 
                    margin: 0 auto; 
                    background: white;
                    padding: 20px;
                    border-radius: 10px;
                    box-shadow: 0 2px 10px rgba(0,0,0,0.1);
                }
                .header { 
                    text-align: center; 
                    margin-bottom: 30px;
                    border-bottom: 2px solid #eee;
                    padding-bottom: 20px;
                }
                .plot-container { 
                    text-align: center; 
                    margin: 20px 0;
                }
                .plot-container img {
                    max-width: 100%;
                    height: auto;
                    border: 1px solid #ddd;
                    border-radius: 5px;
                }
                .stats-container {
                    display: flex;
                    justify-content: space-around;
                    flex-wrap: wrap;
                    margin: 20px 0;
                }
                .price-box, .covariance-box {
                    background: #f8f9fa;
                    padding: 15px;
                    border-radius: 8px;
                    margin: 10px;
                    min-width: 200px;
                    text-align: center;
                    border-left: 4px solid #007bff;
                }
                .covariance-box {
                    border-left-color: #28a745;
                }
                .price {
                    font-size: 24px;
                    font-weight: bold;
                    color: #333;
                }
                .covariance {
                    font-size: 20px;
                    font-weight: bold;
                    color: #28a745;
                }
                .label {
                    font-size: 14px;
                    color: #666;
                    margin-bottom: 5px;
                }
                .last-updated {
                    text-align: center;
                    color: #888;
                    font-size: 12px;
                    margin-top: 20px;
                }
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>Cryptocurrency Price Dashboard</h1>
                    <p>1-minute intervals for the last 50 minutes | Covariance calculated relative to BTC</p>
                </div>
                
                <div class="stats-container">
                    {% for asset, price in current_prices.items() %}
                    <div class="price-box">
                        <div class="label">{{ asset }} Current Price</div>
                        <div class="price">${{ "%.2f"|format(price) }}</div>
                    </div>
                    {% endfor %}
                </div>
                
                <div class="plot-container">
                    <img src="data:image/png;base64,{{ plot_url }}" alt="Price Chart">
                </div>
                
                <div class="stats-container">
                    {% for asset, cov in covariances.items() %}
                    <div class="covariance-box">
                        <div class="label">{{ asset }} Covariance with BTC</div>
                        <div class="covariance">{{ "%.6f"|format(cov) }}</div>
                    </div>
                    {% endfor %}
                </div>
                
                <div class="last-updated">
                    Last updated: {{ last_updated }}
                </div>
            </div>
        </body>
        </html>
        """
        
        return render_template_string(
            html_template,
            plot_url=plot_url,
            covariances=covariances,
            current_prices=current_prices,
            last_updated=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        )
    
    except Exception as e:
        return f"Error: {str(e)}"

if __name__ == '__main__':
    print("Starting Crypto Price Dashboard...")
    print("Access the dashboard at: http://localhost:5000")
    app.run(debug=True, host='0.0.0.0', port=5000)
