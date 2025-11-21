import pandas as pd
import numpy as np
import requests
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Required for headless environments
from flask import Flask, send_file
import io
import datetime
from threading import Thread
import time

app = Flask(__name__)

# Global variable to store the latest data
latest_data = None
latest_covariance = None

def get_most_liquid_cryptos():
    """Fetch the top 10 most liquid cryptocurrencies (excluding stablecoins)"""
    try:
        url = "https://api.coingecko.com/api/v3/coins/markets"
        params = {
            'vs_currency': 'usd',
            'order': 'volume_desc',
            'per_page': 15,  # Get extra to filter out stablecoins
            'page': 1,
            'sparkline': False
        }
        
        response = requests.get(url, params=params, timeout=10)
        data = response.json()
        
        # Filter out stablecoins and get top 10
        stablecoin_keywords = ['usd', 'usdt', 'usdc', 'dai', 'busd', 'ust', 'tusd']
        liquid_cryptos = []
        
        for crypto in data:
            symbol = crypto['symbol'].lower()
            name = crypto['name'].lower()
            
            # Skip if it's a stablecoin
            is_stablecoin = any(keyword in symbol or keyword in name 
                              for keyword in stablecoin_keywords)
            
            if not is_stablecoin and crypto['id'] != 'bitcoin':
                liquid_cryptos.append({
                    'id': crypto['id'],
                    'symbol': crypto['symbol'].upper(),
                    'name': crypto['name']
                })
            
            if len(liquid_cryptos) >= 10:
                break
        
        # Add Bitcoin as the first asset
        bitcoin = {
            'id': 'bitcoin',
            'symbol': 'BTC',
            'name': 'Bitcoin'
        }
        
        return [bitcoin] + liquid_cryptos
    
    except Exception as e:
        print(f"Error fetching liquid cryptos: {e}")
        # Fallback list of major cryptocurrencies
        return [
            {'id': 'bitcoin', 'symbol': 'BTC', 'name': 'Bitcoin'},
            {'id': 'ethereum', 'symbol': 'ETH', 'name': 'Ethereum'},
            {'id': 'binancecoin', 'symbol': 'BNB', 'name': 'Binance Coin'},
            {'id': 'ripple', 'symbol': 'XRP', 'name': 'XRP'},
            {'id': 'cardano', 'symbol': 'ADA', 'name': 'Cardano'},
            {'id': 'solana', 'symbol': 'SOL', 'name': 'Solana'},
            {'id': 'polkadot', 'symbol': 'DOT', 'name': 'Polkadot'},
            {'id': 'dogecoin', 'symbol': 'DOGE', 'name': 'Dogecoin'},
            {'id': 'avalanche-2', 'symbol': 'AVAX', 'name': 'Avalanche'},
            {'id': 'chainlink', 'symbol': 'LINK', 'name': 'Chainlink'},
            {'id': 'litecoin', 'symbol': 'LTC', 'name': 'Litecoin'}
        ]

def fetch_crypto_data(coin_id, start_date, end_date):
    """Fetch historical price data for a cryptocurrency"""
    try:
        url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart"
        params = {
            'vs_currency': 'usd',
            'days': (end_date - start_date).days,
            'interval': 'daily'
        }
        
        response = requests.get(url, params=params, timeout=15)
        data = response.json()
        
        # Extract prices and dates
        prices = [entry[1] for entry in data['prices']]
        dates = [datetime.datetime.fromtimestamp(entry[0]/1000) for entry in data['prices']]
        
        return pd.DataFrame({
            'date': dates,
            'price': prices
        }).set_index('date')
    
    except Exception as e:
        print(f"Error fetching data for {coin_id}: {e}")
        return None

def calculate_relative_returns_and_covariance(cryptos_data):
    """Calculate relative price changes and covariance with Bitcoin"""
    # Combine all prices into a single DataFrame
    combined_data = pd.DataFrame()
    
    for symbol, data in cryptos_data.items():
        combined_data[symbol] = data['price']
    
    # Calculate daily returns
    returns_data = combined_data.pct_change().dropna()
    
    # Calculate covariance with Bitcoin
    btc_returns = returns_data['BTC']
    covariance_data = {}
    
    for symbol in returns_data.columns:
        if symbol != 'BTC':
            cov = returns_data[symbol].cov(btc_returns)
            correlation = returns_data[symbol].corr(btc_returns)
            covariance_data[symbol] = {
                'covariance': cov,
                'correlation': correlation
            }
    
    return returns_data, covariance_data

def create_plot(returns_data, cryptos_info):
    """Create a matplotlib plot of relative price changes"""
    plt.style.use('dark_background')
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Plot 1: Cumulative returns
    cumulative_returns = (1 + returns_data).cumprod()
    
    for symbol in cumulative_returns.columns:
        if symbol != 'BTC':
            crypto_name = next((crypto['name'] for crypto in cryptos_info 
                              if crypto['symbol'] == symbol), symbol)
            ax1.plot(cumulative_returns.index, cumulative_returns[symbol], 
                    label=f'{symbol} ({crypto_name})', linewidth=2)
    
    # Plot Bitcoin separately for emphasis
    ax1.plot(cumulative_returns.index, cumulative_returns['BTC'], 
            label='BTC (Bitcoin)', linewidth=3, color='orange', linestyle='--')
    
    ax1.set_title('Cumulative Relative Price Changes (Jan 2022 - Sep 2023)', 
                 fontsize=14, fontweight='bold', pad=20)
    ax1.set_ylabel('Cumulative Returns', fontsize=12)
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.grid(True, alpha=0.3)
    ax1.tick_params(axis='x', rotation=45)
    
    # Plot 2: Daily returns heatmap-style
    recent_returns = returns_data.tail(30)  # Last 30 days
    symbols = [col for col in recent_returns.columns if col != 'BTC']
    
    # Create a simple line plot of recent returns
    for symbol in symbols:
        crypto_name = next((crypto['name'] for crypto in cryptos_info 
                          if crypto['symbol'] == symbol), symbol)
        ax2.plot(recent_returns.index, recent_returns[symbol] * 100, 
                label=f'{symbol}', alpha=0.7, linewidth=1.5)
    
    ax2.plot(recent_returns.index, recent_returns['BTC'] * 100, 
            label='BTC', linewidth=2, color='orange')
    
    ax2.set_title('Recent Daily Returns (%) - Last 30 Days', 
                 fontsize=14, fontweight='bold', pad=20)
    ax2.set_ylabel('Daily Return (%)', fontsize=12)
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax2.grid(True, alpha=0.3)
    ax2.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    
    # Save plot to bytes
    img_bytes = io.BytesIO()
    plt.savefig(img_bytes, format='png', dpi=150, bbox_inches='tight')
    img_bytes.seek(0)
    plt.close()
    
    return img_bytes

def update_data():
    """Background task to update crypto data periodically"""
    global latest_data, latest_covariance
    
    while True:
        try:
            print("Updating crypto data...")
            
            # Define date range
            start_date = datetime.datetime(2022, 1, 1)
            end_date = datetime.datetime(2023, 9, 30)
            
            # Get most liquid cryptocurrencies
            cryptos = get_most_liquid_cryptos()
            print(f"Fetching data for: {[crypto['symbol'] for crypto in cryptos]}")
            
            # Fetch data for all cryptocurrencies
            cryptos_data = {}
            cryptos_info = []
            
            for crypto in cryptos:
                print(f"Fetching {crypto['symbol']}...")
                data = fetch_crypto_data(crypto['id'], start_date, end_date)
                if data is not None and not data.empty:
                    cryptos_data[crypto['symbol']] = data
                    cryptos_info.append(crypto)
                time.sleep(1)  # Rate limiting
            
            if len(cryptos_data) > 1:
                # Calculate returns and covariance
                returns_data, covariance_data = calculate_relative_returns_and_covariance(cryptos_data)
                
                latest_data = {
                    'returns': returns_data,
                    'cryptos_info': cryptos_info,
                    'covariance_data': covariance_data
                }
                latest_covariance = covariance_data
                
                print("Data update completed successfully!")
                
            else:
                print("Insufficient data fetched")
                
        except Exception as e:
            print(f"Error updating data: {e}")
        
        # Update every hour
        time.sleep(3600)

@app.route('/')
def index():
    """Main page with the graph"""
    global latest_data
    
    if latest_data is None:
        return """
        <html>
            <head><title>Crypto Analysis</title></head>
            <body style="background-color: #1e1e1e; color: white; font-family: Arial;">
                <h1>Cryptocurrency Analysis</h1>
                <p>Data is being loaded. Please refresh in a few moments...</p>
            </body>
        </html>
        """
    
    # Create and return the plot
    img_bytes = create_plot(latest_data['returns'], latest_data['cryptos_info'])
    return send_file(img_bytes, mimetype='image/png')

@app.route('/data')
def get_data():
    """API endpoint to get covariance data"""
    global latest_covariance
    
    if latest_covariance is None:
        return {"error": "Data not available yet"}, 503
    
    return {
        "covariance_data": latest_covariance,
        "last_updated": datetime.datetime.now().isoformat()
    }

@app.route('/status')
def status():
    """Status page"""
    global latest_data
    
    status_info = {
        "status": "running",
        "last_update": datetime.datetime.now().isoformat(),
        "assets_loaded": len(latest_data['cryptos_info']) if latest_data else 0
    }
    
    if latest_data and latest_covariance:
        status_info["covariance_summary"] = {
            symbol: {
                "covariance": float(data['covariance']),
                "correlation": float(data['correlation'])
            }
            for symbol, data in latest_covariance.items()
        }
    
    html = f"""
    <html>
        <head><title>Status - Crypto Analysis</title></head>
        <body style="background-color: #1e1e1e; color: white; font-family: Arial; padding: 20px;">
            <h1>Cryptocurrency Analysis Status</h1>
            <p><strong>Status:</strong> {status_info['status']}</p>
            <p><strong>Last Update:</strong> {status_info['last_update']}</p>
            <p><strong>Assets Loaded:</strong> {status_info['assets_loaded']}</p>
            
            <h2>Covariance with Bitcoin</h2>
    """
    
    if 'covariance_summary' in status_info:
        html += "<table border='1' style='border-collapse: collapse; width: 100%;'>"
        html += "<tr><th>Asset</th><th>Covariance</th><th>Correlation</th></tr>"
        for symbol, data in status_info['covariance_summary'].items():
            html += f"<tr><td>{symbol}</td><td>{data['covariance']:.6f}</td><td>{data['correlation']:.4f}</td></tr>"
        html += "</table>"
    
    html += """
            <br>
            <p><a href="/" style="color: #4CAF50;">View Graph</a></p>
            <p><a href="/data" style="color: #4CAF50;">Raw Data (JSON)</a></p>
        </body>
    </html>
    """
    
    return html

if __name__ == '__main__':
    # Start background data update thread
    update_thread = Thread(target=update_data, daemon=True)
    update_thread.start()
    
    # Wait a bit for initial data load
    print("Starting server... Initial data loading may take a few minutes.")
    time.sleep(5)
    
    # Start Flask server
    app.run(host='0.0.0.0', port=8080, debug=False)
