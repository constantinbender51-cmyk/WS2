from flask import Flask, render_template, jsonify
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io
import base64
import os
import subprocess
import requests
from datetime import datetime, timedelta
import time
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

def fetch_btc_data():
    """Fetch BTC price data from 2018 to 2025"""
    logger.info("Fetching BTC price data from 2018 to 2025...")
    
    if os.path.exists('btc_data.csv'):
        logger.info("Loading existing data file...")
        df = pd.read_csv('btc_data.csv')
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
        return df
    
    base_url = 'https://api.binance.com/api/v3/klines'
    symbol = 'BTCUSDT'
    interval = '1d'
    start_date = datetime(2018, 1, 1)
    end_date = datetime(2025, 12, 31)
    
    all_data = []
    current_start = start_date
    
    while current_start <= end_date:
        start_time = int(current_start.timestamp() * 1000)
        end_time = int((current_start + timedelta(days=1000)).timestamp() * 1000)
        if end_time > int(end_date.timestamp() * 1000):
            end_time = int(end_date.timestamp() * 1000)
        
        params = {
            'symbol': symbol,
            'interval': interval,
            'startTime': start_time,
            'endTime': end_time,
            'limit': 1000
        }
        
        headers = {'User-Agent': 'BTC-Trading-Strategy/1.0'}
        
        try:
            response = requests.get(base_url, params=params, headers=headers, timeout=10)
            if response.status_code == 200:
                data = response.json()
                if data:
                    for candle in data:
                        timestamp = candle[0]
                        open_price = float(candle[1])
                        high = float(candle[2])
                        low = float(candle[3])
                        close = float(candle[4])
                        volume = float(candle[5])
                        date = datetime.fromtimestamp(timestamp / 1000).strftime('%Y-%m-%d')
                        all_data.append([date, open_price, high, low, close, volume])
                    current_start = datetime.fromtimestamp(data[-1][0] / 1000) + timedelta(days=1)
                else:
                    current_start += timedelta(days=1000)
            else:
                logger.error(f"Error fetching data: {response.status_code}")
                break
        except Exception as e:
            logger.error(f"Error during API call: {e}")
            break
        
        time.sleep(0.15)
    
    if all_data:
        df = pd.DataFrame(all_data, columns=['date', 'open', 'high', 'low', 'close', 'volume'])
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
        df.to_csv('btc_data.csv')
        logger.info(f"Data fetched and saved. Total records: {len(df)}")
        return df
    else:
        raise Exception("Failed to fetch BTC data")

def calculate_strategy(df):
    """Calculate trading strategy - long if price above SMA 365, otherwise short"""
    logger.info("Calculating SMA 365 strategy...")
    
    # Calculate 365-day SMA
    df['sma_365'] = df['close'].rolling(window=365).mean()
    
    # Strategy: Long if price above SMA 365, otherwise short
    df['position'] = np.where(df['close'] > df['sma_365'], 1, -1)
    
    # Calculate daily returns
    df['daily_return'] = df['close'].pct_change()
    
    # Calculate strategy returns (position * daily return)
    df['strategy_return'] = df['position'].shift(1) * df['daily_return']
    
    # Calculate cumulative capital development - update only on signal changes
    initial_capital = 1000
    capital = initial_capital
    capital_series = []
    
    for i in range(len(df)):
        # Only update capital when position changes
        if i > 0 and df['position'].iloc[i] != df['position'].iloc[i-1]:
            # Update capital based on the return since last signal change
            capital = capital_series[-1] * (1 + df['strategy_return'].iloc[i])
        elif i == 0:
            # First day
            capital = initial_capital
        else:
            # Maintain same capital until next signal change
            capital = capital_series[-1]
        
        capital_series.append(capital)
    
    df['capital'] = capital_series
    
    # Fill NaN values
    df['capital'] = df['capital'].fillna(initial_capital)
    
    return df

def create_plot(df):
    """Create visualization of strategy results"""
    logger.info("Creating strategy visualization...")
    
    plt.figure(figsize=(12, 10))
    
    # Plot 1: BTC Price and 365-day SMA
    plt.subplot(3, 1, 1)
    plt.plot(df.index, df['close'], label='BTC Price', color='blue', linewidth=1)
    plt.plot(df.index, df['sma_365'], label='365-day SMA', color='orange', linewidth=1)
    plt.title('BTC Price and 365-day Simple Moving Average')
    plt.ylabel('Price (USD)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Trading Positions
    plt.subplot(3, 1, 2)
    # Plot long positions (green) and short positions (red)
    long_mask = df['position'] == 1
    short_mask = df['position'] == -1
    plt.plot(df.index[long_mask], df['close'][long_mask], label='Long Position', color='green', linewidth=2)
    plt.plot(df.index[short_mask], df['close'][short_mask], label='Short Position', color='red', linewidth=2)
    plt.title('Trading Positions (Long if Price > SMA 365, Otherwise Short)')
    plt.ylabel('Price (USD)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 3: Capital Development
    plt.subplot(3, 1, 3)
    plt.plot(df.index, df['capital'], label='Strategy Capital', color='purple', linewidth=2)
    plt.axhline(y=1000, color='gray', linestyle='--', label='Initial Capital ($1000)')
    plt.title('Capital Development - SMA 365 Strategy')
    plt.ylabel('Capital (USD)')
    plt.xlabel('Date')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Add performance metrics to the plot
    final_capital = df['capital'].iloc[-1]
    total_return = (final_capital - 1000) / 1000 * 100
    
    plt.figtext(0.02, 0.02, f'Final Capital: ${final_capital:,.2f} | Total Return: {total_return:+.2f}% | Strategy: Long if Price > SMA 365, Otherwise Short', 
                fontsize=10, bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    
    plt.tight_layout()
    
    # Convert plot to base64 for web display
    img = io.BytesIO()
    plt.savefig(img, format='png', dpi=100, bbox_inches='tight')
    plt.close()
    img.seek(0)
    
    return base64.b64encode(img.getvalue()).decode()

@app.route('/')
def index():
    """Main page with strategy results"""
    try:
        # Fetch and calculate strategy
        df = fetch_btc_data()
        df = calculate_strategy(df)
        
        # Create visualization
        plot_url = create_plot(df)
        
        # Calculate performance metrics
        final_capital = df['capital'].iloc[-1]
        total_return = (final_capital - 1000) / 1000 * 100
        total_days = len(df)
        trading_days = len(df[df['position'].notna()])
        
        return render_template('index.html', 
                             plot_url=plot_url,
                             final_capital=final_capital,
                             total_return=total_return,
                             total_days=total_days,
                             trading_days=trading_days)
    
    except Exception as e:
        logger.error(f"Error in main route: {e}")
        return render_template('index.html', 
                             error=str(e),
                             plot_url=None)

@app.route('/api/strategy_data')
def api_strategy_data():
    """API endpoint for strategy data"""
    try:
        df = fetch_btc_data()
        df = calculate_strategy(df)
        
        # Return basic strategy performance
        final_capital = df['capital'].iloc[-1]
        total_return = (final_capital - 1000) / 1000 * 100
        
        return jsonify({
            'status': 'success',
            'final_capital': final_capital,
            'total_return_percent': total_return,
            'total_days': len(df),
            'data_points': len(df[df['position'].notna()])
        })
    
    except Exception as e:
        return jsonify({
            'status': 'error',
            'error': str(e)
        })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=False)