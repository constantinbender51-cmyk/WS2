import io
import base64
import logging
from datetime import datetime

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from flask import Flask, render_template, request
from binance.client import Client
from sklearn.preprocessing import MinMaxScaler

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

app = Flask(__name__)

# Binance API (using placeholders - replace with your actual keys or environment variables)
# For security, never hardcode API keys directly in production code.
# Consider using environment variables: os.environ.get('BINANCE_API_KEY')
API_KEY = ''  # Your Binance API Key
API_SECRET = '' # Your Binance API Secret

# Initialize Binance client (will work in read-only mode if keys are empty)
try:
    client = Client(API_KEY, API_SECRET)
    logging.info("Binance client initialized.")
except Exception as e:
    logging.error(f"Failed to initialize Binance client: {e}. Data fetching might fail.")

@app.route('/')
def index():
    symbol = request.args.get('symbol', 'BTCUSDT')
    interval = request.args.get('interval', Client.KLINE_INTERVAL_1DAY)
    start_str = request.args.get('start_date', '1 Jan, 2018')

    try:
        df = fetch_binance_data(symbol, interval, start_str)
        if df is None or df.empty:
            return render_template('index.html', plot_image=None, error_message="Could not fetch data or data is empty.")

        processed_df = process_data(df)
        plot_image_base64 = create_plot(processed_df, symbol)
        
        # Get last few rows of data for display details
        details_df = processed_df.tail(5).to_html(classes='table table-striped', justify='left')

        return render_template('index.html', 
                               plot_image=plot_image_base64,
                               symbol=symbol,
                               interval=interval,
                               start_str=start_str,
                               details_df=details_df)
    except Exception as e:
        logging.error(f"An error occurred: {e}")
        return render_template('index.html', plot_image=None, error_message=f"An error occurred: {e}")

def fetch_binance_data(symbol, interval, start_str):
    logging.info(f"Fetching {symbol} {interval} data from {start_str}...")
    try:
        klines = client.get_historical_klines(symbol, interval, start_str)
        
        if not klines:
            logging.warning(f"No historical data found for {symbol} starting {start_str}.")
            return None

        df = pd.DataFrame(klines, columns=[
            'open_time', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_asset_volume', 'number_of_trades',
            'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
        ])

        df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
        df['close_time'] = pd.to_datetime(df['close_time'], unit='ms')
        df.set_index('open_time', inplace=True)

        # Convert OHLCV columns to numeric
        numeric_cols = ['open', 'high', 'low', 'close', 'volume']
        df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')
        
        logging.info(f"Fetched {len(df)} rows of data for {symbol}.")
        return df
    except Exception as e:
        logging.error(f"Error fetching Binance data: {e}")
        return None

def process_data(df):
    logging.info("Processing data: calculating SMAs and strategy returns...")
    # Calculate SMAs
    df['SMA_120'] = df['close'].rolling(window=120).mean()
    df['SMA_365'] = df['close'].rolling(window=365).mean()

    # Calculate daily returns (close-open)/open
    df['daily_return_raw'] = (df['close'] - df['open']) / df['open']

    # Calculate strategy return
    df['strategy_return'] = 0.0
    
    # Condition 1: close price > both SMAs
    cond_above = (df['close'] > df['SMA_120']) & (df['close'] > df['SMA_365'])
    df.loc[cond_above, 'strategy_return'] = df.loc[cond_above, 'daily_return_raw']
    
    # Condition 2: close price < both SMAs
    cond_below = (df['close'] < df['SMA_120']) & (df['close'] < df['SMA_365'])
    df.loc[cond_below, 'strategy_return'] = -df.loc[cond_below, 'daily_return_raw']
    
    # Drop rows with NaN in critical columns after calculations
    df.dropna(subset=['close', 'SMA_120', 'SMA_365', 'strategy_return'], inplace=True)
    logging.info("Data processing complete.")
    return df

def create_plot(df, symbol):
    logging.info("Creating plot...")
    fig, ax1 = plt.subplots(figsize=(14, 7))

    # Plot Close Price and SMAs on the primary Y-axis
    ax1.plot(df.index, df['close'], label=f'{symbol} Close Price', color='blue', linewidth=1)
    ax1.plot(df.index, df['SMA_120'], label='120-day SMA', color='green', linestyle='--', linewidth=1)
    ax1.plot(df.index, df['SMA_365'], label='365-day SMA', color='red', linestyle=':', linewidth=1)
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Price', color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')
    ax1.grid(True, linestyle='--', alpha=0.6)

    # Create a secondary Y-axis for Strategy Returns
    ax2 = ax1.twinx()
    ax2.plot(df.index, df['strategy_return'], label='Strategy Return', color='purple', alpha=0.7, linewidth=0.8)
    ax2.set_ylabel('Strategy Return', color='purple')
    ax2.tick_params(axis='y', labelcolor='purple')
    
    # Combine legends
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc='upper left')

    plt.title(f'{symbol} Price, SMAs and Strategy Returns (Daily)')
    plt.tight_layout()

    # Save plot to a bytes buffer and encode as base64
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close(fig)  # Close the figure to free up memory
    buf.seek(0)
    plot_image = base64.b64encode(buf.read()).decode('utf-8')
    logging.info("Plot created and encoded.")
    return plot_image

# Create templates directory and index.html if they don't exist
# This part is typically done manually or by a build script, 
# but included here to ensure the file exists if running this directly.
import os
if not os.path.exists('templates'):
    os.makedirs('templates')

# Main execution block for Flask app
if __name__ == '__main__':
    # This is for local development. Gunicorn will handle this in production.
    app.run(debug=True, host='0.0.0.0', port=8080)
