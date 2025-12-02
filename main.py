import pandas as pd
import numpy as np
from flask import Flask, render_template
import plotly.graph_objs as go
import plotly.utils
import json
from datetime import datetime
import requests
import io
import sys

app = Flask(__name__)


def fetch_binance_ohlcv(symbol='BTCUSDT', interval='1d', start_date='2017-01-01'):
    """Fetch OHLCV data from Binance public API"""
    base_url = 'https://api.binance.com/api/v3/klines'
    
    # Validate symbol format
    if not symbol.isupper() or not symbol.endswith('USDT'):
        print(f"Warning: Symbol {symbol} may not be valid Binance format")
    
    # Convert start_date to timestamp
    start_timestamp = int(datetime.strptime(start_date, '%Y-%m-%d').timestamp() * 1000)
    
    # Get current timestamp
    end_timestamp = int(datetime.now().timestamp() * 1000)
    
    print(f"Fetching {symbol} data from {start_date} to now...")
    
    all_data = []
    current_start = start_timestamp
    limit = 1000  # Binance max per request
    
    print(f"Fetching data from {start_date} to now...")
    
    while current_start < end_timestamp:
        params = {
            'symbol': symbol,
            'interval': interval,
            'startTime': current_start,
            'limit': limit
        }
        
        try:
            response = requests.get(base_url, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            if not data:
                print(f"No data returned for batch starting at {current_start}")
                break
                
            all_data.extend(data)
            print(f"Fetched {len(data)} candles, total so far: {len(all_data)}")
            
            # Update start time for next batch
            current_start = data[-1][0] + 1
            
            # If we got less than limit, we're done
            if len(data) < limit:
                print(f"Received less than limit ({len(data)} < {limit}), ending fetch")
                break
                
        except requests.exceptions.RequestException as e:
            print(f"Request error: {e}")
            break
        except Exception as e:
            print(f"Unexpected error: {e}")
            break
    
    if not all_data:
        raise Exception("No data fetched from Binance API")
    
    print(f"Total candles fetched: {len(all_data)}")
    
    # Convert to DataFrame
    columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume',
               'close_time', 'quote_asset_volume', 'number_of_trades',
               'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore']
    
    df = pd.DataFrame(all_data, columns=columns)
    
    # Convert types
    numeric_cols = ['open', 'high', 'low', 'close', 'volume']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col])
    
    # Convert timestamp to datetime
    df['date'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('date', inplace=True)
    df.sort_index(inplace=True)
    
    print(f"Data range: {df.index[0]} to {df.index[-1]}")
    print(f"Total rows: {len(df)}")
    
    if len(df) < 365:
        print(f"Warning: Only {len(df)} data points, need at least 365 for full SMA calculations")
    
    return df[['open', 'high', 'low', 'close', 'volume']]
    
    # Check if we have enough data for SMA calculations
    if len(df) < 365:
        print(f"Warning: Only {len(df)} data points, need at least 365 for full SMA calculations")
        print(f"Consider using an earlier start_date or different interval")
    
    return df[['open', 'high', 'low', 'close', 'volume']]


def calculate_returns(df):
    """Calculate returns based on SMA conditions"""
    print(f"\n=== CALCULATING RETURNS ===")
    print(f"Input DataFrame shape: {df.shape}")
    
    # Calculate SMAs
    df['sma_120'] = df['close'].rolling(window=120).mean()
    df['sma_365'] = df['close'].rolling(window=365).mean()
    
    print(f"\nSMA calculation complete")
    print(f"First few SMA values:")
    print(df[['close', 'sma_120', 'sma_365']].head(10))
    
    # Calculate daily returns
    df['daily_return'] = df['close'].pct_change()
    
    # Initialize strategy returns
    df['strategy_return'] = 0.0
    
    # Apply strategy rules
    # Condition 1: Price above both SMAs -> positive returns
    above_both = (df['close'] > df['sma_120']) & (df['close'] > df['sma_365'])
    df.loc[above_both, 'strategy_return'] = df.loc[above_both, 'daily_return']
    
    # Condition 2: Price below both SMAs -> negative returns (inverse of daily return)
    below_both = (df['close'] < df['sma_120']) & (df['close'] < df['sma_365'])
    df.loc[below_both, 'strategy_return'] = -df.loc[below_both, 'daily_return']
    
    # Debug: Show strategy return distribution
    print(f"Strategy return range: [{df['strategy_return'].min():.4f}, {df['strategy_return'].max():.4f}]")
    
    # All other days remain 0 (already initialized)
    
    print(f"\nStrategy conditions:")
    print(f"Days above both SMAs: {above_both.sum()}")
    print(f"Days below both SMAs: {below_both.sum()}")
    print(f"Days with zero returns: {(df['strategy_return'] == 0).sum()}")
    
    # Calculate cumulative returns
    df['cumulative_return'] = (1 + df['strategy_return']).cumprod() - 1
    
    # Calculate buy and hold returns for comparison
    df['buy_hold_return'] = (1 + df['daily_return']).cumprod() - 1
    
    print(f"\nReturn statistics:")
    print(f"Final strategy return: {df['cumulative_return'].iloc[-1]*100:.2f}%")
    print(f"Final buy & hold return: {df['buy_hold_return'].iloc[-1]*100:.2f}%")
    
    return df


@app.route('/')
def index():
    """Main endpoint that displays the analysis"""
    try:
        # Fetch data
        df = fetch_binance_ohlcv()
        
        # Fetch data with earlier start date for better SMA calculations
        df = fetch_binance_ohlcv(start_date='2017-01-01')
        
        if len(df) == 0:
            return "Error: No data fetched from Binance"
        
        # Calculate returns
        df = calculate_returns(df)
        
        # Drop NaN values (from SMA calculations)
        df_clean = df.dropna()
        
        if len(df_clean) == 0:
            return "Error: No valid data after SMA calculations. Need at least 365 days of data."
        
        print(f"Clean data points for charts: {len(df_clean)}")
        
        # Create price chart
        price_trace = go.Scatter(
            x=df_clean.index,
            y=df_clean['close'],
            mode='lines',
            name='BTC Price',
            line=dict(color='blue')
        )
        
        sma_120_trace = go.Scatter(
            x=df_clean.index,
            y=df_clean['sma_120'],
            mode='lines',
            name='120 SMA',
            line=dict(color='orange', dash='dash')
        )
        
        sma_365_trace = go.Scatter(
            x=df_clean.index,
            y=df_clean['sma_365'],
            mode='lines',
            name='365 SMA',
            line=dict(color='red', dash='dash')
        )
        
        price_layout = go.Layout(
            title='BTC Price with SMAs',
            xaxis=dict(title='Date'),
            yaxis=dict(title='Price (USDT)'),
            hovermode='x unified'
        )
        
        price_fig = go.Figure(data=[price_trace, sma_120_trace, sma_365_trace], layout=price_layout)
        price_graph = json.dumps(price_fig, cls=plotly.utils.PlotlyJSONEncoder)
        
        # Create returns chart
        strategy_trace = go.Scatter(
            x=df_clean.index,
            y=df_clean['cumulative_return'] * 100,  # Convert to percentage
            mode='lines',
            name='Strategy Returns',
            line=dict(color='green')
        )
        
        buy_hold_trace = go.Scatter(
            x=df_clean.index,
            y=df_clean['buy_hold_return'] * 100,  # Convert to percentage
            mode='lines',
            name='Buy & Hold Returns',
            line=dict(color='gray', dash='dash')
        )
        
        returns_layout = go.Layout(
            title='Cumulative Returns Comparison',
            xaxis=dict(title='Date'),
            yaxis=dict(title='Cumulative Return (%)'),
            hovermode='x unified'
        )
        
        returns_fig = go.Figure(data=[strategy_trace, buy_hold_trace], layout=returns_layout)
        returns_graph = json.dumps(returns_fig, cls=plotly.utils.PlotlyJSONEncoder)
        
        # Calculate statistics
        total_days = len(df_clean)
        positive_days = len(df_clean[df_clean['strategy_return'] > 0])
        negative_days = len(df_clean[df_clean['strategy_return'] < 0])
        zero_days = len(df_clean[df_clean['strategy_return'] == 0])
        
        final_strategy_return = df_clean['cumulative_return'].iloc[-1] * 100
        final_buy_hold_return = df_clean['buy_hold_return'].iloc[-1] * 100
        
        stats = {
            'total_days': total_days,
            'positive_days': positive_days,
            'negative_days': negative_days,
            'zero_days': zero_days,
            'final_strategy_return': round(final_strategy_return, 2),
            'final_buy_hold_return': round(final_buy_hold_return, 2),
            'data_start': df_clean.index[0].strftime('%Y-%m-%d'),
            'data_end': df_clean.index[-1].strftime('%Y-%m-%d'),
            'current_price': round(df_clean['close'].iloc[-1], 2)
        }
        
        print(f"Stats calculated: {stats}")
        
        return render_template('index.html', 
                             price_graph=price_graph,
                             returns_graph=returns_graph,
                             stats=stats)
        
    except Exception as e:
        print(f"Error in index route: {str(e)}")
        return f"Error: {str(e)}"


if __name__ == '__main__':
    print("Starting Flask server on port 8080...")
    print(f"Python version: {sys.version}")
    print(f"Pandas version: {pd.__version__}")
    print(f"Flask version: {Flask.__version__}")
    print("\nServer will output detailed console logs for debugging.")
    print("Visit http://localhost:8080 to see the analysis.\n")
    app.run(host='0.0.0.0', port=8080, debug=True)