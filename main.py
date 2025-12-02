import pandas as pd
import numpy as np
from flask import Flask, render_template
import plotly.graph_objs as go
import plotly.utils
import json
from datetime import datetime
import requests
import io

app = Flask(__name__)


def fetch_binance_ohlcv(symbol='BTCUSDT', interval='1d', start_date='2018-01-01'):
    """Fetch OHLCV data from Binance public API"""
    base_url = 'https://api.binance.com/api/v3/klines'
    
    # Convert start_date to timestamp
    start_timestamp = int(datetime.strptime(start_date, '%Y-%m-%d').timestamp() * 1000)
    
    # Get current timestamp
    end_timestamp = int(datetime.now().timestamp() * 1000)
    
    all_data = []
    current_start = start_timestamp
    limit = 1000  # Binance max per request
    
    while current_start < end_timestamp:
        params = {
            'symbol': symbol,
            'interval': interval,
            'startTime': current_start,
            'limit': limit
        }
        
        response = requests.get(base_url, params=params)
        if response.status_code != 200:
            raise Exception(f"API request failed: {response.status_code}")
        
        data = response.json()
        if not data:
            break
            
        all_data.extend(data)
        
        # Update start time for next batch
        current_start = data[-1][0] + 1
        
        # If we got less than limit, we're done
        if len(data) < limit:
            break
    
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
    
    return df[['open', 'high', 'low', 'close', 'volume']]


def calculate_returns(df):
    """Calculate returns based on SMA conditions"""
    # Calculate SMAs
    df['sma_120'] = df['close'].rolling(window=120).mean()
    df['sma_365'] = df['close'].rolling(window=365).mean()
    
    # Calculate daily returns
    df['daily_return'] = df['close'].pct_change()
    
    # Initialize strategy returns
    df['strategy_return'] = 0.0
    
    # Apply strategy rules
    # Condition 1: Price above both SMAs -> positive returns
    above_both = (df['close'] > df['sma_120']) & (df['close'] > df['sma_365'])
    df.loc[above_both, 'strategy_return'] = df.loc[above_both, 'daily_return']
    
    # Condition 2: Price below both SMAs -> negative returns
    below_both = (df['close'] < df['sma_120']) & (df['close'] < df['sma_365'])
    df.loc[below_both, 'strategy_return'] = -df.loc[below_both, 'daily_return']
    
    # All other days remain 0 (already initialized)
    
    # Calculate cumulative returns
    df['cumulative_return'] = (1 + df['strategy_return']).cumprod() - 1
    
    # Calculate buy and hold returns for comparison
    df['buy_hold_return'] = (1 + df['daily_return']).cumprod() - 1
    
    return df


@app.route('/')
def index():
    """Main endpoint that displays the analysis"""
    try:
        # Fetch data
        df = fetch_binance_ohlcv()
        
        # Calculate returns
        df = calculate_returns(df)
        
        # Drop NaN values (from SMA calculations)
        df_clean = df.dropna()
        
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
        
        return render_template('index.html', 
                             price_graph=price_graph,
                             returns_graph=returns_graph,
                             stats=stats)
        
    except Exception as e:
        return f"Error: {str(e)}"


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=False)