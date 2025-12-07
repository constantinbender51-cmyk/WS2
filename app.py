import ccxt
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from dash import Dash, dcc, html
from datetime import datetime

# --- Configuration ---
SYMBOL = 'BTC/USDT'
TIMEFRAME = '1d'  # Daily candles. Change to '4h' or '1h' for more granularity.
START_DATE = '2018-01-01 00:00:00'
SMA_PERIODS = sorted(set([round(1.5 ** x) for x in range(1, 23)]))  # x from 1 to 22
PORT = 8080

def fetch_data(symbol, timeframe, start_str):
    """
    Fetches historical OHLCV data from Binance with pagination handling.
    """
    print(f"Fetching {symbol} data since {start_str}...")
    exchange = ccxt.binance({'enableRateLimit': True})
    since = exchange.parse8601(start_str)
    all_ohlcv = []
    
    while True:
        try:
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=1000)
            if not ohlcv:
                break
            
            all_ohlcv.extend(ohlcv)
            since = ohlcv[-1][0] + 1  # Move to next timestamp
            
            # Print progress
            last_date = datetime.fromtimestamp(ohlcv[-1][0] / 1000)
            print(f"Fetched up to {last_date}")
            
            # Stop if we reached current time (approx)
            if len(ohlcv) < 1000:
                break
        except Exception as e:
            print(f"Error fetching data: {e}")
            break

    df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    return df

def compute_sma_metrics(df):
    """
    Vectorized computation of SMAs using periods round(1.25^x) for x=1..22
    and the count below price.
    """
    print("Computing SMAs and regime metrics...")
    
    # optimize: store SMAs in a separate DataFrame for vectorized comparison
    sma_dict = {}
    for period in SMA_PERIODS:
        sma_dict[f'SMA_{period}'] = df['close'].rolling(window=period).mean()
    
    df_smas = pd.DataFrame(sma_dict, index=df.index)
    
    # 1. Count how many SMAs are BELOW the current close price
    # We use .lt() (less than) for broadcasting comparison
    # sum(axis=1) counts the True values per row
    df['count_below'] = df_smas.lt(df['close'], axis=0).sum(axis=1)
    
    # 2. The "Rest" is simply Total SMAs - Count Below
    total_smas = len(SMA_PERIODS)
    df['count_above'] = total_smas - df['count_below']
    
    return df

def create_figure(df):
    """
    Creates the Dual-Axis Plotly figure.
    """
    fig = go.Figure()

    # --- Stacked Area Chart (Counts) on Primary Y-Axis ---
    # "Below" Count (Green area)
    fig.add_trace(go.Scatter(
        x=df.index, y=df['count_below'],
        name='SMAs Below Price',
        stackgroup='one', # Enable stacking
        mode='none',      # No lines, just filled area
        fillcolor='rgba(0, 200, 100, 0.5)', # Semi-transparent Green
        hoverinfo='x+y'
    ))

    # "Above" Count (Red/Grey area) - The "Rest"
    fig.add_trace(go.Scatter(
        x=df.index, y=df['count_above'],
        name='SMAs Above Price',
        stackgroup='one', 
        mode='none',
        fillcolor='rgba(200, 50, 50, 0.3)', # Semi-transparent Red
        hoverinfo='x+y'
    ))

    # --- Price Line on Secondary Y-Axis ---
    fig.add_trace(go.Scatter(
        x=df.index, y=df['close'],
        name='Price (BTC)',
        line=dict(color='white', width=1.5),
        yaxis='y2'
    ))

    # Layout Configuration
    fig.update_layout(
        title=f'Market Regime: SMA Counts ({SMA_PERIODS[0]}-{SMA_PERIODS[-1]}) vs Price',
        template='plotly_dark',
        height=700,
        hovermode='x unified',
        yaxis=dict(
            title='Count of SMAs',
            range=[0, len(SMA_PERIODS)],
            fixedrange=True
        ),
        yaxis2=dict(
            title='Price (USDT)',
            overlaying='y',
            side='right',
            showgrid=False
        ),
        legend=dict(x=0, y=1.05, orientation='h')
    )
    return fig

# --- Main Execution ---
if __name__ == '__main__':
    # 1. Fetch
    df = fetch_data(SYMBOL, TIMEFRAME, START_DATE)
    
    # 2. Compute
    df = compute_sma_metrics(df)
    
    # 3. Server Setup
    app = Dash(__name__)
    
    app.layout = html.Div([
        dcc.Graph(figure=create_figure(df), style={'height': '95vh'})
    ], style={'backgroundColor': '#111', 'margin': '-8px'})

    print(f"Starting server on port {PORT}...")
    app.run(debug=False, port=PORT, host='0.0.0.0')
