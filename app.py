import ccxt
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from dash import Dash, dcc, html
from datetime import datetime

# --- Configuration ---
SYMBOL = 'BTC/USDT'
TIMEFRAME = '1d'  
START_DATE = '2018-01-01 00:00:00'
SMA_PERIODS = sorted(set([2 ** x for x in range(0, 9)]))  # 1, 2, 4, 8, 16, 32, 64, 128, 256
PORT = 8080
INITIAL_CAPITAL = 10000

def fetch_data(symbol, timeframe, start_str):
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
            since = ohlcv[-1][0] + 1
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
    print("Computing SMAs and regime metrics...")
    sma_dict = {}
    for period in SMA_PERIODS:
        sma_dict[f'SMA_{period}'] = df['close'].rolling(window=period).mean()
    
    df_smas = pd.DataFrame(sma_dict, index=df.index)
    
    # Count SMAs BELOW price
    df['count_below'] = df_smas.lt(df['close'], axis=0).sum(axis=1)
    
    # The "Rest" is Total SMAs - Count Below
    total_smas = len(SMA_PERIODS)
    df['count_above'] = total_smas - df['count_below']
    
    return df

def backtest_strategy(df):
    """
    Calculates strategy returns based on the count of SMAs below price.
    Long if Yesterday > 4, Short if Yesterday < 4.
    """
    print("Running backtest...")
    
    # 1. Calculate underlying asset daily returns
    df['pct_change'] = df['close'].pct_change()
    
    # 2. Define Signal (1 = Long, -1 = Short, 0 = Flat)
    # This signal is based on the current row's data
    conditions = [
        (df['count_below'] > 4),
        (df['count_below'] < 4)
    ]
    choices = [1, -1]
    # Default is 0 (Flat) if count == 4
    df['raw_signal'] = np.select(conditions, choices, default=0)
    
    # 3. Shift Signal
    # We trade TODAY based on YESTERDAY's signal. 
    df['position'] = df['raw_signal'].shift(1)
    
    # 4. Calculate Strategy Returns
    df['strategy_return'] = df['position'] * df['pct_change']
    df['strategy_return'] = df['strategy_return'].fillna(0)
    
    # 5. Cumulative Capital
    df['capital'] = INITIAL_CAPITAL * (1 + df['strategy_return']).cumprod()
    
    return df

def create_regime_figure(df):
    fig = go.Figure()

    # Stacked Area: Below (Green)
    fig.add_trace(go.Scatter(
        x=df.index, y=df['count_below'],
        name='SMAs Below Price',
        stackgroup='one',
        mode='none',
        fillcolor='rgba(0, 200, 100, 0.5)', 
        hoverinfo='x+y'
    ))

    # Stacked Area: Above (Red)
    fig.add_trace(go.Scatter(
        x=df.index, y=df['count_above'],
        name='SMAs Above Price',
        stackgroup='one', 
        mode='none',
        fillcolor='rgba(200, 50, 50, 0.3)', 
        hoverinfo='x+y'
    ))

    # Price Line
    fig.add_trace(go.Scatter(
        x=df.index, y=df['close'],
        name='Price (BTC)',
        line=dict(color='white', width=1.5),
        yaxis='y2'
    ))

    fig.update_layout(
        title=f'Market Regime: SMA Counts vs Price',
        template='plotly_dark',
        height=500, # Reduced height to fit both charts
        hovermode='x unified',
        yaxis=dict(title='Count of SMAs', range=[0, len(SMA_PERIODS)], fixedrange=True),
        yaxis2=dict(title='Price (USDT)', overlaying='y', side='right', showgrid=False),
        legend=dict(x=0, y=1.05, orientation='h'),
        margin=dict(l=50, r=50, t=50, b=20)
    )
    return fig

def create_capital_figure(df):
    fig = go.Figure()

    # Capital Curve
    fig.add_trace(go.Scatter(
        x=df.index, y=df['capital'],
        name='Strategy Capital',
        line=dict(color='#00d4ff', width=2),
        fill='tozeroy',
        fillcolor='rgba(0, 212, 255, 0.1)'
    ))

    fig.update_layout(
        title=f'Strategy Performance (Start: ${INITIAL_CAPITAL})',
        template='plotly_dark',
        height=400,
        hovermode='x unified',
        yaxis=dict(title='Capital (USDT)', type='log'), # Log scale usually better for crypto
        margin=dict(l=50, r=50, t=50, b=50)
    )
    return fig

# --- Main Execution ---
if __name__ == '__main__':
    # 1. Fetch
    df = fetch_data(SYMBOL, TIMEFRAME, START_DATE)
    
    # 2. Compute
    df = compute_sma_metrics(df)
    
    # 3. Backtest
    df = backtest_strategy(df)
    
    # 4. Server Setup
    app = Dash(__name__)
    
    app.layout = html.Div([
        # Graph 1: Regime
        dcc.Graph(figure=create_regime_figure(df)),
        
        # Graph 2: Capital
        dcc.Graph(figure=create_capital_figure(df))
    ], style={'backgroundColor': '#111', 'padding': '20px'})

    print(f"Starting server on port {PORT}...")
    app.run(debug=False, port=PORT, host='0.0.0.0')
