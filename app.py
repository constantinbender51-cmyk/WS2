import requests
import pandas as pd
import plotly.graph_objects as go
from dash import Dash, dcc, html
from dash.dependencies import Input, Output
from collections import deque
import statistics

# --- Configuration ---
SYMBOL = "PF_XBTUSD"
URL = "https://futures.kraken.com/derivatives/api/v3/orderbook"
PORT = 8080
UPDATE_INTERVAL_MS = 5000  # 5 seconds
# 60 minutes * 60 seconds / 5 seconds = 720 samples
MAX_HISTORY = 720 

# --- Global State ---
# Storing history in memory (server-side). 
# In a multi-user production app, use dcc.Store or a database (Redis).
ratio_history = deque(maxlen=MAX_HISTORY)

app = Dash(__name__)

def fetch_order_book():
    try:
        response = requests.get(URL, params={'symbol': SYMBOL}, timeout=3)
        response.raise_for_status()
        data = response.json()
        
        if data.get('result') == 'success':
            return data.get('orderBook', {})
        return None
    except Exception as e:
        print(f"Fetch error: {e}")
        return None

def process_data(order_book):
    # Process Bids
    bids = pd.DataFrame(order_book.get('bids', []), columns=['price', 'size'])
    bids['price'] = bids['price'].astype(float)
    bids['size'] = bids['size'].astype(float)
    bids = bids.sort_values(by='price', ascending=False)
    bids['cumulative'] = bids['size'].cumsum()

    # Process Asks
    asks = pd.DataFrame(order_book.get('asks', []), columns=['price', 'size'])
    asks['price'] = asks['price'].astype(float)
    asks['size'] = asks['size'].astype(float)
    asks = asks.sort_values(by='price', ascending=True)
    asks['cumulative'] = asks['size'].cumsum()
    
    return bids, asks

app.layout = html.Div([
    html.H2(f"Kraken Futures: {SYMBOL} | Depth & Imbalance", style={'textAlign': 'center', 'fontFamily': 'sans-serif'}),
    
    # Metrics Container
    html.Div([
        html.Div([
            html.H4("Current Ratio (1 - Bid/Ask)", style={'margin': '5px'}),
            html.H3(id='current-ratio-display', style={'margin': '5px', 'color': '#0074D9'})
        ], style={'width': '45%', 'display': 'inline-block', 'textAlign': 'center', 'border': '1px solid #ddd', 'padding': '10px'}),
        
        html.Div([
            html.H4("60m Average Ratio", style={'margin': '5px'}),
            html.H3(id='avg-ratio-display', style={'margin': '5px', 'color': '#FF851B'})
        ], style={'width': '45%', 'display': 'inline-block', 'textAlign': 'center', 'border': '1px solid #ddd', 'padding': '10px', 'float': 'right'})
    ], style={'width': '80%', 'margin': '0 auto 20px auto', 'padding': '10px'}),

    dcc.Graph(id='depth-chart'),
    dcc.Interval(
        id='interval-component',
        interval=UPDATE_INTERVAL_MS,
        n_intervals=0
    )
])

@app.callback(
    [Output('depth-chart', 'figure'),
     Output('current-ratio-display', 'children'),
     Output('avg-ratio-display', 'children')],
    [Input('interval-component', 'n_intervals')]
)
def update_metrics(n):
    ob = fetch_order_book()
    
    # Default returns if fetch fails
    if not ob:
        return go.Figure(), "N/A", "N/A"

    bids, asks = process_data(ob)
    
    # 1. Calculate Metrics
    total_bid_vol = bids['size'].sum()
    total_ask_vol = asks['size'].sum()
    
    # Avoid division by zero
    if total_ask_vol == 0:
        ratio = 0 
    else:
        ratio = 1 - (total_bid_vol / total_ask_vol)

    # Update History
    ratio_history.append(ratio)
    
    # Calculate 60m Average
    if len(ratio_history) > 0:
        avg_60m = statistics.mean(ratio_history)
        avg_text = f"{avg_60m:.4f}"
    else:
        avg_text = "Wait for data..."

    ratio_text = f"{ratio:.4f}"

    # 2. Build Chart
    mid_price = (bids['price'].iloc[0] + asks['price'].iloc[0]) / 2
    lower_bound = mid_price * 0.98
    upper_bound = mid_price * 1.02
    
    bids_view = bids[bids['price'] >= lower_bound]
    asks_view = asks[asks['price'] <= upper_bound]

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=bids_view['price'], y=bids_view['cumulative'],
        fill='tozeroy', name='Bids', line=dict(color='green')
    ))

    fig.add_trace(go.Scatter(
        x=asks_view['price'], y=asks_view['cumulative'],
        fill='tozeroy', name='Asks', line=dict(color='red')
    ))

    fig.update_layout(
        title=f"Market Depth (Mid: {mid_price:.2f})",
        xaxis_title="Price (USD)",
        yaxis_title="Cumulative Volume",
        template="plotly_dark",
        hovermode="x unified",
        margin=dict(l=40, r=40, t=40, b=40)
    )
    
    return fig, ratio_text, avg_text

if __name__ == '__main__':
    app.run(debug=False, port=PORT, host='0.0.0.0')
