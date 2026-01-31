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
MAX_HISTORY = 720 

# --- Global State ---
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

def get_subset(df, mid_price, percent=0.02, is_bid=True):
    """Filters data to within specific percentage of mid price."""
    if is_bid:
        limit = mid_price * (1 - percent)
        return df[df['price'] >= limit]
    else:
        limit = mid_price * (1 + percent)
        return df[df['price'] <= limit]

def build_figure(bids, asks, title, log_scale=False):
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=bids['price'], y=bids['cumulative'],
        fill='tozeroy', name='Bids', line=dict(color='green')
    ))

    fig.add_trace(go.Scatter(
        x=asks['price'], y=asks['cumulative'],
        fill='tozeroy', name='Asks', line=dict(color='red')
    ))

    layout_args = dict(
        title=title,
        xaxis_title="Price (USD)",
        yaxis_title="Cumulative Volume",
        template="plotly_dark",
        hovermode="x unified",
        margin=dict(l=40, r=40, t=40, b=40),
        height=400
    )
    
    if log_scale:
        layout_args['yaxis_type'] = "log"
        
    fig.update_layout(**layout_args)
    return fig

app.layout = html.Div([
    html.H2(f"Kraken Futures: {SYMBOL} Analysis", style={'textAlign': 'center', 'fontFamily': 'sans-serif', 'color': '#eee'}),
    
    # --- Metrics Section ---
    html.Div([
        html.Div([
            html.H4("Ratio (±2% Depth)", style={'margin': '5px', 'color': '#aaa'}),
            html.H3(id='current-ratio-display', style={'margin': '5px', 'color': '#0074D9'})
        ], style={'width': '45%', 'display': 'inline-block', 'textAlign': 'center', 'border': '1px solid #444', 'padding': '10px', 'borderRadius': '5px'}),
        
        html.Div([
            html.H4("60m Avg Ratio", style={'margin': '5px', 'color': '#aaa'}),
            html.H3(id='avg-ratio-display', style={'margin': '5px', 'color': '#FF851B'})
        ], style={'width': '45%', 'display': 'inline-block', 'textAlign': 'center', 'border': '1px solid #444', 'padding': '10px', 'borderRadius': '5px', 'float': 'right'})
    ], style={'width': '90%', 'margin': '0 auto 20px auto'}),

    # --- Charts Section ---
    html.Div([
        # Graph 1: Zoomed View (Linear)
        dcc.Graph(id='focused-depth-chart'),
        html.Hr(style={'borderColor': '#333', 'margin': '20px 0'}),
        # Graph 2: Full View (Log Scale)
        dcc.Graph(id='full-depth-chart')
    ], style={'width': '95%', 'margin': '0 auto'}),

    dcc.Interval(
        id='interval-component',
        interval=UPDATE_INTERVAL_MS,
        n_intervals=0
    )
], style={'backgroundColor': '#111', 'padding': '20px', 'minHeight': '100vh'})

@app.callback(
    [Output('focused-depth-chart', 'figure'),
     Output('full-depth-chart', 'figure'),
     Output('current-ratio-display', 'children'),
     Output('avg-ratio-display', 'children')],
    [Input('interval-component', 'n_intervals')]
)
def update_dashboard(n):
    ob = fetch_order_book()
    
    if not ob:
        empty = go.Figure()
        return empty, empty, "N/A", "N/A"

    bids, asks = process_data(ob)
    
    # 1. Determine Mid Price
    mid_price = (bids['price'].iloc[0] + asks['price'].iloc[0]) / 2

    # 2. Filter for +/- 2% (Metrics & Focused Plot)
    bids_2pct = get_subset(bids, mid_price, 0.02, is_bid=True)
    asks_2pct = get_subset(asks, mid_price, 0.02, is_bid=False)

    # 3. Calculate Ratio on 2% Subset
    vol_bid_2pct = bids_2pct['size'].sum()
    vol_ask_2pct = asks_2pct['size'].sum()
    
    if vol_ask_2pct == 0:
        ratio = 0
    else:
        ratio = 1 - (vol_bid_2pct / vol_ask_2pct)

    # Update History
    ratio_history.append(ratio)
    if len(ratio_history) > 0:
        avg_60m = statistics.mean(ratio_history)
        avg_text = f"{avg_60m:.4f}"
    else:
        avg_text = "Wait..."

    # 4. Build Plots
    # Focused: Linear Scale, limited to 2% data
    fig_focused = build_figure(bids_2pct, asks_2pct, f"Depth ±2% (Mid: {mid_price:.2f})", log_scale=False)
    
    # Full: Log Scale, all data
    fig_full = build_figure(bids, asks, "Full Order Book (Log Scale)", log_scale=True)
    
    return fig_focused, fig_full, f"{ratio:.4f}", avg_text

if __name__ == '__main__':
    app.run(debug=False, port=PORT, host='0.0.0.0')
