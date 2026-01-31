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

def build_depth_figure(bids, asks, title, focus=False):
    """
    Helper to generate Plotly figures for depth.
    If focus=True, filters data to +/- 2% of mid price.
    """
    if bids.empty or asks.empty:
        return go.Figure()

    # Calculate Mid Price for filtering
    mid_price = (bids['price'].iloc[0] + asks['price'].iloc[0]) / 2

    if focus:
        lower_bound = mid_price * 0.98
        upper_bound = mid_price * 1.02
        bids_plot = bids[bids['price'] >= lower_bound]
        asks_plot = asks[asks['price'] <= upper_bound]
    else:
        bids_plot = bids
        asks_plot = asks

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=bids_plot['price'], y=bids_plot['cumulative'],
        fill='tozeroy', name='Bids', line=dict(color='green')
    ))

    fig.add_trace(go.Scatter(
        x=asks_plot['price'], y=asks_plot['cumulative'],
        fill='tozeroy', name='Asks', line=dict(color='red')
    ))

    fig.update_layout(
        title=title,
        xaxis_title="Price (USD)",
        yaxis_title="Cumulative Volume",
        template="plotly_dark",
        hovermode="x unified",
        margin=dict(l=40, r=40, t=40, b=40),
        height=400 # Fixed height for better stacking
    )
    return fig

app.layout = html.Div([
    html.H2(f"Kraken Futures: {SYMBOL} Analysis", style={'textAlign': 'center', 'fontFamily': 'sans-serif'}),
    
    # --- Metrics Section ---
    html.Div([
        html.Div([
            html.H4("Ratio (1 - Bid/Ask)", style={'margin': '5px'}),
            html.H3(id='current-ratio-display', style={'margin': '5px', 'color': '#0074D9'})
        ], style={'width': '45%', 'display': 'inline-block', 'textAlign': 'center', 'border': '1px solid #555', 'padding': '10px', 'borderRadius': '5px'}),
        
        html.Div([
            html.H4("60m Avg Ratio", style={'margin': '5px'}),
            html.H3(id='avg-ratio-display', style={'margin': '5px', 'color': '#FF851B'})
        ], style={'width': '45%', 'display': 'inline-block', 'textAlign': 'center', 'border': '1px solid #555', 'padding': '10px', 'borderRadius': '5px', 'float': 'right'})
    ], style={'width': '90%', 'margin': '0 auto 20px auto', 'color': 'white'}),

    # --- Charts Section ---
    html.Div([
        # Graph 1: Zoomed View
        dcc.Graph(id='focused-depth-chart'),
        html.Hr(style={'borderColor': '#333'}),
        # Graph 2: Full View
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
        empty_fig = go.Figure()
        return empty_fig, empty_fig, "N/A", "N/A"

    bids, asks = process_data(ob)
    
    # 1. Calculate Metrics
    total_bid_vol = bids['size'].sum()
    total_ask_vol = asks['size'].sum()
    
    if total_ask_vol == 0:
        ratio = 0 
    else:
        ratio = 1 - (total_bid_vol / total_ask_vol)

    ratio_history.append(ratio)
    
    if len(ratio_history) > 0:
        avg_60m = statistics.mean(ratio_history)
        avg_text = f"{avg_60m:.4f}"
    else:
        avg_text = "Wait..."

    ratio_text = f"{ratio:.4f}"

    # 2. Build Figures
    # Zoomed Chart (+/- 2%)
    mid_price = (bids['price'].iloc[0] + asks['price'].iloc[0]) / 2
    fig_focused = build_depth_figure(bids, asks, f"Focused Depth (Mid: {mid_price:.2f})", focus=True)
    
    # Full Chart (Entire Book)
    fig_full = build_depth_figure(bids, asks, "Full Order Book Depth", focus=False)
    
    return fig_focused, fig_full, ratio_text, avg_text

if __name__ == '__main__':
    app.run(debug=False, port=PORT, host='0.0.0.0')
