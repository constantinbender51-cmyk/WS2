import requests
import pandas as pd
import plotly.graph_objects as go
from dash import Dash, dcc, html
from dash.dependencies import Input, Output

# --- Configuration ---
SYMBOL = "PF_XBTUSD"
URL = "https://futures.kraken.com/derivatives/api/v3/orderbook"
PORT = 8080

app = Dash(__name__)

def fetch_order_book():
    try:
        response = requests.get(URL, params={'symbol': SYMBOL})
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
    bids = bids.sort_values(by='price', ascending=False) # Highest bid first
    bids['cumulative'] = bids['size'].cumsum()

    # Process Asks
    asks = pd.DataFrame(order_book.get('asks', []), columns=['price', 'size'])
    asks['price'] = asks['price'].astype(float)
    asks['size'] = asks['size'].astype(float)
    asks = asks.sort_values(by='price', ascending=True) # Lowest ask first
    asks['cumulative'] = asks['size'].cumsum()
    
    return bids, asks

app.layout = html.Div([
    html.H2(f"Kraken Futures Depth Chart: {SYMBOL}", style={'textAlign': 'center'}),
    dcc.Graph(id='depth-chart'),
    dcc.Interval(
        id='interval-component',
        interval=5*1000, # Update every 5 seconds
        n_intervals=0
    )
])

@app.callback(Output('depth-chart', 'figure'),
              [Input('interval-component', 'n_intervals')])
def update_metrics(n):
    ob = fetch_order_book()
    if not ob:
        return go.Figure()

    bids, asks = process_data(ob)
    
    # Filter for better visualization (e.g., +/- 2% from mid price)
    mid_price = (bids['price'].iloc[0] + asks['price'].iloc[0]) / 2
    lower_bound = mid_price * 0.98
    upper_bound = mid_price * 1.02
    
    bids_view = bids[bids['price'] >= lower_bound]
    asks_view = asks[asks['price'] <= upper_bound]

    fig = go.Figure()

    # Bids (Green) - Fill to zero
    fig.add_trace(go.Scatter(
        x=bids_view['price'], 
        y=bids_view['cumulative'],
        fill='tozeroy',
        name='Bids',
        line=dict(color='green')
    ))

    # Asks (Red) - Fill to zero
    fig.add_trace(go.Scatter(
        x=asks_view['price'], 
        y=asks_view['cumulative'],
        fill='tozeroy',
        name='Asks',
        line=dict(color='red')
    ))

    fig.update_layout(
        title=f"Market Depth (Last Updated: {pd.Timestamp.now()})",
        xaxis_title="Price (USD)",
        yaxis_title="Cumulative Volume",
        template="plotly_dark",
        hovermode="x unified"
    )
    
    return fig

if __name__ == '__main__':
    # host='0.0.0.0' allows external access if needed
    app.run_server(debug=False, port=PORT, host='0.0.0.0')
