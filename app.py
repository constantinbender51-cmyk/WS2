import requests
import pandas as pd
import plotly.graph_objects as go
from dash import Dash, dcc, html
from dash.dependencies import Input, Output
from collections import deque
import statistics
import numpy as np
import datetime

# --- Configuration ---
SYMBOL = "PF_XBTUSD"
URL = "https://futures.kraken.com/derivatives/api/v3/orderbook"
PORT = 8080
UPDATE_INTERVAL_MS = 5000  # 5 seconds
MAX_HISTORY = 720  # 60 minutes
ORDER_USD_VALUE = 1000 # $1000 per grid level (adjust as needed)

# --- Paper Trading Engine ---
class PaperTrader:
    def __init__(self, initial_balance=10000):
        self.balance = initial_balance
        self.position = 0.0
        self.avg_entry_price = 0.0
        self.realized_pnl = 0.0
        self.fees_paid = 0.0
        self.active_orders = [] 
        self.trade_log = deque(maxlen=50)
        self.is_strategy_active = False
        self.reference_price = 0.0

        # Cost Model
        # User specified: 0.2% Fee + 0.1% Slippage = 0.3% per side (0.6% Round Trip)
        self.FEE_RATE = 0.002      
        self.SLIPPAGE_RATE = 0.001 
        self.TOTAL_COST_RATE = self.FEE_RATE + self.SLIPPAGE_RATE # 0.003

    def reset_strategy(self):
        """Cancels orders and closes positions (Market Close)."""
        self.active_orders = []
        if self.position != 0:
            self._execute_trade(-self.position, self.reference_price, "Strategy Deactivated")
        self.is_strategy_active = False

    def activate_strategy(self, current_price):
        if self.is_strategy_active:
            return 
            
        self.is_strategy_active = True
        self.reference_price = current_price
        mid = current_price
        
        # FIX: Calculate Quantity in BTC based on USD Target
        # e.g. $1000 / $80000 = 0.0125 BTC
        base_size = ORDER_USD_VALUE / mid 

        # --- Generate Grids with Mapped Stops ---
        
        # 1. Price Arrays
        sells = np.linspace(mid * 1.006, mid * 1.02, 10)
        buys = np.linspace(mid * 0.994, mid * 0.98, 10) # 0.994 is closest, 0.98 furthest

        # 2. Stop Variance Array (0% to 3.88%)
        # Mapped 1:1. First order (closest) has 0% variance. Last (furthest) has 3.88%
        stop_variances = np.linspace(0.0, 0.0388, 10)

        # 3. Create Sell Orders + Companion Buy Stops
        for i, price in enumerate(sells):
            # Recalculate size for this specific price to maintain USD value? 
            # Or use fixed base_size? Using fixed base_size is safer for consistency.
            self.active_orders.append({'side': 'sell', 'type': 'limit', 'price': price, 'size': base_size})
            
            # Stop (Short Protection): Above Entry
            variance = stop_variances[i]
            stop_price = price * (1 + variance)
            self.active_orders.append({'side': 'buy', 'type': 'stop', 'price': stop_price, 'size': base_size})

        # 4. Create Buy Orders + Companion Sell Stops
        for i, price in enumerate(buys):
            self.active_orders.append({'side': 'buy', 'type': 'limit', 'price': price, 'size': base_size})
            
            # Stop (Long Protection): Below Entry
            variance = stop_variances[i]
            stop_price = price * (1 - variance)
            self.active_orders.append({'side': 'sell', 'type': 'stop', 'price': stop_price, 'size': base_size})

    def process_tick(self, bid, ask):
        if not self.is_strategy_active:
            return

        current_mid = (bid + ask) / 2
        self.reference_price = current_mid
        
        filled_indices = []

        for i, order in enumerate(self.active_orders):
            executed = False
            
            # Limit Buy
            if order['type'] == 'limit' and order['side'] == 'buy':
                if ask <= order['price']:
                    self._execute_trade(order['size'], order['price'], "Limit Buy")
                    executed = True
            
            # Limit Sell
            elif order['type'] == 'limit' and order['side'] == 'sell':
                if bid >= order['price']:
                    self._execute_trade(-order['size'], order['price'], "Limit Sell")
                    executed = True
            
            # Stop Buy (Short Protection)
            elif order['type'] == 'stop' and order['side'] == 'buy':
                if ask >= order['price'] and self.position < -1e-9: # Check for non-zero short
                    qty = min(order['size'], abs(self.position))
                    if qty > 1e-9:
                        self._execute_trade(qty, order['price'], "Stop Loss Buy")
                        executed = True 
            
            # Stop Sell (Long Protection)
            elif order['type'] == 'stop' and order['side'] == 'sell':
                if bid <= order['price'] and self.position > 1e-9: # Check for non-zero long
                    qty = min(order['size'], self.position)
                    if qty > 1e-9:
                        self._execute_trade(-qty, order['price'], "Stop Loss Sell")
                        executed = True

            if executed:
                filled_indices.append(i)
        
        for i in sorted(filled_indices, reverse=True):
            del self.active_orders[i]

    def _execute_trade(self, size, price, reason):
        # 1. Calculate Transaction Cost (Fee + Slippage)
        # Notional Value = Quantity * Price
        trade_value = abs(size * price) 
        cost = trade_value * self.TOTAL_COST_RATE # 0.3%
        self.fees_paid += cost
        self.balance -= cost 

        # 2. PnL Calculation
        if (self.position > 1e-9 and size < 0) or (self.position < -1e-9 and size > 0):
            closing_qty = min(abs(size), abs(self.position))
            pnl = (price - self.avg_entry_price) * closing_qty
            if self.position < 0: pnl = -pnl
            
            self.realized_pnl += pnl
            self.balance += pnl
            
        # 3. Update Position
        new_pos = self.position + size
        if abs(new_pos) < 1e-9: # Close enough to zero
            self.position = 0.0
            self.avg_entry_price = 0.0
        elif (self.position >= 0 and size > 0) or (self.position <= 0 and size < 0):
            total_cost = (self.position * self.avg_entry_price) + (size * price)
            self.avg_entry_price = total_cost / new_pos
            self.position = new_pos
        else:
            # Reducing position but not flipping, avg entry doesn't change
            self.position = new_pos

        self.trade_log.append(f"{reason} | {size:+.4f} @ {price:.2f} | Fee: ${cost:.2f}")

    def get_stats(self):
        unrealized = 0.0
        if self.position != 0:
            unrealized = (self.reference_price - self.avg_entry_price) * self.position
        
        return {
            'balance': self.balance,
            'position': self.position,
            'realized': self.realized_pnl,
            'unrealized': unrealized,
            'fees': self.fees_paid,
            'total_equity': self.balance + unrealized
        }

# --- Global State ---
ratio_history = deque(maxlen=MAX_HISTORY)
trader = PaperTrader()

app = Dash(__name__)

def fetch_order_book():
    try:
        response = requests.get(URL, params={'symbol': SYMBOL}, timeout=3)
        response.raise_for_status()
        data = response.json()
        if data.get('result') == 'success':
            return data.get('orderBook', {})
        return None
    except:
        return None

def process_data(order_book):
    bids = pd.DataFrame(order_book.get('bids', []), columns=['price', 'size']).astype(float)
    bids = bids.sort_values(by='price', ascending=False)
    bids['cumulative'] = bids['size'].cumsum()

    asks = pd.DataFrame(order_book.get('asks', []), columns=['price', 'size']).astype(float)
    asks = asks.sort_values(by='price', ascending=True)
    asks['cumulative'] = asks['size'].cumsum()
    return bids, asks

def build_figure(bids, asks, title, log_scale=False, active_orders=None):
    fig = go.Figure()

    fig.add_trace(go.Scatter(x=bids['price'], y=bids['cumulative'], fill='tozeroy', name='Bids', line=dict(color='green')))
    fig.add_trace(go.Scatter(x=asks['price'], y=asks['cumulative'], fill='tozeroy', name='Asks', line=dict(color='red')))

    if active_orders:
        y_max = max(bids['cumulative'].max(), asks['cumulative'].max())
        y_level = y_max * 0.1 if log_scale else y_max * 0.5
        
        buy_prices = [o['price'] for o in active_orders if o['side'] == 'buy' and o['type'] == 'limit']
        sell_prices = [o['price'] for o in active_orders if o['side'] == 'sell' and o['type'] == 'limit']
        stop_prices = [o['price'] for o in active_orders if o['type'] == 'stop']

        if buy_prices:
            fig.add_trace(go.Scatter(x=buy_prices, y=[y_level]*len(buy_prices), mode='markers', marker=dict(symbol='triangle-up', size=10, color='cyan'), name='Limit Buy'))
        if sell_prices:
            fig.add_trace(go.Scatter(x=sell_prices, y=[y_level]*len(sell_prices), mode='markers', marker=dict(symbol='triangle-down', size=10, color='orange'), name='Limit Sell'))
        if stop_prices:
             fig.add_trace(go.Scatter(x=stop_prices, y=[y_level]*len(stop_prices), mode='markers', marker=dict(symbol='x', size=8, color='magenta'), name='Stop Order'))

    layout_args = dict(title=title, xaxis_title="Price", yaxis_title="Vol", template="plotly_dark", height=400, margin=dict(l=40, r=40, t=40, b=40))
    if log_scale: layout_args['yaxis_type'] = "log"
    fig.update_layout(**layout_args)
    return fig

app.layout = html.Div([
    html.H2(f"Kraken: {SYMBOL} + Paper Trader", style={'textAlign': 'center', 'color': '#eee', 'fontFamily': 'sans-serif'}),
    
    # --- Paper Trading Account Panel ---
    html.Div([
        html.Div([
            html.H4("Equity", style={'margin': '0', 'color': '#aaa'}),
            html.H2(id='equity-display', style={'margin': '5px', 'color': '#fff'})
        ], style={'flex': 1, 'textAlign': 'center', 'borderRight': '1px solid #444'}),
        html.Div([
            html.H4("Unrealized PnL", style={'margin': '0', 'color': '#aaa'}),
            html.H2(id='pnl-display', style={'margin': '5px'})
        ], style={'flex': 1, 'textAlign': 'center', 'borderRight': '1px solid #444'}),
        html.Div([
            html.H4("Fees Paid", style={'margin': '0', 'color': '#aaa'}),
            html.H2(id='fees-display', style={'margin': '5px', 'color': '#FF851B'})
        ], style={'flex': 1, 'textAlign': 'center', 'borderRight': '1px solid #444'}),
        html.Div([
            html.H4("Position", style={'margin': '0', 'color': '#aaa'}),
            html.H2(id='pos-display', style={'margin': '5px'})
        ], style={'flex': 1, 'textAlign': 'center'}),
        html.Div([
            html.H4("Status", style={'margin': '0', 'color': '#aaa'}),
            html.H2(id='status-display', style={'margin': '5px'})
        ], style={'flex': 1, 'textAlign': 'center', 'borderLeft': '1px solid #444'}),
    ], style={'display': 'flex', 'backgroundColor': '#222', 'marginBottom': '20px', 'padding': '10px', 'borderRadius': '8px'}),

    # --- Metrics ---
    html.Div([
        html.Div([html.H5("Current Ratio"), html.H3(id='ratio-val')], style={'display': 'inline-block', 'width': '45%', 'textAlign': 'center', 'color': '#ccc'}),
        html.Div([html.H5("60m Avg Ratio"), html.H3(id='avg-val')], style={'display': 'inline-block', 'width': '45%', 'textAlign': 'center', 'color': '#ccc'})
    ], style={'textAlign': 'center'}),

    dcc.Graph(id='focused-chart'),
    dcc.Graph(id='full-chart'),
    dcc.Interval(id='timer', interval=UPDATE_INTERVAL_MS, n_intervals=0)
], style={'backgroundColor': '#111', 'padding': '20px', 'minHeight': '100vh', 'fontFamily': 'sans-serif'})

@app.callback(
    [Output('focused-chart', 'figure'), Output('full-chart', 'figure'),
     Output('ratio-val', 'children'), Output('avg-val', 'children'),
     Output('equity-display', 'children'), Output('pnl-display', 'children'),
     Output('pnl-display', 'style'), Output('fees-display', 'children'),
     Output('pos-display', 'children'), Output('pos-display', 'style'),
     Output('status-display', 'children'), Output('status-display', 'style')],
    [Input('timer', 'n_intervals')]
)
def update(n):
    ob = fetch_order_book()
    if not ob: return go.Figure(), go.Figure(), "-", "-", "-", "-", {}, "-", "-", {}, "OFFLINE", {'color': 'red'}

    bids, asks = process_data(ob)
    best_bid = bids['price'].iloc[0]
    best_ask = asks['price'].iloc[0]
    mid = (best_bid + best_ask) / 2

    # 1. Calc Ratio
    b_sub = bids[bids['price'] >= mid * 0.98]
    a_sub = asks[asks['price'] <= mid * 1.02]
    vb, va = b_sub['size'].sum(), a_sub['size'].sum()
    ratio = 0 if va == 0 else 1 - (vb / va)
    
    ratio_history.append(ratio)
    avg_60m = statistics.mean(ratio_history) if ratio_history else 0

    # 2. Strategy Logic
    active_zone = -0.1 <= avg_60m <= 0.1
    
    if active_zone:
        trader.activate_strategy(mid)
        status_txt = "ACTIVE"
        status_col = {'color': '#2ECC40'}
    else:
        trader.reset_strategy()
        status_txt = "CLOSED"
        status_col = {'color': '#FF4136'}

    trader.process_tick(best_bid, best_ask)
    stats = trader.get_stats()

    pnl_col = {'color': '#2ECC40'} if stats['unrealized'] >= 0 else {'color': '#FF4136'}
    pos_col = {'color': '#2ECC40'} if stats['position'] > 0 else ({'color': '#FF4136'} if stats['position'] < 0 else {'color': '#ccc'})
    
    fig1 = build_figure(b_sub, a_sub, f"Active Depth ±2% ({mid:.1f})", False, trader.active_orders)
    fig2 = build_figure(bids, asks, "Full Book", True, trader.active_orders)

    return (fig1, fig2, f"{ratio:.4f}", f"{avg_60m:.4f}", 
            f"${stats['total_equity']:,.2f}", f"${stats['unrealized']:,.2f}", pnl_col,
            f"${stats['fees']:,.2f}", f"{stats['position']:.4f}", pos_col, status_txt, status_col)

if __name__ == '__main__':
    app.run(debug=False, port=PORT, host='0.0.0.0')
