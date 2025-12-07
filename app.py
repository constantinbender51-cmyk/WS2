import ccxt
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output, callback
import time

# --- Configuration ---
SYMBOL = 'BTC/USDT'
TIMEFRAME = '1d'
START_DATE = '2018-01-01 00:00:00'
MAX_SMA = 365
PORT = 8080

def fetch_data(symbol, timeframe, start_str):
    print(f"--- Fetching {symbol} data since {start_str} ---")
    exchange = ccxt.binance({'enableRateLimit': True})
    since = exchange.parse8601(start_str)
    all_ohlcv = []
    
    while True:
        try:
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=1000)
            if not ohlcv: break
            all_ohlcv.extend(ohlcv)
            since = ohlcv[-1][0] + 1
            if len(ohlcv) < 1000: break
        except Exception as e:
            print(f"Error fetching data: {e}")
            break

    df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    print(f"Data loaded: {len(df)} candles.")
    return df

def calculate_heatmap_matrix(df):
    """
    Computes Sharpe Ratio for ALL pairs of SMAs (SMA_i AND SMA_j).
    Returns a symmetric matrix (MAX_SMA, MAX_SMA).
    """
    print("Computing Heatmap Matrix (this may take 10-20 seconds)...")
    start_t = time.time()
    
    prices = df['close'].to_numpy()
    # Market returns (T,)
    market_returns = df['close'].pct_change().fillna(0).to_numpy()
    n_days = len(prices)
    
    # 1. Pre-calculate SMA Booleans: (T, MAX_SMA)
    # col 0 -> SMA 1, col 1 -> SMA 2...
    sma_long = np.zeros((n_days, MAX_SMA), dtype=bool)
    sma_short = np.zeros((n_days, MAX_SMA), dtype=bool)
    
    for i in range(MAX_SMA):
        sma_val = df['close'].rolling(window=i+1).mean().fillna(0).to_numpy()
        sma_long[:, i] = prices > sma_val
        sma_short[:, i] = prices < sma_val

    # 2. Compute Matrix
    # We want a matrix `sharpe_matrix` where [i, j] is Sharpe of SMA(i+1) & SMA(j+1)
    sharpe_matrix = np.zeros((MAX_SMA, MAX_SMA))
    
    # Optimization: Iterate outer loop i, vectorize inner loop j
    for i in range(MAX_SMA):
        # Base vector for SMA i
        base_l = sma_long[:, i][:, None] # (T, 1)
        base_s = sma_short[:, i][:, None] # (T, 1)
        
        # Combine with ALL columns j
        # (T, 1) & (T, MAX_SMA) -> (T, MAX_SMA)
        comb_l = base_l & sma_long
        comb_s = base_s & sma_short
        
        # Signal: 1 if Long, -1 if Short, 0 else
        sigs = comb_l.astype(np.int8) - comb_s.astype(np.int8)
        
        # Shift signals forward 1 day (trade tomorrow based on today)
        sigs = np.roll(sigs, 1, axis=0)
        sigs[0, :] = 0
        
        # Returns: (T, MAX_SMA)
        strat_rets = sigs * market_returns[:, None]
        
        # Vectorized Sharpe
        means = np.mean(strat_rets, axis=0)
        stds = np.std(strat_rets, axis=0)
        
        with np.errstate(divide='ignore', invalid='ignore'):
            row_sharpes = np.divide(means, stds) * np.sqrt(365)
        
        row_sharpes[np.isnan(row_sharpes)] = 0
        sharpe_matrix[i, :] = row_sharpes
        
    print(f"Heatmap computed in {time.time() - start_t:.2f}s")
    return sharpe_matrix

# --- GLOBAL DATA ---
# We load this once on startup
df_global = fetch_data(SYMBOL, TIMEFRAME, START_DATE)
sharpe_matrix_global = calculate_heatmap_matrix(df_global)

# Find Global Max for default view
max_idx = np.unravel_index(np.argmax(sharpe_matrix_global), sharpe_matrix_global.shape)
best_sma1, best_sma2 = max_idx[0] + 1, max_idx[1] + 1
best_sharpe = sharpe_matrix_global[max_idx]

# --- APP SETUP ---
app = Dash(__name__)

app.layout = html.Div([
    html.Div([
        html.H2("BTC/USDT Strategy Clusters", style={'color': 'white', 'margin': '0'}),
        html.P("Logic: Long if Price > SMA_A & SMA_B. Short if Price < SMA_A & SMA_B. (Double Confirmation)", style={'color': '#888'})
    ], style={'padding': '20px', 'backgroundColor': '#1e1e1e'}),
    
    html.Div([
        # --- LEFT: HEATMAP ---
        html.Div([
            dcc.Graph(id='heatmap-graph', style={'height': '80vh'})
        ], style={'width': '55%', 'display': 'inline-block', 'verticalAlign': 'top'}),
        
        # --- RIGHT: SELECTED STRATEGY ---
        html.Div([
            html.Div(id='stats-panel', style={'padding': '20px', 'color': 'white'}),
            dcc.Graph(id='equity-graph', style={'height': '60vh'})
        ], style={'width': '44%', 'display': 'inline-block', 'verticalAlign': 'top', 'paddingLeft': '1%'})
    ], style={'display': 'flex'})
    
], style={'backgroundColor': '#111', 'minHeight': '100vh', 'margin': '-8px'})

@callback(
    [Output('heatmap-graph', 'figure'),
     Output('equity-graph', 'figure'),
     Output('stats-panel', 'children')],
    [Input('heatmap-graph', 'clickData')]
)
def update_view(clickData):
    # 1. Determine Selected SMAs
    if clickData:
        # Plotly heatmap x/y are 0-indexed or 1-indexed depending on axis setup
        # We set axis explicitly below, so x=SMA1, y=SMA2
        p1 = clickData['points'][0]['x']
        p2 = clickData['points'][0]['y']
        sma1, sma2 = int(p1), int(p2)
    else:
        # Default to global max
        sma1, sma2 = best_sma1, best_sma2

    # --- 2. Generate Heatmap (Only done once technically, but Dash statelessness) ---
    # To optimize, we could store figure, but generating just the trace is fast enough.
    
    hm_fig = go.Figure(data=go.Heatmap(
        z=sharpe_matrix_global,
        x=list(range(1, MAX_SMA + 1)),
        y=list(range(1, MAX_SMA + 1)),
        colorscale='Viridis',
        colorbar=dict(title='Sharpe'),
        hovertemplate='SMA %{x} & SMA %{y}<br>Sharpe: %{z:.3f}<extra></extra>'
    ))
    
    # Highlight selection
    hm_fig.add_trace(go.Scatter(
        x=[sma1], y=[sma2],
        mode='markers',
        marker=dict(color='red', size=12, line=dict(color='white', width=2)),
        name='Selected'
    ))

    hm_fig.update_layout(
        title='Sharpe Ratio Heatmap (All Combinations)',
        template='plotly_dark',
        xaxis_title='SMA 1 Period',
        yaxis_title='SMA 2 Period',
        margin=dict(l=50, r=50, t=50, b=50)
    )

    # --- 3. Backtest Selected Strategy ---
    prices = df_global['close']
    returns = df_global['close'].pct_change().fillna(0)
    
    s_val1 = prices.rolling(sma1).mean()
    s_val2 = prices.rolling(sma2).mean()
    
    # Logic: Confirmation
    cond_long = (prices > s_val1) & (prices > s_val2)
    cond_short = (prices < s_val1) & (prices < s_val2)
    
    sig = cond_long.astype(int) - cond_short.astype(int)
    sig = sig.shift(1).fillna(0) # Trade tomorrow
    
    strat_rets = sig * returns
    cum_equity = (1 + strat_rets).cumprod()
    
    # Metrics
    ann_sharpe = (strat_rets.mean() / strat_rets.std()) * np.sqrt(365) if strat_rets.std() > 0 else 0
    total_ret = (cum_equity.iloc[-1] - 1) * 100
    trades = (sig.diff().abs() > 0).sum()
    
    # --- 4. Equity Curve Figure ---
    eq_fig = go.Figure()
    eq_fig.add_trace(go.Scatter(x=df_global.index, y=cum_equity, mode='lines', name='Strategy', line=dict(color='#00d4ff')))
    eq_fig.add_trace(go.Scatter(x=df_global.index, y=df_global['close']/df_global['close'].iloc[0], 
                                mode='lines', name='Buy & Hold (Norm)', line=dict(color='gray', dash='dot')))
    
    eq_fig.update_layout(
        title=f'Performance: SMA {sma1} & SMA {sma2}',
        template='plotly_dark',
        yaxis_title='Growth (1.0 = Initial)',
        yaxis_type='log',
        margin=dict(l=50, r=50, t=50, b=50)
    )
    
    # --- 5. Stats Panel ---
    stats_html = [
        html.H3(f"Selected: SMA {sma1} + SMA {sma2}"),
        html.Div([
            html.Span("Sharpe Ratio: ", style={'color': '#888'}),
            html.Span(f"{ann_sharpe:.3f}", style={'fontWeight': 'bold', 'color': '#00ff00' if ann_sharpe > 1 else 'white'}),
        ]),
        html.Div([
            html.Span("Total Return: ", style={'color': '#888'}),
            html.Span(f"{total_ret:,.0f}%", style={'fontWeight': 'bold'}),
        ]),
        html.Div([
            html.Span("Est. Trades: ", style={'color': '#888'}),
            html.Span(f"{trades}", style={'fontWeight': 'bold'}),
        ]),
        html.Hr(style={'borderColor': '#333'}),
        html.P("Click any point on the heatmap to analyze that pair.", style={'fontSize': '0.9em', 'color': '#666'})
    ]

    return hm_fig, eq_fig, stats_html

if __name__ == '__main__':
    print("Starting server on port 8080...")
    app.run(debug=False, port=8080, host='0.0.0.0')
