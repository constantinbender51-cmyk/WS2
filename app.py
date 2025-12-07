import ccxt
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from dash import Dash, dcc, html
from plotly.subplots import make_subplots
import time

# --- Configuration ---
SYMBOL = 'BTC/USDT'
TIMEFRAME = '1d'
START_DATE = '2018-01-01 00:00:00'
MAX_SMA = 365
TOP_N_SURVIVORS = 40 # Keep top 40 candidates per level to breed next level
RISK_FREE_RATE = 0.0

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

def get_sharpe_vectorized(returns_matrix):
    """Calculates Annualized Sharpe Ratio for a matrix of returns."""
    means = np.mean(returns_matrix, axis=0)
    stds = np.std(returns_matrix, axis=0)
    with np.errstate(divide='ignore', invalid='ignore'):
        sharpes = np.divide(means, stds) * np.sqrt(365)
    sharpes[np.isnan(sharpes)] = 0
    return sharpes

def run_greedy_optimization(df):
    """
    Performs greedy forward selection to find best SMA combinations.
    Logic: Long if Price > ALL SMAs, Short if Price < ALL SMAs.
    Returns: DataFrame with consensus score and metadata.
    """
    print("--- Starting Greedy Optimization ---")
    start_time = time.time()
    
    prices = df['close'].to_numpy()
    market_returns = df['close'].pct_change().fillna(0).to_numpy()
    n_days = len(prices)
    
    # 1. Pre-calculate all Boolean conditions (Price > SMA)
    # Shape: (T, MAX_SMA)
    print("Pre-calculating SMA matrix...")
    sma_conds_long = np.zeros((n_days, MAX_SMA), dtype=bool)
    sma_conds_short = np.zeros((n_days, MAX_SMA), dtype=bool)
    
    for i in range(MAX_SMA):
        sma = df['close'].rolling(window=i+1).mean().fillna(0).to_numpy()
        sma_conds_long[:, i] = prices > sma
        sma_conds_short[:, i] = prices < sma

    # Master list to store all valid strategies found across all levels
    # List of dicts: {'indices': [1, 50], 'sharpe': 1.2, 'level': 2}
    master_strategy_list = []

    # --- LEVEL 1: Single SMAs ---
    print("Scanning Level 1 (Single SMAs)...")
    # 1 if Long, -1 if Short, 0 else
    l1_sigs = sma_conds_long.astype(np.int8) - sma_conds_short.astype(np.int8)
    l1_sigs = np.roll(l1_sigs, 1, axis=0) # Shift 1 day
    l1_sigs[0, :] = 0
    
    l1_rets = l1_sigs * market_returns[:, None]
    l1_sharpes = get_sharpe_vectorized(l1_rets)
    
    level_results = []
    for i in range(MAX_SMA):
        if l1_sharpes[i] > 0:
            level_results.append({'indices': [i], 'sharpe': l1_sharpes[i], 'level': 1})
            
    level_results.sort(key=lambda x: x['sharpe'], reverse=True)
    master_strategy_list.extend(level_results[:50]) # Add best singles to master list
    
    # Survivors continue to breed in Level 2
    current_survivors = level_results[:TOP_N_SURVIVORS]
    
    # --- LEVEL 2 to 5: Greedy Additions ---
    for level in range(2, 6):
        print(f"Scanning Level {level} (adding confirmation)...")
        next_gen = []
        
        for survivor in current_survivors:
            base_indices = survivor['indices']
            
            # Construct Base Condition (AND logic)
            # Start true, AND with every SMA in the set
            base_L = np.ones(n_days, dtype=bool)
            base_S = np.ones(n_days, dtype=bool)
            for idx in base_indices:
                base_L &= sma_conds_long[:, idx]
                base_S &= sma_conds_short[:, idx]
            
            # Vectorized test adding one more SMA
            # Combined = Base AND New_Column
            combined_L = base_L[:, None] & sma_conds_long
            combined_S = base_S[:, None] & sma_conds_short
            
            sigs = combined_L.astype(np.int8) - combined_S.astype(np.int8)
            sigs = np.roll(sigs, 1, axis=0)
            sigs[0, :] = 0
            
            rets = sigs * market_returns[:, None]
            sharpes = get_sharpe_vectorized(rets)
            
            # Collect results
            # We only look at indices > max(base_indices) to avoid permutations/duplicates 
            # (e.g., [10, 20] is same as [20, 10])
            start_search = max(base_indices) + 1
            if start_search < MAX_SMA:
                for k in range(start_search, MAX_SMA):
                    # Only add if Sharpe improves or is high enough
                    if sharpes[k] > 0.5: 
                        new_idx = base_indices + [k]
                        next_gen.append({'indices': new_idx, 'sharpe': sharpes[k], 'level': level})
        
        # Sort and Keep Best
        next_gen.sort(key=lambda x: x['sharpe'], reverse=True)
        
        # Deduplicate (just in case logic above missed something)
        unique_next = []
        seen = set()
        for item in next_gen:
            t = tuple(sorted(item['indices']))
            if t not in seen:
                seen.add(t)
                unique_next.append(item)
        
        # Add to master list and set survivors
        master_strategy_list.extend(unique_next[:50])
        current_survivors = unique_next[:TOP_N_SURVIVORS]

    # --- SELECT TOP 25 OVERALL ---
    print("Selecting Top 25 Strategies from all levels...")
    # Sort master list by Sharpe
    master_strategy_list.sort(key=lambda x: x['sharpe'], reverse=True)
    
    # Ensure uniqueness (strategies might appear in multiple levels if we aren't careful, 
    # though strict > logic usually prevents it. Just being safe.)
    final_top_25 = []
    seen_sets = set()
    for strat in master_strategy_list:
        k = tuple(sorted(strat['indices']))
        if k not in seen_sets:
            seen_sets.add(k)
            final_top_25.append(strat)
        if len(final_top_25) >= 25:
            break
            
    # Print Top 5 for debug
    print("Top 5 Strategies:")
    for i, s in enumerate(final_top_25[:5]):
        smas = [str(x+1) for x in s['indices']]
        print(f"{i+1}. Sharpe {s['sharpe']:.2f} | SMAs: {smas}")

    # --- CALCULATE CONSENSUS SCORE ---
    print("Calculating Consensus Score...")
    consensus_sum = np.zeros(n_days)
    
    for strat in final_top_25:
        indices = strat['indices']
        
        # Rebuild signal
        is_long = np.ones(n_days, dtype=bool)
        is_short = np.ones(n_days, dtype=bool)
        
        for idx in indices:
            is_long &= sma_conds_long[:, idx]
            is_short &= sma_conds_short[:, idx]
            
        sig = is_long.astype(np.int8) - is_short.astype(np.int8)
        # Shift (Trade tomorrow)
        sig = np.roll(sig, 1)
        sig[0] = 0
        
        consensus_sum += sig
        
    # Scale: Divide by 5 to fit 0-5 range (technically -5 to +5)
    df['consensus_raw'] = consensus_sum
    df['consensus_score'] = consensus_sum / 5.0
    
    print(f"Optimization finished in {time.time() - start_time:.2f}s")
    return df, final_top_25

def create_figure(df):
    # Create subplots: Price on top, Consensus on bottom
    fig = make_subplots(
        rows=2, cols=1, 
        shared_xaxes=True, 
        vertical_spacing=0.03,
        row_heights=[0.7, 0.3],
        specs=[[{"secondary_y": True}], [{"secondary_y": False}]]
    )

    # --- Top Panel: Price ---
    # Candlestick (simplified to line for speed if dense, but user likes details)
    fig.add_trace(go.Scatter(
        x=df.index, y=df['close'],
        name='BTC Price',
        line=dict(color='white', width=1)
    ), row=1, col=1, secondary_y=False)

    # --- Bottom Panel: Consensus Score ---
    # We color the area: Green if > 0, Red if < 0
    
    # Split into positive and negative for coloring
    pos_score = df['consensus_score'].clip(lower=0)
    neg_score = df['consensus_score'].clip(upper=0)

    fig.add_trace(go.Scatter(
        x=df.index, y=pos_score,
        name='Consensus (Long)',
        fill='tozeroy',
        mode='none',
        fillcolor='rgba(0, 255, 100, 0.5)'
    ), row=2, col=1)

    fig.add_trace(go.Scatter(
        x=df.index, y=neg_score,
        name='Consensus (Short)',
        fill='tozeroy',
        mode='none',
        fillcolor='rgba(255, 50, 50, 0.5)'
    ), row=2, col=1)

    # Add a horizontal line at 0
    fig.add_hline(y=0, line_dash="dot", line_color="gray", row=2, col=1)

    # Layout
    fig.update_layout(
        title='Bitcoin Price vs. SMA Consensus Score (Top 25 Strategies)',
        template='plotly_dark',
        height=800,
        hovermode='x unified',
        showlegend=False
    )
    
    fig.update_yaxes(title_text="Price (USDT)", row=1, col=1, secondary_y=False)
    fig.update_yaxes(title_text="Consensus (-5 to +5)", row=2, col=1, range=[-5.5, 5.5])

    return fig

# --- Main ---
if __name__ == '__main__':
    # 1. Fetch
    df = fetch_data(SYMBOL, TIMEFRAME, START_DATE)
    
    # 2. Optimize & Calc
    df, top_strats = run_greedy_optimization(df)
    
    # 3. Server
    app = Dash(__name__)
    
    app.layout = html.Div([
        dcc.Graph(figure=create_figure(df), style={'height': '95vh'}),
        
        # Display Top 5 Strats text
        html.Div([
            html.H4("Top 5 Component Strategies:", style={'color': '#888'}),
            html.Ul([
                html.Li(f"Level {s['level']} | SMAs: {[x+1 for x in s['indices']]} | Sharpe: {s['sharpe']:.3f}") 
                for s in top_strats[:5]
            ], style={'color': '#aaa'})
        ], style={'padding': '20px'})
    ], style={'backgroundColor': '#111', 'margin': '-8px'})
    
    print("Starting server on port 8080...")
    app.run(debug=False, port=8080, host='0.0.0.0')
