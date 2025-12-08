import ccxt
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import http.server
import socketserver
import webbrowser
import os
import threading
import time
from datetime import datetime
import itertools

# --- Configuration ---
SYMBOL = 'BTC/USDT'
TIMEFRAME = '1d'
START_DATE = '2018-01-01 00:00:00'
PORT = 8080
SEQ_LENGTH = 30  # 30 lag features

# Profitable Periods (Start, End) - Inclusive
PROFITABLE_PERIODS = [
    ('2020-09-06', '2021-02-15'),
    ('2021-07-12', '2021-10-11'),
    ('2022-03-28', '2023-01-02'), # Note: This period includes a significant downtrend in history
    ('2023-09-04', '2024-03-04'),
    ('2024-09-02', '2025-02-03'),
    ('2025-03-31', '2025-07-07')
]

def fetch_data():
    print(f"Fetching {SYMBOL} data from Binance starting {START_DATE}...")
    exchange = ccxt.binance()
    since = exchange.parse8601(START_DATE)
    all_ohlcv = []
    
    while True:
        try:
            ohlcv = exchange.fetch_ohlcv(SYMBOL, TIMEFRAME, since=since)
            if not ohlcv:
                break
            all_ohlcv.extend(ohlcv)
            since = ohlcv[-1][0] + 1 
            if len(ohlcv) < 500: 
                break
            time.sleep(0.1) 
        except Exception as e:
            print(f"Error fetching data: {e}")
            break
            
    df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    df = df[~df.index.duplicated(keep='first')].sort_index()
    return df

def add_features(df):
    print("Engineering features...")
    # Calculate SMAs
    df['sma365'] = df['close'].rolling(window=365).mean()
    df['sma120'] = df['close'].rolling(window=120).mean()
    df['sma40'] = df['close'].rolling(window=40).mean()
    
    # Calculate Distances (Percent)
    df['dist_365'] = (df['close'] - df['sma365']) / df['sma365']
    df['dist_120'] = (df['close'] - df['sma120']) / df['sma120']
    df['dist_40'] = (df['close'] - df['sma40']) / df['sma40']
    
    # NEW: Log Return in Percent
    # We use log returns because they are time-additive
    df['log_ret'] = np.log(df['close'] / df['close'].shift(1)) * 100
    
    # Drop NaNs (created by largest SMA and the shift)
    df.dropna(inplace=True)
    return df

def prepare_data(df):
    print("Preparing LSTM sequences...")
    
    # Target Labeling: Periods
    df['target'] = 0
    
    for start_date, end_date in PROFITABLE_PERIODS:
        # Create a mask for the date range
        mask = (df.index >= pd.to_datetime(start_date)) & (df.index <= pd.to_datetime(end_date))
        df.loc[mask, 'target'] = 1

    # Added 'log_ret' to features
    feature_cols = ['dist_365', 'dist_120', 'dist_40', 'log_ret']
    
    data = df[feature_cols].values
    targets = df['target'].values
    dates = df.index
    
    # Scaling
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)
    
    X, y, prediction_dates = [], [], []
    
    # Create sequences
    for i in range(SEQ_LENGTH, len(data)):
        x_seq = data_scaled[i-SEQ_LENGTH:i]
        X.append(x_seq)
        y.append(targets[i])
        prediction_dates.append(dates[i])
        
    return np.array(X), np.array(y), prediction_dates, df

def build_model(input_shape, params):
    units_1 = params.get('units_1', 64)
    units_2 = params.get('units_2', 32)
    dropout = params.get('dropout', 0.2)
    dense_units = params.get('dense_units', 16)
    lr = params.get('learning_rate', 0.001)
    
    model = Sequential([
        Input(shape=input_shape),
        LSTM(units_1, return_sequences=True),
        Dropout(dropout),
        LSTM(units_2),
        Dropout(dropout),
        Dense(dense_units, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy', 'AUC'])
    return model

def run_grid_search(X, y, class_weight_dict):
    print("\n--- Starting Grid Search ---")
    
    # Reduced grid for speed, adjusted for larger dataset of "1"s
    param_grid = {
        'units_1': [50, 100],
        'units_2': [25, 50],
        'dropout': [0.2, 0.3],
        'batch_size': [32, 64],
        'epochs': [20] 
    }
    
    keys, values = zip(*param_grid.items())
    combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
    
    best_score = -1
    best_params = None
    best_model = None
    
    print(f"Testing {len(combinations)} combinations...")
    
    for i, params in enumerate(combinations):
        print(f"\n[{i+1}/{len(combinations)}] Testing: {params}")
        
        model = build_model((X.shape[1], X.shape[2]), params)
        
        model.fit(
            X, y, 
            epochs=params['epochs'], 
            batch_size=params['batch_size'], 
            class_weight=class_weight_dict,
            verbose=0
        )
        
        y_pred_prob = model.predict(X, verbose=0)
        y_pred = (y_pred_prob > 0.5).astype(int)
        
        f1 = f1_score(y, y_pred)
        
        print(f"   -> F1 Score: {f1:.4f}")
        
        if f1 > best_score:
            best_score = f1
            best_params = params
            best_model = model
            print("   *** New Best Model ***")
            
    print("\n--- Grid Search Complete ---")
    print(f"Best Params: {best_params}")
    print(f"Best F1: {best_score:.4f}")
    return best_model, best_params

def create_plot(df, pred_dates, y_true, y_pred_prob):
    print("Generating plot...")
    
    plot_df = df.loc[pred_dates].copy()
    plot_df['prediction_prob'] = y_pred_prob
    plot_df['is_profitable'] = y_true
    
    fig = make_subplots(
        rows=2, cols=1, 
        shared_xaxes=True, 
        vertical_spacing=0.05,
        row_heights=[0.7, 0.3],
        subplot_titles=(f"BTC/USDT Price & Target Periods", "LSTM Exit/Entry Signals")
    )

    # --- Top Panel: Price & SMAs ---
    fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df['close'], name='Price', line=dict(color='white', width=1)), row=1, col=1)
    fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df['sma365'], name='SMA 365', line=dict(color='orange', width=1.5)), row=1, col=1)
    
    # Highlight Profitable Periods (Ground Truth)
    # We construct a signal line that is high during profitable periods
    fig.add_trace(go.Scatter(
        x=plot_df.index, 
        y=plot_df['is_profitable'] * plot_df['close'].max(), # Scale to fit chart roughly
        name='Target Period (1=Active)',
        fill='tozeroy',
        mode='none',
        fillcolor='rgba(0, 255, 0, 0.1)', # Light green shading
    ), row=1, col=1)

    # --- Bottom Panel: Predictions ---
    fig.add_trace(go.Scatter(
        x=plot_df.index, 
        y=plot_df['prediction_prob'], 
        name='AI Probability',
        fill='tozeroy',
        line=dict(color='#00ff00', width=1)
    ), row=2, col=1)
    
    fig.add_hline(y=0.5, line_dash="dash", line_color="gray", annotation_text="Hold Threshold", row=2, col=1)

    fig.update_layout(
        template='plotly_dark',
        title="LSTM Trend Duration Identifier",
        hovermode='x unified',
        height=800,
        margin=dict(l=10, r=10, t=50, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    # Mobile Fullscreen CSS
    custom_style = """
    <style>
        body { margin: 0; padding: 0; background-color: #111; }
        .plotly-graph-div { height: 100vh !important; }
    </style>
    """
    html_content = fig.to_html(full_html=True, include_plotlyjs='cdn')
    html_content = html_content.replace('</head>', f'{custom_style}</head>')
    
    with open("index.html", "w") as f:
        f.write(html_content)
    print("index.html created.")

def run_server():
    class Handler(http.server.SimpleHTTPRequestHandler):
        def log_message(self, format, *args):
            pass 

    with socketserver.TCPServer(("", PORT), Handler) as httpd:
        print(f"\nServing results at http://localhost:{PORT}")
        print("Press Ctrl+C to stop.")
        httpd.serve_forever()

def main():
    # 1. Fetch
    df = fetch_data()
    
    # 2. Features
    df = add_features(df)
    
    # 3. Prepare
    X, y, dates, full_df = prepare_data(df)
    
    print(f"Data shape: {X.shape}")
    print(f"Profitable days (Target=1): {np.sum(y)}")
    
    # 4. Class Weights
    weights = compute_class_weight(class_weight='balanced', classes=np.unique(y), y=y)
    class_weight_dict = dict(enumerate(weights))
    print(f"Class weights: {class_weight_dict}")
    
    # 5. Grid Search
    best_model, best_params = run_grid_search(X, y, class_weight_dict)
    
    if best_model is None:
        best_model = build_model((X.shape[1], X.shape[2]), {})
        best_model.fit(X, y, epochs=50, batch_size=32, class_weight=class_weight_dict, verbose=1)

    # 6. Predict
    y_pred_prob = best_model.predict(X)
    
    # 7. Plot
    create_plot(full_df, dates, y, y_pred_prob.flatten())
    
    # 8. Server
    run_server()

if __name__ == "__main__":
    main()
