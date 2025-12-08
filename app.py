import ccxt
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import http.server
import socketserver
import webbrowser
import os
import threading
import time
from datetime import datetime

# --- Configuration ---
SYMBOL = 'BTC/USDT'
TIMEFRAME = '1d'
START_DATE = '2018-01-01 00:00:00'
PORT = 8080
SEQ_LENGTH = 30  # 30 lag features

# Hyperparameters (Hardcoded based on optimization)
BATCH_SIZE = 64
DROPOUT_RATE = 0.4
EPOCHS = 50
UNITS_1 = 32
UNITS_2 = 16

# Profitable Periods (Start, End) - Inclusive
PROFITABLE_PERIODS = [
    ('2020-09-06', '2021-02-15'),
    ('2021-07-12', '2021-10-11'),
    ('2022-03-28', '2023-01-02'), # Downtrend retest period
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
    
    # Log Return in Percent
    df['log_ret'] = np.log(df['close'] / df['close'].shift(1)) * 100
    
    # Drop NaNs
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

def build_model(input_shape):
    print(f"Building Model (Units: {UNITS_1}/{UNITS_2}, Dropout: {DROPOUT_RATE})...")
    model = Sequential([
        Input(shape=input_shape),
        LSTM(UNITS_1, return_sequences=True),
        Dropout(DROPOUT_RATE),
        LSTM(UNITS_2),
        Dropout(DROPOUT_RATE),
        Dense(16, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy', 'AUC'])
    return model

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
        subplot_titles=(f"BTC/USDT Price & Target Periods", "LSTM Trend Signal")
    )

    # --- Top Panel: Price & SMAs ---
    fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df['close'], name='Price', line=dict(color='white', width=1)), row=1, col=1)
    fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df['sma365'], name='SMA 365', line=dict(color='orange', width=1.5)), row=1, col=1)
    
    # Highlight Profitable Periods (Ground Truth)
    # We construct a signal line that is high during profitable periods
    fig.add_trace(go.Scatter(
        x=plot_df.index, 
        y=plot_df['is_profitable'] * plot_df['close'].max(), # Scale to fit chart roughly
        name='Target Period',
        fill='tozeroy',
        mode='none',
        fillcolor='rgba(0, 255, 0, 0.1)', # Light green shading
        hoverinfo='skip'
    ), row=1, col=1)

    # --- Bottom Panel: Predictions ---
    fig.add_trace(go.Scatter(
        x=plot_df.index, 
        y=plot_df['prediction_prob'], 
        name='AI Confidence',
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
    
    # 5. Build & Train (Direct)
    model = build_model((X.shape[1], X.shape[2]))
    
    model.fit(
        X, y, 
        epochs=EPOCHS, 
        batch_size=BATCH_SIZE, 
        class_weight=class_weight_dict,
        verbose=1
    )

    # 6. Predict
    y_pred_prob = model.predict(X)
    
    # 7. Plot
    create_plot(full_df, dates, y, y_pred_prob.flatten())
    
    # 8. Server
    run_server()

if __name__ == "__main__":
    main()
