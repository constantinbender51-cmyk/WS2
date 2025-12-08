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
from datetime import datetime, timedelta

# --- Configuration ---
SYMBOL = 'BTC/USDT'
TIMEFRAME = '1d'
START_DATE = '2018-01-01 00:00:00'
PORT = 8080
SEQ_LENGTH = 30  # 30 lag features

# Hyperparameters
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

def label_data_by_date(df):
    """
    Pre-labels the dataframe based on hardcoded dates.
    We do this BEFORE augmentation so the labels get warped/augmented with the price.
    """
    df['target'] = 0
    for start_date, end_date in PROFITABLE_PERIODS:
        mask = (df.index >= pd.to_datetime(start_date)) & (df.index <= pd.to_datetime(end_date))
        df.loc[mask, 'target'] = 1
    return df

def augment_data(df):
    """
    Creates a synthetic version of the dataset with smoothing, 
    time-warping (stretch/contract), and noise injection.
    """
    print("Augmenting data (Smoothing, Warping, Noise)...")
    
    # 1. Prepare Source Arrays
    # We will augment OHLCV and the Target.
    src_len = len(df)
    orig_indices = np.arange(src_len)
    
    # Columns to interpolate
    cols_to_warp = ['open', 'high', 'low', 'close', 'volume', 'target']
    data_values = df[cols_to_warp].values

    # 2. Time Warping Logic (Stretch and Contract)
    # We create a new non-linear time index. 
    # Using a Sine wave to alternate between speeding up and slowing down the data.
    # Frequency: How often we switch from stretch to contract.
    # Amplitude: How intense the stretch/contract is.
    freq = 4 * np.pi / src_len  # roughly 2 cycles over the dataset
    amplitude = src_len * 0.1   # Distort time by up to 10% total length in places
    
    warped_indices = orig_indices + amplitude * np.sin(freq * orig_indices)
    
    # Ensure indices remain within bounds for interpolation
    warped_indices = np.clip(warped_indices, 0, src_len - 1)
    
    # Sort indices to maintain causality (time must move forward)
    warped_indices = np.sort(warped_indices)

    # 3. Create Synthetic Data Container
    synthetic_data = np.zeros_like(data_values)

    # 4. Interpolate and Smooth
    # We interpolate the original values onto the warped time steps.
    for i, col_name in enumerate(cols_to_warp):
        # Linear interpolation performs the "Stretch/Contract"
        interp_series = np.interp(orig_indices, warped_indices, data_values[:, i])
        
        # Apply Smoothing (Rolling mean simulation via convolution)
        # Only smooth price/vol, not the binary target
        if col_name != 'target':
            window_size = 5
            kernel = np.ones(window_size) / window_size
            # 'same' mode keeps array size same
            smoothed_series = np.convolve(interp_series, kernel, mode='same')
            
            # Fix edge effects from convolution
            smoothed_series[:window_size] = interp_series[:window_size]
            smoothed_series[-window_size:] = interp_series[-window_size:]
            
            synthetic_data[:, i] = smoothed_series
        else:
            # For target, threshold the interpolated values back to 0 or 1
            synthetic_data[:, i] = (interp_series > 0.5).astype(int)

    # 5. Add Noise
    # Add roughly 1-2% random noise to prices
    noise_level = 0.015
    noise = np.random.normal(0, noise_level, size=synthetic_data.shape)
    
    # Only apply noise to OHLC, not Volume or Target
    # indices 0-3 are OHLC
    synthetic_data[:, 0:4] = synthetic_data[:, 0:4] * (1 + noise[:, 0:4])
    
    # 6. Reconstruct DataFrame
    syn_df = pd.DataFrame(synthetic_data, columns=cols_to_warp)
    
    # 7. Generate New Timestamps
    # Append this data to the end of the original data.
    last_date = df.index[-1]
    # Assume daily data (1D)
    new_dates = pd.date_range(start=last_date + timedelta(days=1), periods=len(syn_df), freq='D')
    syn_df.index = new_dates
    syn_df.index.name = 'timestamp'
    
    # Combine
    combined_df = pd.concat([df, syn_df])
    
    print(f"Dataset expanded from {len(df)} to {len(combined_df)} rows.")
    return combined_df

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
    
    # Simple % Change for Backtesting
    df['pct_change'] = df['close'].pct_change()
    
    # Drop NaNs created by rolling windows (at the very start)
    df.dropna(inplace=True)
    return df

def prepare_data(df):
    print("Preparing LSTM sequences...")
    
    # NOTE: 'target' is now already in the dataframe from the `label_data_by_date` step
    # We do NOT recalculate it here, otherwise we lose the targets for the synthetic data.

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
    print("Generating simulation and plot...")
    
    plot_df = df.loc[pred_dates].copy()
    plot_df['prediction_prob'] = y_pred_prob
    plot_df['is_profitable'] = y_true
    plot_df['pred_signal'] = (plot_df['prediction_prob'] > 0.5).astype(int)
    
    # --- Backtest Simulation ---
    plot_df['prev_close'] = plot_df['close'].shift(1)
    plot_df['prev_sma365'] = plot_df['sma365'].shift(1)
    
    # Conditions
    long_condition = (plot_df['pred_signal'] == 1) & (plot_df['prev_close'] > plot_df['prev_sma365'])
    short_condition = (plot_df['pred_signal'] == 1) & (plot_df['prev_close'] < plot_df['prev_sma365'])
    
    plot_df['strategy_ret'] = 0.0
    plot_df.loc[long_condition, 'strategy_ret'] = plot_df.loc[long_condition, 'pct_change']
    plot_df.loc[short_condition, 'strategy_ret'] = -plot_df.loc[short_condition, 'pct_change']
    
    plot_df['cum_strategy'] = (1 + plot_df['strategy_ret']).cumprod()
    plot_df['cum_bnh'] = (1 + plot_df['pct_change']).cumprod()

    # --- Plotting ---
    # Downsample for plotting performance if dataset is huge
    if len(plot_df) > 10000:
        display_df = plot_df.iloc[::2] 
    else:
        display_df = plot_df

    fig = make_subplots(
        rows=3, cols=1, 
        shared_xaxes=True, 
        vertical_spacing=0.05,
        row_heights=[0.5, 0.25, 0.25],
        subplot_titles=(f"BTC/USDT Price (Real + Synthetic)", "LSTM Signal", "Strategy Equity Curve")
    )

    # Panel 1: Price & SMAs
    fig.add_trace(go.Scatter(x=display_df.index, y=display_df['close'], name='Price', line=dict(color='white', width=1)), row=1, col=1)
    fig.add_trace(go.Scatter(x=display_df.index, y=display_df['sma365'], name='SMA 365', line=dict(color='orange', width=1.5)), row=1, col=1)
    
    # Highlight Target Periods
    fig.add_trace(go.Scatter(
        x=display_df.index, 
        y=display_df['is_profitable'] * display_df['close'].max(), 
        name='Target Period',
        fill='tozeroy',
        mode='none',
        fillcolor='rgba(0, 255, 0, 0.1)',
        hoverinfo='skip'
    ), row=1, col=1)

    # Panel 2: Predictions
    fig.add_trace(go.Scatter(
        x=display_df.index, 
        y=display_df['prediction_prob'], 
        name='AI Confidence',
        fill='tozeroy',
        line=dict(color='#00ff00', width=1)
    ), row=2, col=1)
    fig.add_hline(y=0.5, line_dash="dash", line_color="gray", row=2, col=1)

    # Panel 3: Equity Curve
    fig.add_trace(go.Scatter(
        x=display_df.index, 
        y=display_df['cum_strategy'], 
        name='Strategy Return',
        line=dict(color='#00ffff', width=2)
    ), row=3, col=1)
    
    fig.add_trace(go.Scatter(
        x=display_df.index, 
        y=display_df['cum_bnh'], 
        name='Buy & Hold',
        line=dict(color='gray', width=1, dash='dot')
    ), row=3, col=1)

    fig.update_layout(
        template='plotly_dark',
        title="LSTM Strategy Backtest (With Synthetic Data Augmentation)",
        hovermode='x unified',
        height=1000,
        margin=dict(l=10, r=10, t=50, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
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

    # 2. Label Targets (BEFORE augmentation so labels warp with price)
    df = label_data_by_date(df)
    
    # 3. Augment (Double dataset size, smooth, warp, noise)
    df = augment_data(df)

    # 4. Features (Calculate SMAs on the combined dataset)
    df = add_features(df)
    
    # 5. Prepare (Using existing targets)
    X, y, dates, full_df = prepare_data(df)
    
    print(f"Data shape: {X.shape}")
    print(f"Profitable days (Target=1): {np.sum(y)}")
    
    # 6. Class Weights
    weights = compute_class_weight(class_weight='balanced', classes=np.unique(y), y=y)
    class_weight_dict = dict(enumerate(weights))
    print(f"Class weights: {class_weight_dict}")
    
    # 7. Build & Train
    model = build_model((X.shape[1], X.shape[2]))
    
    model.fit(
        X, y, 
        epochs=EPOCHS, 
        batch_size=BATCH_SIZE, 
        class_weight=class_weight_dict,
        verbose=1
    )

    # 8. Predict
    y_pred_prob = model.predict(X)
    
    # 9. Plot & Simulate
    create_plot(full_df, dates, y, y_pred_prob.flatten())
    
    # 10. Server
    run_server()

if __name__ == "__main__":
    main()
