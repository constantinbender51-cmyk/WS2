import ccxt
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import TimeSeriesSplit
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import http.server
import socketserver
import time

# --- Configuration ---
SYMBOL = 'BTC/USDT'
TIMEFRAME = '1d'
START_DATE = '2018-01-01 00:00:00'
PORT = 8080
SEQ_LENGTH = 30
N_SPLITS = 5  # Number of walk-forward folds

# Hyperparameters
BATCH_SIZE = 64
DROPOUT_RATE = 0.4
EPOCHS = 30  # Reduced slightly for speed during WFV
UNITS_1 = 32
UNITS_2 = 16

# Profitable Periods (Start, End) - Inclusive
PROFITABLE_PERIODS = [
    ('2020-09-06', '2021-02-15'),
    ('2021-07-12', '2021-10-11'),
    ('2022-03-28', '2023-01-02'),
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
    df['sma365'] = df['close'].rolling(window=365).mean()
    df['sma120'] = df['close'].rolling(window=120).mean()
    df['sma40'] = df['close'].rolling(window=40).mean()
    
    df['dist_365'] = (df['close'] - df['sma365']) / df['sma365']
    df['dist_120'] = (df['close'] - df['sma120']) / df['sma120']
    df['dist_40'] = (df['close'] - df['sma40']) / df['sma40']
    
    df['log_ret'] = np.log(df['close'] / df['close'].shift(1)) * 100
    df['pct_change'] = df['close'].pct_change()
    
    df.dropna(inplace=True)
    return df

def prepare_data(df):
    print("Preparing LSTM sequences...")
    
    df['target'] = 0
    for start_date, end_date in PROFITABLE_PERIODS:
        mask = (df.index >= pd.to_datetime(start_date)) & (df.index <= pd.to_datetime(end_date))
        df.loc[mask, 'target'] = 1

    feature_cols = ['dist_365', 'dist_120', 'dist_40', 'log_ret']
    
    data = df[feature_cols].values
    targets = df['target'].values
    dates = df.index
    
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)
    
    X, y, prediction_dates = [], [], []
    
    # Need to keep track of original indices to map back to dataframe
    original_indices = []

    for i in range(SEQ_LENGTH, len(data)):
        x_seq = data_scaled[i-SEQ_LENGTH:i]
        X.append(x_seq)
        y.append(targets[i])
        prediction_dates.append(dates[i])
        original_indices.append(i)
        
    return np.array(X), np.array(y), np.array(prediction_dates), df, np.array(original_indices)

def build_model(input_shape):
    # Reduced verbosity for loop
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
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    return model

def walk_forward_validation(X, y, dates, original_indices):
    """
    Performs TimeSeriesSplit Walk-Forward Validation.
    Returns arrays of collected test predictions and their corresponding dates.
    """
    tscv = TimeSeriesSplit(n_splits=N_SPLITS)
    
    wf_dates = []
    wf_preds = []
    wf_true = []
    
    print(f"\n--- Starting Walk-Forward Validation ({N_SPLITS} Splits) ---")
    
    fold = 1
    for train_index, test_index in tscv.split(X):
        print(f"Fold {fold}/{N_SPLITS} | Train Size: {len(train_index)} | Test Size: {len(test_index)}")
        
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        dates_test = dates[test_index]
        
        # Calculate weights for this specific fold
        classes = np.unique(y_train)
        if len(classes) > 1:
            weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_train)
            class_weight_dict = dict(enumerate(weights))
        else:
            class_weight_dict = None

        # Build fresh model for each fold to avoid leakage
        model = build_model((X.shape[1], X.shape[2]))
        
        # Train
        model.fit(
            X_train, y_train, 
            epochs=EPOCHS, 
            batch_size=BATCH_SIZE, 
            class_weight=class_weight_dict,
            verbose=0  # Silent training
        )
        
        # Predict on unseen test data
        preds = model.predict(X_test, verbose=0)
        
        # Store results
        wf_dates.extend(dates_test)
        wf_preds.extend(preds.flatten())
        wf_true.extend(y_test)
        
        fold += 1
        
    return np.array(wf_dates), np.array(wf_preds), np.array(wf_true)

def create_plot(df, pred_dates, y_true, y_pred_prob):
    print("Generating simulation and plot...")
    
    # Subset DF to only the Walk-Forward Validation period
    plot_df = df.loc[pred_dates].copy()
    plot_df['prediction_prob'] = y_pred_prob
    plot_df['is_profitable'] = y_true
    plot_df['pred_signal'] = (plot_df['prediction_prob'] > 0.5).astype(int)
    
    plot_df['prev_close'] = plot_df['close'].shift(1)
    plot_df['prev_sma365'] = plot_df['sma365'].shift(1)
    
    long_condition = (plot_df['pred_signal'] == 1) & (plot_df['prev_close'] > plot_df['prev_sma365'])
    short_condition = (plot_df['pred_signal'] == 1) & (plot_df['prev_close'] < plot_df['prev_sma365'])
    
    plot_df['strategy_ret'] = 0.0
    plot_df.loc[long_condition, 'strategy_ret'] = plot_df.loc[long_condition, 'pct_change']
    plot_df.loc[short_condition, 'strategy_ret'] = -plot_df.loc[short_condition, 'pct_change']
    
    # Re-calc Cumulative Returns for just this period
    plot_df['cum_strategy'] = (1 + plot_df['strategy_ret']).cumprod()
    plot_df['cum_bnh'] = (1 + plot_df['pct_change']).cumprod()

    # --- Plotting ---
    fig = make_subplots(
        rows=3, cols=1, 
        shared_xaxes=True, 
        vertical_spacing=0.05,
        row_heights=[0.5, 0.25, 0.25],
        subplot_titles=(f"BTC/USDT Price (Walk-Forward Period)", "OOS LSTM Signal", "OOS Equity Curve")
    )

    fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df['close'], name='Price', line=dict(color='white', width=1)), row=1, col=1)
    fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df['sma365'], name='SMA 365', line=dict(color='orange', width=1.5)), row=1, col=1)
    
    fig.add_trace(go.Scatter(
        x=plot_df.index, 
        y=plot_df['is_profitable'] * plot_df['close'].max(), 
        name='Target Period',
        fill='tozeroy',
        mode='none',
        fillcolor='rgba(0, 255, 0, 0.1)',
        hoverinfo='skip'
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=plot_df.index, 
        y=plot_df['prediction_prob'], 
        name='AI Confidence (OOS)',
        fill='tozeroy',
        line=dict(color='#00ff00', width=1)
    ), row=2, col=1)
    fig.add_hline(y=0.5, line_dash="dash", line_color="gray", row=2, col=1)

    fig.add_trace(go.Scatter(
        x=plot_df.index, 
        y=plot_df['cum_strategy'], 
        name='Strategy Return',
        line=dict(color='#00ffff', width=2)
    ), row=3, col=1)
    
    fig.add_trace(go.Scatter(
        x=plot_df.index, 
        y=plot_df['cum_bnh'], 
        name='Buy & Hold',
        line=dict(color='gray', width=1, dash='dot')
    ), row=3, col=1)

    fig.update_layout(
        template='plotly_dark',
        title="Walk-Forward Validation Results (Out-of-Sample Only)",
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
    df = fetch_data()
    df = add_features(df)
    X, y, dates, full_df, indices = prepare_data(df)
    
    print(f"Total Data shape: {X.shape}")
    
    # Perform Walk-Forward Validation
    wf_dates, wf_preds, wf_true = walk_forward_validation(X, y, dates, indices)
    
    # Plot results using ONLY the validation data
    create_plot(full_df, wf_dates, wf_true, wf_preds)
    
    run_server()

if __name__ == "__main__":
    main()
