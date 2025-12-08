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
import time

# --- Configuration ---
SYMBOL = 'BTC/USDT'
TIMEFRAME = '1d'
START_DATE = '2018-01-01 00:00:00'
PORT = 8080
SEQ_LENGTH = 30
NOISE_LEVEL = 0.002  # 0.2% Standard Deviation noise added to returns

# Hyperparameters
BATCH_SIZE = 64
DROPOUT_RATE = 0.4
EPOCHS = 50
UNITS_1 = 32
UNITS_2 = 16

# Profitable Periods
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

def generate_mock_data(original_df, noise_std=0.02):
    """
    Generates a synthetic price history by adding Gaussian noise to daily returns
    and applying a 0.9 to 1.1 oscillator with 100-day frequency.
    Preserves general trend but changes specific price action.
    """
    print(f"Generating Mock Data with {noise_std*100}% return noise and 0.9-1.1 oscillator (100-day freq)...")
    df = original_df.copy()
    
    # 1. Calculate original returns
    df['pct_change'] = df['close'].pct_change()
    df['pct_change'].fillna(0, inplace=True)
    
    # 2. Add Noise to returns
    noise = np.random.normal(0, noise_std, size=len(df))
    df['distorted_returns'] = df['pct_change'] + noise
    
    # 3. Create oscillator: sine wave from 0.9 to 1.1 with 100-day period
    # One complete wave (0.9->1.1->0.9) takes 100 days
    # Sine wave oscillates between -1 and 1, we scale to 0.9-1.1
    days = np.arange(len(df))
    # Frequency: 2π / period (100 days)
    frequency = 2 * np.pi / 100
    # Sine wave oscillating between -1 and 1
    sine_wave = np.sin(frequency * days)
    # Scale to 0.9-1.1 range: (sine_wave + 1) * 0.1 + 0.9
    oscillator = (sine_wave + 1) * 0.1 + 0.9
    
    # 4. Apply oscillator to distorted returns
    df['oscillated_returns'] = df['distorted_returns'] * oscillator
    
    # 5. Reconstruct Price Path
    # Start from original first close
    start_price = df['close'].iloc[0]
    # Cumulatively apply oscillated returns
    # (1 + r1) * (1 + r2) ...
    price_path = start_price * (1 + df['oscillated_returns']).cumprod()
    
    # 6. Reconstruct OHLC roughly to maintain candle structure relative to Close
    # We assume the ratio of High/Close, Low/Close stays similar to original
    df['high_ratio'] = df['high'] / df['close']
    df['low_ratio'] = df['low'] / df['close']
    df['open_ratio'] = df['open'] / df['close']
    
    df['mock_close'] = price_path
    df['mock_high'] = df['mock_close'] * df['high_ratio']
    df['mock_low'] = df['mock_close'] * df['low_ratio']
    df['mock_open'] = df['mock_close'] * df['open_ratio']
    
    # Replace original columns for feature engineering
    mock_df = pd.DataFrame(index=df.index)
    mock_df['open'] = df['mock_open']
    mock_df['high'] = df['mock_high']
    mock_df['low'] = df['mock_low']
    mock_df['close'] = df['mock_close']
    mock_df['volume'] = df['volume'] # Keep volume same
    
    return mock_df

def add_features(df):
    # Recalculate features on the passed dataframe (Real or Mock)
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
    
    for i in range(SEQ_LENGTH, len(data)):
        x_seq = data_scaled[i-SEQ_LENGTH:i]
        X.append(x_seq)
        y.append(targets[i])
        prediction_dates.append(dates[i])
        
    return np.array(X), np.array(y), prediction_dates, df

def build_model(input_shape):
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

def create_comparison_plot(real_df, mock_df, pred_dates, y_true, y_pred_prob):
    print("Generating comparison plot...")
    
    # Align mock_df to prediction dates
    plot_df = mock_df.loc[pred_dates].copy()
    plot_df['prediction_prob'] = y_pred_prob
    plot_df['is_profitable'] = y_true
    plot_df['pred_signal'] = (plot_df['prediction_prob'] > 0.5).astype(int)
    
    # Get Real prices for comparison background
    real_prices = real_df.loc[pred_dates]['close']

    # Strategy Calculation on MOCK data
    plot_df['prev_close'] = plot_df['close'].shift(1)
    plot_df['prev_sma365'] = plot_df['sma365'].shift(1)
    
    long_condition = (plot_df['pred_signal'] == 1) & (plot_df['prev_close'] > plot_df['prev_sma365'])
    short_condition = (plot_df['pred_signal'] == 1) & (plot_df['prev_close'] < plot_df['prev_sma365'])
    
    plot_df['strategy_ret'] = 0.0
    plot_df.loc[long_condition, 'strategy_ret'] = plot_df.loc[long_condition, 'pct_change']
    plot_df.loc[short_condition, 'strategy_ret'] = -plot_df.loc[short_condition, 'pct_change']
    
    plot_df['cum_strategy'] = (1 + plot_df['strategy_ret']).cumprod()
    plot_df['cum_bnh'] = (1 + plot_df['pct_change']).cumprod()

    # --- Plotting ---
    fig = make_subplots(
        rows=3, cols=1, 
        shared_xaxes=True, 
        vertical_spacing=0.05,
        row_heights=[0.5, 0.25, 0.25],
        subplot_titles=(f"Mock vs Real Price (Noise: {NOISE_LEVEL*100}%)", "Model Confidence on Mock Data", "Equity Curve (Mock Data)")
    )

    # Panel 1: Mock Price vs Real Price
    fig.add_trace(go.Scatter(x=plot_df.index, y=real_prices, name='Original Price', line=dict(color='rgba(255,255,255,0.2)', width=1)), row=1, col=1)
    fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df['close'], name='Distorted Price', line=dict(color='cyan', width=1)), row=1, col=1)
    fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df['sma365'], name='Mock SMA 365', line=dict(color='orange', width=1.5)), row=1, col=1)
    
    # Highlight Target Periods
    fig.add_trace(go.Scatter(
        x=plot_df.index, 
        y=plot_df['is_profitable'] * plot_df['close'].max(), 
        name='Target Period',
        fill='tozeroy',
        mode='none',
        fillcolor='rgba(0, 255, 0, 0.1)',
        hoverinfo='skip'
    ), row=1, col=1)

    # Panel 2: Predictions
    fig.add_trace(go.Scatter(
        x=plot_df.index, 
        y=plot_df['prediction_prob'], 
        name='AI Confidence',
        fill='tozeroy',
        line=dict(color='#00ff00', width=1)
    ), row=2, col=1)
    fig.add_hline(y=0.5, line_dash="dash", line_color="gray", row=2, col=1)

    # Panel 3: Equity Curve
    fig.add_trace(go.Scatter(
        x=plot_df.index, 
        y=plot_df['cum_strategy'], 
        name='Strategy (Mock)',
        line=dict(color='#00ffff', width=2)
    ), row=3, col=1)
    
    fig.add_trace(go.Scatter(
        x=plot_df.index, 
        y=plot_df['cum_bnh'], 
        name='Buy & Hold (Mock)',
        line=dict(color='gray', width=1, dash='dot')
    ), row=3, col=1)

    fig.update_layout(
        template='plotly_dark',
        title="Stress Test: Model Robustness on Distorted Data",
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
    # 1. Fetch & Prepare ORIGINAL Training Data
    print("--- Training Phase (Original Data) ---")
    raw_df = fetch_data()
    train_df = add_features(raw_df.copy()) # Feature engineering on original
    X_train, y_train, _, _ = prepare_data(train_df)
    
    # 2. Class Weights
    weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
    class_weight_dict = dict(enumerate(weights))
    
    # 3. Train Model on Original Data
    model = build_model((X_train.shape[1], X_train.shape[2]))
    model.fit(
        X_train, y_train, 
        epochs=EPOCHS, 
        batch_size=BATCH_SIZE, 
        class_weight=class_weight_dict,
        verbose=1
    )
    
    # 4. Generate MOCK Test Data
    print("\n--- Testing Phase (Mock/Distorted Data) ---")
    mock_raw_df = generate_mock_data(raw_df, noise_std=NOISE_LEVEL)
    mock_featured_df = add_features(mock_raw_df) # Recalculate indicators on distorted price!
    
    # 5. Predict on Mock Data
    X_mock, y_mock, dates_mock, full_mock_df = prepare_data(mock_featured_df)
    
    print(f"Predicting on {len(X_mock)} distorted samples...")
    y_pred_prob = model.predict(X_mock)
    
    # 6. Plot Results
    # We pass 'raw_df' just to plot the faint original line for visual comparison
    create_comparison_plot(add_features(raw_df.copy()), full_mock_df, dates_mock, y_mock, y_pred_prob.flatten())
    
    run_server()

if __name__ == "__main__":
    main()
