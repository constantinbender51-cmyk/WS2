import ccxt
import pandas as pd
import numpy as np
import matplotlib
# Set backend to Agg to avoid needing a GUI window
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io
import flask
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime, timedelta
import threading

# ==========================================
# 1. CONFIGURATION
# ==========================================
SYMBOL = 'BTC/USDT'
TIMEFRAME = '1d'
START_DATE = '2018-01-01 00:00:00'
SMA_PERIOD = 150
LOOK_AHEAD = 150 # Target is 150 days into the future
WINDOW_SIZE = 60 # Input window (lags)
PORT = 8080
HOST = '0.0.0.0'

# ==========================================
# 2. DATA FETCHING (CCXT)
# ==========================================
def fetch_data():
    print(f"Fetching {SYMBOL} data from Binance starting {START_DATE}...")
    exchange = ccxt.binance()
    since = exchange.parse8601(START_DATE)
    all_ohlcv = []
    
    while True:
        try:
            ohlcv = exchange.fetch_ohlcv(SYMBOL, TIMEFRAME, since)
            if not ohlcv:
                break
            all_ohlcv.extend(ohlcv)
            since = ohlcv[-1][0] + 86400000 # move to next day
            print(f"Fetched {len(all_ohlcv)} candles...", end='\r')
            
            # Break if we reached current time
            if since > exchange.milliseconds():
                break
        except Exception as e:
            print(f"\nError fetching data: {e}")
            break
            
    df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['date'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('date', inplace=True)
    df = df[~df.index.duplicated(keep='first')]
    print(f"\nTotal candles: {len(df)}")
    return df

# ==========================================
# 3. PREPROCESSING
# ==========================================
def prepare_data(df):
    print("Preprocessing data...")
    
    # Calculate the 150-day SMA
    df['sma_150'] = df['close'].rolling(window=SMA_PERIOD).mean()
    
    # Calculate Log Returns of Price (Feature)
    # log_ret_t = ln(Price_t / Price_{t-1})
    df['log_ret'] = np.log(df['close'] / df['close'].shift(1))
    
    # Create the Target: 
    # The goal is to predict the SMA 150 days in the future.
    # To make this stationary for the LSTM, we predict the Log Return 
    # from Current Price (t) to Future SMA (t + 150).
    # Target_t = ln(SMA_{t+150} / Close_t)
    df['future_sma'] = df['sma_150'].shift(-LOOK_AHEAD)
    df['target'] = np.log(df['future_sma'] / df['close'])

    # Drop NaNs created by rolling SMA, shifting target, and log return calculation
    df_clean = df.dropna().copy()
    
    feature_data = df_clean['log_ret'].values.reshape(-1, 1)
    target_data = df_clean['target'].values.reshape(-1, 1)
    
    # Scale data
    scaler_features = MinMaxScaler(feature_range=(-1, 1))
    scaler_target = MinMaxScaler(feature_range=(-1, 1))
    
    scaled_features = scaler_features.fit_transform(feature_data)
    scaled_target = scaler_target.fit_transform(target_data)
    
    X, y = [], []
    # Create sequences
    # We need WINDOW_SIZE past days of features to predict the target associated with the current time
    for i in range(WINDOW_SIZE, len(df_clean)):
        X.append(scaled_features[i-WINDOW_SIZE:i])
        y.append(scaled_target[i])
        
    X = np.array(X)
    y = np.array(y)
    
    # Reference data for reconstruction later (Close prices aligned with prediction index)
    # The 'y' array starts at index WINDOW_SIZE of df_clean
    ref_prices = df_clean['close'].values[WINDOW_SIZE:]
    ref_dates = df_clean.index[WINDOW_SIZE:]
    
    return X, y, scaler_features, scaler_target, ref_prices, ref_dates

# ==========================================
# 4. MODEL TRAINING
# ==========================================
def train_model(X, y):
    # Split 80/20 non-shuffled for time series
    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    
    print(f"Training shapes: X={X_train.shape}, y={y_train.shape}")
    
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
        Dropout(0.2),
        LSTM(50, return_sequences=False),
        Dropout(0.2),
        Dense(25),
        Dense(1)
    ])
    
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
    
    history = model.fit(
        X_train, y_train, 
        batch_size=32, 
        epochs=10, # Kept low for demonstration speed; increase for performance
        validation_data=(X_test, y_test),
        verbose=1
    )
    
    return model, X_train, X_test, y_train, y_test

# ==========================================
# 5. VISUALIZATION & SERVER
# ==========================================
def create_plot(model, X, y, scaler_target, ref_prices, ref_dates, split_index):
    # Predict
    preds_scaled = model.predict(X)
    
    # Inverse transform predictions and actuals (Back to log return scale)
    preds_log_ret = scaler_target.inverse_transform(preds_scaled).flatten()
    actual_log_ret = scaler_target.inverse_transform(y).flatten()
    
    # Reconstruct the absolute Future SMA values
    # Future_SMA = Current_Close * exp(Predicted_Log_Ret)
    pred_sma_values = ref_prices * np.exp(preds_log_ret)
    actual_sma_values = ref_prices * np.exp(actual_log_ret)
    
    # Plotting
    plt.figure(figsize=(12, 6))
    
    # We plot the Future SMA targets. 
    # Note: These values represent the SMA 150 days AFTER the date on the x-axis.
    
    plt.plot(ref_dates, actual_sma_values, label='Actual 150-day SMA (Target)', color='black', alpha=0.6, linewidth=1)
    
    # Plot Training Predictions
    plt.plot(ref_dates[:split_index], pred_sma_values[:split_index], label='Training Predictions', color='blue', alpha=0.7)
    
    # Plot Validation Predictions
    plt.plot(ref_dates[split_index:], pred_sma_values[split_index:], label='Validation Predictions', color='orange', alpha=0.8)
    
    plt.title(f'{SYMBOL} LSTM Prediction: Target is 150-day SMA (150 days ahead)')
    plt.xlabel('Date (Observation Time)')
    plt.ylabel('Price Level (USDT)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save to BytesIO object
    img = io.BytesIO()
    plt.savefig(img, format='png', dpi=100)
    img.seek(0)
    plt.close()
    return img

app = flask.Flask(__name__)
plot_image = None

@app.route('/')
def home():
    return """
    <html>
        <head>
            <title>BTC LSTM Monitor</title>
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <style>
                body { font-family: sans-serif; text-align: center; margin: 0; padding: 20px; background: #f4f4f9; }
                img { max-width: 100%; height: auto; border: 1px solid #ccc; box-shadow: 0 4px 8px rgba(0,0,0,0.1); }
                .container { max-width: 1200px; margin: 0 auto; }
            </style>
        </head>
        <body>
            <div class="container">
                <h1>LSTM Regressor: BTC/USDT</h1>
                <p>Target: 150-day SMA (150 days in the future) | Inputs: 60-day Log Returns</p>
                <img src="/plot.png" alt="Prediction Plot">
                <p><i>Predictions show the model's estimate of the SMA 150 days relative to the date on the X-axis.</i></p>
            </div>
        </body>
    </html>
    """

@app.route('/plot.png')
def plot_png():
    if plot_image:
        return flask.send_file(plot_image, mimetype='image/png')
    else:
        return "Training in progress...", 503

def run_pipeline():
    global plot_image
    
    # 1. Fetch
    df = fetch_data()
    
    # 2. Prep
    X, y, _, scaler_target, ref_prices, ref_dates = prepare_data(df)
    
    # 3. Train
    model, X_train, X_test, y_train, y_test = train_model(X, y)
    
    # 4. Generate Plot (Reconstruct full dataset for continuity)
    split_index = len(X_train)
    plot_image = create_plot(model, X, y, scaler_target, ref_prices, ref_dates, split_index)
    
    print("\n\nPipeline Complete. Server is ready.")

# ==========================================
# 6. MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    # Run training in a separate thread so the server can start listening immediately (or blocking, up to preference)
    # We will run training first, then start server, to ensure data is ready.
    try:
        run_pipeline()
        print(f"Starting Web Server on {HOST}:{PORT}...")
        app.run(host=HOST, port=PORT)
    except KeyboardInterrupt:
        print("Stopping...")
