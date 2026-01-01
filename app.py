import os
import io
import base64
import threading
import time
import logging
from flask import Flask, render_template_string, jsonify

# Matplotlib backend for non-interactive environments (servers)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import gdown
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

# --- Configuration ---
torch.manual_seed(42)
np.random.seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

HORIZONS_DAYS = [1, 2, 3, 4, 5, 6, 7, 14, 30]

# IMPLEMENTATION 6: Sparse Slopes (Fibonacci sequence) to reduce collinearity
LOOKBACK_SLOPES = [1, 2, 3, 5, 8, 13, 21, 30]

SEQ_LEN = 24

# IMPLEMENTATION 3: Change Resampling to 1 Day
RESAMPLE_FREQ = '1d'

BATCH_SIZE = 64
EPOCHS = 10
FILE_ID = '1kDCl_29nXyW1mLNUAS-nsJe0O2pOuO6o'

# --- Global State ---
app = Flask(__name__)
training_state = {
    'status': 'idle', # idle, downloading, processing, training, complete, error
    'progress': [],
    'plot_image': None,
    'error_msg': None
}

# --- Data & Model Functions ---

def get_and_process_data():
    log("Starting data download...")
    url = f'https://drive.google.com/uc?id={FILE_ID}'
    output_file = 'ohlcv_data.csv'

    if not os.path.exists(output_file):
        try:
            gdown.download(url, output_file, quiet=False)
        except Exception as e:
            log(f"Download warning: {e}")

    log("Loading and resampling data...")
    try:
        df = pd.read_csv(output_file)
        df.columns = [c.lower().strip() for c in df.columns]
        date_col = next((c for c in df.columns if 'date' in c or 'time' in c), None)
        df[date_col] = pd.to_datetime(df[date_col])
        df = df.set_index(date_col).sort_index()
        
        # Resampling based on config
        df_resampled = df['close'].resample(RESAMPLE_FREQ).last().dropna().to_frame()
        return df_resampled
    except Exception as e:
        log(f"Data processing error: {e}")
        return None

def calculate_rolling_slope_fast(series, window_size):
    y = series.values
    x = np.arange(window_size)
    n = window_size
    sum_x = x.sum()
    sum_x2 = (x ** 2).sum()
    delta = n * sum_x2 - sum_x ** 2
    
    sum_y = np.convolve(y, np.ones(n), 'valid')
    sum_xy = np.convolve(y, x[::-1], 'valid')
    
    numerator = n * sum_xy - sum_x * sum_y
    m = numerator / delta
    
    pad = np.full(n - 1, np.nan)
    return np.concatenate([pad, m])

def get_rows_per_day():
    if RESAMPLE_FREQ == '1h':
        return 24
    elif RESAMPLE_FREQ == '1d':
        return 1
    else:
        return 1440 # Default to minutes

def feature_engineering(df):
    log("Feature Engineering (Calculating Slopes)...")
    feature_data = pd.DataFrame(index=df.index)
    
    rows_per_day = get_rows_per_day()
    log(f"Rows per day calculated as: {rows_per_day}")

    for day in LOOKBACK_SLOPES:
        window_size = day * rows_per_day
        if len(df) > window_size:
            feature_data[f'slope_{day}d'] = calculate_rolling_slope_fast(df['close'], window_size)
            
    log("Creating Targets...")
    target_data = pd.DataFrame(index=df.index)
    for h_days in HORIZONS_DAYS:
        h_steps = h_days * rows_per_day
        future_close = df['close'].shift(-h_steps)
        target_data[f'target_{h_days}d'] = (future_close > df['close']).astype(int)
        target_data.loc[df.index[-h_steps:], f'target_{h_days}d'] = np.nan

    full_data = pd.concat([feature_data, target_data], axis=1).dropna()
    return full_data

def create_sequences(data, feature_cols, target_cols, seq_len):
    X_vals = data[feature_cols].values
    y_vals = data[target_cols].values
    X, y = [], []
    for i in range(len(data) - seq_len):
        X.append(X_vals[i : i+seq_len])
        y.append(y_vals[i+seq_len])
    return torch.FloatTensor(np.array(X)), torch.FloatTensor(np.array(y))

class TrendLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2):
        super(TrendLSTM, self).__init__()
        # Keeping Dropout at 0.6 as per previous instruction
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=0.6)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :] 
        out = self.fc(out)
        return self.sigmoid(out)

# --- Pipeline Logic ---

def log(message):
    print(message)
    training_state['progress'].append(message)

def run_pipeline():
    try:
        training_state['status'] = 'downloading'
        df = get_and_process_data()
        if df is None:
            raise ValueError("Data could not be loaded")

        training_state['status'] = 'processing'
        data = feature_engineering(df)
        log(f"Dataset Shape: {data.shape}")

        if len(data) < SEQ_LEN + BATCH_SIZE:
             raise ValueError("Dataset too small after resampling/processing.")

        # Split
        n = len(data)
        train_end = int(n * 0.6)
        val_end = int(n * 0.8)
        
        feature_cols = [c for c in data.columns if 'slope' in c]
        target_cols = [c for c in data.columns if 'target' in c]
        
        # Scale
        scaler = StandardScaler()
        scaler.fit(data.iloc[:train_end][feature_cols])
        data[feature_cols] = scaler.transform(data[feature_cols])
        
        train_data = data.iloc[:train_end]
        val_data = data.iloc[train_end:val_end]
        test_data = data.iloc[val_end:]
        
        log("Generating Sequences...")
        X_train, y_train = create_sequences(train_data, feature_cols, target_cols, SEQ_LEN)
        X_val, y_val = create_sequences(val_data, feature_cols, target_cols, SEQ_LEN)
        X_test, y_test = create_sequences(test_data, feature_cols, target_cols, SEQ_LEN)
        
        train_loader = DataLoader(TensorDataset(X_train, y_train), shuffle=True, batch_size=BATCH_SIZE)
        val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=BATCH_SIZE)
        
        model = TrendLSTM(input_dim=len(feature_cols), hidden_dim=64, output_dim=len(target_cols)).to(device)
        criterion = nn.BCELoss()
        
        # IMPLEMENTATION 1: Reduced Learning Rate to 0.0001
        optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-4)
        
        train_losses = []
        val_losses = []

        training_state['status'] = 'training'
        log(f"Starting Training on {device} for {EPOCHS} epochs...")
        
        total_batches = len(train_loader)
        for epoch in range(EPOCHS):
            model.train()
            batch_loss = []
            
            for i, (X_batch, y_batch) in enumerate(train_loader):
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                optimizer.zero_grad()
                y_pred = model(X_batch)
                loss = criterion(y_pred, y_batch)
                loss.backward()
                
                # IMPLEMENTATION 2: Gradient Clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                optimizer.step()
                batch_loss.append(loss.item())
                
                # Logging progress every 10 batches (or every batch if dataset is small)
                log_freq = 10 if total_batches > 10 else 1
                if (i + 1) % log_freq == 0 or (i + 1) == total_batches:
                    log(f"Epoch {epoch+1} | Batch {i+1}/{total_batches} | Loss: {loss.item():.4f}")
            
            # Validation
            model.eval()
            val_batch_loss = []
            with torch.no_grad():
                for X_v, y_v in val_loader:
                    X_v, y_v = X_v.to(device), y_v.to(device)
                    y_p = model(X_v)
                    loss = criterion(y_p, y_v)
                    val_batch_loss.append(loss.item())
                    
            avg_train = np.mean(batch_loss)
            avg_val = np.mean(val_batch_loss)
            train_losses.append(avg_train)
            val_losses.append(avg_val)
            
            log(f"FINISH EPOCH {epoch+1}/{EPOCHS} | Avg Train: {avg_train:.4f} | Avg Val: {avg_val:.4f}")

        # Visualization
        log("Generating Visualization...")
        plt.style.use('dark_background')
        fig = plt.figure(figsize=(14, 10))
        
        # Plot 1: Loss
        plt.subplot(2, 1, 1)
        plt.plot(train_losses, label='Train Loss', color='cyan')
        plt.plot(val_losses, label='Val Loss', color='orange')
        plt.title('Training Metrics')
        plt.legend()
        plt.grid(alpha=0.2)
        
        # Plot 2: Predictions
        model.eval()
        with torch.no_grad():
            test_preds = model(X_test.to(device)).cpu().numpy()
            y_test_cpu = y_test.numpy()
        
        pred_df = pd.DataFrame(test_preds, columns=target_cols)
        actual_df = pd.DataFrame(y_test_cpu, columns=target_cols)
        
        plt.subplot(2, 1, 2)
        subset = 200
        
        # 1 Day Certainty
        plt.plot(actual_df.iloc[:subset, 0], color='lime', alpha=0.3, linewidth=1, label='Actual 1D')
        plt.plot(pred_df.iloc[:subset, 0], color='lime', linestyle='--', linewidth=1.5, label='Pred 1D Probability')
        
        # 30 Day Certainty
        plt.plot(actual_df.iloc[:subset, -1], color='magenta', alpha=0.3, linewidth=1, label='Actual 30D')
        plt.plot(pred_df.iloc[:subset, -1], color='magenta', linestyle='--', linewidth=1.5, label='Pred 30D Probability')
        
        plt.title(f'Prediction Confidence (First {subset} Test Hours)')
        plt.ylabel('Probability (1=Up, 0=Down)')
        plt.legend()
        plt.grid(alpha=0.2)
        plt.tight_layout()

        # Save to buffer
        img = io.BytesIO()
        plt.savefig(img, format='png', bbox_inches='tight', facecolor='#121212')
        img.seek(0)
        plot_url = base64.b64encode(img.getvalue()).decode()
        plt.close(fig)

        training_state['plot_image'] = plot_url
        training_state['status'] = 'complete'
        log("Pipeline complete!")

    except Exception as e:
        training_state['status'] = 'error'
        training_state['error_msg'] = str(e)
        log(f"CRITICAL ERROR: {e}")

# --- Flask Routes ---

@app.route('/status')
def status():
    return jsonify({
        'status': training_state['status'],
        'logs': training_state['progress'][-10:] # Return last 10 logs
    })

@app.route('/')
def index():
    if training_state['status'] == 'complete' and training_state['plot_image']:
        html = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Trend LSTM Results</title>
            <style>
                body { background-color: #121212; color: #ffffff; font-family: monospace; text-align: center; }
                img { max-width: 95%; height: auto; border: 2px solid #333; border-radius: 8px; margin-top: 20px; }
                .status { color: #00ff00; margin-bottom: 10px; }
            </style>
        </head>
        <body>
            <h1>Trend Prediction LSTM Results</h1>
            <div class="status">Training Complete</div>
            <img src="data:image/png;base64,{{ plot_url }}" alt="Chart">
        </body>
        </html>
        """
        return render_template_string(html, plot_url=training_state['plot_image'])
    
    elif training_state['status'] == 'error':
        return f"<h1>Error Occurred</h1><p>{training_state['error_msg']}</p>"
        
    else:
        # Refreshing loading page
        html = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Training Model...</title>
            <meta http-equiv="refresh" content="3">
            <style>
                body { background-color: #121212; color: #00ff00; font-family: monospace; padding: 50px; }
                .box { border: 1px solid #333; padding: 20px; max-width: 600px; margin: 0 auto; }
                h1 { color: #ffffff; }
            </style>
        </head>
        <body>
            <div class="box">
                <h1>System Training...</h1>
                <p>Status: {{ status }}</p>
                <hr style="border-color: #333;">
                <h3>Live Logs:</h3>
                <ul>
                {% for log in logs %}
                    <li>> {{ log }}</li>
                {% endfor %}
                </ul>
                <p><i>Page will auto-refresh every 3 seconds until results are ready.</i></p>
            </div>
        </body>
        </html>
        """
        return render_template_string(html, status=training_state['status'], logs=training_state['progress'][-15:])

# Start background training thread on import/start
threading.Thread(target=run_pipeline, daemon=True).start()

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)