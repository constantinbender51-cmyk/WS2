import gdown
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import os
from tqdm import tqdm  # Progress bar

# --- Configuration ---
torch.manual_seed(42)
np.random.seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

HORIZONS_DAYS = [1, 2, 3, 4, 5, 6, 7, 14, 30]  # Target horizons
LOOKBACK_SLOPES = range(1, 31)                 # Feature lookbacks (1d to 30d)
SEQ_LEN = 24                                   # LSTM Sequence length
RESAMPLE_FREQ = '1h'                           # kept at 1h, but now safe for 1min if you change it
BATCH_SIZE = 64
EPOCHS = 20                                    

# --- 1. Data Loading ---
def get_and_process_data():
    file_id = '1kDCl_29nXyW1mLNUAS-nsJe0O2pOuO6o'
    url = f'https://drive.google.com/uc?id={file_id}'
    output_file = 'ohlcv_data.csv'

    if not os.path.exists(output_file):
        print(f"Downloading data...")
        try:
            gdown.download(url, output_file, quiet=False)
        except:
            print("Download failed or file exists.")

    print("Loading and resampling data...")
    try:
        df = pd.read_csv(output_file)
        df.columns = [c.lower().strip() for c in df.columns]
        date_col = next((c for c in df.columns if 'date' in c or 'time' in c), None)
        df[date_col] = pd.to_datetime(df[date_col])
        df = df.set_index(date_col).sort_index()
        
        # Resample
        df_resampled = df['close'].resample(RESAMPLE_FREQ).last().dropna().to_frame()
        return df_resampled
    except Exception as e:
        print(f"Error: {e}")
        return None

# --- 2. Ultra-Fast Convolution Slope Calculation ---
def calculate_rolling_slope_fast(series, window_size):
    """
    Calculates rolling slope using convolution (FFT based or sliding window sum).
    Complexity: O(N) instead of O(N*W).
    Massively faster and memory efficient for large windows.
    """
    y = series.values
    x = np.arange(window_size)
    
    # Constants for the window
    n = window_size
    sum_x = x.sum()
    sum_x2 = (x ** 2).sum()
    delta = n * sum_x2 - sum_x ** 2
    
    # Rolling Sum of Y (Sigma y)
    # convolve with ones
    sum_y = np.convolve(y, np.ones(n), 'valid')
    
    # Rolling Sum of X*Y (Sigma xy)
    # convolve y with x reversed acts as a sliding dot product
    sum_xy = np.convolve(y, x[::-1], 'valid')
    
    # Slope formula: (N * Sigma(xy) - Sigma(x) * Sigma(y)) / Delta
    numerator = n * sum_xy - sum_x * sum_y
    m = numerator / delta
    
    # Pad the beginning with NaNs to align with original index
    pad = np.full(n - 1, np.nan)
    return np.concatenate([pad, m])

def feature_engineering(df):
    print("Feature Engineering...")
    
    feature_data = pd.DataFrame(index=df.index)
    
    # Using tqdm to show progress for feature creation
    for day in tqdm(LOOKBACK_SLOPES, desc="Calculating Slopes"):
        # Calculate window size based on resample freq
        # If 1h freq: 1 day = 24 rows
        # If 1min freq: 1 day = 1440 rows
        rows_per_day = 24 if RESAMPLE_FREQ == '1h' else 1440
        window_size = day * rows_per_day
        
        if len(df) > window_size:
            feature_data[f'slope_{day}d'] = calculate_rolling_slope_fast(df['close'], window_size)
            
    print("Creating Targets...")
    target_data = pd.DataFrame(index=df.index)
    for h_days in HORIZONS_DAYS:
        rows_per_day = 24 if RESAMPLE_FREQ == '1h' else 1440
        h_steps = h_days * rows_per_day
        
        future_close = df['close'].shift(-h_steps)
        # 1 if Up, 0 if Down
        target_data[f'target_{h_days}d'] = (future_close > df['close']).astype(int)
        
        # Mask end
        target_data.loc[df.index[-h_steps:], f'target_{h_days}d'] = np.nan

    full_data = pd.concat([feature_data, target_data], axis=1).dropna()
    return full_data

# --- 3. Dataset Prep ---
def create_sequences(data, feature_cols, target_cols, seq_len):
    X_vals = data[feature_cols].values
    y_vals = data[target_cols].values
    
    X, y = [], []
    for i in range(len(data) - seq_len):
        X.append(X_vals[i : i+seq_len])
        y.append(y_vals[i+seq_len])
        
    return torch.FloatTensor(np.array(X)), torch.FloatTensor(np.array(y))

# --- 4. Model Definition ---
class TrendLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2):
        super(TrendLSTM, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :] 
        out = self.fc(out)
        return self.sigmoid(out)

# --- 5. Main Execution ---
def main():
    df = get_and_process_data()
    if df is None: return

    data = feature_engineering(df)
    print(f"Dataset Shape after processing: {data.shape}")

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
    
    print("Generating Sequences...")
    X_train, y_train = create_sequences(train_data, feature_cols, target_cols, SEQ_LEN)
    X_val, y_val = create_sequences(val_data, feature_cols, target_cols, SEQ_LEN)
    X_test, y_test = create_sequences(test_data, feature_cols, target_cols, SEQ_LEN)
    
    # Loaders
    train_loader = DataLoader(TensorDataset(X_train, y_train), shuffle=True, batch_size=BATCH_SIZE)
    val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=BATCH_SIZE)
    
    # Model
    model = TrendLSTM(input_dim=len(feature_cols), hidden_dim=64, output_dim=len(target_cols)).to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    train_losses = []
    val_losses = []

    print(f"\n--- Starting Training on {device} ---")
    
    for epoch in range(EPOCHS):
        model.train()
        batch_loss = []
        
        # Progress bar for training steps
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}", unit="batch", leave=False)
        
        for X_batch, y_batch in pbar:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            optimizer.zero_grad()
            y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch)
            loss.backward()
            optimizer.step()
            
            batch_loss.append(loss.item())
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
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
        
        print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {avg_train:.4f} | Val Loss: {avg_val:.4f}")

    # Visualization
    print("\n--- Visualizing Results ---")
    plt.style.use('dark_background')
    plt.figure(figsize=(14, 10))
    
    # Loss
    plt.subplot(2, 1, 1)
    plt.plot(train_losses, label='Train Loss', color='cyan')
    plt.plot(val_losses, label='Val Loss', color='orange')
    plt.title('Training Metrics')
    plt.legend()
    plt.grid(alpha=0.2)
    
    # Preds
    model.eval()
    with torch.no_grad():
        test_preds = model(X_test.to(device)).cpu().numpy()
        y_test_cpu = y_test.numpy()
    
    pred_df = pd.DataFrame(test_preds, columns=target_cols)
    actual_df = pd.DataFrame(y_test_cpu, columns=target_cols)
    
    plt.subplot(2, 1, 2)
    subset = 200
    
    # Plot 1 Day Certainty
    plt.plot(actual_df.iloc[:subset, 0], color='lime', alpha=0.3, linewidth=1, label='Actual 1D')
    plt.plot(pred_df.iloc[:subset, 0], color='lime', linestyle='--', linewidth=1.5, label='Pred 1D Probability')
    
    # Plot 30 Day Certainty
    plt.plot(actual_df.iloc[:subset, -1], color='magenta', alpha=0.3, linewidth=1, label='Actual 30D')
    plt.plot(pred_df.iloc[:subset, -1], color='magenta', linestyle='--', linewidth=1.5, label='Pred 30D Probability')
    
    plt.title(f'Prediction Confidence (First {subset} Test Hours)')
    plt.ylabel('Probability (1=Up, 0=Down)')
    plt.legend()
    plt.grid(alpha=0.2)
    
    plt.tight_layout()
    plt.show()
    
    torch.save(model.state_dict(), 'trend_lstm_model.pth')

if __name__ == '__main__':
    main()
