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

# --- Configuration ---
torch.manual_seed(42)
np.random.seed(42)

HORIZONS_DAYS = [1, 2, 3, 4, 5, 6, 7, 14, 30]  # Target horizons
LOOKBACK_SLOPES = range(1, 31)                 # Feature lookbacks (1d to 30d)
SEQ_LEN = 24                                   # LSTM Sequence length (e.g., 24 hours context)
RESAMPLE_FREQ = '1h'                           # Resample 1-min data to 1-hour
BATCH_SIZE = 64
EPOCHS = 20                                    # Keep low for demo speed, increase for results

# --- 1. Data Loading & Resampling ---
def get_and_process_data():
    file_id = '1kDCl_29nXyW1mLNUAS-nsJe0O2pOuO6o'
    url = f'https://drive.google.com/uc?id={file_id}'
    output_file = 'ohlcv_data.csv'

    if not os.path.exists(output_file):
        print(f"Downloading data...")
        gdown.download(url, output_file, quiet=False)

    print("Loading and resampling data...")
    try:
        df = pd.read_csv(output_file)
        df.columns = [c.lower().strip() for c in df.columns]
        date_col = next((c for c in df.columns if 'date' in c or 'time' in c), None)
        df[date_col] = pd.to_datetime(df[date_col])
        df = df.set_index(date_col).sort_index()
        
        # Resample to Hourly to make LSTM training feasible
        # Using 'last' for close price, 'max' for high, etc.
        df_resampled = df['close'].resample(RESAMPLE_FREQ).last().dropna().to_frame()
        
        return df_resampled
    except Exception as e:
        print(f"Error: {e}")
        return None

# --- 2. Optimized Rolling Slope Calculation ---
def calculate_rolling_slope(series, window_size):
    """
    Calculates rolling slope efficiently using vectorized operations.
    Slope = Cov(x, y) / Var(x)
    Since x is linear (0, 1, 2...), Var(x) is constant for a fixed window.
    """
    y = series.values
    x = np.arange(window_size)
    
    # Pre-calculate x stats
    sx = x.sum()
    sx2 = (x ** 2).sum()
    var_x = sx2 - (sx ** 2) / window_size
    
    # We need rolling sum of y and rolling sum of xy
    # Using pandas rolling is easiest for y sum
    s_y = series.rolling(window_size).sum()
    
    # For xy, we can't use simple rolling because x resets [0..W] at every step.
    # However, for a linear trend, we can use stride_tricks for speed or a loop for simplicity.
    # Given the dataset size (~8000 hours), a simple loop with numpy is fast enough if optimized.
    # Let's use stride_tricks for max speed.
    
    strides = y.strides + y.strides
    shape = (len(y) - window_size + 1, window_size)
    y_strided = np.lib.stride_tricks.as_strided(y, shape=shape, strides=strides)
    
    # x is constant [0, 1, ... W-1] broadcasted
    # Calculate numerator: N * sum(xy) - sum(x)*sum(y)
    # But slope formula: (sum((x-mx)(y-my))) / sum((x-mx)^2)
    # Simplified: (N*sum(xy) - sum(x)sum(y)) / (N*sum(x^2) - (sum(x))^2)
    
    denominator = window_size * sx2 - sx**2
    
    # Vectorized dot product for sum(xy) across all windows
    sum_xy = y_strided.dot(x)
    sum_y = s_y.dropna().values
    
    numerator = (window_size * sum_xy) - (sx * sum_y)
    
    slopes = numerator / denominator
    
    # Pad beginning with NaNs to match index
    pad = np.full(window_size - 1, np.nan)
    return np.concatenate([pad, slopes])

def feature_engineering(df):
    print("Calculating 30 feature slopes (this may take a moment)...")
    
    # 1 day = 24 hours (given 1h resampling)
    feature_data = pd.DataFrame(index=df.index)
    
    for day in LOOKBACK_SLOPES:
        window_hours = day * 24
        # Need at least window_size data points
        if len(df) > window_hours:
            feature_data[f'slope_{day}d'] = calculate_rolling_slope(df['close'], window_hours)
            
    # Targets: Price direction at future horizons
    # 1 if Price(t+h) > Price(t), else 0
    target_data = pd.DataFrame(index=df.index)
    for h_days in HORIZONS_DAYS:
        h_hours = h_days * 24
        # Shift close backwards to compare future to current
        future_close = df['close'].shift(-h_hours)
        target_data[f'target_{h_days}d'] = (future_close > df['close']).astype(int)
        
        # Mask the last h_hours where target is unknown
        target_data.loc[df.index[-h_hours:], f'target_{h_days}d'] = np.nan

    # Combine and drop NaNs
    full_data = pd.concat([feature_data, target_data], axis=1).dropna()
    return full_data

# --- 3. Dataset Prep ---
def create_sequences(data, feature_cols, target_cols, seq_len):
    X_vals = data[feature_cols].values
    y_vals = data[target_cols].values
    
    X, y = [], []
    for i in range(len(data) - seq_len):
        X.append(X_vals[i : i+seq_len])
        y.append(y_vals[i+seq_len]) # Target is the outcome at the END of the sequence step
        
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

# --- 4. Model Definition ---
class TrendLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2):
        super(TrendLSTM, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        # x shape: (batch, seq_len, features)
        out, _ = self.lstm(x)
        # Take last time step
        out = out[:, -1, :] 
        out = self.fc(out)
        return self.sigmoid(out)

# --- 5. Main Execution ---
def main():
    # A. Get Data
    df = get_and_process_data()
    if df is None: return

    # B. Feature Engineering
    data = feature_engineering(df)
    print(f"Processed Dataset Shape: {data.shape}")

    # C. Split 60/20/20
    n = len(data)
    train_end = int(n * 0.6)
    val_end = int(n * 0.8)
    
    feature_cols = [c for c in data.columns if 'slope' in c]
    target_cols = [c for c in data.columns if 'target' in c]
    
    # Scaling Features (Fit on Train only)
    scaler = StandardScaler()
    scaler.fit(data.iloc[:train_end][feature_cols])
    
    data[feature_cols] = scaler.transform(data[feature_cols])
    
    train_data = data.iloc[:train_end]
    val_data = data.iloc[train_end:val_end]
    test_data = data.iloc[val_end:]
    
    # Create Sequences
    X_train, y_train = create_sequences(train_data, feature_cols, target_cols, SEQ_LEN)
    X_val, y_val = create_sequences(val_data, feature_cols, target_cols, SEQ_LEN)
    X_test, y_test = create_sequences(test_data, feature_cols, target_cols, SEQ_LEN)
    
    # Convert to Tensors
    train_loader = DataLoader(TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train)), shuffle=True, batch_size=BATCH_SIZE)
    val_loader = DataLoader(TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val)), batch_size=BATCH_SIZE)
    
    # D. Model Setup
    model = TrendLSTM(input_dim=len(feature_cols), hidden_dim=64, output_dim=len(target_cols))
    criterion = nn.BCELoss() # Binary Cross Entropy for Probability
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    train_losses = []
    val_losses = []

    print("\n--- Starting Training ---")
    for epoch in range(EPOCHS):
        model.train()
        batch_loss = []
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch)
            loss.backward()
            optimizer.step()
            batch_loss.append(loss.item())
        
        # Validation
        model.eval()
        val_batch_loss = []
        with torch.no_grad():
            for X_v, y_v in val_loader:
                y_p = model(X_v)
                loss = criterion(y_p, y_v)
                val_batch_loss.append(loss.item())
                
        avg_train = np.mean(batch_loss)
        avg_val = np.mean(val_batch_loss)
        train_losses.append(avg_train)
        val_losses.append(avg_val)
        
        if (epoch+1) % 2 == 0:
            print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {avg_train:.4f} | Val Loss: {avg_val:.4f}")

    # E. Visualization
    print("\n--- Visualizing Results ---")
    
    # 1. Loss Curve
    plt.figure(figsize=(14, 10))
    plt.subplot(2, 1, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Training History')
    plt.xlabel('Epoch')
    plt.ylabel('BCE Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 2. Test Predictions (Sample)
    # We will predict on Test set and show certainty for 1 Day and 30 Day horizons
    model.eval()
    with torch.no_grad():
        test_preds = model(torch.from_numpy(X_test)).numpy()
    
    # Create DataFrame for easier plotting
    pred_df = pd.DataFrame(test_preds, columns=target_cols)
    actual_df = pd.DataFrame(y_test, columns=target_cols)
    
    # Plotting first 200 test points for 1-Day vs 30-Day prediction
    plt.subplot(2, 1, 2)
    
    # Horizon indices
    h1_idx = 0 # 1 Day
    h30_idx = -1 # 30 Day
    
    subset = 150
    x_axis = range(subset)
    
    # 1 Day Horizon
    plt.plot(x_axis, actual_df.iloc[:subset, h1_idx], 'g-', alpha=0.3, label='Actual 1D Direction')
    plt.plot(x_axis, pred_df.iloc[:subset, h1_idx], 'g--', label='Predicted 1D Probability')
    
    # 30 Day Horizon
    plt.plot(x_axis, actual_df.iloc[:subset, h30_idx], 'b-', alpha=0.3, label='Actual 30D Direction')
    plt.plot(x_axis, pred_df.iloc[:subset, h30_idx], 'b--', label='Predicted 30D Probability')
    
    plt.title(f'Test Set Predictions (First {subset} samples) - Up/Down Certainty')
    plt.ylabel('Probability (0=Down, 1=Up)')
    plt.xlabel('Time Steps (Hours)')
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

    # F. Save Model
    torch.save(model.state_dict(), 'trend_lstm_model.pth')
    print("Model saved to trend_lstm_model.pth")

if __name__ == '__main__':
    main()