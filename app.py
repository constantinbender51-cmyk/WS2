import time
import sys
import os
import gdown
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler

# ==========================================
# 1. HYPERPARAMETERS & CONFIG
# ==========================================
# --- HARDWARE ---
# With 32 vCPUs, we want to use them!
# We leave some overhead for the OS/Container orchestration.
torch.set_num_threads(24)         # Use 24 cores for Matrix Math
LOADER_WORKERS = 8                # Use 8 processes for fetching data
DEVICE = torch.device('cpu')      # Keep CPU to avoid CUDA driver timeouts in Railway

# --- DATASET PARAMETERS ---
FILE_ID = '1RreTtTTGZCRcLqk6Ixl85sJAu8izj8DW'
DOWNLOAD_OUTPUT = 'market_data.csv'
MAX_ROWS = None       # Set to an integer (e.g., 50000) to limit data for fast testing. None = Full Dataset.
SEQ_LENGTH = 60       # Lookback window
TRAIN_SPLIT = 0.8     # 80% Training, 20% Validation

# --- MODEL PARAMETERS ---
INPUT_DIM = 1         # Log Returns
HIDDEN_DIM = 128      # Larger hidden state since we have compute power
NUM_LAYERS = 2        # Stacked LSTM
DROPOUT = 0.2
NUM_CLASSES = 3       # -1, 0, 1

# --- TRAINING PARAMETERS ---
BATCH_SIZE = 256      # Increased batch size for parallel throughput
EPOCHS = 10
LEARNING_RATE = 0.001

def log(msg):
    """Timestamped log with mandatory flush and small delay."""
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)
    time.sleep(0.1)

# ==========================================
# 2. DATA PIPELINE
# ==========================================
def get_data():
    # 1. Download
    if not os.path.exists(DOWNLOAD_OUTPUT):
        log(f"Downloading data from ID: {FILE_ID}...")
        try:
            url = f'https://drive.google.com/uc?id={FILE_ID}'
            gdown.download(url, DOWNLOAD_OUTPUT, quiet=False, fuzzy=True)
        except Exception as e:
            log(f"Download Error: {e}")
            sys.exit(1)

    # 2. Load
    log("Reading CSV...")
    df = pd.read_csv(DOWNLOAD_OUTPUT)
    
    # Optional Truncation
    if MAX_ROWS is not None:
        log(f"Truncating dataset to last {MAX_ROWS} rows (User Config).")
        df = df.tail(MAX_ROWS).copy()
    else:
        log(f"Using Full Dataset: {len(df)} rows.")

    # 3. Preprocess
    log("Calculating Log Returns...")
    # ln(Close_t / Close_{t-1})
    df['log_ret'] = np.log(df['close'] / df['close'].shift(1))
    
    # Map Signals
    label_map = {-1: 0, 0: 1, 1: 2}
    df['target_class'] = df['signal'].map(label_map)
    
    # Clean
    df.dropna(subset=['log_ret', 'target_class'], inplace=True)
    df = df[~df.isin([np.nan, np.inf, -np.inf]).any(axis=1)]
    df['target_class'] = df['target_class'].astype(int)
    
    # Extract
    data_val = df['log_ret'].values.reshape(-1, 1)
    labels_val = df['target_class'].values
    
    # Normalize
    log("Normalizing Features...")
    scaler = StandardScaler()
    data_val = scaler.fit_transform(data_val)
    
    # Sequence Creation (Vectorized for speed)
    log(f"Generating Sequences (Length {SEQ_LENGTH})...")
    from numpy.lib.stride_tricks import sliding_window_view
    
    flat_data = data_val.flatten()
    # Create windows: (N_windows, SEQ_LENGTH)
    windows = sliding_window_view(flat_data, window_shape=SEQ_LENGTH)
    X = windows.reshape(*windows.shape, 1)
    
    # Align labels (Predict label at end of sequence)
    y = labels_val[SEQ_LENGTH-1:]
    
    # Trim to match
    min_len = min(len(X), len(y))
    X = X[:min_len]
    y = y[:min_len]
    
    log(f"Final Data Shapes | X: {X.shape} | y: {y.shape}")
    return X, y

# ==========================================
# 3. MODEL
# ==========================================
class LSTMClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(INPUT_DIM, HIDDEN_DIM, NUM_LAYERS, 
                            batch_first=True, dropout=DROPOUT)
        self.fc = nn.Linear(HIDDEN_DIM, NUM_CLASSES)
        
    def forward(self, x):
        # x: (batch, seq, feature)
        out, _ = self.lstm(x)
        # Take last time step
        out = self.fc(out[:, -1, :])
        return out

# ==========================================
# 4. MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    log("==========================================")
    log(f" Starting High-Performance Training")
    log(f" vCPUs Available: {os.cpu_count()}")
    log(f" Threads Configured: {torch.get_num_threads()}")
    log("==========================================")
    
    # 1. Data
    X, y = get_data()
    
    # 2. Split
    split_idx = int(len(X) * TRAIN_SPLIT)
    
    X_train = torch.from_numpy(X[:split_idx]).float()
    y_train = torch.from_numpy(y[:split_idx]).long()
    
    X_test = torch.from_numpy(X[split_idx:]).float()
    y_test = torch.from_numpy(y[split_idx:]).long()
    
    log(f"Training Samples:   {len(X_train)}")
    log(f"Validation Samples: {len(X_test)}")
    
    train_ds = TensorDataset(X_train, y_train)
    test_ds = TensorDataset(X_test, y_test)
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=LOADER_WORKERS)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=LOADER_WORKERS)
    
    # 3. Model
    model = LSTMClassifier().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()
    
    log("Model Initialized. Starting Epochs...")
    
    # 4. Loop
    for epoch in range(EPOCHS):
        start_time = time.time()
        
        # --- TRAIN ---
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        for i, (bx, by) in enumerate(train_loader):
            bx, by = bx.to(DEVICE), by.to(DEVICE)
            
            optimizer.zero_grad()
            out = model(bx)
            loss = criterion(out, by)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, pred = torch.max(out, 1)
            train_correct += (pred == by).sum().item()
            train_total += by.size(0)
            
            # Log progress every 10% of batches
            if i > 0 and i % (len(train_loader) // 10 + 1) == 0:
                print(f"\r[Ep {epoch+1} Train] {i}/{len(train_loader)} batches...", end="", flush=True)

        # --- VALIDATE ---
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for i, (bx, by) in enumerate(test_loader):
                bx, by = bx.to(DEVICE), by.to(DEVICE)
                out = model(bx)
                loss = criterion(out, by)
                val_loss += loss.item()
                
                _, pred = torch.max(out, 1)
                val_correct += (pred == by).sum().item()
                val_total += by.size(0)
                
                # CRITICAL: Log inside validation loop to prove it's running
                if i > 0 and i % (len(test_loader) // 5 + 1) == 0:
                     print(f"\r[Ep {epoch+1} Val]   {i}/{len(test_loader)} batches...", end="", flush=True)

        # Metrics
        time_taken = time.time() - start_time
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(test_loader)
        train_acc = 100 * train_correct / train_total
        val_acc = 100 * val_correct / val_total
        
        print("") # Clear carriage return line
        log(f"Epoch {epoch+1}/{EPOCHS} | Time: {time_taken:.1f}s")
        log(f"  > Train Loss: {avg_train_loss:.4f} | Acc: {train_acc:.2f}%")
        log(f"  > Val   Loss: {avg_val_loss:.4f} | Acc: {val_acc:.2f}%")

    log("Saving Model...")
    torch.save(model.state_dict(), 'lstm_32core.pth')
    log("Complete.")