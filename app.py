import time
import sys
import os
import gdown
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import http.server
import socketserver
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import RobustScaler
from sklearn.utils.class_weight import compute_class_weight

# ==========================================
# 1. OPTIMIZED CONFIG
# ==========================================
torch.set_num_threads(12)         
DEVICE = torch.device('cpu')      

FILE_ID = '1zmPWQo5MAxgyDyvaFTpf_NiqN2o6lswa'
DOWNLOAD_OUTPUT = 'market_data.csv'
SEQ_LENGTH = 20
TRAIN_SPLIT = 0.8     

# --- MODEL PARAMETERS ---
INPUT_DIM = 2          # Feature 1: Log Returns, Feature 2: Z-Scored Month
HIDDEN_DIM = 256
NUM_LAYERS = 3
DROPOUT = 0.2
NUM_CLASSES = 3       

# --- TRAINING PARAMETERS ---
BATCH_SIZE = 4096      
EPOCHS = 300         
MAX_LR = 1e-2          
WEIGHT_DECAY = 1e-4
MODEL_FILENAME = 'gru_full_dataset.pth'

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

# ==========================================
# 2. DATA PIPELINE (Full Dataset)
# ==========================================
def get_focused_data():
    if not os.path.exists(DOWNLOAD_OUTPUT):
        log(f"Downloading data from ID: {FILE_ID}...")
        try:
            url = f'https://drive.google.com/uc?id={FILE_ID}'
            gdown.download(url, DOWNLOAD_OUTPUT, quiet=False, fuzzy=True)
        except Exception as e:
            log(f"Download Error: {e}")
            sys.exit(1)

    # Read CSV
    df = pd.read_csv(DOWNLOAD_OUTPUT)
    
    # Normalize column names to lowercase to be safe
    df.columns = df.columns.str.strip().str.lower()
    log(f"Raw data loaded. Shape: {df.shape}. Columns: {df.columns.tolist()}")

    # --- Feature Engineering ---
    
    # 1. Determine Price Column for Log Returns
    # User specified 'value', but we check for 'close' as fallback
    if 'value' in df.columns:
        price_col = 'value'
    elif 'close' in df.columns:
        price_col = 'close'
    else:
        log("CRITICAL ERROR: Could not find 'value' or 'close' column for price data.")
        sys.exit(1)

    log(f"Using column '{price_col}' for return calculations.")
    
    # Calculate Log Returns
    # log(price_t / price_t-1)
    df['log_ret'] = np.log(df[price_col] / df[price_col].shift(1))
    
    # 2. Month Processing
    if 'month' not in df.columns:
        # Fallback if 'month' is missing but 'datetime' exists
        if 'datetime' in df.columns:
            df['month'] = pd.to_datetime(df['datetime']).dt.month
        else:
            log("Warning: No 'month' column found. Defaulting to 0.")
            df['month'] = 0

    # Z-Score Normalization for Month
    month_mean = df['month'].mean()
    month_std = df['month'].std()
    
    if month_std == 0: 
        month_std = 1.0
        
    df['month_norm'] = (df['month'] - month_mean) / month_std
    log(f"Month Stats: Mean={month_mean:.2f}, Std={month_std:.2f}")

    # 3. Label Map (Signal)
    if 'signal' not in df.columns:
        log("CRITICAL ERROR: No 'signal' column found.")
        sys.exit(1)

    # Map signals: -1 -> 0, 0 -> 1, 1 -> 2
    label_map = {-1: 0, 0: 1, 1: 2}
    df['target_class'] = df['signal'].map(label_map)
    
    # Drop NaNs created by shift() or bad data
    df.dropna(subset=['log_ret', 'target_class', 'month_norm'], inplace=True)
    df = df[~df.isin([np.nan, np.inf, -np.inf]).any(axis=1)]
    
    # --- Feature Assembly ---
    # Feature 1: Robust Scaled Log Returns
    log_ret_vals = df['log_ret'].values.reshape(-1, 1)
    scaler = RobustScaler()
    log_ret_scaled = scaler.fit_transform(log_ret_vals)
    
    # Feature 2: Z-Scored Month
    month_scaled = df['month_norm'].values.reshape(-1, 1)
    
    # Stack features: Shape (N, 2)
    data_val = np.hstack([log_ret_scaled, month_scaled])
    labels_val = df['target_class'].values.astype(int)
    
    log(f"Dataset Size after filtering: {len(df)} rows.")
    log(f"Features shape: {data_val.shape}")
    
    # Generate Sequences
    log(f"Generating Sequences (Length {SEQ_LENGTH})...")
    from numpy.lib.stride_tricks import sliding_window_view
    
    # sliding_window_view creates shape (Num_Windows, Window_Size, Features)
    windows = sliding_window_view(data_val, window_shape=SEQ_LENGTH, axis=0) 
    
    # Ensure dimensions are (Batch, Seq_Len, Features)
    # The default behavior of sliding_window_view on 2D array:
    # Input (N, F) -> Output (N - W + 1, W, F) is automatic if we don't mess up axes.
    # Actually, for 2D input (N, F), sliding_window_view(x, w, axis=0) results in (N-w+1, F, w).
    # We need (N-w+1, w, F).
    
    windows = sliding_window_view(data_val, window_shape=SEQ_LENGTH, axis=0)
    # Current shape: (N_samples, Features, Seq_Len) -> e.g. (X, 2, 20)
    # We want: (N_samples, Seq_Len, Features) -> (X, 20, 2)
    X = windows.transpose(0, 2, 1)
    
    # Align labels (taking the label at the end of the sequence)
    y = labels_val[SEQ_LENGTH-1:]
    
    min_len = min(len(X), len(y))
    return X[:min_len], y[:min_len]

# ==========================================
# 3. MODEL (GRU)
# ==========================================
class GRUClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.gru = nn.GRU(INPUT_DIM, HIDDEN_DIM, NUM_LAYERS, 
                          batch_first=True, dropout=DROPOUT)
        self.bn = nn.BatchNorm1d(HIDDEN_DIM)
        self.dropout = nn.Dropout(DROPOUT)
        self.fc = nn.Linear(HIDDEN_DIM, NUM_CLASSES)
        
    def forward(self, x):
        # Input shape: (Batch, Seq, Feature)
        out, _ = self.gru(x)
        # GRU returns (output, hidden). Output is (Batch, Seq, Hidden)
        # We take the last time step
        out = out[:, -1, :] 
        out = self.bn(out)
        out = torch.relu(out)
        out = self.dropout(out)
        out = self.fc(out)
        return out

# ==========================================
# 4. MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    log("==========================================")
    log(f" Starting Full Dataset GRU Training")
    log(f" Features: LogRet(from 'value') + Month(Z-Scored)")
    log("==========================================")
    
    X, y = get_focused_data()
    
    # Train/Test Split
    split_idx = int(len(X) * TRAIN_SPLIT)
    
    # Convert to Tensor
    X_train = torch.tensor(X[:split_idx], dtype=torch.float32)
    y_train = torch.tensor(y[:split_idx], dtype=torch.long)
    X_test = torch.tensor(X[split_idx:], dtype=torch.float32)
    y_test = torch.tensor(y[split_idx:], dtype=torch.long)

    # Class Weights for Imbalance handling
    unique_classes = np.unique(y[:split_idx])
    class_weights = compute_class_weight('balanced', classes=unique_classes, y=y[:split_idx])
    weights_tensor = torch.zeros(NUM_CLASSES).to(DEVICE)
    for i, cls in enumerate(unique_classes):
        weights_tensor[cls] = class_weights[i]
        
    log(f"Class Weights: {weights_tensor}")

    # Dataloaders
    train_ds = TensorDataset(X_train, y_train)
    test_ds = TensorDataset(X_test, y_test)
    
    # Check if we have enough data for the batch size
    actual_batch_size = min(BATCH_SIZE, len(train_ds))
    
    train_loader = DataLoader(train_ds, batch_size=actual_batch_size, shuffle=True, 
                              num_workers=0) # num_workers=0 is safer for simple scripts/CPU
    test_loader = DataLoader(test_ds, batch_size=actual_batch_size, shuffle=False)
    
    # Instantiate GRU Model
    model = GRUClassifier().to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=MAX_LR, weight_decay=WEIGHT_DECAY)
    criterion = nn.CrossEntropyLoss(weight=weights_tensor)

    # OneCycleLR Scheduler
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, 
        max_lr=MAX_LR,
        steps_per_epoch=len(train_loader),
        epochs=EPOCHS,
        pct_start=0.3
    )
    
    log(f"Beginning training on {len(X_train)} samples for {EPOCHS} epochs.")
    
    best_acc = 0
    for epoch in range(EPOCHS):
        start_time = time.time()
        model.train()
        train_loss = 0
        
        for bx, by in train_loader:
            bx, by = bx.to(DEVICE), by.to(DEVICE)
            optimizer.zero_grad()
            out = model(bx)
            loss = criterion(out, by)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            train_loss += loss.item()

        # Evaluation
        model.eval()
        val_loss = 0
        val_correct = 0
        with torch.no_grad():
            for bx, by in test_loader:
                bx, by = bx.to(DEVICE), by.to(DEVICE)
                out = model(bx)
                val_loss += criterion(out, by).item()
                val_correct += (out.argmax(1) == by).sum().item()

        time_taken = time.time() - start_time
        val_acc = 100 * val_correct / len(test_ds) if len(test_ds) > 0 else 0
        
        # Periodic Logging
        if (epoch + 1) % 5 == 0 or epoch == 0:
            avg_train_loss = train_loss / len(train_loader)
            avg_val_loss = val_loss / len(test_loader)
            print(f"Ep {epoch+1:03d} | T: {time_taken:.2f}s | "
                  f"TLoss: {avg_train_loss:.4f} | "
                  f"VLoss: {avg_val_loss:.4f} | VAcc: {val_acc:.2f}%")
        
        # Save Best Model
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), MODEL_FILENAME)

    log(f"Training Complete. Best Val Accuracy: {best_acc:.2f}%")
    
    # Server logic for deployment environment (Keep-Alive)
    PORT = int(os.environ.get("PORT", 8080))
    log(f"Serving on port {PORT}...")
    with socketserver.TCPServer(("0.0.0.0", PORT), http.server.SimpleHTTPRequestHandler) as httpd:
        httpd.serve_forever()