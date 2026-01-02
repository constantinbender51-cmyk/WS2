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

FILE_ID = '12Q2CI1Jbv3Sr-8S0pCnbNQ5EnhxpKFnk'
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
WEIGHT_DECAY = 1e-0
MODEL_FILENAME = 'gru_full_dataset.pth' # Updated filename to reflect full data usage

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

    df = pd.read_csv(DOWNLOAD_OUTPUT)
    log(f"Raw data loaded. Shape: {df.shape}")

    # --- LOGIC UPDATE: Use Full Dataset ---
    # We bypass the "Last 100 Realities" filter to ensure we get all data.
    # The previous logic was slicing the data too aggressively.
    
    # 1. Log returns
    df['log_ret'] = np.log(df['close'] / df['close'].shift(1))
    
    # 2. Month Z-Score Normalization
    if 'month' not in df.columns:
        if 'datetime' in df.columns:
            # log("Attempting to extract month from 'datetime' column...")
            df['month'] = pd.to_datetime(df['datetime']).dt.month
        else:
            log("Warning: No 'month' or 'datetime' column. Defaulting month to 0.")
            df['month'] = 0

    month_mean = df['month'].mean()
    month_std = df['month'].std()
    
    if month_std == 0: 
        month_std = 1.0
        
    df['month_norm'] = (df['month'] - month_mean) / month_std
    log(f"Month Stats: Mean={month_mean:.2f}, Std={month_std:.2f}")

    # Label Map
    label_map = {-1: 0, 0: 1, 1: 2}
    df['target_class'] = df['signal'].map(label_map)
    
    # Drop NaNs and Infs
    df.dropna(subset=['log_ret', 'target_class', 'month_norm'], inplace=True)
    df = df[~df.isin([np.nan, np.inf, -np.inf]).any(axis=1)]
    
    # --- Feature Assembly ---
    log_ret_vals = df['log_ret'].values.reshape(-1, 1)
    scaler = RobustScaler()
    log_ret_scaled = scaler.fit_transform(log_ret_vals)
    
    month_scaled = df['month_norm'].values.reshape(-1, 1)
    
    # Stack features: Shape (N, 2)
    data_val = np.hstack([log_ret_scaled, month_scaled])
    labels_val = df['target_class'].values.astype(int)
    
    log(f"Dataset Size after filtering: {len(df)} rows.")
    log(f"Features shape: {data_val.shape}")
    
    log(f"Generating Sequences (Length {SEQ_LENGTH})...")
    from numpy.lib.stride_tricks import sliding_window_view
    
    windows = sliding_window_view(data_val, window_shape=SEQ_LENGTH, axis=0) 
    X = windows.transpose(0, 2, 1)
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
        # GRU returns (output, hidden), we use last timestep of output
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
    log(f" Features: LogRet + Month(Z-Scored)")
    log("==========================================")
    
    X, y = get_focused_data()
    
    # Train/Test Split
    split_idx = int(len(X) * TRAIN_SPLIT)
    X_train, y_train = torch.tensor(X[:split_idx], dtype=torch.float32), torch.tensor(y[:split_idx], dtype=torch.long)
    X_test, y_test = torch.tensor(X[split_idx:], dtype=torch.float32), torch.tensor(y[split_idx:], dtype=torch.long)

    # Class Weights for Imbalance
    unique_classes = np.unique(y[:split_idx])
    class_weights = compute_class_weight('balanced', classes=unique_classes, y=y[:split_idx])
    weights_tensor = torch.zeros(NUM_CLASSES).to(DEVICE)
    for i, cls in enumerate(unique_classes):
        weights_tensor[cls] = class_weights[i]

    # Dataloaders
    train_ds = TensorDataset(X_train, y_train)
    test_ds = TensorDataset(X_test, y_test)
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, 
                              num_workers=4, pin_memory=True, persistent_workers=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)
    
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
    
    # Server logic for deployment environment
    PORT = int(os.environ.get("PORT", 8080))
    log(f"Serving on port {PORT}...")
    with socketserver.TCPServer(("0.0.0.0", PORT), http.server.SimpleHTTPRequestHandler) as httpd:
        httpd.serve_forever()