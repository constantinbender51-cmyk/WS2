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
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.utils.class_weight import compute_class_weight

# ==========================================
# 1. OPTIMIZED CONFIG
# ==========================================
# --- HARDWARE ---
# LIMIT THREADS: On small models, 8-12 threads often beat 32 due to less overhead.
torch.set_num_threads(12)         
DEVICE = torch.device('cpu')      

# --- DATASET PARAMETERS ---
FILE_ID = '1_2IDMRsQCalNn-SIT7nWvqbRMfI1ZFBb'
DOWNLOAD_OUTPUT = 'market_data.csv'
MAX_ROWS = None       
# INCREASED: Give the LSTM enough history to see a pattern
SEQ_LENGTH = 30     
TRAIN_SPLIT = 0.8     

# --- MODEL PARAMETERS ---
INPUT_DIM = 1         
# INCREASED: 12 is too small to capture market complexity
HIDDEN_DIM = 128   
NUM_LAYERS = 2
DROPOUT = 0.2         
NUM_CLASSES = 3       

# --- TRAINING PARAMETERS ---
# INCREASED: Massive batch size to saturate 32 vCPUs
BATCH_SIZE = 4096      
EPOCHS = 100            # Reduced epochs, we aim for faster convergence
MAX_LR = 0.01           # Higher max LR for OneCycle
WEIGHT_DECAY = 1e-1
MODEL_FILENAME = 'lstm_optimized.pth'

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

# ==========================================
# 2. DATA PIPELINE (With Robust Scaling)
# ==========================================
def get_data():
    if not os.path.exists(DOWNLOAD_OUTPUT):
        log(f"Downloading data from ID: {FILE_ID}...")
        try:
            url = f'https://drive.google.com/uc?id={FILE_ID}'
            gdown.download(url, DOWNLOAD_OUTPUT, quiet=False, fuzzy=True)
        except Exception as e:
            log(f"Download Error: {e}")
            sys.exit(1)

    df = pd.read_csv(DOWNLOAD_OUTPUT)
    
    if MAX_ROWS: df = df.tail(MAX_ROWS).copy()

    # Log returns
    df['log_ret'] = np.log(df['close'] / df['close'].shift(1))
    
    # Label Map
    label_map = {-1: 0, 0: 1, 1: 2}
    df['target_class'] = df['signal'].map(label_map)
    
    # Drop NaNs
    df.dropna(subset=['log_ret', 'target_class'], inplace=True)
    df = df[~df.isin([np.nan, np.inf, -np.inf]).any(axis=1)]
    
    data_val = df['log_ret'].values.reshape(-1, 1)
    labels_val = df['target_class'].values.astype(int)
    
    log("Normalizing Features (RobustScaler)...")
    # CHANGED: RobustScaler handles financial "fat tails" (outliers) better than Standard
    scaler = RobustScaler()
    data_val = scaler.fit_transform(data_val)
    
    log(f"Generating Sequences (Length {SEQ_LENGTH})...")
    from numpy.lib.stride_tricks import sliding_window_view
    
    # Efficient windowing
    windows = sliding_window_view(data_val.flatten(), window_shape=SEQ_LENGTH)
    X = windows.reshape(*windows.shape, 1)
    # Align labels: The label for a window is the label of the *last* timestamp in that window
    y = labels_val[SEQ_LENGTH-1:]
    
    min_len = min(len(X), len(y))
    return X[:min_len], y[:min_len]

# ==========================================
# 3. MODEL
# ==========================================
class LSTMClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(INPUT_DIM, HIDDEN_DIM, NUM_LAYERS, 
                            batch_first=True) # Dropout removed from LSTM if num_layers=1
        self.bn = nn.BatchNorm1d(HIDDEN_DIM)
        self.dropout = nn.Dropout(DROPOUT)
        self.fc = nn.Linear(HIDDEN_DIM, NUM_CLASSES)
        
    def forward(self, x):
        # x shape: (Batch, Seq, Feature)
        out, _ = self.lstm(x)
        
        # Take the last time step output
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
    log(f" Starting Optimized Training (OneCycle)")
    log("==========================================")
    
    X, y = get_data()
    
    # Split
    split_idx = int(len(X) * TRAIN_SPLIT)
    X_train, y_train = torch.tensor(X[:split_idx], dtype=torch.float32), torch.tensor(y[:split_idx], dtype=torch.long)
    X_test, y_test = torch.tensor(X[split_idx:], dtype=torch.float32), torch.tensor(y[split_idx:], dtype=torch.long)

    # Weights
    class_weights = compute_class_weight('balanced', classes=np.unique(y[:split_idx]), y=y[:split_idx])
    weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(DEVICE)

    # Dataloaders - Increased workers, Pin Memory
    # CHANGED: persistent_workers=True keeps threads alive between epochs
    train_ds = TensorDataset(X_train, y_train)
    test_ds = TensorDataset(X_test, y_test)
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, 
                              num_workers=8, pin_memory=True, persistent_workers=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, 
                             num_workers=4, pin_memory=True)
    
    model = LSTMClassifier().to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=MAX_LR, weight_decay=WEIGHT_DECAY)
    criterion = nn.CrossEntropyLoss(weight=weights_tensor)

    # CHANGED: OneCycleLR for Super-Convergence
    # This warms up LR to MAX_LR then cools down, significantly speeding up early training
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, 
        max_lr=MAX_LR,
        steps_per_epoch=len(train_loader),
        epochs=EPOCHS,
        pct_start=0.3  # Spend 30% of time warming up
    )
    
    log("Model Initialized. Starting Training...")
    
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
            
            # Clip gradients prevents "exploding" gradients in early LSTM training
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            optimizer.step()
            scheduler.step() # Step per batch for OneCycle
            train_loss += loss.item()

        # Validation (Every epoch)
        model.eval()
        val_loss = 0
        val_correct = 0
        with torch.no_grad():
            for bx, by in test_loader:
                bx, by = bx.to(DEVICE), by.to(DEVICE)
                out = model(bx)
                val_loss += criterion(out, by).item()
                val_correct += (out.argmax(1) == by).sum().item()

        # Metrics
        time_taken = time.time() - start_time
        curr_lr = optimizer.param_groups[0]['lr']
        val_acc = 100 * val_correct / len(test_ds)
        
        print(f"Ep {epoch+1:03d} | T: {time_taken:.2f}s | LR: {curr_lr:.5f} | "
              f"TLoss: {train_loss/len(train_loader):.4f} | "
              f"VLoss: {val_loss/len(test_loader):.4f} | VAcc: {val_acc:.2f}%")

    # Save
    torch.save(model.state_dict(), MODEL_FILENAME)
    
    # Server (Unchanged)
    PORT = int(os.environ.get("PORT", 8080))
    log(f"Serving on port {PORT}...")
    with socketserver.TCPServer(("0.0.0.0", PORT), http.server.SimpleHTTPRequestHandler) as httpd:
        httpd.serve_forever()
