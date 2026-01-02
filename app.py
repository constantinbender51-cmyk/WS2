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
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight

# ==========================================
# 1. HYPERPARAMETERS & CONFIG
# ==========================================
# --- HARDWARE ---
torch.set_num_threads(24)         
LOADER_WORKERS = 8                
DEVICE = torch.device('cpu')      

# --- DATASET PARAMETERS ---
FILE_ID = '1_2IDMRsQCalNn-SIT7nWvqbRMfI1ZFBb'
DOWNLOAD_OUTPUT = 'market_data.csv'
MAX_ROWS = None       
SEQ_LENGTH = 30
TRAIN_SPLIT = 0.8     

# --- MODEL PARAMETERS ---
INPUT_DIM = 1         
HIDDEN_DIM = 512
NUM_LAYERS = 1
DROPOUT = 0.4           # High dropout for regularization
NUM_CLASSES = 3       

# --- TRAINING PARAMETERS ---
BATCH_SIZE = 256      
EPOCHS = 500
LEARNING_RATE = 0.001
WEIGHT_DECAY = 1e-3     # L2 Regularization
PATIENCE = 7            # Early stopping patience
MODEL_FILENAME = 'lstm_regularized.pth'

def log(msg):
    """Timestamped log with mandatory flush."""
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

# ==========================================
# 2. DATA PIPELINE
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

    log("Reading CSV...")
    df = pd.read_csv(DOWNLOAD_OUTPUT)
    
    if MAX_ROWS is not None:
        df = df.tail(MAX_ROWS).copy()
    else:
        log(f"Using Full Dataset: {len(df)} rows.")

    log("Calculating Log Returns...")
    # Add small epsilon to avoid log(0) if exists, though unlikely in price
    df['log_ret'] = np.log(df['close'] / df['close'].shift(1))
    
    label_map = {-1: 0, 0: 1, 1: 2}
    df['target_class'] = df['signal'].map(label_map)
    
    df.dropna(subset=['log_ret', 'target_class'], inplace=True)
    df = df[~df.isin([np.nan, np.inf, -np.inf]).any(axis=1)]
    df['target_class'] = df['target_class'].astype(int)
    
    data_val = df['log_ret'].values.reshape(-1, 1)
    labels_val = df['target_class'].values
    
    log("Normalizing Features...")
    scaler = StandardScaler()
    data_val = scaler.fit_transform(data_val)
    
    log(f"Generating Sequences (Length {SEQ_LENGTH})...")
    from numpy.lib.stride_tricks import sliding_window_view
    
    flat_data = data_val.flatten()
    windows = sliding_window_view(flat_data, window_shape=SEQ_LENGTH)
    X = windows.reshape(*windows.shape, 1)
    y = labels_val[SEQ_LENGTH-1:]
    
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
        # LSTM Layer
        self.lstm = nn.LSTM(INPUT_DIM, HIDDEN_DIM, NUM_LAYERS, 
                            batch_first=True, dropout=DROPOUT)
        
        # Batch Normalization: Stabilizes inputs to the linear layer
        self.bn = nn.BatchNorm1d(HIDDEN_DIM)
        
        # Dropout: Randomly zero out neurons to prevent co-adaptation
        self.dropout = nn.Dropout(DROPOUT)
        
        # Final Classification Layer
        self.fc = nn.Linear(HIDDEN_DIM, NUM_CLASSES)
        
    def forward(self, x):
        # LSTM output: (batch, seq, hidden)
        out, _ = self.lstm(x)
        
        # Take the last time step
        out = out[:, -1, :]
        
        # Apply Batch Norm -> Activation -> Dropout -> FC
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
    log(f" Starting Regularized Training")
    log(f" vCPUs Available: {os.cpu_count()}")
    log("==========================================")
    
    X, y = get_data()
    
    # --- SPLIT ---
    split_idx = int(len(X) * TRAIN_SPLIT)
    
    X_train_np = X[:split_idx]
    y_train_np = y[:split_idx]
    X_test_np = X[split_idx:]
    y_test_np = y[split_idx:]

    # --- CALCULATE CLASS WEIGHTS ---
    class_weights = compute_class_weight(
        class_weight='balanced', 
        classes=np.unique(y_train_np), 
        y=y_train_np
    )
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float).to(DEVICE)
    log(f"Class Weights calculated: {class_weights}")

    # --- TENSORS ---
    X_train = torch.from_numpy(X_train_np).float()
    y_train = torch.from_numpy(y_train_np).long()
    X_test = torch.from_numpy(X_test_np).float()
    y_test = torch.from_numpy(y_test_np).long()
    
    train_ds = TensorDataset(X_train, y_train)
    test_ds = TensorDataset(X_test, y_test)
    
    # Pin memory speeds up host-to-device transfer
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, 
                              num_workers=LOADER_WORKERS, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, 
                             num_workers=LOADER_WORKERS, pin_memory=True)
    
    model = LSTMClassifier().to(DEVICE)
    
    # Use AdamW (Decoupled Weight Decay) for better regularization
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    
    # Scheduler: Reduce LR if validation loss plateaus
    # REMOVED verbose=True to fix TypeError
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3
    )
    
    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
    
    # --- TRAINING LOOP WITH EARLY STOPPING ---
    best_val_loss = float('inf')
    patience_counter = 0
    
    log("Model Initialized. Starting Training Loop...")
    
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
            
            # Gradient Clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            train_loss += loss.item()
            _, pred = torch.max(out, 1)
            train_correct += (pred == by).sum().item()
            train_total += by.size(0)

        # --- VALIDATE ---
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for bx, by in test_loader:
                bx, by = bx.to(DEVICE), by.to(DEVICE)
                out = model(bx)
                loss = criterion(out, by)
                val_loss += loss.item()
                
                _, pred = torch.max(out, 1)
                val_correct += (pred == by).sum().item()
                val_total += by.size(0)

        # --- METRICS ---
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(test_loader)
        train_acc = 100 * train_correct / train_total
        val_acc = 100 * val_correct / val_total
        time_taken = time.time() - start_time
        
        # Get current Learning Rate manually
        current_lr = optimizer.param_groups[0]['lr']
        
        # --- LOGGING ---
        print(f"Epoch {epoch+1:02d} | T: {time_taken:.1f}s | LR: {current_lr:.6f}")
        print(f"  > TrLoss: {avg_train_loss:.4f} TrAcc: {train_acc:.2f}%")
        print(f"  > ValLoss: {avg_val_loss:.4f} ValAcc: {val_acc:.2f}%")
        
        # --- SCHEDULER STEP ---
        scheduler.step(avg_val_loss)
        
        # --- EARLY STOPPING CHECK ---
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), MODEL_FILENAME)
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                log(f"Early Stopping triggered at Epoch {epoch+1}")
                break

    log(f"Training Complete. Best Validation Loss: {best_val_loss:.4f}")

    # ==========================================
    # 5. WEB SERVER FOR DOWNLOAD
    # ==========================================
    PORT = int(os.environ.get("PORT", 8080))
    Handler = http.server.SimpleHTTPRequestHandler
    
    log("="*40)
    log(f"STARTING DOWNLOAD SERVER")
    log(f"Serving file: {MODEL_FILENAME}")
    log(f"URL: https://your-app.up.railway.app/{MODEL_FILENAME}")
    log("="*40)
    
    socketserver.TCPServer.allow_reuse_address = True
    with socketserver.TCPServer(("0.0.0.0", PORT), Handler) as httpd:
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            log("Server stopped.")