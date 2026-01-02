import time
import os
import gdown
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler

# ==========================================
# 1. HYPERPARAMETERS & CONFIGURATION
# ==========================================
FILE_ID = '1RreTtTTGZCRcLqk6Ixl85sJAu8izj8DW'
DOWNLOAD_OUTPUT = 'market_data.csv'

# Data Params
SEQ_LENGTH = 2       # Number of time steps to look back
TRAIN_SPLIT = 0.1     # 80% training, 20% validation

# Model Params
INPUT_DIM = 1         # Using 1 feature: Log Returns
HIDDEN_DIM = 32       # Neurons in LSTM hidden layer
NUM_LAYERS = 2        # Number of stacked LSTM layers
DROPOUT = 0.2         # Dropout probability
NUM_CLASSES = 3       # Classes: 0, 1, 2 (mapped from -1, 0, 1)

# Training Params
BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 0.001

# System
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def slow_print(msg):
    """Prints a message and pauses for 0.1 seconds."""
    print(msg, flush=True)
    time.sleep(0.1)

# ==========================================
# 2. DATA DOWNLOAD & PREPARATION
# ==========================================
def download_data():
    url = f'https://drive.google.com/uc?id={FILE_ID}'
    if not os.path.exists(DOWNLOAD_OUTPUT):
        slow_print(f"Downloading file from Google Drive to {DOWNLOAD_OUTPUT}...")
        gdown.download(url, DOWNLOAD_OUTPUT, quiet=False)
        slow_print("Download complete.")
    else:
        slow_print("File already exists. Skipping download.")

def prepare_data():
    slow_print("Loading CSV...")
    df = pd.read_csv(DOWNLOAD_OUTPUT)
    
    # Ensure columns exist
    if 'close' not in df.columns or 'signal' not in df.columns:
        raise ValueError("CSV must contain 'close' and 'signal' columns")

    slow_print("Calculating Log Returns...")
    # Log Returns = ln(Close_t / Close_{t-1})
    df['log_ret'] = np.log(df['close'] / df['close'].shift(1))
    
    # Drop the first row which will be NaN due to shift
    df = df.dropna()
    
    # Map signals (-1, 0, 1) to (0, 1, 2) for CrossEntropyLoss
    # Mapping: -1 -> 0, 0 -> 1, 1 -> 2
    label_map = {-1: 0, 0: 1, 1: 2}
    df['target_class'] = df['signal'].map(label_map)
    
    # Filter out any rows where signal might not be -1, 0, or 1 (just in case)
    df = df.dropna(subset=['target_class'])
    df['target_class'] = df['target_class'].astype(int)

    features = df['log_ret'].values.reshape(-1, 1)
    labels = df['target_class'].values

    # Normalize features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    slow_print(f"Data processed. Total rows: {len(df)}")
    return features_scaled, labels

def create_sequences(data, labels, seq_length):
    xs, ys = [], []
    for i in range(len(data) - seq_length):
        x = data[i:(i + seq_length)]
        y = labels[i + seq_length] # Predict the label immediately following the sequence
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

# ==========================================
# 3. MODEL DEFINITION
# ==========================================
class LSTMClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, num_classes, dropout):
        super(LSTMClassifier, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, 
                            batch_first=True, dropout=dropout)
        
        self.fc = nn.Linear(hidden_dim, num_classes)
        
    def forward(self, x):
        # Initialize hidden state and cell state with zeros
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(DEVICE)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(DEVICE)
        
        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))
        
        # Decode the hidden state of the last time step
        out = self.fc(out[:, -1, :])
        return out

# ==========================================
# 4. MAIN EXECUTION
# ==========================================
def main():
    slow_print("Starting script...")
    slow_print(f"Using device: {DEVICE}")

    # 1. Download
    download_data()

    # 2. Preprocess
    features, labels = prepare_data()
    
    slow_print(f"Creating sequences with length {SEQ_LENGTH}...")
    X, y = create_sequences(features, labels, SEQ_LENGTH)
    
    # Convert to PyTorch tensors
    X_tensor = torch.from_numpy(X).float()
    y_tensor = torch.from_numpy(y).long() # Long required for CrossEntropy classes

    # Split Train/Test
    dataset_size = len(X_tensor)
    train_size = int(dataset_size * TRAIN_SPLIT)
    test_size = dataset_size - train_size

    slow_print(f"Dataset size: {dataset_size}")
    slow_print(f"Training set: {train_size}, Validation set: {test_size}")

    train_dataset = TensorDataset(X_tensor[:train_size], y_tensor[:train_size])
    test_dataset = TensorDataset(X_tensor[train_size:], y_tensor[train_size:])

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=True)

    # 3. Initialize Model
    model = LSTMClassifier(INPUT_DIM, HIDDEN_DIM, NUM_LAYERS, NUM_CLASSES, DROPOUT).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    slow_print("Model initialized. Starting training...")

    # 4. Training Loop
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        correct_train = 0
        total_train = 0
        
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(DEVICE), batch_y.to(DEVICE)
            
            # Forward pass
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            # Calculate training accuracy for this batch
            _, predicted = torch.max(outputs.data, 1)
            total_train += batch_y.size(0)
            correct_train += (predicted == batch_y).sum().item()
        
        avg_loss = total_loss / len(train_loader)
        train_acc = 100 * correct_train / total_train

        # Validation phase
        model.eval()
        correct_val = 0
        total_val = 0
        with torch.no_grad():
            for batch_X, batch_y in test_loader:
                batch_X, batch_y = batch_X.to(DEVICE), batch_y.to(DEVICE)
                outputs = model(batch_X)
                _, predicted = torch.max(outputs.data, 1)
                total_val += batch_y.size(0)
                correct_val += (predicted == batch_y).sum().item()
        
        val_acc = 100 * correct_val / total_val if total_val > 0 else 0
        
        slow_print(f"Epoch [{epoch+1}/{EPOCHS}] | Loss: {avg_loss:.4f} | Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}%")

    slow_print("Training complete.")
    slow_print("Saving model to lstm_model.pth...")
    torch.save(model.state_dict(), 'lstm_model.pth')
    slow_print("Script finished successfully.")

if __name__ == "__main__":
    main()