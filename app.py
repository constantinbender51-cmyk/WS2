import os
import io
import time
import threading
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib
import matplotlib.pyplot as plt
from flask import Flask, send_file, Response

# --- Configuration & Setup ---
# Force matplotlib to not use any Xwindow backend (headless mode)
matplotlib.use('Agg') 

app = Flask(__name__)

# Global state to share data between Training Thread and Web Server
class GlobalState:
    def __init__(self):
        self.current_epoch = 0
        self.total_epochs = 0
        self.current_loss = 0.0
        self.latest_plot_buffer = None
        self.training_finished = False
        self.lock = threading.Lock()

state = GlobalState()

# --- Model Definition ---
class SimpleLSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size=32, output_size=1):
        super(SimpleLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # Initialize hidden and cell states
        h0 = torch.zeros(1, x.size(0), self.hidden_size)
        c0 = torch.zeros(1, x.size(0), self.hidden_size)
        
        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))
        
        # Decode the hidden state of the last time step
        out = self.fc(out[:, -1, :])
        return out

# --- Training Logic ---
def train_model():
    print("--- Starting Training Script ---")
    
    # 1. Data Generation (Sine wave + Noise)
    N = 1000
    t = np.linspace(0, 100, N)
    data = np.sin(t) + 0.1 * np.random.normal(size=N)
    
    # Prepare windowed data for LSTM
    window_size = 20
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i:i+window_size])
        y.append(data[i+window_size])
    
    X = torch.tensor(X, dtype=torch.float32).unsqueeze(-1) # (Samples, Window, Features)
    y = torch.tensor(y, dtype=torch.float32).unsqueeze(-1)
    
    # 2. Setup Model
    model = SimpleLSTM()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    
    epochs = 500
    with state.lock:
        state.total_epochs = epochs

    # 3. Training Loop
    for epoch in range(epochs):
        model.train()
        outputs = model(X)
        loss = criterion(outputs, y)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Update State
        with state.lock:
            state.current_epoch = epoch + 1
            state.current_loss = loss.item()

        # Visualization Step (Every 20 epochs to save resources)
        if epoch % 20 == 0 or epoch == epochs - 1:
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")
            generate_plot(model, X, y, t, window_size, loss.item(), epoch)
            
        time.sleep(0.05) # Slight delay just to make the visualization observable in real-time

    with state.lock:
        state.training_finished = True
    print("--- Training Finished ---")

def generate_plot(model, X, y, t, window_size, loss, epoch):
    """Generates a plot and saves it to the global buffer."""
    model.eval()
    with torch.no_grad():
        predictions = model(X).numpy()
    
    fig, ax = plt.subplots(figsize=(10, 5))
    
    # Plot Actual
    ax.plot(t[window_size:], y.numpy(), label='Actual (Noisy Sine)', alpha=0.5)
    
    # Plot Prediction
    ax.plot(t[window_size:], predictions, label='LSTM Prediction', color='red', linewidth=2)
    
    ax.set_title(f'LSTM Training Progress - Epoch {epoch} - Loss: {loss:.4f}')
    ax.legend()
    ax.grid(True)
    
    # Save to memory buffer instead of disk
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.close(fig) # Close to free memory
    
    with state.lock:
        state.latest_plot_buffer = buf

# --- Web Server Logic ---
@app.route('/')
def dashboard():
    """Simple HTML Dashboard that auto-refreshes the image."""
    
    with state.lock:
        status = "Finished" if state.training_finished else "Training..."
        epoch_str = f"{state.current_epoch} / {state.total_epochs}"
        loss_str = f"{state.current_loss:.5f}"

    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Railway LSTM Monitor</title>
        <meta http-equiv="refresh" content="2"> <style>
            body {{ font-family: sans-serif; text-align: center; padding: 20px; background: #f4f4f4; }}
            .card {{ background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); max-width: 800px; margin: 0 auto; }}
            img {{ max-width: 100%; height: auto; border: 1px solid #ddd; }}
            .stats {{ display: flex; justify-content: space-around; margin-bottom: 20px; }}
            .stat-box {{ text-align: center; }}
            .val {{ font-size: 1.5em; font-weight: bold; color: #333; }}
            .label {{ color: #666; font-size: 0.9em; }}
        </style>
    </head>
    <body>
        <div class="card">
            <h1>LSTM Training Monitor</h1>
            <div class="stats">
                <div class="stat-box">
                    <div class="val">{status}</div>
                    <div class="label">Status</div>
                </div>
                <div class="stat-box">
                    <div class="val">{epoch_str}</div>
                    <div class="label">Epoch</div>
                </div>
                <div class="stat-box">
                    <div class="val">{loss_str}</div>
                    <div class="label">Current Loss</div>
                </div>
            </div>
            
            <h3>Live Visualization</h3>
            <img src="/plot.png?t={time.time()}" alt="Waiting for first epoch..." />
            <p><i>The page refreshes automatically every 2 seconds.</i></p>
        </div>
    </body>
    </html>
    """
    return html

@app.route('/plot.png')
def plot_image():
    """Returns the raw image bytes from the global buffer."""
    buf = None
    with state.lock:
        if state.latest_plot_buffer:
            # We copy the buffer to avoid IO closed errors if writing happens simultaneously
            buf = io.BytesIO(state.latest_plot_buffer.getvalue())
    
    if buf:
        return send_file(buf, mimetype='image/png')
    else:
        # Return a placeholder or 404 if not ready
        return Response("Plot not ready", status=503)

# --- Entry Point ---
if __name__ == "__main__":
    # 1. Start the Training in a Background Thread
    training_thread = threading.Thread(target=train_model, daemon=True)
    training_thread.start()
    
    # 2. Start the Web Server (Blocking, Main Thread)
    # Railway provides the PORT environment variable
    port = int(os.environ.get("PORT", 5000))
    print(f"--- Starting Web Server on Port {port} ---")
    app.run(host='0.0.0.0', port=port)
