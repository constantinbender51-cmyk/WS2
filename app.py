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
matplotlib.use('Agg') 

app = Flask(__name__)

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
        h0 = torch.zeros(1, x.size(0), self.hidden_size)
        c0 = torch.zeros(1, x.size(0), self.hidden_size)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

# --- Training Logic ---
def train_model():
    print("--- Starting Training with Slope Features ---")
    
    # 1. Data Generation
    N = 1000
    t = np.linspace(0, 100, N)
    raw_data = np.sin(t) + 0.2 * np.random.normal(size=N)
    
    # 2. Calculate Slopes (Differences)
    # slope[i] = raw_data[i] - raw_data[i-1]
    # This results in N-1 points.
    slopes = np.diff(raw_data)
    
    # 3. Prepare windowed data
    # We want to use 20 "slopes" to predict the amplitude at the end of that window
    window_size = 20
    X, y = [], []
    
    # Because slopes is len N-1, we start indices accordingly
    # We look at slopes from i to i+window_size
    # We predict the amplitude at raw_data[i + window_size]
    for i in range(len(slopes) - window_size):
        X.append(slopes[i : i + window_size])
        y.append(raw_data[i + window_size])
    
    X = torch.tensor(np.array(X), dtype=torch.float32).unsqueeze(-1) 
    y = torch.tensor(np.array(y), dtype=torch.float32).unsqueeze(-1)
    
    model = SimpleLSTM()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    
    epochs = 500
    with state.lock:
        state.total_epochs = epochs

    for epoch in range(epochs):
        model.train()
        outputs = model(X)
        loss = criterion(outputs, y)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        with state.lock:
            state.current_epoch = epoch + 1
            state.current_loss = loss.item()

        if epoch % 20 == 0 or epoch == epochs - 1:
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")
            generate_plot(model, X, y, t, window_size, loss.item(), epoch, slopes)
            
        time.sleep(0.02)

    with state.lock:
        state.training_finished = True

def generate_plot(model, X, y, t, window_size, loss, epoch, slopes):
    model.eval()
    with torch.no_grad():
        predictions = model(X).numpy()
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), gridspec_kw={'height_ratios': [2, 1]})
    
    # Top Plot: Predictions vs Actual Amplitudes
    # Note: X starts at index 0 of slopes, so the target y corresponds to index window_size of t
    plot_t = t[window_size : window_size + len(predictions)]
    ax1.plot(plot_t, y.numpy(), label='Actual Amplitude', alpha=0.4, color='gray')
    ax1.plot(plot_t, predictions, label='LSTM Prediction (from Slopes)', color='red', linewidth=1.5)
    ax1.set_title(f'Epoch {epoch} | Loss: {loss:.4f}')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Bottom Plot: The Feature (Slopes)
    # Just showing a snippet of the slopes to visualize the input "noise"
    ax2.plot(plot_t, slopes[0:len(predictions)], label='Input Feature (Slope/Diff)', color='blue', alpha=0.6)
    ax2.set_title('Input Features: Sequential Slopes ($y_t - y_{t-1}$)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.close(fig)
    
    with state.lock:
        state.latest_plot_buffer = buf

@app.route('/')
def dashboard():
    with state.lock:
        status = "Finished" if state.training_finished else "Training..."
        epoch_str = f"{state.current_epoch} / {state.total_epochs}"
        loss_str = f"{state.current_loss:.5f}"

    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Slope-Based LSTM Monitor</title>
        <meta http-equiv="refresh" content="2">
        <style>
            body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; text-align: center; padding: 20px; background: #eceff1; color: #37474f; }}
            .card {{ background: white; padding: 30px; border-radius: 12px; box-shadow: 0 4px 20px rgba(0,0,0,0.08); max-width: 900px; margin: 0 auto; }}
            img {{ max-width: 100%; height: auto; border-radius: 4px; margin-top: 20px; border: 1px solid #cfd8dc; }}
            .stats {{ display: flex; justify-content: space-around; margin-bottom: 30px; border-bottom: 1px solid #eee; padding-bottom: 20px; }}
            .val {{ font-size: 1.8em; font-weight: bold; color: #2196f3; }}
            .label {{ color: #90a4ae; font-size: 0.85em; text-transform: uppercase; letter-spacing: 1px; }}
            h1 {{ margin-top: 0; color: #263238; }}
        </style>
    </head>
    <body>
        <div class="card">
            <h1>Slope-Feature LSTM</h1>
            <div class="stats">
                <div class="stat-box"><div class="val">{status}</div><div class="label">Status</div></div>
                <div class="stat-box"><div class="val">{epoch_str}</div><div class="label">Epoch</div></div>
                <div class="stat-box"><div class="val">{loss_str}</div><div class="label">Loss</div></div>
            </div>
            <img src="/plot.png?t={time.time()}" alt="Generating visualization..." />
            <p style="color: #78909c; font-size: 0.9em; margin-top: 20px;">
                Using $\Delta y$ (Slope) as input to predict absolute $y$ (Amplitude).
            </p>
        </div>
    </body>
    </html>
    """
    return html

@app.route('/plot.png')
def plot_image():
    buf = None
    with state.lock:
        if state.latest_plot_buffer:
            buf = io.BytesIO(state.latest_plot_buffer.getvalue())
    if buf:
        return send_file(buf, mimetype='image/png')
    return Response("Not Ready", status=503)

if __name__ == "__main__":
    training_thread = threading.Thread(target=train_model, daemon=True)
    training_thread.start()
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)