import torch
import torch.nn as nn
import torch.optim as optim
import math
import os
import logging
import time
import uuid
import json
import sys
import threading
import queue
import shutil
from flask import Flask, request, jsonify, render_template_string, send_file, Response, stream_with_context

# --- 1. Real-Time Logging & Data Architecture ---

# Global broadcaster to send logs and metrics to multiple connected web clients
class LogBroadcaster:
    def __init__(self):
        self.listeners = []
        self.history = [] # Cache recent logs for new connections

    def listen(self):
        """Returns a queue for a new web client to consume"""
        q = queue.Queue()
        self.listeners.append(q)
        # Replay history so new users see what happened recently
        for msg in self.history[-50:]:
            q.put(msg)
        return q

    def broadcast(self, data_dict):
        """Push a message to all active listeners"""
        # Format as Server-Sent Event (SSE) data
        # data_dict should have a 'type': 'log' or 'metric'
        payload = f"data: {json.dumps(data_dict)}\n\n"
        
        self.history.append(payload)
        if len(self.history) > 200: self.history.pop(0)
        
        dead_listeners = []
        for q in self.listeners:
            try:
                q.put(payload)
            except:
                dead_listeners.append(q)
        
        # Cleanup disconnected clients
        for d in dead_listeners:
            if d in self.listeners: self.listeners.remove(d)

broadcaster = LogBroadcaster()

# Custom Handler to write to Stdout (Railway) AND Web UI
class StreamLogger(logging.StreamHandler):
    def emit(self, record):
        try:
            msg = self.format(record)
            
            # 1. Write to Railway/Terminal (Force flush for immediate display)
            sys.stdout.write(msg + '\n')
            sys.stdout.flush()
            
            # 2. Send to Web Frontend as a 'log' type event
            broadcaster.broadcast({'type': 'log', 'text': msg})
            
            # 3. Small delay to prevent event flooding
            time.sleep(0.01) 
        except Exception:
            self.handleError(record)

# Configure Root Logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    handlers=[StreamLogger()]
)
logger = logging.getLogger()

# Silence noisy libraries
logging.getLogger('werkzeug').setLevel(logging.ERROR)
logging.getLogger('urllib3').setLevel(logging.ERROR)

app = Flask(__name__)

# --- Configuration ---
FILE_PATH = 'text.txt'
MODEL_PATH = 'financial_model.pth' 
BATCH_SIZE = 64
BLOCK_SIZE = 256
MAX_ITERS = 100000  # Increased to 100k for long training
LEARNING_RATE = 3e-4
EMBED_DIM = 384
NUM_HEADS = 6
NUM_LAYERS = 5
DROPOUT = 0.1
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Global Training State
training_active = False
model_ready = False

# --- 2. Data & Tokenizer ---

def ensure_file_exists():
    if not os.path.exists(FILE_PATH):
        # Create dummy financial data
        base_text = "Asset: Resource with economic value. Equity: Ownership interest. Liability: Financial obligation. "
        data = base_text * 200 
        with open(FILE_PATH, 'w', encoding='utf-8') as f:
            f.write(data)

ensure_file_exists()
with open(FILE_PATH, 'r', encoding='utf-8') as f:
    text = f.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s if c in stoi]
decode = lambda l: ''.join([itos[i] for i in l])

data_tensor = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data_tensor))
train_data = data_tensor[:n]
val_data = data_tensor[n:]

def get_batch(split='train'):
    data = train_data if split == 'train' else val_data
    max_idx = len(data) - BLOCK_SIZE
    if max_idx <= 0: return torch.zeros((BATCH_SIZE, BLOCK_SIZE), dtype=torch.long).to(DEVICE), torch.zeros((BATCH_SIZE, BLOCK_SIZE), dtype=torch.long).to(DEVICE)
    ix = torch.randint(max_idx, (BATCH_SIZE,))
    x = torch.stack([data[i:i+BLOCK_SIZE] for i in ix])
    y = torch.stack([data[i+1:i+BLOCK_SIZE+1] for i in ix])
    return x.to(DEVICE), y.to(DEVICE)

# --- 3. Model Architecture ---

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
    def forward(self, x):
        return x + self.pe[:x.size(0), :].unsqueeze(1)

class TransformerModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, nhead, num_layers, dropout):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_encoder = PositionalEncoding(embed_dim, max_len=BLOCK_SIZE)
        encoder_layers = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=nhead, dim_feedforward=embed_dim*4, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        self.decoder = nn.Linear(embed_dim, vocab_size)
        self.embed_dim = embed_dim
    def forward(self, src):
        src = self.embedding(src) * math.sqrt(self.embed_dim)
        src = src.permute(1, 0, 2) 
        src = self.pos_encoder(src)
        sz = src.size(0)
        mask = nn.Transformer.generate_square_subsequent_mask(sz).to(DEVICE)
        output = self.transformer_encoder(src, mask=mask)
        return self.decoder(output.permute(1, 0, 2))

model = TransformerModel(vocab_size, EMBED_DIM, NUM_HEADS, NUM_LAYERS, DROPOUT).to(DEVICE)

# --- 4. Background Training Logic ---

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(5)
        for k in range(5):
            X, Y = get_batch(split)
            logits = model(X)
            loss = nn.functional.cross_entropy(logits.view(-1, vocab_size), Y.view(-1))
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out

def save_checkpoint(filename):
    """Atomic save to avoid corruption during download"""
    temp_name = filename + ".tmp"
    torch.save(model.state_dict(), temp_name)
    os.replace(temp_name, filename)

def training_thread_target():
    global training_active, model_ready
    training_active = True
    
    logger.info("Initializing training sequence...")
    logger.info(f"Target: {MAX_ITERS} iterations | Device: {DEVICE}")

    # Persistence: Load existing model if available to resume/finetune
    if os.path.exists(MODEL_PATH):
        try:
            logger.info("Found existing model. Loading weights to resume training...")
            model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
            model_ready = True
        except Exception as e:
            logger.error(f"Could not load existing model: {e}")

    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    model.train()
    
    start_time = time.time()
    
    for iter in range(MAX_ITERS + 1):
        # Monitoring: Every 100 iters, check loss and update graph
        if iter % 100 == 0:
            losses = estimate_loss()
            elapsed = time.time() - start_time
            
            # Log to console
            logger.info(f"Iter {iter:06d} | Train: {losses['train']:.4f} | Val: {losses['val']:.4f} | T: {elapsed:.1f}s")
            
            # Send data to Graph
            broadcaster.broadcast({
                'type': 'metric',
                'iter': iter,
                'train_loss': losses['train'],
                'val_loss': losses['val']
            })
        
        # Checkpointing: Every 500 iters, save to disk so user can download
        if iter > 0 and iter % 500 == 0:
            save_checkpoint(MODEL_PATH)
            # logger.info("Checkpoint saved.") 

        xb, yb = get_batch('train')
        logits = model(xb)
        loss = nn.functional.cross_entropy(logits.view(-1, vocab_size), yb.view(-1))
        
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    save_checkpoint(MODEL_PATH)
    logger.info(f"Training Complete. Final model saved to {MODEL_PATH}")
    training_active = False
    model_ready = True

def start_training_background():
    if not training_active:
        t = threading.Thread(target=training_thread_target)
        t.daemon = True 
        t.start()

# --- 5. Web Interface & Routes ---

HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>AI Training Monitor</title>
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <!-- Chart.js for Graphing -->
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        :root { --primary: #2563eb; --bg: #f8fafc; --card: #ffffff; --text: #1e293b; --mono: #0f172a; }
        body { font-family: 'Inter', system-ui, sans-serif; background: var(--bg); color: var(--text); padding: 20px; display: flex; flex-direction: column; align-items: center; min-height: 100vh; margin: 0; }
        .container { width: 100%; max-width: 900px; display: grid; gap: 20px; }
        
        /* Card Styles */
        .card { background: var(--card); padding: 24px; border-radius: 12px; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1); border: 1px solid #e2e8f0; }
        h2 { margin-top: 0; font-size: 1.25rem; font-weight: 700; color: #0f172a; display: flex; justify-content: space-between; align-items: center; }
        
        /* Console Styles */
        .console-window { background: #1e1e1e; color: #10b981; font-family: 'Fira Code', monospace; padding: 16px; border-radius: 8px; height: 200px; overflow-y: auto; font-size: 12px; line-height: 1.5; border: 1px solid #334155; display: flex; flex-direction: column-reverse; }
        .log-entry { border-bottom: 1px solid #333; padding: 2px 0; }
        
        /* Graph Styles */
        .chart-container { position: relative; height: 300px; width: 100%; }

        /* Input Styles */
        textarea { width: 100%; height: 80px; padding: 12px; border: 1px solid #cbd5e1; border-radius: 8px; font-family: inherit; margin: 10px 0; box-sizing: border-box; resize: vertical; }
        .btn-group { display: flex; gap: 10px; flex-wrap: wrap; margin-top: 10px;}
        button, .btn { padding: 10px 20px; border-radius: 6px; font-weight: 600; cursor: pointer; border: none; font-size: 0.9rem; text-decoration: none; display: inline-block; text-align: center; }
        .btn-primary { background: var(--primary); color: white; }
        .btn-primary:hover { opacity: 0.9; }
        .btn-secondary { background: #e2e8f0; color: #475569; }
        .btn-secondary:hover { background: #cbd5e1; }
        
        /* Output Area */
        #output { background: #f1f5f9; padding: 16px; border-radius: 8px; min-height: 60px; white-space: pre-wrap; margin-top: 10px; border-left: 4px solid var(--primary); }
    </style>
</head>
<body>
    <div class="container">
        
        <!-- Live Graph -->
        <div class="card">
            <h2>📈 Training Loss Monitor</h2>
            <div class="chart-container">
                <canvas id="lossChart"></canvas>
            </div>
        </div>

        <!-- Live Console -->
        <div class="card">
            <h2>🚀 System Logs <span style="font-size:0.8em; font-weight:400; color:#64748b">Iter: <span id="iter-count">0</span></span></h2>
            <div id="console" class="console-window"></div>
        </div>

        <!-- Controls -->
        <div class="card">
            <h2>Model Controls</h2>
            <textarea id="prompt">Equity</textarea>
            
            <div class="btn-group">
                <button class="btn btn-primary" onclick="generate()">Test Model</button>
                <button class="btn btn-secondary" onclick="triggerTrain()">Restart Training</button>
                <a href="/download" class="btn btn-secondary">💾 Download Checkpoint (.pth)</a>
            </div>
            <div id="output">Ready.</div>
        </div>
    </div>

    <script>
        // 1. Setup Chart.js
        const ctx = document.getElementById('lossChart').getContext('2d');
        const lossChart = new Chart(ctx, {
            type: 'line',
            data: {
                labels: [],
                datasets: [{
                    label: 'Train Loss',
                    borderColor: '#2563eb',
                    backgroundColor: 'rgba(37, 99, 235, 0.1)',
                    data: [],
                    tension: 0.4
                }, {
                    label: 'Val Loss',
                    borderColor: '#dc2626',
                    backgroundColor: 'rgba(220, 38, 38, 0.1)',
                    data: [],
                    tension: 0.4
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                animation: false, // Disable animation for performance
                scales: {
                    x: { title: { display: true, text: 'Iterations' } },
                    y: { title: { display: true, text: 'Loss' } }
                }
            }
        });

        // 2. Setup SSE for Real-time Logging & Metrics
        const consoleDiv = document.getElementById('console');
        const iterSpan = document.getElementById('iter-count');
        const eventSource = new EventSource('/logs');
        
        eventSource.onmessage = function(event) {
            const data = JSON.parse(event.data);
            
            if (data.type === 'log') {
                // Handle text log
                const line = document.createElement('div');
                line.className = 'log-entry';
                line.innerText = data.text;
                consoleDiv.insertBefore(line, consoleDiv.firstChild);
            } 
            else if (data.type === 'metric') {
                // Handle graph update
                iterSpan.innerText = data.iter;
                
                // Add data point
                lossChart.data.labels.push(data.iter);
                lossChart.data.datasets[0].data.push(data.train_loss);
                lossChart.data.datasets[1].data.push(data.val_loss);
                
                // Keep chart from getting too crowded (keep last 100 points)
                if (lossChart.data.labels.length > 100) {
                    lossChart.data.labels.shift();
                    lossChart.data.datasets[0].data.shift();
                    lossChart.data.datasets[1].data.shift();
                }
                
                lossChart.update();
            }
        };

        eventSource.onerror = function() {
            console.log("Stream lost, retrying...");
        };

        // 3. Interaction Logic
        async function generate() {
            const prompt = document.getElementById('prompt').value;
            const output = document.getElementById('output');
            output.innerText = "Processing...";
            try {
                const res = await fetch('/generate', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ prompt })
                });
                const data = await res.json();
                output.innerText = data.text;
            } catch (e) {
                output.innerText = "Error communicating with server.";
            }
        }

        async function triggerTrain() {
            if(confirm("Restart training? This will continue from current weights but reset the loop.")) {
                fetch('/train', {method: 'POST'});
            }
        }
    </script>
</body>
</html>
"""

@app.route('/')
def home(): 
    return render_template_string(HTML_TEMPLATE, device=DEVICE)

@app.route('/logs')
def stream_logs():
    """SSE Endpoint for streaming logs to browser"""
    def generate():
        q = broadcaster.listen()
        while True:
            msg = q.get()
            yield msg
    return Response(stream_with_context(generate()), mimetype='text/event-stream')

@app.route('/train', methods=['POST'])
def manual_train():
    if not training_active:
        start_training_background()
        return jsonify({"status": "Training started"})
    return jsonify({"status": "Training already in progress"})

@app.route('/download')
def download_file():
    if os.path.exists(MODEL_PATH):
        return send_file(MODEL_PATH, as_attachment=True)
    return "Model file not found. Please wait for training to initialize.", 404

@app.route('/generate', methods=['POST'])
def api_generate():
    global model_ready
    req_id = str(uuid.uuid4())[:8]
    prompt = request.json.get('prompt', 'Equity')
    
    logger.info(f"API Request {req_id} | In: {prompt[:20]}...")

    # Auto-load if exists but not loaded
    if not model_ready and os.path.exists(MODEL_PATH):
        try:
            model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
            model_ready = True
        except:
             pass

    if not model_ready:
        return jsonify({'text': 'Model is currently training/initializing. Please wait...'})

    try:
        context = torch.tensor(encode(prompt), dtype=torch.long, device=DEVICE).unsqueeze(0)
    except KeyError:
        return jsonify({'text': 'Error: Prompt contains unknown characters.'})
        
    generated = prompt
    model.eval()
    for _ in range(150):
        with torch.no_grad():
            logits = model(context[:, -BLOCK_SIZE:])[:, -1, :]
            probs = torch.nn.functional.softmax(logits, dim=-1)
            next_char = torch.multinomial(probs, num_samples=1)
            context = torch.cat((context, next_char), dim=1)
            generated += decode([next_char.item()])
            
    logger.info(f"API Request {req_id} | Complete.")
    return jsonify({'text': generated})

if __name__ == '__main__':
    # On startup, start background training (which will load existing model if found)
    start_training_background()

    port = int(os.environ.get('PORT', 8080))
    app.run(debug=False, host='0.0.0.0', port=port, threaded=True)
