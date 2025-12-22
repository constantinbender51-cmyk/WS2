import torch
import torch.nn as nn
import torch.optim as optim
import math
import os
from flask import Flask, request, jsonify, render_template_string, send_file

# --- Configuration ---
FILE_PATH = 'text.txt'
MODEL_PATH = 'financial_model.pth' # The weights file
BATCH_SIZE = 32
BLOCK_SIZE = 64
MAX_ITERS = 5000
LEARNING_RATE = 3e-4
EMBED_DIM = 128
NUM_HEADS = 4
NUM_LAYERS = 4
DROPOUT = 0.1
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

app = Flask(__name__)

# --- 1. Data & Tokenizer ---

def ensure_file_exists():
    if not os.path.exists(FILE_PATH):
        data = "Asset: A resource... Bond: A loan... Equity: Ownership... Finance: Management..."
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
train_data = data_tensor[:int(0.9 * len(data_tensor))]

def get_batch():
    ix = torch.randint(len(train_data) - BLOCK_SIZE, (BATCH_SIZE,))
    x = torch.stack([train_data[i:i+BLOCK_SIZE] for i in ix])
    y = torch.stack([train_data[i+1:i+BLOCK_SIZE+1] for i in ix])
    return x.to(DEVICE), y.to(DEVICE)

# --- 2. Model Architecture ---

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

# Initialize model
model = TransformerModel(vocab_size, EMBED_DIM, NUM_HEADS, NUM_LAYERS, DROPOUT).to(DEVICE)

# --- 3. Persistence Logic ---

def train_and_save():
    print(f"Starting training...")
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()
    model.train()
    for iter in range(MAX_ITERS):
        xb, yb = get_batch()
        logits = model(xb)
        loss = criterion(logits.view(-1, vocab_size), yb.view(-1))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if iter % 500 == 0: print(f"Iter {iter}: Loss {loss.item():.4f}")
    
    torch.save(model.state_dict(), MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")

def load_model():
    if os.path.exists(MODEL_PATH):
        print(f"Loading existing model...")
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        model.eval()
    else:
        train_and_save()
        model.eval()

# --- 4. Web Interface & Routes ---

HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>AI Control Center</title>
    <style>
        body { font-family: 'Inter', system-ui, sans-serif; background: #f8fafc; display: flex; justify-content: center; padding: 40px 20px; color: #1e293b; }
        .card { background: white; padding: 32px; border-radius: 16px; box-shadow: 0 10px 25px -5px rgba(0,0,0,0.1); width: 100%; max-width: 550px; border: 1px solid #e2e8f0; }
        h2 { margin-top: 0; font-weight: 700; color: #0f172a; }
        label { display: block; margin-bottom: 8px; font-weight: 500; font-size: 14px; color: #64748b; }
        textarea { width: 100%; height: 100px; padding: 12px; border: 1px solid #cbd5e1; border-radius: 8px; box-sizing: border-box; font-size: 16px; margin-bottom: 16px; outline: none; transition: border 0.2s; }
        textarea:focus { border-color: #3b82f6; }
        .btn-group { display: flex; gap: 10px; flex-direction: column; }
        .primary-btn { background: #2563eb; color: white; border: none; padding: 14px; border-radius: 8px; font-weight: 600; cursor: pointer; transition: 0.2s; }
        .primary-btn:hover { background: #1d4ed8; }
        .secondary-btn { background: #f1f5f9; color: #475569; border: 1px solid #e2e8f0; padding: 10px; border-radius: 8px; font-weight: 500; cursor: pointer; text-decoration: none; text-align: center; font-size: 14px; }
        .secondary-btn:hover { background: #e2e8f0; }
        #output { margin-top: 24px; background: #f1f5f9; padding: 16px; border-radius: 10px; min-height: 80px; white-space: pre-wrap; font-size: 15px; border: 1px dashed #cbd5e1; color: #334155; }
        .status { font-size: 12px; margin-top: 20px; color: #94a3b8; text-align: center; }
    </style>
</head>
<body>
    <div class="card">
        <h2>Financial AI Interface</h2>
        <label for="prompt">Starting Text (Prompt)</label>
        <textarea id="prompt">Equity</textarea>
        
        <div class="btn-group">
            <button class="primary-btn" onclick="generate()">Generate Prediction</button>
            <a href="/download" class="secondary-btn">💾 Download Model Weights (.pth)</a>
        </div>

        <div id="output">Prediction will appear here...</div>
        <div class="status">Running on {{ device }}</div>
    </div>
    <script>
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
    </script>
</body>
</html>
"""

@app.route('/')
def home(): 
    return render_template_string(HTML_TEMPLATE, device=DEVICE)

@app.route('/download')
def download_file():
    if os.path.exists(MODEL_PATH):
        return send_file(MODEL_PATH, as_attachment=True)
    return "Model file not found. Please wait for training to complete.", 404

@app.route('/generate', methods=['POST'])
def api_generate():
    prompt = request.json.get('prompt', 'Equity')
    try:
        context = torch.tensor(encode(prompt), dtype=torch.long, device=DEVICE).unsqueeze(0)
    except KeyError:
        return jsonify({'text': 'Error: Prompt contains characters not seen in training.'})
        
    generated = prompt
    model.eval()
    for _ in range(150):
        with torch.no_grad():
            logits = model(context[:, -BLOCK_SIZE:])[:, -1, :]
            probs = torch.nn.functional.softmax(logits, dim=-1)
            next_char = torch.multinomial(probs, num_samples=1)
            context = torch.cat((context, next_char), dim=1)
            generated += decode([next_char.item()])
    return jsonify({'text': generated})

if __name__ == '__main__':
    load_model()
    print(f"\nServer ready at http://0.0.0.0:5000")
    app.run(debug=False, host='0.0.0.0', port=5000)
