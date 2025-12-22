import torch
import torch.nn as nn
import torch.optim as optim
import math
import os
import sys
from flask import Flask, request, jsonify, render_template_string

# --- Configuration ---
FILE_PATH = 'text.txt'
BATCH_SIZE = 32
BLOCK_SIZE = 64      # Context window size
MAX_ITERS = 5000     # Training iterations
LEARNING_RATE = 3e-4
EMBED_DIM = 128
NUM_HEADS = 4
NUM_LAYERS = 4
DROPOUT = 0.1
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Initialize Flask App
app = Flask(__name__)

# --- 1. Data Preparation ---

def ensure_file_exists():
    """Creates a dummy dictionary file if it doesn't exist."""
    if not os.path.exists(FILE_PATH):
        print(f"'{FILE_PATH}' not found. Creating a dummy dictionary...")
        data = """
Asset: A resource with economic value that an individual, corporation, or country owns or controls with the expectation that it will provide a future benefit.
Bond: A fixed income instrument that represents a loan made by an investor to a borrower.
Capital: Financial assets, such as funds held in deposit accounts and/or funds obtained from special financing sources.
Debt: An amount of money borrowed by one party from another.
Equity: The value of the shares issued by a company. It represents the ownership interest held by shareholders.
Finance: The management of large amounts of money, especially by governments or large companies.
Gold: A yellow precious metal, the chemical element of atomic number 79, used especially in jewelry and decoration and to guarantee the value of currencies.
Hedge: An investment to reduce the risk of adverse price movements in an asset.
Income: Money received, especially on a regular basis, for work or through investments.
Liability: Something a person or company owes, usually a sum of money.
Market: A composition of systems, institutions, procedures, social relations or infrastructures whereby parties engage in exchange.
Profit: A financial gain, especially the difference between the amount earned and the amount spent in buying, operating, or producing something.
Stock: The capital raised by a business or corporation through the issue and subscription of shares.
Yield: The earnings generated and realized on an investment over a particular period of time.
"""
        with open(FILE_PATH, 'w', encoding='utf-8') as f:
            f.write(data.strip())

ensure_file_exists()

# Read the file
with open(FILE_PATH, 'r', encoding='utf-8') as f:
    text = f.read()

# Simple Character-level Tokenizer
chars = sorted(list(set(text)))
vocab_size = len(chars)
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}

encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

# Prepare Train/Validation sets
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

def get_batch(split):
    data_src = train_data if split == 'train' else val_data
    max_idx = len(data_src) - BLOCK_SIZE - 1
    if max_idx <= 0:
        ix = torch.randint(0, len(data_src) - 1, (BATCH_SIZE,))
        x = torch.stack([data_src[i:i+1] for i in ix])
        y = torch.stack([data_src[i+1:i+2] for i in ix])
    else:
        ix = torch.randint(len(data_src) - BLOCK_SIZE, (BATCH_SIZE,))
        x = torch.stack([data_src[i:i+BLOCK_SIZE] for i in ix])
        y = torch.stack([data_src[i+1:i+BLOCK_SIZE+1] for i in ix])
    return x.to(DEVICE), y.to(DEVICE)

# --- 2. Model Definition ---

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
        # x: [Seq Len, Batch Size, Embed Dim]
        # Crop pe to the current sequence length of x
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

    def forward(self, src, src_mask=None):
        src = self.embedding(src) * math.sqrt(self.embed_dim)
        src = src.permute(1, 0, 2) 
        src = self.pos_encoder(src)
        
        if src_mask is None:
            sz = src.size(0)
            src_mask = nn.Transformer.generate_square_subsequent_mask(sz).to(DEVICE)

        output = self.transformer_encoder(src, mask=src_mask)
        output = output.permute(1, 0, 2)
        output = self.decoder(output)
        return output

# --- 3. Training & Core Logic ---

# Initialize model globally so Flask can access it
model = TransformerModel(vocab_size, EMBED_DIM, NUM_HEADS, NUM_LAYERS, DROPOUT).to(DEVICE)

def train_model():
    print(f"Starting training on {DEVICE} for {MAX_ITERS} iterations...")
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()
    model.train()

    for iter in range(MAX_ITERS):
        xb, yb = get_batch('train')
        logits = model(xb)
        
        B, T, C = logits.shape
        logits = logits.view(B*T, C)
        targets = yb.view(B*T)
        
        loss = criterion(logits, targets)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if iter % 500 == 0:
            print(f"Iter {iter}: Loss {loss.item():.4f}")
            
    print("Training complete!")

def generate_text(model, start_str, max_new_tokens=100, temperature=0.8):
    model.eval()
    
    # Handle characters not in training data
    try:
        start_tokens = encode(start_str)
    except KeyError:
        # Fallback for unknown chars: strip them or use a default
        valid_chars = [c for c in start_str if c in stoi]
        if not valid_chars: return "Error: Prompt contains unknown characters."
        start_tokens = encode("".join(valid_chars))

    context = torch.tensor(start_tokens, dtype=torch.long, device=DEVICE).unsqueeze(0)
    generated_text = start_str

    for _ in range(max_new_tokens):
        context_cond = context[:, -BLOCK_SIZE:]
        
        with torch.no_grad():
            logits = model(context_cond)
        
        logits = logits[:, -1, :]
        
        # Apply temperature
        if temperature > 0:
            logits = logits / temperature
            probs = torch.nn.functional.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
        else:
            # Greedy decoding if temp is 0
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)
        
        context = torch.cat((context, idx_next), dim=1)
        generated_text += decode([idx_next.item()])
    
    return generated_text

# --- 4. Web Server (Flask) ---

# Simple HTML Interface
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Financial AI Model</title>
    <style>
        body { font-family: sans-serif; max-width: 800px; margin: 40px auto; padding: 20px; line-height: 1.6; }
        h1 { color: #333; }
        textarea { width: 100%; height: 100px; padding: 10px; border-radius: 5px; border: 1px solid #ccc; font-size: 16px; }
        button { background: #007bff; color: white; border: none; padding: 10px 20px; border-radius: 5px; cursor: pointer; font-size: 16px; margin-top: 10px; }
        button:hover { background: #0056b3; }
        #output { margin-top: 20px; padding: 20px; background: #f4f4f4; border-radius: 5px; white-space: pre-wrap; min-height: 100px; border: 1px solid #ddd; }
        .loading { color: #666; font-style: italic; }
    </style>
</head>
<body>
    <h1>Financial Text Generator</h1>
    <p>Model trained on {iters} iterations. Enter a prompt below:</p>
    
    <textarea id="prompt" placeholder="Type something like 'Market' or 'Equity'...">Equity</textarea>
    <br>
    <label>Temperature (Creativity): <input type="range" id="temp" min="0.1" max="1.5" step="0.1" value="0.8"></label>
    <span id="temp-val">0.8</span>
    <br><br>
    <button onclick="generate()">Generate</button>
    
    <div id="output"></div>

    <script>
        document.getElementById('temp').oninput = function() {
            document.getElementById('temp-val').innerHTML = this.value;
        }

        async function generate() {
            const prompt = document.getElementById('prompt').value;
            const temp = parseFloat(document.getElementById('temp').value);
            const outputDiv = document.getElementById('output');
            
            outputDiv.innerHTML = '<span class="loading">Generating...</span>';
            
            try {
                const response = await fetch('/generate', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ prompt: prompt, temperature: temp })
                });
                const data = await response.json();
                outputDiv.innerText = data.text;
            } catch (e) {
                outputDiv.innerText = "Error: " + e;
            }
        }
    </script>
</body>
</html>
"""

@app.route('/')
def home():
    return render_template_string(HTML_TEMPLATE, iters=MAX_ITERS)

@app.route('/generate', methods=['POST'])
def api_generate():
    data = request.json
    start_str = data.get('prompt', 'Equity')
    temperature = data.get('temperature', 0.8)
    
    # Generate text using the global model
    result = generate_text(model, start_str, max_new_tokens=200, temperature=temperature)
    return jsonify({'text': result})

if __name__ == '__main__':
    # 1. Run training
    train_model()
    
    # 2. Start Web Server
    print("\nStarting Web Server at http://127.0.0.1:5000")
    app.run(debug=False, host='0.0.0.0', port=5000)
