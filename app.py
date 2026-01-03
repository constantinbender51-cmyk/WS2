import os
import torch
import torch.nn as nn
import torch.optim as optim
import random
import string
import requests
import re
import time
from bs4 import BeautifulSoup
from flask import Flask, request, jsonify, render_template_string

# ==========================================
# 1. CONFIGURATION
# ==========================================
app = Flask(__name__)
device = torch.device("cpu") # Use CPU for web deployment reliability

# Model Hyperparameters
HIDDEN_SIZE = 128
EMBEDDING_SIZE = 64
LEARNING_RATE = 0.005
MAX_LENGTH = 20
# Training for 25,000 iterations ensures it boots in < 60s for Railway
N_ITERATIONS = 25000 

# Character Dictionary
ALL_CHARS = string.ascii_lowercase
SOS_token = 0
EOS_token = 1
PAD_token = 2
char_to_index = {"SOS": 0, "EOS": 1, "PAD": 2}
index_to_char = {0: "SOS", 1: "EOS", 2: "PAD"}
for c in ALL_CHARS:
    if c not in char_to_index:
        idx = len(char_to_index)
        char_to_index[c] = idx
        index_to_char[idx] = c
VOCAB_SIZE = len(char_to_index)

# Global Model Variables (populated on startup)
encoder = None
decoder = None

# ==========================================
# 2. MODEL CLASSES
# ==========================================
class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(input_size, EMBEDDING_SIZE)
        self.gru = nn.GRU(EMBEDDING_SIZE, hidden_size)

    def forward(self, input, hidden):
        embedded = self.embedding(input).view(1, 1, -1)
        output, hidden = self.gru(embedded, hidden)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)

class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(output_size, EMBEDDING_SIZE)
        self.gru = nn.GRU(EMBEDDING_SIZE, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        output = self.embedding(input).view(1, 1, -1)
        output = torch.relu(output)
        output, hidden = self.gru(output, hidden)
        output = self.softmax(self.out(output[0]))
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)

# ==========================================
# 3. HELPER FUNCTIONS
# ==========================================
def tensor_from_word(word):
    indexes = [char_to_index[char] for char in word if char in char_to_index]
    indexes.append(EOS_token)
    return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)

def inject_noise(word):
    if len(word) < 3: return word
    chars = list(word)
    error_type = random.randint(0, 3)
    idx = random.randint(0, len(chars) - 1)
    
    if error_type == 0: # Insert
        chars.insert(idx, random.choice(ALL_CHARS))
    elif error_type == 1: # Delete
        if len(chars) > 2: del chars[idx]
    elif error_type == 2: # Swap
        if idx < len(chars) - 1: chars[idx], chars[idx+1] = chars[idx+1], chars[idx]
    elif error_type == 3: # Replace
        chars[idx] = random.choice(ALL_CHARS)
    return "".join(chars)

def evaluate_word(word):
    with torch.no_grad():
        input_tensor = tensor_from_word(word)
        input_length = input_tensor.size()[0]
        encoder_hidden = encoder.initHidden()
        for ei in range(input_length):
            _, encoder_hidden = encoder(input_tensor[ei], encoder_hidden)

        decoder_input = torch.tensor([[SOS_token]], device=device)
        decoder_hidden = encoder_hidden
        decoded_chars = []

        for di in range(MAX_LENGTH):
            decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
            _, topi = decoder_output.data.topk(1)
            if topi.item() == EOS_token: break
            decoded_chars.append(index_to_char[topi.item()])
            decoder_input = topi.squeeze().detach()
        return "".join(decoded_chars)

# ==========================================
# 4. TRAINING & STARTUP
# ==========================================
def setup_and_train():
    global encoder, decoder
    print("--- STARTUP: Downloading Vocabulary ---")
    
    # Download words
    clean_words = ["apple", "banana", "computer", "python", "neural"] # Fallback
    try:
        url = "https://www.top10000words.com/english/top-10000-english-words"
        r = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'}, timeout=5)
        if r.status_code == 200:
            soup = BeautifulSoup(r.text, 'html.parser')
            text = soup.get_text()
            # Get top 1000 words to ensure training is fast for demo
            clean_words = list(set(re.findall(r'\b[a-z]{3,}\b', text.lower())))[:1000]
            print(f"Downloaded {len(clean_words)} words.")
    except Exception as e:
        print(f"Using fallback list due to error: {e}")

    # Generate Data
    training_pairs = []
    for word in clean_words:
        training_pairs.append((word, word))
        for _ in range(5):
            training_pairs.append((inject_noise(word), word))

    # Init Models
    print("--- STARTUP: Initializing Models ---")
    encoder = EncoderRNN(VOCAB_SIZE, HIDDEN_SIZE).to(device)
    decoder = DecoderRNN(HIDDEN_SIZE, VOCAB_SIZE).to(device)
    enc_opt = optim.Adam(encoder.parameters(), lr=LEARNING_RATE)
    dec_opt = optim.Adam(decoder.parameters(), lr=LEARNING_RATE)
    criterion = nn.NLLLoss()

    # Train Loop
    print(f"--- STARTUP: Training on {len(training_pairs)} pairs for {N_ITERATIONS} iterations ---")
    for i in range(1, N_ITERATIONS + 1):
        pair = random.choice(training_pairs)
        input_tensor = tensor_from_word(pair[0])
        target_tensor = tensor_from_word(pair[1])

        enc_hidden = encoder.initHidden()
        enc_opt.zero_grad()
        dec_opt.zero_grad()

        loss = 0
        input_len = input_tensor.size(0)
        target_len = target_tensor.size(0)

        for ei in range(input_len):
            _, enc_hidden = encoder(input_tensor[ei], enc_hidden)

        dec_input = torch.tensor([[SOS_token]], device=device)
        dec_hidden = enc_hidden

        # Teacher forcing 50%
        use_tf = True if random.random() < 0.5 else False
        
        for di in range(target_len):
            dec_out, dec_hidden = decoder(dec_input, dec_hidden)
            loss += criterion(dec_out, target_tensor[di])
            if use_tf:
                dec_input = target_tensor[di]
            else:
                _, topi = dec_out.topk(1)
                dec_input = topi.squeeze().detach()
                if dec_input.item() == EOS_token: break
        
        loss.backward()
        enc_opt.step()
        dec_opt.step()

        if i % 5000 == 0:
            print(f"Iteration {i}/{N_ITERATIONS} complete.")

    print("--- STARTUP: Training Complete. Server Ready. ---")

# ==========================================
# 5. FLASK WEB SERVER
# ==========================================

# Run training immediately when script loads
setup_and_train()

# HTML Template
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AI Spelling Corrector</title>
    <style>
        body { font-family: -apple-system, system-ui, sans-serif; background: #f0f2f5; display: flex; justify-content: center; align-items: center; height: 100vh; margin: 0; }
        .card { background: white; padding: 2rem; border-radius: 12px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); width: 100%; max-width: 400px; text-align: center; }
        h1 { margin-bottom: 1.5rem; color: #1a1a1a; font-size: 1.5rem; }
        input { width: 100%; padding: 12px; margin-bottom: 1rem; border: 2px solid #e1e4e8; border-radius: 8px; font-size: 1rem; box-sizing: border-box; transition: 0.2s; }
        input:focus { border-color: #3b82f6; outline: none; }
        button { background: #3b82f6; color: white; border: none; padding: 12px 24px; border-radius: 8px; font-size: 1rem; cursor: pointer; width: 100%; font-weight: 600; transition: 0.2s; }
        button:hover { background: #2563eb; }
        #result { margin-top: 1.5rem; font-size: 1.25rem; min-height: 1.5rem; font-weight: 500; }
        .original { color: #ef4444; text-decoration: line-through; margin-right: 10px; }
        .corrected { color: #10b981; }
    </style>
</head>
<body>
    <div class="card">
        <h1>AI Spelling Corrector</h1>
        <input type="text" id="wordInput" placeholder="Type a misspelled word..." autocomplete="off">
        <button onclick="correctWord()">Auto Correct</button>
        <div id="result"></div>
    </div>

    <script>
        async function correctWord() {
            const word = document.getElementById('wordInput').value;
            const resultDiv = document.getElementById('result');
            
            if (!word) return;
            
            resultDiv.innerHTML = '<span style="color:#666">Thinking...</span>';
            
            try {
                const response = await fetch('/predict', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({word: word})
                });
                const data = await response.json();
                
                resultDiv.innerHTML = `
                    <span class="original">${word}</span>
                    <span class="corrected">${data.corrected}</span>
                `;
            } catch (e) {
                resultDiv.innerText = "Error connecting to server.";
            }
        }
        
        // Allow Enter key
        document.getElementById('wordInput').addEventListener('keypress', function (e) {
            if (e.key === 'Enter') correctWord();
        });
    </script>
</body>
</html>
"""

@app.route('/')
def home():
    return render_template_string(HTML_TEMPLATE)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    word = data.get('word', '').lower().strip()
    if not word:
        return jsonify({'corrected': ''})
    
    correction = evaluate_word(word)
    return jsonify({'corrected': correction})

if __name__ == "__main__":
    # Railway provides PORT via environment variable
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)