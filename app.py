import torch
import torch.nn as nn
import torch.optim as optim
import math
import os

# --- Configuration ---
FILE_PATH = 'text.txt'
BATCH_SIZE = 64
BLOCK_SIZE = 128     # Context window
MAX_ITERS = 5000     
EVAL_INTERVAL = 500  # How often to check validation loss
LEARNING_RATE = 3e-4
EMBED_DIM = 256      # Slightly larger for better generalization
NUM_HEADS = 6
NUM_LAYERS = 6
DROPOUT = 0.2        # Higher dropout to prevent memorization
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# --- 1. Data Loading (No Dummy Creation) ---

if not os.path.exists(FILE_PATH):
    raise FileNotFoundError(f"Please ensure '{FILE_PATH}' exists in the directory.")

with open(FILE_PATH, 'r', encoding='utf-8') as f:
    text = f.read()

# Character-level Tokenizer
chars = sorted(list(set(text)))
vocab_size = len(chars)
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}

encode = lambda s: [stoi[c] for c in s if c in stoi] # safe encode
decode = lambda l: ''.join([itos[i] for i in l])

data = torch.tensor(encode(text), dtype=torch.long)

# 90% Train, 10% Validation (Crucial for Generalization)
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

def get_batch(split):
    data_src = train_data if split == 'train' else val_data
    if len(data_src) <= BLOCK_SIZE:
        raise ValueError(f"Dataset for {split} is too small for BLOCK_SIZE={BLOCK_SIZE}")
        
    ix = torch.randint(len(data_src) - BLOCK_SIZE, (BATCH_SIZE,))
    x = torch.stack([data_src[i:i+BLOCK_SIZE] for i in ix])
    y = torch.stack([data_src[i+1:i+BLOCK_SIZE+1] for i in ix])
    return x.to(DEVICE), y.to(DEVICE)

# --- 2. Evaluation Helper ---

@torch.no_grad()
def estimate_loss(model):
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(100) # Average over 100 batches
        for k in range(100):
            X, Y = get_batch(split)
            logits = model(X)
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = Y.view(B*T)
            loss = nn.CrossEntropyLoss()(logits, targets)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# --- 3. Model Definition ---

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

# --- 4. Training ---

model = TransformerModel(vocab_size, EMBED_DIM, NUM_HEADS, NUM_LAYERS, DROPOUT).to(DEVICE)
optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
criterion = nn.CrossEntropyLoss()

print(f"Training on {DEVICE} for {MAX_ITERS} iterations...")
print(f"Vocab size: {vocab_size}")

for iter in range(MAX_ITERS):
    
    # Every so often, evaluate loss on train and val sets
    if iter % EVAL_INTERVAL == 0 or iter == MAX_ITERS - 1:
        losses = estimate_loss(model)
        print(f"Step {iter}: Train Loss {losses['train']:.4f}, Val Loss {losses['val']:.4f}")

    xb, yb = get_batch('train')

    logits = model(xb)
    B, T, C = logits.shape
    logits = logits.view(B*T, C)
    targets = yb.view(B*T)
    
    loss = criterion(logits, targets)

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

print("Training complete.")

# --- 5. Generation ---

def generate(model, start_str, max_new_tokens=200, temperature=0.8):
    model.eval()
    try:
        context = torch.tensor(encode(start_str), dtype=torch.long, device=DEVICE).unsqueeze(0)
    except KeyError:
        print(f"Error: Prompt '{start_str}' contains characters not seen in training data.")
        return

    print(f"\nPrompt: '{start_str}' (Temp: {temperature})")
    print("-" * 40)
    print(start_str, end='', flush=True)

    for _ in range(max_new_tokens):
        context_cond = context[:, -BLOCK_SIZE:]
        
        with torch.no_grad():
            logits = model(context_cond)
        
        logits = logits[:, -1, :]
        logits = logits / temperature
        probs = torch.nn.functional.softmax(logits, dim=-1)
        idx_next = torch.multinomial(probs, num_samples=1)
        
        context = torch.cat((context, idx_next), dim=1)
        print(decode([idx_next.item()]), end='', flush=True)
    
    print("\n" + "-" * 40)

# Higher temperature for generalization/structure completion
generate(model, "Equity", temperature=0.8)
