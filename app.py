import torch
import torch.nn as nn
import torch.optim as optim
import math
import os

# --- Configuration ---
FILE_PATH = 'text.txt'
BATCH_SIZE = 32
BLOCK_SIZE = 64      # Context window size
MAX_ITERS = 500      # Training iterations
LEARNING_RATE = 3e-4
EMBED_DIM = 128
NUM_HEADS = 4
NUM_LAYERS = 4
DROPOUT = 0.1
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

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
    # Ensure we don't go out of bounds
    max_idx = len(data_src) - BLOCK_SIZE - 1
    if max_idx <= 0:
        # Fallback for very short text files
        ix = torch.randint(0, len(data_src) - 1, (BATCH_SIZE,))
        x = torch.stack([data_src[i:i+1] for i in ix]) # Context of 1
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
        return x + self.pe[:x.size(0), :].unsqueeze(1)

class TransformerModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, nhead, num_layers, dropout):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_encoder = PositionalEncoding(embed_dim, max_len=BLOCK_SIZE)
        
        # Using TransformerEncoder with causal mask acts as a Decoder for GPT-style generation
        encoder_layers = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=nhead, dim_feedforward=embed_dim*4, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        
        self.decoder = nn.Linear(embed_dim, vocab_size)
        self.embed_dim = embed_dim

    def forward(self, src, src_mask=None):
        # src shape: [Batch, Seq] -> Transpose for nn.Transformer [Seq, Batch, Embed]
        src = self.embedding(src) * math.sqrt(self.embed_dim)
        src = src.permute(1, 0, 2) 
        src = self.pos_encoder(src)
        
        if src_mask is None:
            # Generate causal mask
            sz = src.size(0)
            src_mask = nn.Transformer.generate_square_subsequent_mask(sz).to(DEVICE)

        output = self.transformer_encoder(src, mask=src_mask)
        # Transpose back to [Batch, Seq, Embed] for Linear layer
        output = output.permute(1, 0, 2)
        output = self.decoder(output)
        return output

# --- 3. Training ---

model = TransformerModel(vocab_size, EMBED_DIM, NUM_HEADS, NUM_LAYERS, DROPOUT).to(DEVICE)
optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
criterion = nn.CrossEntropyLoss()

print(f"Training on {DEVICE} for {MAX_ITERS} iterations...")
model.train()

for iter in range(MAX_ITERS):
    xb, yb = get_batch('train')

    # Forward pass
    logits = model(xb)
    
    # Reshape for loss calculation: [Batch * Seq, Vocab]
    B, T, C = logits.shape
    logits = logits.view(B*T, C)
    targets = yb.view(B*T)
    
    loss = criterion(logits, targets)

    # Backward pass
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

    if iter % 100 == 0:
        print(f"Iter {iter}: Loss {loss.item():.4f}")

print("Training complete.")

# --- 4. Generation ---

def generate(model, start_str, max_new_tokens=100):
    model.eval()
    # Encode start string
    context = torch.tensor(encode(start_str), dtype=torch.long, device=DEVICE).unsqueeze(0) # [1, Seq]
    
    print(f"\nPrompt: '{start_str}'")
    print("-" * 40)
    print(start_str, end='', flush=True)

    for _ in range(max_new_tokens):
        # Crop context to block size if it gets too long
        context_cond = context[:, -BLOCK_SIZE:]
        
        with torch.no_grad():
            logits = model(context_cond)
        
        # Focus only on the last time step
        logits = logits[:, -1, :] # [Batch, Vocab]
        
        # Apply softmax to get probabilities
        probs = torch.nn.functional.softmax(logits, dim=-1)
        
        # Sample from the distribution
        idx_next = torch.multinomial(probs, num_samples=1)
        
        # Append to current sequence
        context = torch.cat((context, idx_next), dim=1)
        
        # Print the new character
        print(decode([idx_next.item()]), end='', flush=True)
    
    print("\n" + "-" * 40)

# Run the prompt
generate(model, "Equity")
