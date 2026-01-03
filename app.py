import torch
import torch.nn as nn
import torch.optim as optim
import random
import string
import requests
import time
import os
import sys
import base64
import json
import multiprocessing
import math
from torch.utils.data import Dataset, DataLoader

# ==========================================
# 1. CONFIGURATION (STABILITY MODE)
# ==========================================
torch.set_num_threads(4)

if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"⚙️ Running on: CUDA ({torch.cuda.get_device_name(0)})")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
    print("⚙️ Running on: MPS (Apple Silicon)")
else:
    device = torch.device("cpu")
    print(f"⚙️ Running on: CPU")

# Hyperparameters
HIDDEN_SIZE = 256
EMBEDDING_SIZE = 128
LEARNING_RATE = 0.001     
GRADIENT_CLIP = 1.0       # Tighter clip for stability
MAX_LENGTH = 20
TARGET_DATASET_SIZE = 300000 
BATCH_SIZE = 256 
N_EPOCHS = 5

# Vocabulary
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

# ==========================================
# 2. DATASET & GENERATION
# ==========================================
def inject_noise(word):
    if len(word) < 3: return word
    chars = list(word)
    error_type = random.randint(0, 3)
    idx = random.randint(0, len(chars) - 1)
    
    if error_type == 0: 
        chars.insert(idx, random.choice(string.ascii_lowercase))
    elif error_type == 1: 
        if len(chars) > 2: del chars[idx]
    elif error_type == 2: 
        if idx < len(chars) - 1: chars[idx], chars[idx+1] = chars[idx+1], chars[idx]
    elif error_type == 3: 
        chars[idx] = random.choice(string.ascii_lowercase)
    return "".join(chars)

def generate_chunk(args):
    words_chunk, num_typos = args
    pairs = []
    for word in words_chunk:
        pairs.append((word, word)) 
        for _ in range(num_typos): 
            pairs.append((inject_noise(word), word))
    return pairs

class SpellingDataset(Dataset):
    def __init__(self, raw_data):
        self.data = raw_data
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        inp_word, target_word = self.data[idx]
        return (self.word_to_tensor(inp_word), self.word_to_tensor(target_word))

    def word_to_tensor(self, word):
        indexes = [char_to_index.get(c, PAD_token) for c in word]
        indexes.append(EOS_token)
        if len(indexes) < MAX_LENGTH:
            indexes += [PAD_token] * (MAX_LENGTH - len(indexes))
        else:
            indexes = indexes[:MAX_LENGTH]
        return torch.tensor(indexes, dtype=torch.long)

# ==========================================
# 3. MODEL ARCHITECTURE (STABLE)
# ==========================================
def init_weights(m):
    """Initialize weights to prevent NaN start"""
    for name, param in m.named_parameters():
        if 'weight' in name:
            nn.init.normal_(param.data, mean=0, std=0.01)
        else:
            nn.init.constant_(param.data, 0)

class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(EncoderRNN, self).__init__()
        self.embedding = nn.Embedding(input_size, EMBEDDING_SIZE)
        self.gru = nn.GRU(EMBEDDING_SIZE, hidden_size, batch_first=True)
        self.apply(init_weights) # Init weights

    def forward(self, input):
        embedded = self.embedding(input) 
        output, hidden = self.gru(embedded)
        return output, hidden

class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(DecoderRNN, self).__init__()
        self.embedding = nn.Embedding(output_size, EMBEDDING_SIZE)
        self.gru = nn.GRU(EMBEDDING_SIZE, hidden_size, batch_first=True)
        self.out = nn.Linear(hidden_size, output_size)
        # NOTE: Removed LogSoftmax. We output raw logits for CrossEntropyLoss
        self.apply(init_weights) # Init weights

    def forward(self, input, hidden):
        output = self.embedding(input)
        output = torch.relu(output)
        output, hidden = self.gru(output, hidden)
        output = self.out(output[:, 0]) # Raw logits
        return output, hidden

# ==========================================
# 4. GITHUB UTILS
# ==========================================
def get_pat_from_env():
    if not os.path.exists(".env"): return None
    try:
        with open(".env", "r") as f:
            for line in f:
                if "PAT" in line and "=" in line:
                    return line.split("=", 1)[1].strip().strip('"\'')
    except: pass
    return None

def upload_to_github(file_path, token):
    OWNER = "constantinbender51-cmyk"
    REPO = "Models"
    FILE_NAME = os.path.basename(file_path)
    API_URL = f"https://api.github.com/repos/{OWNER}/{REPO}/contents/{FILE_NAME}"
    
    print(f"\n🚀 Uploading {FILE_NAME} to GitHub...")
    with open(file_path, "rb") as f: content = base64.b64encode(f.read()).decode("utf-8")
    
    headers = {"Authorization": f"Bearer {token}", "Accept": "application/vnd.github.v3+json"}
    
    try:
        r_check = requests.get(API_URL, headers=headers)
        sha = r_check.json().get("sha") if r_check.status_code == 200 else None
    except: sha = None
    
    data = {"message": f"Update Model: {FILE_NAME}", "content": content}
    if sha: data["sha"] = sha
        
    r = requests.put(API_URL, headers=headers, data=json.dumps(data))
    if r.status_code in [200, 201]: print(f"✅ Upload successful: {r.json().get('html_url')}")
    else: print(f"❌ Upload failed: {r.status_code} {r.text}")

# ==========================================
# 5. MAIN PIPELINE
# ==========================================
def main():
    # --- STEP 1: DOWNLOAD ---
    print("\n[1/4] Downloading Vocabulary...")
    url = "https://raw.githubusercontent.com/first20hours/google-10000-english/refs/heads/master/20k.txt"
    try:
        r = requests.get(url, timeout=10)
        clean_words = [w.strip().lower() for w in r.text.splitlines() if w.isalpha() and len(w) >= 3]
    except:
        clean_words = []
    
    clean_words.extend(["mouse", "lock", "local", "harry", "hurry"])
    clean_words = list(set(clean_words))
    vocab_len = len(clean_words)
    print(f"✅ Vocabulary Base: {vocab_len} words")

    # --- STEP 2: DATA GEN ---
    num_cores = multiprocessing.cpu_count()
    gen_cores = min(num_cores, 8) 
    
    pairs_per_word = math.ceil(TARGET_DATASET_SIZE / vocab_len)
    typos_per_word = max(1, pairs_per_word - 1)
    
    print(f"\n[2/4] Generating Data ({gen_cores} cores)...")
    
    chunk_size = vocab_len // gen_cores
    chunks = []
    for i in range(0, vocab_len, chunk_size):
        chunks.append((clean_words[i:i + chunk_size], typos_per_word))
    
    with multiprocessing.Pool(processes=gen_cores) as pool:
        results = pool.map(generate_chunk, chunks)
    
    training_data = [item for sublist in results for item in sublist]
    random.shuffle(training_data)
    if len(training_data) > TARGET_DATASET_SIZE:
        training_data = training_data[:TARGET_DATASET_SIZE]
    
    print(f"✅ Generated {len(training_data)} pairs.")

    # --- STEP 3: TRAINING (STABLE LOSS) ---
    print(f"\n[3/4] Training (Batch Size: {BATCH_SIZE})...")
    
    dataset = SpellingDataset(training_data)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

    encoder = EncoderRNN(VOCAB_SIZE, HIDDEN_SIZE).to(device)
    decoder = DecoderRNN(HIDDEN_SIZE, VOCAB_SIZE).to(device)
    
    enc_opt = optim.Adam(encoder.parameters(), lr=LEARNING_RATE)
    dec_opt = optim.Adam(decoder.parameters(), lr=LEARNING_RATE)
    
    # FIX: CrossEntropyLoss is numerically stable for Raw Logits
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_token)

    for epoch in range(N_EPOCHS):
        total_loss = 0
        batches_count = 0
        
        for i, (input_tensor, target_tensor) in enumerate(dataloader):
            input_tensor = input_tensor.to(device)
            target_tensor = target_tensor.to(device)
            
            enc_opt.zero_grad()
            dec_opt.zero_grad()
            
            _, hidden = encoder(input_tensor)
            
            dec_input = torch.tensor([[SOS_token]] * input_tensor.size(0), device=device)
            dec_hidden = hidden
            
            loss = 0
            use_tf = True if random.random() < 0.5 else False
            
            for t in range(MAX_LENGTH):
                dec_out, dec_hidden = decoder(dec_input, dec_hidden)
                
                # loss += criterion(logits, target_class_index)
                if t < target_tensor.size(1):
                     loss += criterion(dec_out, target_tensor[:, t])
                
                if use_tf:
                    if t < target_tensor.size(1):
                        dec_input = target_tensor[:, t].unsqueeze(1)
                else:
                    _, topi = dec_out.topk(1)
                    dec_input = topi
            
            loss.backward()
            
            # Clip Gradients to prevent NaN
            torch.nn.utils.clip_grad_norm_(encoder.parameters(), GRADIENT_CLIP)
            torch.nn.utils.clip_grad_norm_(decoder.parameters(), GRADIENT_CLIP)
            
            enc_opt.step()
            dec_opt.step()
            
            total_loss += loss.item()
            batches_count += 1
            
            if i % 100 == 0:
                avg_loss = total_loss / (batches_count if batches_count > 0 else 1)
                print(f"Epoch {epoch+1} | Batch {i}/{len(dataloader)} | Avg Loss: {avg_loss/MAX_LENGTH:.4f}")

    # --- STEP 4: SAVE & UPLOAD ---
    save_path = "spelling_model.pth"
    torch.save({'encoder': encoder.state_dict(), 'decoder': decoder.state_dict()}, save_path)
    print(f"\n[4/4] Model saved locally to {save_path}")

    pat = get_pat_from_env()
    if pat: 
        upload_to_github(save_path, pat)
    else:
        print("⚠️  Skipping Upload: .env file missing or PAT not found.")

    # --- TEST ---
    def predict(word):
        encoder.eval()
        decoder.eval()
        with torch.no_grad():
            inp = dataset.word_to_tensor(word.lower()).unsqueeze(0).to(device)
            _, hidden = encoder(inp)
            dec_input = torch.tensor([[SOS_token]], device=device)
            res = []
            for _ in range(MAX_LENGTH):
                out, hidden = decoder(dec_input, hidden)
                
                # Since we use CrossEntropyLoss now, output is raw logits.
                # Use Softmax to get probabilities, or just topk on logits (same result)
                _, topi = out.topk(1)
                
                if topi.item() == EOS_token: break
                res.append(index_to_char[topi.item()])
                dec_input = topi
            return "".join(res)

    print("\n--- TEST RESULTS ---")
    for w in ["Mouses", "Loc", "Harry", "intelgnec"]:
        print(f"Input: {w:<10} -> Correction: {predict(w).title()}")

if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()