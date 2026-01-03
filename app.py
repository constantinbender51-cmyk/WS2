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

# ==========================================
# 1. CONFIGURATION
# ==========================================
# Hardware Acceleration
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print(f"⚙️ Running on: {device}")

# Hyperparameters
HIDDEN_SIZE = 256
EMBEDDING_SIZE = 128
LEARNING_RATE = 0.005
MAX_LENGTH = 20
TARGET_DATASET_SIZE = 200000 
N_EPOCHS = 1 

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
# 2. MODEL ARCHITECTURE
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
# 3. HELPERS
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
    if error_type == 0: chars.insert(idx, random.choice(ALL_CHARS))
    elif error_type == 1: 
        if len(chars) > 2: del chars[idx]
    elif error_type == 2: 
        if idx < len(chars) - 1: chars[idx], chars[idx+1] = chars[idx+1], chars[idx]
    elif error_type == 3: chars[idx] = random.choice(ALL_CHARS)
    return "".join(chars)

def evaluate(encoder, decoder, word):
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
# 4. GITHUB UPLOAD LOGIC
# ==========================================
def get_pat_from_env():
    """Reads PAT from .env file manually to avoid dependencies"""
    env_path = ".env"
    if not os.path.exists(env_path):
        print("⚠️  .env file not found.")
        return None
    
    try:
        with open(env_path, "r") as f:
            for line in f:
                # Basic parsing for PAT=...
                if "PAT" in line and "=" in line:
                    key, value = line.split("=", 1)
                    if key.strip() == "PAT":
                        return value.strip().strip('"').strip("'")
    except Exception as e:
        print(f"⚠️  Error reading .env: {e}")
    return None

def upload_to_github(file_path, token):
    """Uploads file to GitHub using the contents API"""
    OWNER = "constantinbender51-cmyk"
    REPO = "Models"
    FILE_NAME = os.path.basename(file_path)
    API_URL = f"https://api.github.com/repos/{OWNER}/{REPO}/contents/{FILE_NAME}"
    
    print(f"\n🚀 Uploading {FILE_NAME} to GitHub ({OWNER}/{REPO})...")

    # 1. Read and Encode File
    with open(file_path, "rb") as f:
        content = base64.b64encode(f.read()).decode("utf-8")

    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github.v3+json",
        "Content-Type": "application/json"
    }

    # 2. Check if file exists (we need the SHA to update it)
    sha = None
    r_check = requests.get(API_URL, headers=headers)
    if r_check.status_code == 200:
        sha = r_check.json().get("sha")
        print("ℹ️  File exists, performing update...")
    
    # 3. Upload Payload
    data = {
        "message": f"Update AI Model: {FILE_NAME}",
        "content": content
    }
    if sha:
        data["sha"] = sha
        
    # 4. Perform Request
    r = requests.put(API_URL, headers=headers, data=json.dumps(data))
    
    if r.status_code in [200, 201]:
        print("✅ Upload successful!")
        print(f"🔗 Link: {r.json().get('html_url')}")
    else:
        print(f"❌ Upload failed: {r.status_code}")
        print(r.text)

# ==========================================
# 5. MAIN PIPELINE
# ==========================================
def main():
    # --- STEP 1: DOWNLOAD DATA ---
    print("\n[1/4] Downloading Vocabulary...")
    url = "https://raw.githubusercontent.com/first20hours/google-10000-english/refs/heads/master/20k.txt"
    try:
        r = requests.get(url, timeout=10)
        clean_words = [w.strip().lower() for w in r.text.splitlines() if w.isalpha() and len(w) >= 3]
        print(f"✅ Loaded {len(clean_words)} words.")
    except:
        print("❌ Download failed. Using fallback.")
        clean_words = ["apple", "mouse", "lock", "harry", "hurry", "local", "house"]

    # --- STEP 2: PREPARE DATASET ---
    print("\n[2/4] Generating Synthetic Typos...")
    training_data = []
    pairs_needed = TARGET_DATASET_SIZE // len(clean_words)
    for word in clean_words:
        training_data.append((word, word)) # Identity
        for _ in range(max(1, pairs_needed)):
            training_data.append((inject_noise(word), word))
    
    random.shuffle(training_data)
    training_data = training_data[:TARGET_DATASET_SIZE]
    print(f"✅ Created {len(training_data)} training pairs.")

    # --- STEP 3: TRAIN ---
    print("\n[3/4] Training Neural Network...")
    encoder = EncoderRNN(VOCAB_SIZE, HIDDEN_SIZE).to(device)
    decoder = DecoderRNN(HIDDEN_SIZE, VOCAB_SIZE).to(device)
    enc_opt = optim.Adam(encoder.parameters(), lr=LEARNING_RATE)
    dec_opt = optim.Adam(decoder.parameters(), lr=LEARNING_RATE)
    criterion = nn.NLLLoss()

    start_time = time.time()
    
    for i, (inp, target) in enumerate(training_data):
        input_tensor = tensor_from_word(inp)
        target_tensor = tensor_from_word(target)
        
        enc_hidden = encoder.initHidden()
        enc_opt.zero_grad()
        dec_opt.zero_grad()
        
        input_len = input_tensor.size(0)
        target_len = target_tensor.size(0)
        loss = 0
        
        for ei in range(input_len):
            _, enc_hidden = encoder(input_tensor[ei], enc_hidden)
            
        dec_input = torch.tensor([[SOS_token]], device=device)
        dec_hidden = enc_hidden
        
        use_tf = True if random.random() < 0.5 else False
        
        for di in range(target_len):
            dec_out, dec_hidden = decoder(dec_input, dec_hidden)
            loss += criterion(dec_out, target_tensor[di])
            if use_tf: dec_input = target_tensor[di]
            else:
                _, topi = dec_out.topk(1)
                dec_input = topi.squeeze().detach()
                if dec_input.item() == EOS_token: break
                
        loss.backward()
        enc_opt.step()
        dec_opt.step()
        
        if i % 10000 == 0 and i > 0:
            print(f"   Processed {i} samples...")

    print(f"✅ Training Complete ({time.time() - start_time:.0f}s).")

    # --- STEP 4: SAVE, PREDICT & UPLOAD ---
    print("\n[4/4] Saving, Testing & Uploading...")
    
    # Save Logic
    save_path = "spelling_model.pth"
    torch.save({
        'encoder': encoder.state_dict(),
        'decoder': decoder.state_dict()
    }, save_path)
    print(f"💾 Model saved locally to: {os.path.abspath(save_path)}")

    # GitHub Upload Logic
    pat = get_pat_from_env()
    if pat:
        upload_to_github(save_path, pat)
    else:
        print("⚠️  Skipping GitHub upload: 'PAT' not found in .env file.")

    # Inference on specific prompts
    test_words = ["Mouses", "Loc", "Harry"]
    print("\n--- 🤖 MODEL PREDICTIONS ---")
    
    for word in test_words:
        # Preprocessing: Model trains on lowercase
        clean_input = word.lower()
        output = evaluate(encoder, decoder, clean_input)
        
        # Postprocessing: Restore title case if input was title case
        final_output = output.title() if word[0].isupper() else output
        
        print(f"Input: '{word}' \t-> Correction: '{final_output}'")

if __name__ == "__main__":
    main()