import torch
import torch.nn as nn
import torch.optim as optim
import random
import string
import requests
import time
import sys

# ==========================================
# 1. CONFIGURATION
# ==========================================
# Detect Hardware
if torch.cuda.is_available():
    device = torch.device("cuda")
    print("✅ Training on GPU (CUDA)")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
    print("✅ Training on Mac GPU (MPS)")
else:
    device = torch.device("cpu")
    print("⚠️ Training on CPU")

# Hyperparameters
HIDDEN_SIZE = 256
EMBEDDING_SIZE = 128
LEARNING_RATE = 0.005
MAX_LENGTH = 20
TARGET_DATASET_SIZE = 500000  # As requested: 500k pairs
BATCH_SIZE = 1 # SGD for simplicity in this architecture

# Vocabulary Setup
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
# 2. DATA PREPARATION
# ==========================================
def fetch_words():
    """Downloads the 20k word list."""
    url = "https://raw.githubusercontent.com/first20hours/google-10000-english/refs/heads/master/20k.txt"
    print(f"📥 Downloading words from {url}...")
    try:
        r = requests.get(url, timeout=20)
        if r.status_code == 200:
            words = r.text.splitlines()
            # Clean data: alpha only, 3+ chars
            clean_words = [w.strip().lower() for w in words if w.isalpha() and len(w) >= 3]
            print(f"✅ Loaded {len(clean_words)} unique words.")
            return clean_words
    except Exception as e:
        print(f"❌ Error downloading: {e}")
        sys.exit(1)

def inject_noise(word):
    """Generates a typo: Insert, Delete, Swap, or Replace."""
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

def create_dataset(words, target_size):
    """Creates a list of (misspelled, correct) pairs."""
    print(f"🔨 Generating {target_size} training pairs...")
    dataset = []
    
    # Calculate how many typos per word we need to hit target_size
    # e.g., 500,000 / 20,000 words = ~25 pairs per word
    pairs_per_word = max(1, target_size // len(words))
    
    for word in words:
        # 1. Add the correct word (identity mapping) - important for stability
        dataset.append((word, word))
        
        # 2. Add noisy versions
        for _ in range(pairs_per_word):
            typo = inject_noise(word)
            dataset.append((typo, word))
            
    # Trim or fill to match exact target if needed
    if len(dataset) > target_size:
        dataset = dataset[:target_size]
        
    random.shuffle(dataset)
    print(f"✅ Dataset created with {len(dataset)} pairs.")
    return dataset

def tensor_from_word(word):
    indexes = [char_to_index[char] for char in word if char in char_to_index]
    indexes.append(EOS_token)
    return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)

# ==========================================
# 3. MODEL ARCHITECTURE
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
# 4. MAIN TRAINING LOOP
# ==========================================
def train_model():
    # 1. Prepare Data
    words = fetch_words()
    training_data = create_dataset(words, TARGET_DATASET_SIZE)
    
    # 2. Initialize Model
    print("🚀 Initializing Model...")
    encoder = EncoderRNN(VOCAB_SIZE, HIDDEN_SIZE).to(device)
    decoder = DecoderRNN(HIDDEN_SIZE, VOCAB_SIZE).to(device)
    
    encoder_optimizer = optim.Adam(encoder.parameters(), lr=LEARNING_RATE)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=LEARNING_RATE)
    criterion = nn.NLLLoss()
    
    # 3. Train
    print(f"🚀 Starting training on {len(training_data)} samples...")
    start_time = time.time()
    total_loss = 0
    print_every = 5000
    
    for i, (input_word, target_word) in enumerate(training_data, 1):
        input_tensor = tensor_from_word(input_word)
        target_tensor = tensor_from_word(target_word)
        
        encoder_hidden = encoder.initHidden()
        encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()
        
        input_length = input_tensor.size(0)
        target_length = target_tensor.size(0)
        
        loss = 0
        
        # Encoder Pass
        for ei in range(input_length):
            _, encoder_hidden = encoder(input_tensor[ei], encoder_hidden)
            
        # Decoder Pass
        decoder_input = torch.tensor([[SOS_token]], device=device)
        decoder_hidden = encoder_hidden
        
        # Teacher Forcing: 50% chance to use correct previous letter
        use_teacher_forcing = True if random.random() < 0.5 else False
        
        for di in range(target_length):
            decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
            loss += criterion(decoder_output, target_tensor[di])
            
            if use_teacher_forcing:
                decoder_input = target_tensor[di]
            else:
                _, topi = decoder_output.topk(1)
                decoder_input = topi.squeeze().detach()
                if decoder_input.item() == EOS_token:
                    break
                    
        loss.backward()
        encoder_optimizer.step()
        decoder_optimizer.step()
        
        total_loss += (loss.item() / target_length)
        
        # Logging
        if i % print_every == 0:
            avg_loss = total_loss / print_every
            elapsed = time.time() - start_time
            print(f"Step {i}/{len(training_data)} | Loss: {avg_loss:.4f} | Time: {elapsed:.0f}s")
            total_loss = 0

    print("✨ Training Complete.")
    
    # 4. Save Model
    print("💾 Saving model to 'spelling_model.pth'...")
    torch.save({
        'encoder_state': encoder.state_dict(),
        'decoder_state': decoder.state_dict(),
        'hidden_size': HIDDEN_SIZE,
        'vocab_size': VOCAB_SIZE,
        'char_to_index': char_to_index,
        'index_to_char': index_to_char
    }, 'spelling_model.pth')
    print("✅ Model Saved.")

if __name__ == "__main__":
    train_model()