import numpy as np
import requests
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GRU, TimeDistributed, Bidirectional, Embedding
from tensorflow.keras.preprocessing.sequence import pad_sequences
import random

# 1. Download and Prepare Data
url = "https://raw.githubusercontent.com/first20hours/google-10000-english/refs/heads/master/20k.txt"
print(f"Downloading data from {url}...")
try:
    response = requests.get(url)
    response.raise_for_status()
    all_words = response.text.splitlines()
except Exception as e:
    print(f"Error downloading data: {e}")
    # Fallback for demonstration if download fails
    all_words = ["the", "of", "and", "to", "in", "a", "is", "that", "for", "it"] * 10 

# Take the first 100 words
words = [w.strip().lower() for w in all_words[:100] if w.strip()]
print(f"Training on first {len(words)} words.")

# 2. Data Preprocessing & Noise Injection
# Create a character set including a padding token
chars = sorted(list(set("".join(words) + "abcdefghijklmnopqrstuvwxyz")))
char_to_int = {c: i + 1 for i, c in enumerate(chars)} # +1 for padding (0)
int_to_char = {i + 1: c for i, c in enumerate(chars)}
vocab_size = len(chars) + 1

# Determine max length for padding
max_len = max([len(w) for w in words]) + 2 # +2 buffer for insertions

def add_noise(word):
    """Introduces random spelling errors (substitution, deletion, insertion)."""
    if len(word) < 2: return word
    word = list(word)
    mutation = random.choice(['sub', 'del', 'ins', 'swap'])
    
    idx = random.randint(0, len(word) - 1)
    
    if mutation == 'sub':
        word[idx] = random.choice(chars)
    elif mutation == 'del':
        del word[idx]
    elif mutation == 'ins' and len(word) < max_len:
        word.insert(idx, random.choice(chars))
    elif mutation == 'swap' and idx < len(word) - 1:
        word[idx], word[idx+1] = word[idx+1], word[idx]
        
    return "".join(word)

# Generate synthetic training data
# We expand the dataset by creating multiple noisy versions of the 100 words
num_augmentations = 50
X_raw = []
y_raw = []

for word in words:
    # Add the word itself (identity mapping)
    X_raw.append(word)
    y_raw.append(word)
    # Add noisy versions
    for _ in range(num_augmentations):
        X_raw.append(add_noise(word))
        y_raw.append(word)

# Encode sequences
def encode_sequence(seq_list, max_len):
    encoded = [[char_to_int.get(c, 0) for c in w] for w in seq_list]
    return pad_sequences(encoded, maxlen=max_len, padding='post')

X_train = encode_sequence(X_raw, max_len)
y_train = encode_sequence(y_raw, max_len)

# One-hot encode targets for the Dense layer (sparse_categorical_crossentropy is also an option)
# but explicitly expanding dims is clearer for shape understanding here.
y_train = tf.keras.utils.to_categorical(y_train, num_classes=vocab_size)

print(f"Dataset shape: X={X_train.shape}, y={y_train.shape}")

# 3. Build the GRU Model
# Simple Sequence-to-Sequence approach using TimeDistributed Dense layer
# Note: A real production system would use an Encoder-Decoder with Attention.
# This architecture assumes input and output length are aligned (handled by padding).
model = Sequential([
    Embedding(input_dim=vocab_size, output_dim=64, input_length=max_len, mask_zero=True),
    Bidirectional(GRU(128, return_sequences=True)),
    GRU(128, return_sequences=True),
    TimeDistributed(Dense(vocab_size, activation='softmax'))
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# Train
print("Training model (this may take a moment)...")
model.fit(X_train, y_train, epochs=30, batch_size=64, verbose=1, validation_split=0.1)

# 4. Inference Function
def correct_spelling(word_list):
    encoded = encode_sequence(word_list, max_len)
    predictions = model.predict(encoded, verbose=0)
    
    corrected_words = []
    for i, pred in enumerate(predictions):
        # Argmax to get the most likely character index
        predicted_indices = np.argmax(pred, axis=-1)
        # Convert back to characters
        result = ""
        for idx in predicted_indices:
            if idx != 0: # Skip padding
                result += int_to_char[idx]
        corrected_words.append(result)
    return corrected_words

# 5. Test Prompt
# Generate 5 misspelled words from the top 100 list manually or randomly
# For clarity, let's pick 5 known words from the list and mess them up
test_targets = [words[0], words[10], words[20], words[30], words[50]] 
misspelled_inputs = [add_noise(w) for w in test_targets]

# Or hardcode specific test cases if they match the vocabulary
# Note: The model only knows the first 100 words.
hardcoded_tests = ["thier", "abut", "wich", "peopel", "firt"] # Examples that might resemble top 100 words

print("\n--- Testing Model ---")
corrections = correct_spelling(misspelled_inputs)

for wrong, correct_pred, target in zip(misspelled_inputs, corrections, test_targets):
    print(f"Input: {wrong:<15} -> Pred: {correct_pred:<15} (Target: {target})")