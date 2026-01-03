import numpy as np
import requests
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GRU, TimeDistributed, Bidirectional, Embedding, Dropout
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.regularizers import l1_l2
import random

# --- CONFIGURATION & PARAMETERS ---
DATA_URL = "https://raw.githubusercontent.com/first20hours/google-10000-english/refs/heads/master/20k.txt"
NUM_WORDS_TO_LOAD = 100       # How many words from the top of the list to train on
AUGMENTATIONS_PER_WORD = 50   # How many misspelled versions to generate per word
VALIDATION_SPLIT = 0.1        # Portion of data to use for validation
BATCH_SIZE = 512
EPOCHS = 300
EMBEDDING_DIM = 64
GRU_UNITS = 128
PADDING_BUFFER = 2            # Extra space in sequence length for insertions
RANDOM_SEED = 42

# Regularization Hyperparameters
DROPOUT_RATE = 0.5       # Probability of dropping a unite
L1_REG = 1e-3          # L1 regularization factor
L2_REG = 1e-2                 # L2 regularization factor

# Set seeds for reproducibility
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

# --- 1. DATA LOADING ---
print(f"Downloading data from {DATA_URL}...")
try:
    response = requests.get(DATA_URL)
    response.raise_for_status()
    all_words = response.text.splitlines()
except Exception as e:
    print(f"Error downloading data: {e}")
    # Fallback dataset
    all_words = ["the", "of", "and", "to", "in", "a", "is", "that", "for", "it"] * 10

# Filter and select top words
words = [w.strip().lower() for w in all_words[:NUM_WORDS_TO_LOAD] if w.strip()]
print(f"Training on first {len(words)} words.")

# --- 2. PREPROCESSING & NOISE INJECTION ---
# Create character mappings
chars = sorted(list(set("".join(words) + "abcdefghijklmnopqrstuvwxyz")))
char_to_int = {c: i + 1 for i, c in enumerate(chars)} # +1 for padding (0)
int_to_char = {i + 1: c for i, c in enumerate(chars)}
vocab_size = len(chars) + 1

# Determine max sequence length
max_len = max([len(w) for w in words]) + PADDING_BUFFER

def add_noise(word):
    """Introduces random spelling errors (substitution, deletion, insertion)."""
    if len(word) < 2: return word
    word_list = list(word)
    mutation = random.choice(['sub', 'del', 'ins', 'swap'])
    idx = random.randint(0, len(word_list) - 1)
    
    if mutation == 'sub':
        word_list[idx] = random.choice(chars)
    elif mutation == 'del':
        del word_list[idx]
    elif mutation == 'ins' and len(word_list) < max_len:
        word_list.insert(idx, random.choice(chars))
    elif mutation == 'swap' and idx < len(word_list) - 1:
        word_list[idx], word_list[idx+1] = word_list[idx+1], word_list[idx]
        
    return "".join(word_list)

# Generate synthetic training data
X_raw = []
y_raw = []

for word in words:
    # Add the clean word (identity mapping)
    X_raw.append(word)
    y_raw.append(word)
    # Add noisy versions
    for _ in range(AUGMENTATIONS_PER_WORD):
        X_raw.append(add_noise(word))
        y_raw.append(word)

# Encode sequences
def encode_sequence(seq_list, max_len):
    encoded = [[char_to_int.get(c, 0) for c in w] for w in seq_list]
    return pad_sequences(encoded, maxlen=max_len, padding='post')

X_train = encode_sequence(X_raw, max_len)
y_train_seq = encode_sequence(y_raw, max_len)

# One-hot encode targets
y_train = tf.keras.utils.to_categorical(y_train_seq, num_classes=vocab_size)

print(f"Dataset shape: X={X_train.shape}, y={y_train.shape}")

# --- 3. MODEL ARCHITECTURE ---
# Added Dropout and Regularization to combat overfitting
reg = l1_l2(l1=L1_REG, l2=L2_REG)

model = Sequential([
    Embedding(input_dim=vocab_size, output_dim=EMBEDDING_DIM, input_length=max_len, mask_zero=True),
    Dropout(DROPOUT_RATE),
    
    Bidirectional(GRU(GRU_UNITS, return_sequences=True, kernel_regularizer=reg)),
    Dropout(DROPOUT_RATE),
    
    GRU(GRU_UNITS, return_sequences=True, kernel_regularizer=reg),
    Dropout(DROPOUT_RATE),
    
    TimeDistributed(Dense(vocab_size, activation='softmax', kernel_regularizer=reg))
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# --- 4. TRAINING ---
print("Starting training...")
model.fit(
    X_train, 
    y_train, 
    epochs=EPOCHS, 
    batch_size=BATCH_SIZE, 
    verbose=1, 
    validation_split=VALIDATION_SPLIT
)

# --- 5. INFERENCE & TESTING ---
def correct_spelling(word_list):
    encoded = encode_sequence(word_list, max_len)
    predictions = model.predict(encoded, verbose=0)
    
    corrected_words = []
    for i, pred in enumerate(predictions):
        predicted_indices = np.argmax(pred, axis=-1)
        result = ""
        for idx in predicted_indices:
            if idx != 0: # Skip padding
                result += int_to_char[idx]
        corrected_words.append(result)
    return corrected_words

print("\n--- Testing Model with 5 Generated Misspelled Words ---")

# Pick 5 random words from our training vocabulary to distort
test_targets = random.sample(words, 5)
misspelled_inputs = [add_noise(w) for w in test_targets]

corrections = correct_spelling(misspelled_inputs)

print(f"{'Input (Misspelled)':<20} | {'Prediction':<20} | {'Target':<20}")
print("-" * 66)
for wrong, correct_pred, target in zip(misspelled_inputs, corrections, test_targets):
    print(f"{wrong:<20} | {correct_pred:<20} | {target:<20}")