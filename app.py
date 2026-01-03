import numpy as np
import requests
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Embedding, Dropout
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.regularizers import l1_l2
import random
import matplotlib.pyplot as plt
import os
import base64
from dotenv import load_dotenv
from datetime import datetime

# Load environment variables (specifically PAT)
load_dotenv()

# --- CONFIGURATION & PARAMETERS ---
DATA_URL = "https://raw.githubusercontent.com/first20hours/google-10000-english/refs/heads/master/20k.txt"
NUM_WORDS_TO_LOAD = 100       # Reverted to 100
AUGMENTATIONS_PER_WORD = 200   
VALIDATION_SPLIT = 0.2
BATCH_SIZE = 1024              # Changed to 512
EPOCHS = 10                 # Increased epochs for autoregressive convergence
LATENT_DIM = 512              # Internal state size
EMBEDDING_DIM = 256
RANDOM_SEED = 42

# Regularization Hyperparameters
L1_REG = 1e-1
L2_REG = 1e-4

# Special Tokens
SOS_TOKEN = '\t' # Start of Sequence
EOS_TOKEN = '\n' # End of Sequence

# Regularization
DROPOUT_RATE = 0.3

# Set seeds
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
    print(f"Error downloading: {e}. Using fallback.")
    all_words = ["the", "quick", "brown", "fox", "jumps", "over", "lazy", "dog"] * 20

# Select words
words = [w.strip().lower() for w in all_words[:NUM_WORDS_TO_LOAD] if w.strip()]
print(f"Training on first {len(words)} words.")

# --- 2. PREPROCESSING & TOKENIZATION ---

# Create vocabulary including special tokens
chars = sorted(list(set("".join(words) + "abcdefghijklmnopqrstuvwxyz" + SOS_TOKEN + EOS_TOKEN)))
char_to_int = {c: i for i, c in enumerate(chars)} # 0 is now a valid index for a character
int_to_char = {i: c for i, c in enumerate(chars)}
vocab_size = len(chars)

# Determine max sequence length (add space for SOS/EOS)
max_len = max([len(w) for w in words]) + 4

def add_noise(word):
    """Introduces noise for the encoder input."""
    if len(word) < 2: return word
    word_list = list(word)
    mutation = random.choice(['sub', 'del', 'ins', 'swap', 'none']) # Added 'none' for identity mapping
    idx = random.randint(0, len(word_list) - 1)
    
    if mutation == 'sub':
        word_list[idx] = random.choice(chars[:26]) # Only letters
    elif mutation == 'del':
        del word_list[idx]
    elif mutation == 'ins' and len(word_list) < max_len:
        word_list.insert(idx, random.choice(chars[:26]))
    elif mutation == 'swap' and idx < len(word_list) - 1:
        word_list[idx], word_list[idx+1] = word_list[idx+1], word_list[idx]
        
    return "".join(word_list)

# Prepare lists for Encoder Input, Decoder Input, and Decoder Target
encoder_input_data = []
decoder_input_data = []
decoder_target_data = []

print("Generating synthetic data...")
for word in words:
    for _ in range(AUGMENTATIONS_PER_WORD):
        # 1. Encoder Input: The noisy word
        noisy_word = add_noise(word)
        encoder_input_data.append([char_to_int[c] for c in noisy_word])
        
        # 2. Decoder Input: SOS + correct word
        dec_in = [char_to_int[SOS_TOKEN]] + [char_to_int[c] for c in word]
        decoder_input_data.append(dec_in)
        
        # 3. Decoder Target: correct word + EOS
        dec_target = [char_to_int[c] for c in word] + [char_to_int[EOS_TOKEN]]
        decoder_target_data.append(dec_target)

# Pad Sequences
encoder_input_data = pad_sequences(encoder_input_data, maxlen=max_len, padding='post')
decoder_input_data = pad_sequences(decoder_input_data, maxlen=max_len, padding='post')
decoder_target_data = pad_sequences(decoder_target_data, maxlen=max_len, padding='post')

# One-hot encode targets for Softmax
decoder_target_one_hot = to_categorical(decoder_target_data, num_classes=vocab_size)

print(f"Encoder Input Shape: {encoder_input_data.shape}")
print(f"Decoder Input Shape: {decoder_input_data.shape}")
print(f"Decoder Target Shape: {decoder_target_one_hot.shape}")

# --- 3. MODEL ARCHITECTURE (Seq2Seq) ---

# Define Regularizer
reg = l1_l2(l1=L1_REG, l2=L2_REG)

# --- Encoder ---
encoder_inputs = Input(shape=(None,), name="Encoder_Input")

# Separate definition from call to reuse layer later
enc_emb_layer = Embedding(vocab_size, EMBEDDING_DIM, mask_zero=True, name="Encoder_Embedding")
encoder_embedding = enc_emb_layer(encoder_inputs)

encoder_dropout = Dropout(DROPOUT_RATE)(encoder_embedding)

# return_state=True to get the internal state vectors (h, c)
encoder_lstm = LSTM(LATENT_DIM, return_state=True, dropout=DROPOUT_RATE, kernel_regularizer=reg, name="Encoder_LSTM")
encoder_outputs, state_h, state_c = encoder_lstm(encoder_dropout)

# We discard `encoder_outputs` and only keep the states.
encoder_states = [state_h, state_c]

# --- Decoder ---
decoder_inputs = Input(shape=(None,), name="Decoder_Input")

# Separate definition from call to reuse layer later in inference
dec_emb_layer = Embedding(vocab_size, EMBEDDING_DIM, mask_zero=True, name="Decoder_Embedding")
decoder_embedding = dec_emb_layer(decoder_inputs)

decoder_dropout = Dropout(DROPOUT_RATE)(decoder_embedding)

# return_sequences=True to output the whole sequence
# return_state=True is needed for inference later, though ignored during training
decoder_lstm = LSTM(LATENT_DIM, return_sequences=True, return_state=True, dropout=DROPOUT_RATE, kernel_regularizer=reg, name="Decoder_LSTM")
decoder_outputs, _, _ = decoder_lstm(decoder_dropout, initial_state=encoder_states)

decoder_dense = Dense(vocab_size, activation='softmax', kernel_regularizer=reg, name="Decoder_Output")
decoder_outputs = decoder_dense(decoder_outputs)

# --- Define Training Model ---
model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# --- 4. TRAINING ---
print("Starting training...")
# Note: We pass TWO inputs: [encoder_input_data, decoder_input_data]
history = model.fit(
    [encoder_input_data, decoder_input_data],
    decoder_target_one_hot,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    validation_split=VALIDATION_SPLIT
)

# --- 5. INFERENCE SETUP (Sampling Models) ---
# To generate text, we need to separate the Encoder and Decoder so we can loop manually.

# Encoder Inference Model
encoder_model = Model(encoder_inputs, encoder_states)

# Decoder Inference Model
decoder_state_input_h = Input(shape=(LATENT_DIM,))
decoder_state_input_c = Input(shape=(LATENT_DIM,))
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]

# Reuse the embedding layer defined above (dec_emb_layer), not the tensor
dec_emb2 = dec_emb_layer(decoder_inputs) 

# Reuse the LSTM layer
dec_outputs2, state_h2, state_c2 = decoder_lstm(dec_emb2, initial_state=decoder_states_inputs)
decoder_states2 = [state_h2, state_c2]

# Reuse the Dense layer
decoder_outputs2 = decoder_dense(dec_outputs2)

decoder_model = Model(
    [decoder_inputs] + decoder_states_inputs,
    [decoder_outputs2] + decoder_states2
)

def decode_sequence(input_seq):
    # Encode the input as state vectors.
    states_value = encoder_model.predict(input_seq, verbose=0)

    # Generate empty target sequence of length 1.
    target_seq = np.zeros((1, 1))
    # Populate the first character of target sequence with the start character.
    target_seq[0, 0] = char_to_int[SOS_TOKEN]

    # Sampling loop for a batch of sequences
    # (to simplify, here we assume a batch of size 1).
    stop_condition = False
    decoded_sentence = ''
    
    while not stop_condition:
        output_tokens, h, c = decoder_model.predict([target_seq] + states_value, verbose=0)

        # Sample a token
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_char = int_to_char[sampled_token_index]

        if sampled_char != EOS_TOKEN:
            decoded_sentence += sampled_char

        # Exit condition: either hit max length or find stop character.
        if (sampled_char == EOS_TOKEN or
           len(decoded_sentence) > max_len):
            stop_condition = True

        # Update the target sequence (of length 1).
        target_seq = np.zeros((1, 1))
        target_seq[0, 0] = sampled_token_index

        # Update states
        states_value = [h, c]

    return decoded_sentence

# --- 6. PLOTTING & GITHUB UPLOAD ---
def upload_plot_to_github(history):
    print("\n--- Generating and Uploading Plot ---")
    
    # Generate Plot
    plt.figure(figsize=(6, 4))
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.plot(history.history['loss'], label='Training Loss')
    plt.title('Encoder-Decoder Loss Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    filename = f"seq2seq_loss_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    plt.savefig(filename, dpi=60)
    plt.close()
    print(f"Plot saved locally as {filename}")

    # Prepare for Upload
    repo_owner = "constantinbender51-cmyk"
    repo_name = "models"
    github_token = os.getenv("PAT")
    
    if not github_token:
        print("Error: PAT not found in .env. Skipping upload.")
        return

    api_url = f"https://api.github.com/repos/{repo_owner}/{repo_name}/contents/{filename}"
    
    with open(filename, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode('utf-8')

    headers = {
        "Authorization": f"Bearer {github_token}",
        "Accept": "application/vnd.github+json"
    }
    
    payload = {
        "message": f"Upload seq2seq plot {filename}",
        "content": encoded_string
    }

    try:
        response = requests.put(api_url, json=payload, headers=headers)
        if response.status_code in [200, 201]:
            print("Upload successful!")
        else:
            print(f"Upload failed: {response.status_code}")
    except Exception as e:
        print(f"Error uploading: {e}")

upload_plot_to_github(history)

# --- 7. TESTING ---
print("\n--- Testing Autoregressive Model ---")

test_indices = np.random.choice(len(encoder_input_data), 5, replace=False)

print(f"{'Input (Misspelled)':<20} | {'Decoded (Prediction)':<20}")
print("-" * 45)

for idx in test_indices:
    input_seq = encoder_input_data[idx:idx+1]
    
    # Reconstruct input string for display (remove padding)
    input_str = ""
    for i in input_seq[0]:
        if i != 0: input_str += int_to_char[i]
            
    decoded_sentence = decode_sequence(input_seq)
    
    print(f"{input_str:<20} | {decoded_sentence:<20}")