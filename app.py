import numpy as np
import tensorflow as tf
import requests
import random
import string
import os

# Ensure reproducible results
seed = 42
np.random.seed(seed)
tf.random.set_seed(seed)
random.seed(seed)

def download_word_list(url):
    print(f"Downloading word list from {url}...")
    try:
        response = requests.get(url)
        response.raise_for_status()
        text = response.text
        # Filter for purely alphabetic words to keep vocab simple
        words = [w.strip().lower() for w in text.splitlines() if w.strip().isalpha()]
        print(f"Loaded {len(words)} words.")
        return words
    except Exception as e:
        print(f"Error downloading: {e}")
        # Fallback to a small list if download fails (for safety)
        return ["hello", "world", "python", "neural", "network", "coding", "intelligence"]

def generate_misspelling(word):
    """Introduces a single error: insert, delete, or swap."""
    chars = list(word)
    if not chars: return word
    
    op = random.choice(['insert', 'delete', 'swap'])
    
    if op == 'insert':
        # Insert a random letter at a random position
        pos = random.randint(0, len(chars))
        char = random.choice(string.ascii_lowercase)
        chars.insert(pos, char)
    
    elif op == 'delete':
        # Delete a random character
        if len(chars) > 1:
            pos = random.randint(0, len(chars) - 1)
            del chars[pos]
            
    elif op == 'swap':
        # Swap two adjacent characters
        if len(chars) > 1:
            pos = random.randint(0, len(chars) - 2)
            chars[pos], chars[pos+1] = chars[pos+1], chars[pos]
            
    return "".join(chars)

def create_dataset(words, num_samples=500000):
    print(f"Generating {num_samples} misspelled-correct pairs...")
    inputs = []
    targets = []
    
    # We use a set for target texts to include start/end tokens later
    # Input: "helo"
    # Target Input (Decoder): "\t" + "hello"
    # Target Output (Label): "hello" + "\n"
    
    for _ in range(num_samples):
        word = random.choice(words)
        misspelled = generate_misspelling(word)
        
        inputs.append(misspelled)
        targets.append(word)
        
    return inputs, targets

# --- Main Execution ---

# 1. Load Data
url = "https://raw.githubusercontent.com/first20hours/google-10000-english/refs/heads/master/20k.txt"
clean_words = download_word_list(url)

# 2. Create 500k pairs
input_texts, target_texts = create_dataset(clean_words, num_samples=500000)

# 3. Vectorization logic
# Define valid characters
characters = sorted(list(string.ascii_lowercase))
num_encoder_tokens = len(characters) + 1 # +1 for padding/unknown if needed, though we restrict to ascii
num_decoder_tokens = len(characters) + 3 # +1 pad, +1 start '\t', +1 end '\n'

# Add start and end tokens to targets for the decoder
target_texts_input = ['\t' + text for text in target_texts]
target_texts_output = [text + '\n' for text in target_texts]

# Determine max lengths
max_encoder_seq_length = max([len(txt) for txt in input_texts])
max_decoder_seq_length = max([len(txt) for txt in target_texts_output])

print(f"Max Encoder Length: {max_encoder_seq_length}")
print(f"Max Decoder Length: {max_decoder_seq_length}")

# Token dictionaries
input_token_index = {char: i+1 for i, char in enumerate(characters)} # 0 reserved for padding
target_token_index = {char: i+2 for i, char in enumerate(characters)}
target_token_index['\t'] = 1 # Start token
target_token_index['\n'] = 0 # End token (and padding logic usually uses 0, but we will mask)
# Actually, standard keras padding uses 0. Let's adjust.
# Encoder: 0=pad, 1..26=a-z
# Decoder: 0=pad, 1=start, 2=end, 3..28=a-z

input_token_index = {char: i+1 for i, char in enumerate(characters)}
reverse_input_char_index = {i: char for char, i in input_token_index.items()}

target_token_index = {'\t': 1, '\n': 2}
for i, char in enumerate(characters):
    target_token_index[char] = i + 3
reverse_target_char_index = {i: char for char, i in target_token_index.items()}

num_encoder_tokens = len(input_token_index) + 1
num_decoder_tokens = len(target_token_index) + 1

print("Vectorizing data...")
encoder_input_data = np.zeros((len(input_texts), max_encoder_seq_length), dtype="float32")
decoder_input_data = np.zeros((len(input_texts), max_decoder_seq_length), dtype="float32")
decoder_target_data = np.zeros((len(input_texts), max_decoder_seq_length, num_decoder_tokens), dtype="float32")

for i, (input_text, target_text, target_out) in enumerate(zip(input_texts, target_texts, target_texts_output)):
    # Encoder Input
    for t, char in enumerate(input_text):
        if char in input_token_index:
            encoder_input_data[i, t] = input_token_index[char]
    
    # Decoder Input (Teacher Forcing: input is previous correct char)
    decoder_input_text = '\t' + target_text
    for t, char in enumerate(decoder_input_text):
        if t < max_decoder_seq_length:
            if char in target_token_index:
                decoder_input_data[i, t] = target_token_index[char]
    
    # Decoder Target (One-hot encoded for softmax)
    # Offset by 1 from input (predict next char)
    for t, char in enumerate(target_out):
        if t < max_decoder_seq_length:
            if char in target_token_index:
                decoder_target_data[i, t, target_token_index[char]] = 1.0

# 3. Train GRU Encoder-Decoder
latent_dim = 128 # Hidden layer size

# Encoder
encoder_inputs = tf.keras.Input(shape=(None,))
encoder_embedding = tf.keras.layers.Embedding(num_encoder_tokens, latent_dim, mask_zero=True)(encoder_inputs)
encoder_gru = tf.keras.layers.GRU(latent_dim, return_state=True)
encoder_outputs, state_h = encoder_gru(encoder_embedding)

# Decoder
decoder_inputs = tf.keras.Input(shape=(None,))
decoder_embedding = tf.keras.layers.Embedding(num_decoder_tokens, latent_dim, mask_zero=True)(decoder_inputs)
decoder_gru = tf.keras.layers.GRU(latent_dim, return_sequences=True, return_state=True)
decoder_outputs, _ = decoder_gru(decoder_embedding, initial_state=state_h)
decoder_dense = tf.keras.layers.Dense(num_decoder_tokens, activation="softmax")
decoder_outputs = decoder_dense(decoder_outputs)

model = tf.keras.Model([encoder_inputs, decoder_inputs], decoder_outputs)

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

print("Training model (this may take a moment)...")
# Using a small batch size and 1 epoch for demonstration speed on CPU
# Increase epochs for better accuracy
batch_size = 128
epochs = 5 

# Train on a subset if needed, but request asked for 500k pairs. 
# We will train on all, but keep epochs low.
model.fit(
    [encoder_input_data, decoder_input_data],
    decoder_target_data,
    batch_size=batch_size,
    epochs=epochs,
    validation_split=0.2,
    verbose=1
)

# 4. Inference Setup
# To decode, we need separate models to step through the sequence manually

# Encoder Model
encoder_model = tf.keras.Model(encoder_inputs, state_h)

# Decoder Model
decoder_state_input_h = tf.keras.Input(shape=(latent_dim,))
decoder_inputs_single = tf.keras.Input(shape=(None,)) # One char at a time
decoder_emb_single = decoder_embedding(decoder_inputs_single)
decoder_outputs_single, state_h_single = decoder_gru(decoder_emb_single, initial_state=decoder_state_input_h)
decoder_outputs_single = decoder_dense(decoder_outputs_single)
decoder_model = tf.keras.Model(
    [decoder_inputs_single, decoder_state_input_h],
    [decoder_outputs_single, state_h_single]
)

def decode_sequence(input_seq):
    # Encode the input as state vectors.
    states_value = encoder_model.predict(input_seq, verbose=0)

    # Generate empty target sequence of length 1.
    target_seq = np.zeros((1, 1))
    # Populate the first character of target sequence with the start character.
    target_seq[0, 0] = target_token_index['\t']

    # Sampling loop
    stop_condition = False
    decoded_sentence = ""
    
    while not stop_condition:
        output_tokens, h = decoder_model.predict([target_seq, states_value], verbose=0)

        # Sample a token
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        
        # Handle unknown token case
        if sampled_token_index in reverse_target_char_index:
            sampled_char = reverse_target_char_index[sampled_token_index]
        else:
            sampled_char = ''
            
        if sampled_char == '\n' or len(decoded_sentence) > max_decoder_seq_length:
            stop_condition = True
        else:
            decoded_sentence += sampled_char

        # Update the target sequence (of length 1).
        target_seq = np.zeros((1, 1))
        target_seq[0, 0] = sampled_token_index

        # Update states
        states_value = h

    return decoded_sentence

# 5. Prompt 5 Misspelled Words
test_words = ["compputer", "scienc", "intellgence", "mashine", "languag"]
print("\n--- Testing Model ---")

for word in test_words:
    # Vectorize input
    input_seq = np.zeros((1, max_encoder_seq_length), dtype="float32")
    for t, char in enumerate(word):
        if char in input_token_index:
            input_seq[0, t] = input_token_index[char]
            
    corrected = decode_sequence(input_seq)
    print(f"Input: {word:15} -> Predicted: {corrected}")

print("\nDone.")