import numpy as np
import tensorflow as tf
import requests
import random
import string
import os
import base64
import json

# Try to import dotenv, handle if missing
try:
    from dotenv import load_dotenv
except ImportError:
    print("python-dotenv not found. Assuming environment variables are set or parsing manually.")
    def load_dotenv():
        if os.path.exists('.env'):
            with open('.env') as f:
                for line in f:
                    if '=' in line:
                        k, v = line.strip().split('=', 1)
                        os.environ[k] = v

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
        return ["hello", "world", "python", "neural", "network", "coding", "intelligence"]

def generate_misspelling(word):
    chars = list(word)
    if not chars: return word
    
    op = random.choice(['insert', 'delete', 'swap'])
    
    if op == 'insert':
        pos = random.randint(0, len(chars))
        char = random.choice(string.ascii_lowercase)
        chars.insert(pos, char)
    
    elif op == 'delete':
        if len(chars) > 1:
            pos = random.randint(0, len(chars) - 1)
            del chars[pos]
            
    elif op == 'swap':
        if len(chars) > 1:
            pos = random.randint(0, len(chars) - 2)
            chars[pos], chars[pos+1] = chars[pos+1], chars[pos]
            
    return "".join(chars)

def create_dataset(words, num_samples=500000):
    print(f"Generating {num_samples} misspelled-correct pairs...")
    inputs = []
    targets = []
    for _ in range(num_samples):
        word = random.choice(words)
        misspelled = generate_misspelling(word)
        inputs.append(misspelled)
        targets.append(word)
    return inputs, targets

def upload_model_to_github(file_path, repo_url, token_key='PAT'):
    print("\n--- Initiating GitHub Upload ---")
    
    # 1. Load Token
    load_dotenv()
    token = os.getenv(token_key)
    
    if not token:
        print(f"ERROR: No token found in environment variable '{token_key}'. Upload skipped.")
        return

    # 2. Parse Repo Info
    # Expected: https://github.com/OWNER/REPO
    try:
        parts = repo_url.strip("/").split("/")
        owner = parts[-2]
        repo = parts[-1]
    except IndexError:
        print("ERROR: Invalid repository URL format.")
        return

    file_name = os.path.basename(file_path)
    api_url = f"https://api.github.com/repos/{owner}/{repo}/contents/{file_name}"
    
    print(f"Target API URL: {api_url}")

    # 3. Read and Encode File
    try:
        with open(file_path, "rb") as f:
            content = f.read()
            encoded_content = base64.b64encode(content).decode("utf-8")
    except FileNotFoundError:
        print(f"ERROR: Model file {file_path} not found.")
        return

    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github.v3+json"
    }

    # 4. Check if file exists (to get SHA for update)
    sha = None
    check_resp = requests.get(api_url, headers=headers)
    if check_resp.status_code == 200:
        print("File already exists in repo. Fetching SHA to overwrite...")
        sha = check_resp.json().get("sha")

    # 5. Prepare Payload
    data = {
        "message": f"Update model: {file_name}",
        "content": encoded_content,
        "branch": "main" # Or master, depending on repo default
    }
    if sha:
        data["sha"] = sha

    # 6. Upload (PUT)
    print("Uploading file (this may take a few seconds)...")
    put_resp = requests.put(api_url, headers=headers, json=data)

    if put_resp.status_code in [200, 201]:
        print(f"SUCCESS: Model uploaded to {repo_url}/blob/main/{file_name}")
    else:
        print(f"ERROR: Upload failed. Status: {put_resp.status_code}")
        print(put_resp.text)


# --- Main Execution ---

# 1. Load Data
url = "https://raw.githubusercontent.com/first20hours/google-10000-english/refs/heads/master/20k.txt"
clean_words = download_word_list(url)

# 2. Create 500k pairs
input_texts, target_texts = create_dataset(clean_words, num_samples=500000)

# 3. Vectorization
characters = sorted(list(string.ascii_lowercase))
input_token_index = {char: i+1 for i, char in enumerate(characters)}
reverse_input_char_index = {i: char for char, i in input_token_index.items()}

target_token_index = {'\t': 1, '\n': 2}
for i, char in enumerate(characters):
    target_token_index[char] = i + 3
reverse_target_char_index = {i: char for char, i in target_token_index.items()}

num_encoder_tokens = len(input_token_index) + 1
num_decoder_tokens = len(target_token_index) + 1

max_encoder_seq_length = max([len(txt) for txt in input_texts])
# Approximate max decoder length (word + start + end + margin)
max_decoder_seq_length = max([len(txt) for txt in target_texts]) + 5 

print("Vectorizing data...")
encoder_input_data = np.zeros((len(input_texts), max_encoder_seq_length), dtype="float32")
decoder_input_data = np.zeros((len(input_texts), max_decoder_seq_length), dtype="float32")
decoder_target_data = np.zeros((len(input_texts), max_decoder_seq_length, num_decoder_tokens), dtype="float32")

target_texts_output = [text + '\n' for text in target_texts]

for i, (input_text, target_text, target_out) in enumerate(zip(input_texts, target_texts, target_texts_output)):
    for t, char in enumerate(input_text):
        if char in input_token_index:
            encoder_input_data[i, t] = input_token_index[char]
    
    decoder_input_text = '\t' + target_text
    for t, char in enumerate(decoder_input_text):
        if t < max_decoder_seq_length:
            if char in target_token_index:
                decoder_input_data[i, t] = target_token_index[char]
    
    for t, char in enumerate(target_out):
        if t < max_decoder_seq_length:
            if char in target_token_index:
                decoder_target_data[i, t, target_token_index[char]] = 1.0

# 4. Train Model
latent_dim = 128

encoder_inputs = tf.keras.Input(shape=(None,))
encoder_embedding_layer = tf.keras.layers.Embedding(num_encoder_tokens, latent_dim, mask_zero=True)
encoder_embedding = encoder_embedding_layer(encoder_inputs)
encoder_gru = tf.keras.layers.GRU(latent_dim, return_state=True)
encoder_outputs, state_h = encoder_gru(encoder_embedding)

decoder_inputs = tf.keras.Input(shape=(None,))
decoder_embedding_layer = tf.keras.layers.Embedding(num_decoder_tokens, latent_dim, mask_zero=True)
decoder_embedding = decoder_embedding_layer(decoder_inputs)
decoder_gru = tf.keras.layers.GRU(latent_dim, return_sequences=True, return_state=True)
decoder_outputs, _ = decoder_gru(decoder_embedding, initial_state=state_h)
decoder_dense = tf.keras.layers.Dense(num_decoder_tokens, activation="softmax")
decoder_outputs = decoder_dense(decoder_outputs)

model = tf.keras.Model([encoder_inputs, decoder_inputs], decoder_outputs)
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

print("Training model...")
model.fit(
    [encoder_input_data, decoder_input_data],
    decoder_target_data,
    batch_size=128,
    epochs=5,
    validation_split=0.2,
    verbose=1
)

# 5. Inference
encoder_model = tf.keras.Model(encoder_inputs, state_h)

decoder_state_input_h = tf.keras.Input(shape=(latent_dim,))
decoder_inputs_single = tf.keras.Input(shape=(None,))
decoder_emb_single = decoder_embedding_layer(decoder_inputs_single)
decoder_outputs_single, state_h_single = decoder_gru(decoder_emb_single, initial_state=decoder_state_input_h)
decoder_outputs_single = decoder_dense(decoder_outputs_single)
decoder_model = tf.keras.Model(
    [decoder_inputs_single, decoder_state_input_h],
    [decoder_outputs_single, state_h_single]
)

def decode_sequence(input_seq):
    states_value = encoder_model.predict(input_seq, verbose=0)
    target_seq = np.zeros((1, 1))
    target_seq[0, 0] = target_token_index['\t']

    stop_condition = False
    decoded_sentence = ""
    
    while not stop_condition:
        output_tokens, h = decoder_model.predict([target_seq, states_value], verbose=0)
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        
        if sampled_token_index in reverse_target_char_index:
            sampled_char = reverse_target_char_index[sampled_token_index]
        else:
            sampled_char = ''
            
        if sampled_char == '\n' or len(decoded_sentence) > max_decoder_seq_length:
            stop_condition = True
        else:
            decoded_sentence += sampled_char

        target_seq = np.zeros((1, 1))
        target_seq[0, 0] = sampled_token_index
        states_value = h

    return decoded_sentence

test_words = ["compputer", "scienc", "intellgence", "mashine", "languag"]
print("\n--- Testing Model ---")
for word in test_words:
    input_seq = np.zeros((1, max_encoder_seq_length), dtype="float32")
    for t, char in enumerate(word):
        if char in input_token_index:
            input_seq[0, t] = input_token_index[char]
    print(f"Input: {word:15} -> Predicted: {decode_sequence(input_seq)}")

# 6. Save and Upload
model_filename = "spell_corrector.keras"
print(f"\nSaving model to {model_filename}...")
model.save(model_filename)

target_repo = "https://github.com/constantinbender51-cmyk/models"
upload_model_to_github(model_filename, target_repo, token_key="PAT")