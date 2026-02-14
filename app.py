import os
import requests
import json
import http.server
import socketserver
import time
import threading
from openai import OpenAI
from dotenv import load_dotenv

# Config
# 128k tokens ~= 500k chars. Setting to 350k leaves ample buffer for system prompts + output.
MAX_CHUNK_CHARS = 350000  
MODEL = "deepseek-chat"

def get_client():
    load_dotenv()
    return OpenAI(api_key=os.getenv("DSAPI"), base_url="https://api.deepseek.com")

def analyze_segment(client, text, context_type="partial"):
    """
    Analyzes a text segment.
    context_type: 'partial' for chunks, 'final' for synthesis.
    """
    if context_type == "partial":
        sys_prompt = (
            "You are a specialized data analyst. Analyze this segment of a larger social media dataset. "
            "Provide a sentiment analysis, identify dominant themes, and emotional trends for this specific chunk. "
            "Return the analysis as a structured JSON object."
        )
    else:
        sys_prompt = (
            "You are a lead data analyst. You are provided with multiple partial analysis reports from a large dataset. "
            "Synthesize these reports into one comprehensive final JSON analysis. "
            "Merge consensus, resolve conflicts, and provide a unified sentiment and thematic overview."
        )

    try:
        completion = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": text}
            ],
            response_format={'type': 'json_object'}
        )
        return completion.choices[0].message.content
    except Exception as e:
        print(f"Segment Analysis Error: {e}")
        return None

def process_with_metrics():
    db_url = "https://try3btc.up.railway.app/"
    
    try:
        response = requests.get(db_url)
        response.raise_for_status()
        data = response.json()
    except Exception as e:
        print(f"Data Fetch Error: {e}")
        return None

    # 1. Aggregate fragments
    corpus_fragments = []
    results = data.get("results", {})

    for subreddit, posts in results.items():
        for post in posts:
            fragment = f"Subreddit: {subreddit}\nTitle: {post.get('title')}\nBody: {post.get('body')}\n"
            for comment in post.get("comments", []):
                fragment += f"Comment: {comment.get('text', '')}\n"
            corpus_fragments.append(fragment)

    total_chars = sum(len(f) for f in corpus_fragments)
    print(f"--- DATA METRICS ---")
    print(f"Total Characters: {total_chars}")
    print(f"Total Fragments: {len(corpus_fragments)}")
    print(f"--------------------")

    client = get_client()

    # 2. Chunking Logic
    chunks = []
    current_chunk = ""
    
    for frag in corpus_fragments:
        if len(current_chunk) + len(frag) > MAX_CHUNK_CHARS:
            chunks.append(current_chunk)
            current_chunk = frag
        else:
            current_chunk += "\n---\n" + frag if current_chunk else frag
    
    if current_chunk:
        chunks.append(current_chunk)

    print(f"Processing {len(chunks)} chunks...")

    # 3. Map (Analyze Chunks)
    if len(chunks) == 0:
        return None
    
    if len(chunks) == 1:
        return analyze_segment(client, chunks[0], "partial")

    partial_results = []
    for i, chunk in enumerate(chunks):
        print(f"Analyzing chunk {i+1}/{len(chunks)}...")
        res = analyze_segment(client, chunk, "partial")
        if res:
            partial_results.append(f"Analysis Part {i+1}: {res}")

    # 4. Reduce (Synthesize)
    print("Synthesizing final report...")
    synthesis_input = "\n\n".join(partial_results)
    return analyze_segment(client, synthesis_input, "final")

def start_server():
    PORT = 8080
    class QuietHandler(http.server.SimpleHTTPRequestHandler):
        def log_message(self, format, *args): return

    socketserver.TCPServer.allow_reuse_address = True
    with socketserver.TCPServer(("", PORT), QuietHandler) as httpd:
        print(f"Serving aggregate analysis at http://localhost:{PORT}/analysis_result.json")
        httpd.serve_forever()

if __name__ == "__main__":
    if not os.path.exists("analysis_result.json"):
        with open("analysis_result.json", "w") as f:
            f.write(json.dumps({"status": "initializing"}))

    server_thread = threading.Thread(target=start_server, daemon=True)
    server_thread.start()

    while True:
        print(f"Starting analysis cycle: {time.ctime()}")
        analysis_content = process_with_metrics()
        
        if analysis_content:
            with open("analysis_result.json", "w") as f:
                f.write(analysis_content)
            print("Analysis updated and saved.")
        
        time.sleep(3600)
