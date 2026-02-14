import os
import requests
import json
import http.server
import socketserver
import time
import threading
from openai import OpenAI
from dotenv import load_dotenv

def process_with_metrics():
    load_dotenv()
    api_key = os.getenv("DSAPI")
    db_url = "https://try3btc.up.railway.app/"
    
    try:
        response = requests.get(db_url)
        response.raise_for_status()
        data = response.json()
    except Exception as e:
        print(f"Data Fetch Error: {e}")
        return None

    # Aggregate the entire corpus
    corpus_fragments = []
    results = data.get("results", {})

    for subreddit, posts in results.items():
        for post in posts:
            fragment = f"Subreddit: {subreddit}\nTitle: {post.get('title')}\nBody: {post.get('body')}\n"
            for comment in post.get("comments", []):
                fragment += f"Comment: {comment.get('text', '')}\n"
            corpus_fragments.append(fragment)

    full_corpus = "\n---\n".join(corpus_fragments)
    
    print(f"--- DATA METRICS ---")
    print(f"Total Characters: {len(full_corpus)}")
    print(f"Total Logical Fragments: {len(corpus_fragments)}")
    print(f"--------------------")

    client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")

    try:
        completion = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {
                    "role": "system", 
                    "content": "You are a specialized data analyst. Analyze the following social media dataset. "
                               "Provide a comprehensive sentiment analysis of the entire community. "
                               "Identify dominant themes, emotional trends, and collective consensus. "
                               "Return the analysis as a structured JSON object."
                },
                {"role": "user", "content": full_corpus}
            ],
            response_format={'type': 'json_object'}
        )
        return completion.choices[0].message.content
    except Exception as e:
        print(f"Analysis Error: {e}")
        return None

def start_server():
    PORT = 8080
    class QuietHandler(http.server.SimpleHTTPRequestHandler):
        def log_message(self, format, *args): return

    # Allow reuse of address to prevent 'Address already in use' on restart
    socketserver.TCPServer.allow_reuse_address = True
    
    with socketserver.TCPServer(("", PORT), QuietHandler) as httpd:
        print(f"Serving aggregate analysis at http://localhost:{PORT}/analysis_result.json")
        httpd.serve_forever()

if __name__ == "__main__":
    # Initialize file if missing
    if not os.path.exists("analysis_result.json"):
        with open("analysis_result.json", "w") as f:
            f.write(json.dumps({"status": "initializing"}))

    # Start server in a separate daemon thread
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
