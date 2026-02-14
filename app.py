import os
import requests
import json
import http.server
import socketserver
from openai import OpenAI
from dotenv import load_dotenv

def generate_community_synthesis():
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

    # Aggregate all text into a single corpus for global context
    corpus = []
    results = data.get("results", {})

    for subreddit in results:
        for post in results[subreddit]:
            corpus.append(f"POST: {post.get('title')} | {post.get('body')}")
            for comment in post.get("comments", []):
                corpus.append(f"COMMENT: {comment.get('text', '')}")

    full_text = "\n".join(corpus)
    
    print(f"--- DATA METRICS ---")
    print(f"Total Text Length: {len(full_text)} characters")
    print(f"Total Subreddits: {len(results)}")
    print(f"--------------------")

    client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")

    try:
        completion = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {
                    "role": "system", 
                    "content": (
                        "You are an objective data analyst. Analyze the following Reddit data corpus. "
                        "Do not score individual items. Provide a high-level community sentiment analysis. "
                        "Identify: 1. Overall sentiment trend. 2. Primary themes of discussion. "
                        "3. Notable conflicts or consensus. 4. Emergent patterns. "
                        "Return strictly in JSON format."
                    )
                },
                {"role": "user", "content": full_text[:32000]} # Truncate to stay within context window
            ],
            response_format={'type': 'json_object'}
        )
        return completion.choices[0].message.content
    except Exception as e:
        print(f"Analysis Error: {e}")
        return None

def serve_response(content):
    with open("sentiment_report.json", "w") as f:
        f.write(content)

    PORT = 8080
    Handler = http.server.SimpleHTTPRequestHandler

    with socketserver.TCPServer(("", PORT), Handler) as httpd:
        print(f"Serving Synthesis Report at http://localhost:{PORT}/sentiment_report.json")
        httpd.serve_forever()

if __name__ == "__main__":
    report = generate_community_synthesis()
    if report:
        serve_response(report)
