import os
import requests
import json
import http.server
import socketserver
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

    posts_to_send = []
    comments_to_send = []
    results = data.get("results", {})

    for subreddit in results:
        for post in results[subreddit]:
            posts_to_send.append({
                "id": post.get("id"),
                "type": "post",
                "text": f"T: {post.get('title')} B: {post.get('body')}"
            })
            for comment in post.get("comments", []):
                comments_to_send.append({
                    "id": post.get("id"),
                    "type": "comment",
                    "text": comment.get("text", "")
                })

    total_posts = len(posts_to_send)
    total_comments = len(comments_to_send)
    total_items = total_posts + total_comments
    
    print(f"--- PRE-FLIGHT METRICS ---")
    print(f"Total Posts: {total_posts}")
    print(f"Total Comments: {total_comments}")
    print(f"Total Payload Items: {total_items}")
    print(f"--------------------------")

    client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
    full_payload = posts_to_send + comments_to_send

    try:
        completion = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": "Analyze the sentiment of EVERY item in this list. Return a comprehensive JSON summary."},
                {"role": "user", "content": json.dumps(full_payload)}
            ],
            response_format={'type': 'json_object'}
        )
        return completion.choices[0].message.content
    except Exception as e:
        print(f"Analysis Error: {e}")
        return None

def serve_response(content):
    with open("analysis_result.json", "w") as f:
        f.write(content)

    PORT = 8080
    Handler = http.server.SimpleHTTPRequestHandler

    with socketserver.TCPServer(("", PORT), Handler) as httpd:
        print(f"Serving analysis at port {PORT}")
        httpd.serve_forever()

if __name__ == "__main__":
    analysis_content = process_with_metrics()
    if analysis_content:
        serve_response(analysis_content)
