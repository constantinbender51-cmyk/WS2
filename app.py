import os
import requests
import json
from openai import OpenAI
from dotenv import load_dotenv

def process_with_metrics():
    load_dotenv()
    api_key = os.getenv("DSAPI")
    db_url = "https://try3btc.up.railway.app/"
    
    # 1. Fetch data from the utility
    try:
        response = requests.get(db_url)
        response.raise_for_status()
        data = response.json()
    except Exception as e:
        print(f"Data Fetch Error: {e}")
        return

    # 2. Extract and count
    posts_to_send = []
    comments_to_send = []
    results = data.get("results", {})

    for subreddit in results:
        for post in results[subreddit]:
            # Capture Post
            posts_to_send.append({
                "id": post.get("id"),
                "type": "post",
                "text": f"T: {post.get('title')} B: {post.get('body')}"
            })
            # Capture Comments
            for comment in post.get("comments", []):
                comments_to_send.append({
                    "id": post.get("id"),
                    "type": "comment",
                    "text": comment.get("text", "")
                })

    # 3. Pre-flight metrics for Jimmy
    total_posts = len(posts_to_send)
    total_comments = len(comments_to_send)
    total_items = total_posts + total_comments
    
    print(f"--- PRE-FLIGHT METRICS ---")
    print(f"Total Posts: {total_posts}")
    print(f"Total Comments: {total_comments}")
    print(f"Total Payload Items: {total_items}")
    print(f"--------------------------")

    # 4. Transmit aggregate payload
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
        print("\n--- ANALYSIS RESULT ---")
        print(completion.choices[0].message.content)
    except Exception as e:
        print(f"Analysis Error: {e}")

if __name__ == "__main__":
    process_with_metrics()
