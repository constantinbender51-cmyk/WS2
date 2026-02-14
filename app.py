import os
import requests
from openai import OpenAI
from dotenv import load_dotenv

def process_reddit_sentiment():
    load_dotenv()
    api_key = os.getenv("DSAPI")
    
    db_url = "https://try3btc.up.railway.app/"
    try:
        response = requests.get(db_url)
        response.raise_for_status()
        data = response.json()
    except Exception as e:
        print(f"Fetch Error: {e}")
        return

    content_to_analyze = []
    results = data.get("results", {})
    
    for subreddit in results:
        for post in results[subreddit]:
            content_to_analyze.append({
                "id": post.get("id"),
                "text": f"Title: {post.get('title', '')} Body: {post.get('body', '')}"
            })
            for comment in post.get("comments", []):
                content_to_analyze.append({
                    "id": post.get("id"),
                    "text": comment.get("text", "")
                })

    client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")

    for item in content_to_analyze:
        try:
            completion = client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": "Analyze sentiment. Output JSON: {'sentiment': 'string', 'score': float}"},
                    {"role": "user", "content": item["text"]}
                ],
                response_format={'type': 'json_object'}
            )
            print(f"ID: {item['id']} | Result: {completion.choices[0].message.content}")
        except Exception as e:
            print(f"Analysis Error {item['id']}: {e}")

if __name__ == "__main__":
    process_reddit_sentiment()
