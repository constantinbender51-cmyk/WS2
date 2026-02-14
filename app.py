import os
import requests
import json
from openai import OpenAI
from dotenv import load_dotenv

def analyze_entire_community():
    load_dotenv()
    api_key = os.getenv("DSAPI")
    db_url = "https://try3btc.up.railway.app/"
    
    try:
        raw_data = requests.get(db_url).json()
    except Exception as e:
        print(f"Connection Error: {e}")
        return

    # 1. Flattening logic to catch EVERYTHING
    payload = []
    results = raw_data.get("results", {})
    
    for sub, posts in results.items():
        for p in posts:
            # Append Post
            payload.append({
                "type": "post",
                "id": p.get("id", "N/A"),
                "content": f"TITLE: {p.get('title')} BODY: {p.get('body')}"
            })
            # Append Every Comment
            for c in p.get("comments", []):
                payload.append({
                    "type": "comment",
                    "id": p.get("id"), # Linking to parent post ID
                    "content": c.get("text")
                })

    total_items = len(payload)
    print(f"Transmitting {total_items} discrete items to DeepSeek...")

    client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
    
    # 2. Strict Prompt to prevent partial analysis
    system_msg = (
        f"You are an objective engine. You are receiving a dataset of {total_items} items. "
        "You MUST analyze every item. Do not provide samples. "
        "Return a JSON object: "
        "{ 'meta': {'total_processed': int, 'overall_sentiment': float}, "
        "'analysis': [{'id': string, 'sentiment': string, 'score': float}] }"
    )

    try:
        completion = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": json.dumps(payload)}
            ],
            response_format={'type': 'json_object'}
        )
        print(completion.choices[0].message.content)
    except Exception as e:
        print(f"API Error: {e}")

if __name__ == "__main__":
    analyze_entire_community()
