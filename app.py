import os
import requests
from openai import OpenAI

def process_reddit_sentiment():
    # Fetch database content
    db_url = "https://try3btc.up.railway.app/"
    try:
        response = requests.get(db_url)
        response.raise_for_status()
        data = response.json()
    except Exception as e:
        print(f"Failed to fetch database: {e}")
        return

    # Extract posts and comments for analysis
    # The structure includes a 'results' key with subreddits (e.g., 'Bitcoin')
    content_to_analyze = []
    results = data.get("results", {})
    
    for subreddit, posts in results.items():
        for post in posts:
            # Add post title and body
            post_text = f"Title: {post.get('title', '')}\nBody: {post.get('body', '')}"
            content_to_analyze.append({"id": post.get("id"), "type": "post", "text": post_text})
            
            # Add individual comments
            for comment in post.get("comments", []):
                content_to_analyze.append({
                    "id": post.get("id"), 
                    "type": "comment", 
                    "text": comment.get("text", ""),
                    "user": comment.get("user")
                })

    # Initialize DeepSeek client
    client = OpenAI(
        api_key=os.getenv("DSAPI"), 
        base_url="https://api.deepseek.com"
    )

    # Perform sentiment analysis
    for item in content_to_analyze:
        try:
            completion = client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": "You are a sentiment analysis engine. Return only a JSON object with 'sentiment' (Positive/Negative/Neutral) and 'score' (0.0 to 1.0)."},
                    {"role": "user", "content": item["text"]}
                ],
                response_format={'type': 'json_object'}
            )
            
            sentiment_result = completion.choices[0].message.content
            print(f"Type: {item['type']} | ID: {item['id']} | Analysis: {sentiment_result}")
            
        except Exception as e:
            print(f"Error analyzing item {item['id']}: {e}")

if __name__ == "__main__":
    process_reddit_sentiment()
