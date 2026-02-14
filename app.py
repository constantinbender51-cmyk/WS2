import os
import requests
import json
from openai import OpenAI
from dotenv import load_dotenv

def analyze_community_psyche():
    load_dotenv()
    api_key = os.getenv("DSAPI")
    db_url = "https://try3btc.up.railway.app/"
    
    try:
        data = requests.get(db_url).json()
    except Exception as e:
        print(f"Data Retrieval Failed: {e}")
        return

    # Aggregate into a single text block for holistic analysis
    full_corpus = []
    results = data.get("results", {})
    for sub, posts in results.items():
        for p in posts:
            full_corpus.append(f"SUB: {sub} | POST: {p.get('title')} - {p.get('body')}")
            for c in p.get("comments", []):
                full_corpus.append(f"COMMENT: {c.get('text')}")

    client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
    
    # Specific instruction to prevent "skimming"
    prompt = (
        "You are an objective analytical engine. I am providing the entire database of a community. "
        "1. Identify the core psychological drivers of this community. "
        "2. Detail the primary conflicts or disagreements found in the text. "
        "3. Provide a quantitative sentiment score (-1.0 to 1.0) for the aggregate. "
        "4. Highlight the top 3 most influential posts/comments that define the current mood."
    )

    try:
        completion = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": "Output strictly in JSON format."},
                {"role": "user", "content": f"{prompt}\n\nDATASET:\n{json.dumps(full_corpus)}"}
            ],
            response_format={'type': 'json_object'}
        )
        print(completion.choices[0].message.content)
    except Exception as e:
        print(f"Analysis Failed: {e}")

if __name__ == "__main__":
    analyze_community_psyche()
