"""
Debug script to list available Gemini models and test Google Trends RSS.
Updated for Gemini 3 and google-genai SDK.
"""

import os
from google import genai
import feedparser
from dotenv import load_dotenv

load_dotenv()

# Test Gemini Models
print("="*70)
print("GEMINI MODELS INVESTIGATION (SDK v1.0+)")
print("="*70)

api_key = os.environ.get("GEMINI_API_KEY")
if api_key:
    client = genai.Client(api_key=api_key)
    print(f"\nAPI Key configured: {api_key[:10]}...{api_key[-4:]}")

    print("\nListing available models...")
    try:
        models = list(client.models.list())
        print(f"\nFound {len(models)} models:\n")

        print("MODELS LIST:")
        print("-" * 70)
        for model in models:
            print(f"Name: {model.name}")
            print(f"Display Name: {model.display_name}")
            print(f"Description: {model.description}")
            print("-" * 70)

    except Exception as e:
        print(f"Error listing models: {e}")
else:
    print("GEMINI_API_KEY not found in environment")

# Test Google Trends RSS
print("\n" + "="*70)
print("GOOGLE TRENDS RSS INVESTIGATION")
print("="*70)

geo = os.environ.get("TRENDS_GEO", "US")
rss_url = f"https://trends.google.com/trends/trendingsearches/daily/rss?geo={geo}"

print(f"\nFetching RSS from: {rss_url}")

try:
    feed = feedparser.parse(rss_url)

    print(f"\nFeed status:")
    print(f"  Feed title: {feed.feed.get('title', 'N/A')}")
    print(f"  Entries count: {len(feed.entries)}")

    if feed.entries:
        print(f"\nTop 3 trending topics:")
        for i, entry in enumerate(feed.entries[:3], 1):
            print(f"\n  {i}. {entry.title}")
            print(f"     Description: {entry.description[:100]}...")
    else:
        print("\n  No entries found!")

except Exception as e:
    print(f"Error fetching RSS: {e}")

print("\n" + "="*70)
