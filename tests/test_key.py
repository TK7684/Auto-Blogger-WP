import os
from google import genai
from dotenv import load_dotenv

load_dotenv()
api_key = os.environ.get('GEMINI_API_KEY')
print(f"Testing key starting with: {api_key[:10]}...")

client = genai.Client(api_key=api_key)

try:
    # Test Gemini 3 Flash (fast)
    print("\nTesting Gemini 3 Flash...")
    response = client.models.generate_content(
        model="gemini-3-flash-preview",
        contents="Say hello"
    )
    print(f"Success! Response: {response.text}")
except Exception as e:
    print(f"Failed with gemini-3-flash-preview: {e}")

try:
    # Test Gemini 3 Pro (deep reasoning)
    print("\nTesting Gemini 3 Pro...")
    response = client.models.generate_content(
        model="gemini-3-pro-preview",
        contents="Say hello"
    )
    print(f"Success! Response: {response.text}")
except Exception as e:
    print(f"Failed with gemini-3-pro-preview: {e}")

try:
    print("\nListing models with this key...")
    for m in client.models.list():
        print(f"Available: {m.name}")
except Exception as e3:
    print(f"Could not list models: {e3}")
