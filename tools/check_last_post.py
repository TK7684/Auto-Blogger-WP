import requests
import os
import base64
from dotenv import load_dotenv

load_dotenv(override=True)

wp_url = os.environ.get("WP_URL").rstrip('/')
wp_user = os.environ.get("WP_USER")
wp_pass = os.environ.get("WP_APP_PASSWORD")

url = f"{wp_url}/wp-json/wp/v2/posts?per_page=1"
credentials = f"{wp_user}:{wp_pass}"
token = base64.b64encode(credentials.encode()).decode()
headers = {'Authorization': f'Basic {token}'}

try:
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        posts = response.json()
        if posts:
            print(f"Latest Post: {posts[0]['title']['rendered']}")
        else:
            print("No posts found.")
    else:
        print(f"Error: {response.status_code}")
except Exception as e:
    print(f"Exception: {e}")
