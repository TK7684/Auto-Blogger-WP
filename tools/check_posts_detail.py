import requests
import os
import base64
from dotenv import load_dotenv

load_dotenv(override=True)

wp_url = os.environ.get("WP_URL").rstrip('/')
wp_user = os.environ.get("WP_USER")
wp_pass = os.environ.get("WP_APP_PASSWORD")

url = f"{wp_url}/wp-json/wp/v2/posts?per_page=10"
credentials = f"{wp_user}:{wp_pass}"
token = base64.b64encode(credentials.encode()).decode()
headers = {'Authorization': f'Basic {token}'}

try:
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        posts = response.json()
        print(f"Checking last 5 posts (any status):")
        for post in posts:
            print(f"ID: {post['id']} | Status: {post['status']} | Date: {post['date']} | Title: {post['title']['rendered']}")
    else:
        print(f"Error: {response.status_code} - {response.text}")
except Exception as e:
    print(f"Exception: {e}")
