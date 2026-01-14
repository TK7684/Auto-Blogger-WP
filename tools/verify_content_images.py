import requests
import os
from dotenv import load_dotenv

load_dotenv(override=True)

wp_url = os.environ.get("WP_URL").rstrip('/')
wp_user = os.environ.get("WP_USER")
wp_pass = os.environ.get("WP_APP_PASSWORD")
post_id = 1381

url = f"{wp_url}/wp-json/wp/v2/posts/{post_id}"
auth = (wp_user, wp_pass)

try:
    response = requests.get(url, auth=auth)
    if response.status_code == 200:
        content = response.json().get('content', {}).get('rendered', '')
        if '<img' in content:
            print("✅ FOUND IMAGES IN CONTENT!")
            import re
            imgs = re.findall(r'<img[^>]+src="([^">]+)"', content)
            for i, src in enumerate(imgs):
                print(f"Image {i+1}: {src}")
        else:
            print("❌ NO IMAGES FOUND IN CONTENT.")
    else:
        print(f"Error: {response.status_code}")
except Exception as e:
    print(f"Exception: {e}")
