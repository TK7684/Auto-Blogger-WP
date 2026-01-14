import requests
import os
import base64
import json
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
    auth = (wp_user, wp_pass)
    r = requests.get(f"{wp_url}/wp-json/wp/v2/users/me", auth=auth)
    if r.status_code == 200:
        print(f"SUCCESS! Logged in as: {r.json().get('name')}")
    else:
        print(f"FAILED: {r.status_code} - {r.text}")
except Exception as e:
    print(f"ERROR: {str(e)}")
