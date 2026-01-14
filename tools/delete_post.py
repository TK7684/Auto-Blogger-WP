"""
Tool to delete or update a problematic WordPress post.
"""
import os
import base64
import requests
import logging
from dotenv import load_dotenv

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

WP_URL = os.environ.get("WP_URL")
WP_USER = os.environ.get("WP_USER")
WP_APP_PASSWORD = os.environ.get("WP_APP_PASSWORD")


def delete_post(post_id: int) -> bool:
    """Delete a WordPress post by ID."""
    if not all([WP_URL, WP_USER, WP_APP_PASSWORD]):
        logger.error("Missing WordPress credentials in environment variables")
        return False

    url = f"{WP_URL.rstrip('/')}/wp-json/wp/v2/posts/{post_id}"
    credentials = f"{WP_USER}:{WP_APP_PASSWORD}"
    token = base64.b64encode(credentials.encode()).decode('utf-8')
    headers = {"Authorization": f"Basic {token}"}

    try:
        response = requests.delete(url, headers=headers, timeout=20)
        if response.status_code in [200, 204]:
            logger.info(f"✅ Successfully deleted Post {post_id}")
            return True
        else:
            logger.error(f"❌ Failed to delete Post {post_id}: {response.status_code} - {response.text}")
            return False
    except Exception as e:
        logger.error(f"❌ Error deleting post: {e}")
        return False


def get_post(post_id: int) -> dict:
    """Get a WordPress post by ID."""
    if not all([WP_URL, WP_USER, WP_APP_PASSWORD]):
        logger.error("Missing WordPress credentials in environment variables")
        return None

    url = f"{WP_URL.rstrip('/')}/wp-json/wp/v2/posts/{post_id}"
    credentials = f"{WP_USER}:{WP_APP_PASSWORD}"
    token = base64.b64encode(credentials.encode()).decode('utf-8')
    headers = {"Authorization": f"Basic {token}"}

    try:
        response = requests.get(url, headers=headers, timeout=20)
        if response.status_code == 200:
            return response.json()
        else:
            logger.error(f"Failed to get Post {post_id}: {response.status_code}")
            return None
    except Exception as e:
        logger.error(f"Error getting post: {e}")
        return None


def update_post_status(post_id: int, status: str = "draft") -> bool:
    """Update post status to draft or trash."""
    if not all([WP_URL, WP_USER, WP_APP_PASSWORD]):
        logger.error("Missing WordPress credentials in environment variables")
        return False

    url = f"{WP_URL.rstrip('/')}/wp-json/wp/v2/posts/{post_id}"
    credentials = f"{WP_USER}:{WP_APP_PASSWORD}"
    token = base64.b64encode(credentials.encode()).decode('utf-8')
    headers = {
        "Authorization": f"Basic {token}",
        "Content-Type": "application/json"
    }

    try:
        response = requests.post(url, headers=headers, json={"status": status}, timeout=20)
        if response.status_code == 200:
            logger.info(f"✅ Successfully updated Post {post_id} to '{status}'")
            return True
        else:
            logger.error(f"❌ Failed to update Post {post_id}: {response.status_code} - {response.text}")
            return False
    except Exception as e:
        logger.error(f"❌ Error updating post: {e}")
        return False


if __name__ == "__main__":
    import sys

    # Post 1450 has the error content
    POST_ID = 1450

    print(f"Post 1450 Handler")
    print("=" * 50)

    # First, check what's in the post
    post = get_post(POST_ID)
    if post:
        print(f"Title: {post.get('title', {}).get('rendered', 'N/A')}")
        print(f"Status: {post.get('status', 'N/A')}")
        print(f"Content Preview: {post.get('content', {}).get('rendered', '')[:200]}...")

    print("\n" + "=" * 50)
    print("Choose an action:")
    print("1. Move to draft")
    print("2. Move to trash")
    print("3. Permanently delete")
    print("4. Exit without changes")

    if len(sys.argv) > 1:
        choice = sys.argv[1]
    else:
        choice = input("Enter choice (1-4): ").strip()

    if choice == "1":
        update_post_status(POST_ID, "draft")
    elif choice == "2":
        update_post_status(POST_ID, "trash")
    elif choice == "3":
        delete_post(POST_ID)
    else:
        print("No changes made.")
