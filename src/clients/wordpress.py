import base64
import logging
import requests
from typing import List, Dict, Optional, Tuple

logger = logging.getLogger(__name__)

class WordPressClient:
    """Centralized client for WordPress REST API interactions."""
    
    def __init__(self, wp_url: Optional[str], wp_user: Optional[str], wp_app_password: Optional[str]):
        if not wp_url:
            self.wp_url = ""
            logger.warning("WordPress URL is missing.")
        else:
            self.wp_url = wp_url.rstrip('/')
            
        self.wp_user = wp_user
        self.wp_app_password = wp_app_password
        self.session = requests.Session()
        
        if wp_user and wp_app_password:
            auth = f"{wp_user}:{wp_app_password}"
            self.token = base64.b64encode(auth.encode()).decode('utf-8')
            self.headers = {
                "Authorization": f"Basic {self.token}"
            }
        else:
            self.headers = {}
            logger.warning("WordPress credentials missing.")

    def fetch_posts(self, params: Dict = None) -> List[Dict]:
        if not self.wp_url: return []
        url = f"{self.wp_url}/wp-json/wp/v2/posts"
        params = params or {"per_page": 20}
        try:
            response = self.session.get(url, headers=self.headers, params=params, timeout=30)
            if response.status_code == 200:
                return response.json()
        except Exception as e:
            logger.error(f"Error fetching posts: {e}")
        return []

    def fetch_terms(self, taxonomy: str = "categories", params: Dict = None) -> List[Dict]:
        if not self.wp_url: return []
        url = f"{self.wp_url}/wp-json/wp/v2/{taxonomy}"
        params = params or {"per_page": 100}
        try:
            response = self.session.get(url, headers=self.headers, params=params, timeout=20)
            if response.status_code == 200:
                return response.json()
        except Exception as e:
            logger.warning(f"Failed to fetch {taxonomy}: {e}")
        return []

    def create_term(self, name: str, taxonomy: str = "categories") -> Optional[int]:
        if not self.wp_url: return None
        url = f"{self.wp_url}/wp-json/wp/v2/{taxonomy}"
        try:
            response = self.session.post(url, headers=self.headers, json={"name": name}, timeout=20)
            if response.status_code == 201:
                return response.json().get('id')
            elif response.status_code == 400: # Term might already exist
                # Fetch existing to get ID
                existing = self.fetch_terms(taxonomy, params={"search": name})
                for term in existing:
                    if term['name'].lower() == name.lower():
                        return term['id']
        except Exception as e:
            logger.warning(f"Failed to create {taxonomy} '{name}': {e}")
        return None

    def upload_media(self, image_path: str, alt_text: str, title: str) -> Tuple[Optional[int], Optional[str]]:
        if not self.wp_url: return None, None
        import os
        try:
            with open(image_path, 'rb') as f:
                image_data = f.read()
            
            headers = self.headers.copy()
            headers.update({
                "Content-Type": "image/jpeg",
                "Content-Disposition": f'attachment; filename="{os.path.basename(image_path)}"'
            })
            
            url = f"{self.wp_url}/wp-json/wp/v2/media"
            response = self.session.post(url, headers=headers, data=image_data, timeout=60)
            if response.status_code == 201:
                res_data = response.json()
                mid = res_data.get('id')
                source_url = res_data.get('source_url')
                self.session.post(f"{url}/{mid}", headers=self.headers, json={"alt_text": alt_text, "title": title})
                return mid, source_url
        except Exception as e:
            logger.error(f"Media upload error: {e}")
        return None, None

    def create_post(self, data: Dict) -> Optional[int]:
        if not self.wp_url: return None
        url = f"{self.wp_url}/wp-json/wp/v2/posts"
        try:
            response = self.session.post(url, headers=self.headers, json=data, timeout=30)
            if response.status_code == 201:
                return response.json().get('id')
            logger.error(f"Failed to create post: {response.status_code} - {response.text}")
        except Exception as e:
            logger.error(f"Error creating post: {e}")
        return None

    def get_existing_links(self, limit: int = 100) -> List[Dict[str, str]]:
        """Fetch list of titles and URLs/slugs for internal linking."""
        posts = self.fetch_posts(params={"per_page": limit, "status": "publish"})
        return [{"title": p['title']['rendered'], "url": p['link'], "slug": p['slug']} for p in posts]

    def update_post(self, post_id: int, data: Dict) -> bool:
        if not self.wp_url: return False
        url = f"{self.wp_url}/wp-json/wp/v2/posts/{post_id}"
        try:
            response = self.session.post(url, headers=self.headers, json=data, timeout=30)
            return response.status_code == 200
        except Exception as e:
            logger.error(f"Error updating post {post_id}: {e}")
        return False
