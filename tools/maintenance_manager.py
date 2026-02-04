import os
import sys
import logging
import argparse
import time
import re
from typing import List, Dict, Optional
from dotenv import load_dotenv
import requests
import base64

# Add parent directory to path to import src modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.image_generator import ImageGenerator, WordPressMediaUploader
from src.clients.gemini import GeminiClient

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("maintenance.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("MaintenanceManager")

class MaintenanceManager:
    def __init__(self, dry_run: bool = False):
        load_dotenv(override=True)
        self.dry_run = dry_run
        self.wp_url = os.environ.get("WP_URL").rstrip('/')
        self.wp_user = os.environ.get("WP_USER")
        self.wp_pass = os.environ.get("WP_APP_PASSWORD")
        self.auth = (self.wp_user, self.wp_pass)
        
        # Initialize clients
        self.gemini_client = GeminiClient()
        self.image_generator = ImageGenerator(gemini_client=self.gemini_client)
        self.media_uploader = WordPressMediaUploader(self.wp_url, self.wp_user, self.wp_pass)
        
        logger.info(f"Maintenance Manager initialized (Dry Run: {self.dry_run})")
        
        if not os.environ.get("HUGGINGFACE_API_KEY"):
            logger.warning("⚠️ HUGGINGFACE_API_KEY is missing. Hugging Face generation will be skipped.")
        if not os.environ.get("OPENAI_API_KEY"):
            logger.warning("⚠️ OPENAI_API_KEY is missing. DALL-E 3 generation will be skipped.")
        
        # Check Gemini status
        if self.gemini_client.is_using_vertexai():
            logger.info("ℹ️ Using Vertex AI. Gemini Consumer API (Image Preview) may be unavailable.")

    def scan_posts(self, limit: int = 100, status: str = 'publish') -> List[Dict]:
        """Fetch posts from WordPress."""
        logger.info(f"Scanning up to {limit} posts with status '{status}'...")
        all_posts = []
        page = 1
        per_page = 20
        
        while len(all_posts) < limit:
            try:
                url = f"{self.wp_url}/wp-json/wp/v2/posts?status={status}&per_page={per_page}&page={page}"
                response = requests.get(url, auth=self.auth)
                
                if response.status_code != 200:
                    logger.error(f"Error fetching posts: {response.status_code} - {response.text}")
                    break
                    
                posts = response.json()
                if not posts:
                    break
                    
                all_posts.extend(posts)
                logger.info(f"Fetched {len(posts)} posts (Total: {len(all_posts)})")
                
                page += 1
                if len(posts) < per_page:
                    break
                    
            except Exception as e:
                logger.error(f"Exception during scan: {e}")
                break
                
        return all_posts[:limit]

    def process_post(self, post: Dict):
        """Analyze and fix a single post."""
        post_id = post.get('id')
        title = post.get('title', {}).get('rendered', 'Untitled')
        content = post.get('content', {}).get('rendered', '')
        link = post.get('link', '')
        
        logger.info(f"Processing Post {post_id}: '{title}'")
        
        updates_needed = False
        new_content = content
        
        # 1. Check Featured Image
        if not post.get('featured_media'):
            logger.info(f"  [MISSING] Featured Image")
            if not self.dry_run:
                media_id = self.fix_featured_image(post_id, title)
                if media_id:
                    # We can't update featured_media via content update, need separate call or separate logic
                    # standard WP API allows updating 'featured_media' in the POST body
                    self.update_post_meta(post_id, {'featured_media': media_id})
                    logger.info(f"  [FIXED] Set featured media to ID {media_id}")
        
        # 2. Check Image Placeholders
        placeholders = re.findall(r'\[IMAGE_PLACEHOLDER_\d+\]', content)
        if placeholders:
            logger.info(f"  [FOUND] {len(placeholders)} image placeholders: {placeholders}")
            new_content = self.fix_placeholders(content, title, placeholders)
            if new_content != content:
                updates_needed = True
        
        # 3. Apply Content Updates
        if updates_needed:
            if not self.dry_run:
                success = self.update_post_content(post_id, new_content)
                if success:
                    logger.info(f"  [SUCCESS] Post {post_id} content updated.")
                else:
                    logger.error(f"  [FAILED] Could not update post {post_id}.")
            else:
                logger.info(f"  [DRY RUN] Would update post {post_id} content.")
        else:
            logger.info(f"  [OK] No content updates needed for post {post_id}.")

    def fix_featured_image(self, post_id: int, title: str) -> Optional[int]:
        """Generate and upload a featured image."""
        prompt = f"featured image for blog post titled '{title}', professional, high quality, 4k"
        logger.info(f"  Generating featured image for '{title}'...")
        
        image_data = self.image_generator.generate_image(prompt)
        if not image_data:
            logger.error("  Failed to generate image.")
            return None
            
        filename = f"featured_{post_id}_{int(time.time())}.jpg"
        filepath = self.image_generator.save_image(image_data, filename)
        
        if filepath:
            media_id = self.media_uploader.upload_media(filepath, alt_text=title, title=title)
            return media_id
        return None

    def fix_placeholders(self, content: str, title: str, placeholders: List[str]) -> str:
        """Replace placeholders with generated images."""
        fixed_content = content
        
        for placeholder in placeholders:
            # Try to find context (heading before the placeholder)
            # This is a simple heuristic; regex could be improved to find nearest H2/H3
            # For now, we'll use the title + placeholder index as context to vary images
            
            prompt = f"illustration for '{title}', section context {placeholder}, detailed, professional"
             
            if self.dry_run:
                logger.info(f"  [DRY RUN] Would generate image for {placeholder} with prompt: {prompt}")
                continue
                
            logger.info(f"  Generating image for {placeholder}...")
            image_data = self.image_generator.generate_image(prompt)
            
            if image_data:
                filename = f"content_{int(time.time())}_{placeholder.strip('[]')}.jpg"
                filepath = self.image_generator.save_image(image_data, filename)
                
                if filepath:
                    media_id = self.media_uploader.upload_media(filepath, alt_text=f"Image for {title}", title=title)
                    if media_id:
                        # Fetch the source URL of the uploaded media
                        src_url = self.get_media_url(media_id)
                        if src_url:
                            img_html = f'<figure class="wp-block-image"><img src="{src_url}" alt="{title}" class="wp-image-{media_id}"/></figure>'
                            fixed_content = fixed_content.replace(placeholder, img_html)
                            logger.info(f"  Replaced {placeholder} with image ID {media_id}")
            else:
                logger.warning(f"  Failed to generate image for {placeholder}")
                
        return fixed_content

    def get_media_url(self, media_id: int) -> Optional[str]:
        """Fetch the source URL for a media item."""
        try:
            url = f"{self.wp_url}/wp-json/wp/v2/media/{media_id}"
            response = requests.get(url, auth=self.auth)
            if response.status_code == 200:
                return response.json().get('source_url')
        except Exception as e:
            logger.error(f"Error fetching media URL: {e}")
        return None

    def update_post_content(self, post_id: int, new_content: str) -> bool:
        """Update post content."""
        url = f"{self.wp_url}/wp-json/wp/v2/posts/{post_id}"
        try:
            response = requests.post(url, auth=self.auth, json={'content': new_content})
            return response.status_code == 200
        except Exception as e:
            logger.error(f"Error updating post content: {e}")
            return False

    def update_post_meta(self, post_id: int, data: Dict) -> bool:
        """Update arbitrary post fields."""
        url = f"{self.wp_url}/wp-json/wp/v2/posts/{post_id}"
        try:
            response = requests.post(url, auth=self.auth, json=data)
            return response.status_code == 200
        except Exception as e:
            logger.error(f"Error updating post meta: {e}")
            return False

def main():
    parser = argparse.ArgumentParser(description="WordPress Auto-Blogging Maintenance Manager")
    parser.add_argument("--dry-run", action="store_true", help="Scan without making changes")
    parser.add_argument("--limit", type=int, default=10, help="Max posts to scan")
    parser.add_argument("--post-id", type=int, help="Process a specific post ID only")
    
    args = parser.parse_args()
    
    manager = MaintenanceManager(dry_run=args.dry_run)
    
    if args.post_id:
        # Process single post logic would need to be added to scan_posts or handled here
        # For now, let's reuse scan logic but filter or fetch single
        logger.info(f"Fetching single post {args.post_id}...")
        url = f"{manager.wp_url}/wp-json/wp/v2/posts/{args.post_id}"
        try:
            resp = requests.get(url, auth=manager.auth)
            if resp.status_code == 200:
                manager.process_post(resp.json())
            else:
                logger.error(f"Post {args.post_id} not found.")
        except Exception as e:
            logger.error(f"Error fetching post: {e}")
    else:
        posts = manager.scan_posts(limit=args.limit)
        for post in posts:
            manager.process_post(post)

if __name__ == "__main__":
    main()
