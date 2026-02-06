import os
import sys
import logging
import argparse
import time
import re
from typing import List, Dict, Optional
from dotenv import load_dotenv

# Add parent directory to path to import src modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.image_generator import ImageGenerator, WordPressMediaUploader
from src.clients.gemini import GeminiClient
from src.clients.wordpress import WordPressClient

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
        
        # Initialize clients
        self.wp_client = WordPressClient(
            os.environ.get("WP_URL"),
            os.environ.get("WP_USER"),
            os.environ.get("WP_APP_PASSWORD")
        )
        self.gemini_client = GeminiClient(os.environ.get("GEMINI_API_KEY"))
        self.image_generator = ImageGenerator(gemini_client=self.gemini_client)
        self.media_uploader = WordPressMediaUploader(
            os.environ.get("WP_URL"),
            os.environ.get("WP_USER"),
            os.environ.get("WP_APP_PASSWORD")
        )
        
        logger.info(f"Maintenance Manager initialized (Dry Run: {self.dry_run})")
        
        if not os.environ.get("HUGGINGFACE_API_KEY"):
            logger.warning("⚠️ HUGGINGFACE_API_KEY is missing. Hugging Face generation will be skipped.")
        if not os.environ.get("OPENAI_API_KEY"):
            logger.warning("⚠️ OPENAI_API_KEY is missing. DALL-E 3 generation will be skipped.")
        
        # Check Gemini status
        if self.gemini_client.is_using_vertexai():
            logger.info("ℹ️ Using Vertex AI. Gemini Consumer API (Image Preview) may be unavailable.")

    def scan_posts(self, limit: int = 100, status: str = 'publish') -> List[Dict]:
        """Fetch posts from WordPress using standardized client."""
        logger.info(f"Scanning up to {limit} posts with status '{status}'...")
        return self.wp_client.fetch_posts(params={"status": status, "per_page": limit})

    def process_post(self, post: Dict):
        """Analyze and fix a single post."""
        post_id = post.get('id')
        title = post.get('title', {}).get('rendered', 'Untitled')
        content = post.get('content', {}).get('rendered', '')
        
        logger.info(f"Processing Post {post_id}: '{title}'")
        
        updates_needed = False
        new_content = content
        
        # 1. Check Featured Image
        if not post.get('featured_media'):
            logger.info(f"  [MISSING] Featured Image")
            if not self.dry_run:
                media_id = self.fix_featured_image(post_id, title)
                if media_id:
                    self.wp_client.update_post(post_id, {'featured_media': media_id})
                    logger.info(f"  [FIXED] Set featured media to ID {media_id}")
        
        # 2. Check Image Placeholders
        placeholders = re.findall(r'\[IMAGE_PLACE_HOLDER_\d+\]|\[IMAGE_PLACEHOLDER_\d+\]', content)
        if placeholders:
            logger.info(f"  [FOUND] {len(placeholders)} image placeholders: {placeholders}")
            new_content = self.fix_placeholders(content, title, placeholders)
            if new_content != content:
                updates_needed = True
        
        # 3. Apply Content Updates
        if updates_needed:
            if not self.dry_run:
                success = self.wp_client.update_post(post_id, {'content': new_content})
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
            prompt = f"illustration for '{title}', section context {placeholder}, detailed, professional"
             
            if self.dry_run:
                logger.info(f"  [DRY RUN] Would generate image for {placeholder} with prompt: {prompt}")
                continue
                
            logger.info(f"  Generating image for {placeholder}...")
            image_data = self.image_generator.generate_image(prompt)
            
            if image_data:
                filename = f"content_{int(time.time())}_{placeholder.strip('[]').replace('_', '')}.jpg"
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
            # We don't have a direct get_media in WordPressClient, but we can use fetch_terms or a generic request
            # Let's use a simpler way since WP client doesn't have it yet
            url = f"{self.wp_client.wp_url}/wp-json/wp/v2/media/{media_id}"
            response = self.wp_client.session.get(url, headers=self.wp_client.headers)
            if response.status_code == 200:
                return response.json().get('source_url')
        except Exception as e:
            logger.error(f"Error fetching media URL: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description="WordPress Auto-Blogging Maintenance Manager")
    parser.add_argument("--dry-run", action="store_true", help="Scan without making changes")
    parser.add_argument("--limit", type=int, default=10, help="Max posts to scan")
    parser.add_argument("--post-id", type=int, help="Process a specific post ID only")
    
    args = parser.parse_args()
    
    manager = MaintenanceManager(dry_run=args.dry_run)
    
    if args.post_id:
        logger.info(f"Fetching single post {args.post_id}...")
        post = manager.wp_client.get_post(args.post_id)
        if post:
            manager.process_post(post)
        else:
            logger.error(f"Post {args.post_id} not found.")
    else:
        posts = manager.scan_posts(limit=args.limit)
        for post in posts:
            manager.process_post(post)

if __name__ == "__main__":
    main()
