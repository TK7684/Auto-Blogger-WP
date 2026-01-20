"""
WordPress Content Auditor & Maintenance Module.
Optimizes old posts for SEO, updates outdated information, fixes links, and ensures featured images exist.
"""

import logging
import re
from typing import List, Dict, Optional, Any
from pydantic import BaseModel, Field
from datetime import datetime
from .clients.wordpress import WordPressClient
from .clients.gemini import GeminiClient
from .image_generator import ImageGenerator, WordPressMediaUploader
from .yoast_seo import YoastSEOIntegrator

logger = logging.getLogger(__name__)

class AuditResult(BaseModel):
    updated_content: str = Field(description="The full optimized HTML content")
    updated_title: str = Field(description="Optimized, high-CTR title")
    focus_keyword: str = Field(description="The primary keyword (2-4 words)")
    meta_description: str = Field(description="SEO meta description (150-160 chars)")
    fact_check_notes: str = Field(description="Notes on what was updated")
    seo_improvements: str = Field(description="Summary of SEO changes")
    fixed_links: str = Field(description="Summary of link fixes")

class ContentAuditor:
    def __init__(self, wp_client: WordPressClient, gemini_client: GeminiClient, 
                 image_gen: Optional[ImageGenerator] = None, 
                 media_uploader: Optional[WordPressMediaUploader] = None,
                 yoast_integrator: Optional[YoastSEOIntegrator] = None):
        self.wp_client = wp_client
        self.gemini_client = gemini_client
        self.image_gen = image_gen
        self.media_uploader = media_uploader
        self.yoast_integrator = yoast_integrator

    def fetch_all_posts(self, per_page: int = 20) -> List[Dict]:
        """Fetch latest published posts."""
        return self.wp_client.fetch_posts(params={"per_page": per_page, "status": "publish"})

    def clean_links(self, content: str) -> str:
        """Fix broken links and formatting."""
        # 1. Fix placeholders
        def replace_placeholder(match):
            keyword = match.group(1)
            return f'<a href="/?s={keyword.replace(" ", "+")}">{keyword}</a>'
        
        content = re.sub(r'\[INSERT_INTERNAL_LINK:(.*?)\]', replace_placeholder, content)
        # 2. Basic cleanup of source tags if they are just text
        content = re.sub(r'\[Source: (.*?)\]', r'<blockquote>Source: \1</blockquote>', content)
        return content

    def process_inline_images(self, content: str, post_title: str) -> str:
        """Find [Image: ...] tags, generate images, and replace with <img> tags."""
        if not self.image_gen or not self.media_uploader:
            return content

        patterns = re.findall(r'\[Image: (.*?)\]', content)
        for prompt in patterns:
            logger.info(f"🎨 Generating inline image for prompt: {prompt}")
            try:
                # Generate
                img_bytes = self.image_gen.generate_image(f"{prompt} - photorealistic, high quality, related to {post_title}")
                if img_bytes:
                    fname = f"inline_{datetime.now().strftime('%M%S_%f')}.jpg"
                    path = self.image_gen.save_image(img_bytes, fname)
                    if path:
                        # Upload
                        media_id = self.media_uploader.upload_media(path, prompt, prompt)
                        if media_id:
                            # Verify URL
                            # We need the source URL. media_uploader returns ID. 
                            # We can try to guess or fetch. 
                            # Since we don't have a lookup, let's assume we can't easily get the URL without an extra call.
                            # But wait, WordPressMediaUploader logic?
                            # Let's verify if we can get the URL.
                            # For simplicity, we just won't be able to insert the <img src> without the URL.
                            # We'll fetch the media details.
                            media_url = self._fetch_media_url(media_id)
                            if media_url:
                                img_tag = f'<img src="{media_url}" alt="{prompt}" class="wp-image-{media_id}" />'
                                content = content.replace(f'[Image: {prompt}]', img_tag)
            except Exception as e:
                logger.error(f"Failed inline image: {e}")
        
        return content

    def _fetch_media_url(self, media_id: int) -> str:
        """Helper to get source URL for connection."""
        try:
             # This is a raw call, ideally we'd add to WPClient, but local helper is fine
             resp = self.wp_client.session.get(f"{self.wp_client.api_url}/media/{media_id}")
             if resp.status_code == 200:
                 return resp.json().get("source_url", "")
        except:
            pass
        return ""

    def check_and_fix_image(self, post_id: int, post_title: str, featured_media: int) -> Optional[int]:
        """Check if post has featured image. If not (id=0), generate and attach one."""
        if featured_media != 0:
            return None 

        if not self.image_gen or not self.media_uploader:
            logger.warning("Skipping image fix: components not available")
            return None

        logger.info(f"🖼️ Missing featured image for '{post_title}'. Generating...")
        try:
            img_bytes = self.image_gen.generate_image(f"Featured image for blog post: {post_title}")
            if img_bytes:
                fname = f"fix_{post_id}_{datetime.now().strftime('%M%S')}.jpg"
                path = self.image_gen.save_image(img_bytes, fname)
                if path:
                    media_id = self.media_uploader.upload_media(path, post_title, post_title)
                    return media_id
        except Exception as e:
            logger.error(f"Image fix failed: {e}")
        return None

    def audit_post(self, post_id: int, title: str, content: str, focus_keyword: str = "") -> Optional[AuditResult]:
        """Audit a single post for SEO, factual accuracy, and links."""
        logger.info(f"🧐 Auditing post {post_id}: {title}")
        
        # Pre-cleaning
        content = self.clean_links(content)
        # Inline images processing
        content = self.process_inline_images(content, title)

        prompt = f"""
        You are a senior SEO editor. Audit and optimize these blog post elements.
        
        TITLE: {title}
        CONTENT (HTML): {content}
        
        REQUIREMENTS:
        1. **SEO Optimization (CRITICAL)**:
           - Identify a strong **Focus Keyword** (if not clear, pick the best one).
           - Write a compelling **Meta Description** (max 160 chars, include keyword).
           - **Rewrite the First Paragraph**: Ensure the Focus Keyword appears naturally in the first 2 sentences.
           - Ensure H2/H3 hierarchy is perfect.
        
        2. **Link & Format Fixing**: 
           - Convert raw URLs to HTML links.
           - Replace any remaining `[INSERT_INTERNAL_LINK:keyword]` with `<a href="/?s=keyword">keyword</a>`.
           - Format data tables or lists properly.
        
        3. **Fact Check**: Update old years (2024->2026).
        
        Return the structured JSON including the full updated HTML content.
        """
        
        try:
            response = self.gemini_client.generate_structured_output(
                model="gemini-2.0-flash",
                prompt=prompt,
                schema=AuditResult.model_json_schema()
            )
            
            if response:
                return AuditResult.model_validate_json(response.text)
        except Exception as e:
            logger.error(f"Audit generation failed: {e}")
        return None

    def run_maintenance(self, limit: int = 5):
        """Run full maintenance: audit content + fix images."""
        logger.info(f"🧹 Starting Maintenance Run (Limit: {limit})...")
        posts = self.fetch_all_posts(per_page=limit)
        
        for post in posts:
            pid = post['id']
            title = post['title']['rendered']
            # Get raw content if available (context edit), else rendered
            content = post.get('content', {}).get('raw', post.get('content', {}).get('rendered', ''))
            featured_media = post.get('featured_media', 0)
            
            # 1. Fix Featured Image
            new_media_id = self.check_and_fix_image(pid, title, featured_media)
            
            # 2. Audit Content & SEO
            audit_res = self.audit_post(pid, title, content)
            
            # 3. Apply Updates
            if audit_res or new_media_id:
                data = {}
                if new_media_id:
                    data['featured_media'] = new_media_id
                    
                if audit_res:
                    data['title'] = audit_res.updated_title
                    data['content'] = audit_res.updated_content
                    logger.info(f"📝 SEO: KW='{audit_res.focus_keyword}' | Desc='{audit_res.meta_description}'")

                if self.wp_client.update_post(pid, data):
                    logger.info(f"✅ Post {pid} updated content.")
                    
                    # 4. Update Yoast Data explicitly
                    if self.yoast_integrator and audit_res:
                        self.yoast_integrator.update_yoast_meta_fields(pid, {
                            "focus_keyword": audit_res.focus_keyword,
                            "seo_title": audit_res.updated_title,
                            "meta_description": audit_res.meta_description
                        })
                else:
                    logger.error(f"Failed to update Post {pid}")

