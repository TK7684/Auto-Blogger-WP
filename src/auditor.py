"""
WordPress Content Auditor Module.
Optimizes old posts for SEO and updates outdated information using Gemini.
"""

import logging
from typing import List, Dict, Optional
from pydantic import BaseModel, Field
from .clients.wordpress import WordPressClient
from .clients.gemini import GeminiClient

logger = logging.getLogger(__name__)

class AuditResult(BaseModel):
    updated_content: str = Field(description="The full optimized HTML content")
    updated_title: str = Field(description="Optimized, high-CTR title (if the current one is weak)")
    fact_check_notes: str = Field(description="Notes on what was updated or fact-checked")
    seo_improvements: str = Field(description="Summary of SEO changes made")

class ContentAuditor:
    def __init__(self, wp_client: WordPressClient, gemini_client: GeminiClient):
        self.wp_client = wp_client
        self.gemini_client = gemini_client

    def fetch_all_posts(self, per_page: int = 20) -> List[Dict]:
        """Fetch latest published posts."""
        return self.wp_client.fetch_posts(params={"per_page": per_page, "status": "publish"})

    def audit_post(self, post_id: int, title: str, content: str) -> Optional[AuditResult]:
        """Audit a single post for SEO and factual accuracy."""
        logger.info(f"🧐 Auditing post {post_id}: {title}")
        
        prompt = f"""
        You are a senior SEO editor and fact-checker. Audit the following WordPress post:
        
        TITLE: {title}
        CONTENT: {content}
        
        TASKS:
        1. **Fact Check & Debug**: Search for any outdated information, broken logic, or old statistics. Update them to be current for 2026. This is CRITICAL.
        2. **Title Optimization**: Review the title. If it does not grab attention in 3 seconds (boring, too long), rewrite it to be a high-CTR, "click-bait" style professional title (max 60 chars).
        3. **SEO Optimization**: Improve heading hierarchy (H2, H3), ensure keywords are used naturally, and add alt text placeholders if missing.
        4. **Readability**: Break up long paragraphs and ensure the tone is professional yet engaging.
        5. **Internal Linking**: If you see opportunities to link to generic topics, use the search-style link: <a href="/?s=topic">topic</a>.
        
        Return the optimized content, new title, and notes in JSON format.
        """
        
        try:
            # Using the centralized Gemini client with fixed model configuration
            response = self.gemini_client.generate_structured_output(
                model="gemini-2.0-flash", # Using the latest stable model
                prompt=prompt,
                schema=AuditResult.model_json_schema()
            )
            
            if response:
                return AuditResult.model_validate_json(response.text)
        except Exception as e:
            logger.error(f"Audit generation failed: {e}")
        return None

    def update_post(self, post_id: int, audit_result: AuditResult):
        """Update the post in WordPress."""
        data = {
            "title": audit_result.updated_title,
            "content": audit_result.updated_content
        }
        if self.wp_client.update_post(post_id, data):
            logger.info(f"✅ Success! Post {post_id} updated.")
            logger.info(f"📝 Notes: {audit_result.fact_check_notes}")
        else:
            logger.error(f"Update failed for {post_id}")

    def run_full_audit(self, limit: int = 5):
        """Run audit on the latest posts."""
        posts = self.fetch_all_posts(per_page=limit)
        for post in posts:
            title = post['title']['rendered']
            content = post['content']['raw'] if 'raw' in post['content'] else post['content']['rendered']
            res = self.audit_post(post['id'], title, content)
            if res:
                self.update_post(post['id'], res)
