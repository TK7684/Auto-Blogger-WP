"""
WordPress Content Auditor & Maintenance Module.
Optimizes old posts for SEO, updates outdated information, and fixes links.
Focuses on content quality and SEO - skips image generation.
"""

import logging
import re
from typing import List, Dict, Optional
from pydantic import BaseModel, Field
from .clients.wordpress import WordPressClient
from .clients.gemini import GeminiClient
from .yoast_seo import YoastSEOIntegrator
from .utils.linking import resolve_internal_links, clean_remaining_placeholders

logger = logging.getLogger(__name__)


class AuditResult(BaseModel):
    """Result of auditing and optimizing a blog post."""
    updated_content: str = Field(description="The full optimized HTML content")
    updated_title: str = Field(description="Optimized, high-CTR title")
    focus_keyword: str = Field(description="The primary keyword (2-4 words)")
    meta_description: str = Field(description="SEO meta description (150-160 chars)")
    excerpt: str = Field(description="Short summary for archive pages")
    fact_check_notes: str = Field(description="Notes on what was updated")
    seo_improvements: str = Field(description="Summary of SEO changes")
    links_resolved: int = Field(description="Number of internal links resolved", default=0)


class ContentAuditor:
    """Audits and optimizes WordPress content for SEO and quality."""

    def __init__(self, wp_client: WordPressClient, gemini_client: GeminiClient,
                 yoast_integrator: Optional[YoastSEOIntegrator] = None):
        self.wp_client = wp_client
        self.gemini_client = gemini_client
        self.yoast_integrator = yoast_integrator
        self._existing_posts_cache: Optional[List[Dict[str, str]]] = None

    def fetch_all_posts(self, per_page: int = 100, status: str = "publish") -> List[Dict]:
        """Fetch posts from WordPress."""
        logger.info(f"📥 Fetching posts (status={status}, per_page={per_page})...")
        posts = self.wp_client.fetch_posts(params={"per_page": per_page, "status": status})
        logger.info(f"📥 Fetched {len(posts)} posts")
        return posts

    def get_existing_posts_for_linking(self, force_refresh: bool = False) -> List[Dict[str, str]]:
        """
        Get existing posts for internal linking. Caches results to avoid repeated API calls.

        Returns:
            List of dicts with 'title' and 'url' keys
        """
        if self._existing_posts_cache is None or force_refresh:
            self._existing_posts_cache = self.wp_client.get_existing_links(limit=100)
            logger.info(f"🔗 Loaded {len(self._existing_posts_cache)} posts for internal linking")
        return self._existing_posts_cache

    def resolve_content_links(self, content: str) -> str:
        """
        Resolve all internal link placeholders in content.

        Args:
            content: HTML content with potential placeholders

        Returns:
            Content with placeholders resolved to actual links
        """
        existing_posts = self.get_existing_posts_for_linking()
        resolved = resolve_internal_links(content, existing_posts)
        return clean_remaining_placeholders(resolved)

    def audit_post(self, post_id: int, title: str, content: str) -> Optional[AuditResult]:
        """
        Audit a single post for SEO, content quality, and internal linking.

        Args:
            post_id: WordPress post ID
            title: Current post title
            content: Current post content (HTML)

        Returns:
            AuditResult with optimized content and metadata
        """
        logger.info(f"🧐 Auditing post {post_id}: '{title}'")

        # First, resolve any remaining link placeholders
        pre_processed_content = self.resolve_content_links(content)

        # Count placeholders before
        placeholders_before = len(re.findall(r'\[INSERT_INTERNAL_LINK:', content))
        placeholders_after = len(re.findall(r'\[INSERT_INTERNAL_LINK:', pre_processed_content))
        links_resolved = placeholders_before - placeholders_after

        if placeholders_before > 0:
            logger.info(f"🔗 Pre-processed {links_resolved} link placeholders")

        # Build the audit prompt
        prompt = f"""You are a senior SEO editor and content strategist. Analyze and optimize this blog post.

CURRENT TITLE: {title}

CURRENT CONTENT (HTML):
{pre_processed_content[:8000]}

YOUR TASK - Optimize for SEO, readability, and user engagement:

1. **IDENTIFY FOCUS KEYWORD**:
   - Extract the primary 2-4 word focus keyword
   - This should be the main topic/phrase the article targets

2. **OPTIMIZE TITLE** (max 60 characters):
   - Make it compelling and click-worthy
   - Include the focus keyword near the beginning
   - Use power words: "Ultimate", "Complete", "Essential", "Proven"
   - Create curiosity or urgency

3. **WRITE META DESCRIPTION** (150-160 characters):
   - Include the focus keyword
   - Tease the value/benefit
   - Include a call-to-action

4. **WRITE EXCERPT** (2 sentences):
   - Hook readers immediately
   - Promise value

5. **OPTIMIZE CONTENT**:
   - Ensure focus keyword appears in first paragraph naturally
   - Fix heading hierarchy (H1 → H2 → H3)
   - Break long paragraphs into shorter ones (2-3 sentences max)
   - Add bullet points or numbered lists where appropriate
   - Convert raw URLs (http/https) to proper HTML links
   - Ensure proper HTML formatting

6. **FACT CHECK**:
   - Update outdated years (2024 → 2026)
   - Verify statistics are current
   - Update any temporal references

7. **PRESERVE**:
   - All existing HTML img tags exactly as-is
   - All existing valid internal links
   - Schema markup and any scripts

Return the result as JSON following the schema exactly."""

        try:
            response = self.gemini_client.generate_structured_output(
                model="gemini-2.0-flash",
                prompt=prompt,
                schema=AuditResult.model_json_schema()
            )

            if response:
                result = AuditResult.model_validate_json(response.text)
                result.links_resolved = links_resolved
                logger.info(f"✅ Audit complete: KW='{result.focus_keyword}' | Title='{result.updated_title}'")
                return result

        except Exception as e:
            logger.error(f"❌ Audit generation failed for post {post_id}: {e}")

        return None

    def apply_audit_results(self, post_id: int, audit_result: AuditResult) -> bool:
        """
        Apply audit results to a WordPress post.

        Args:
            post_id: WordPress post ID
            audit_result: AuditResult with optimized content

        Returns:
            True if update was successful
        """
        update_data = {
            "title": audit_result.updated_title,
            "content": audit_result.updated_content,
            "excerpt": audit_result.excerpt
        }

        logger.info(f"💾 Updating post {post_id}...")

        if self.wp_client.update_post(post_id, update_data):
            logger.info(f"✅ Post {post_id} content updated")

            # Update Yoast SEO fields
            if self.yoast_integrator:
                self.yoast_integrator.update_yoast_meta_fields(post_id, {
                    "focus_keyword": audit_result.focus_keyword,
                    "seo_title": audit_result.updated_title,
                    "meta_description": audit_result.meta_description
                })
                logger.info(f"✅ Yoast SEO fields updated for post {post_id}")

            return True

        logger.error(f"❌ Failed to update post {post_id}")
        return False

    def run_maintenance(self, limit: int = 10) -> Dict[str, any]:
        """
        Run full maintenance: audit and optimize multiple posts.

        Args:
            limit: Maximum number of posts to process

        Returns:
            Summary dict with stats
        """
        logger.info(f"🧹 Starting Maintenance Run (Limit: {limit})...")
        logger.info(f"🔗 Loading existing posts for internal linking...")

        # Pre-load posts cache for internal linking
        self.get_existing_posts_for_linking(force_refresh=True)

        posts = self.fetch_all_posts(per_page=limit)

        stats = {
            "total": len(posts),
            "processed": 0,
            "updated": 0,
            "failed": 0,
            "links_resolved": 0
        }

        for post in posts:
            post_id = post['id']
            title = post['title']['rendered']
            content = post.get('content', {}).get('raw', post.get('content', {}).get('rendered', ''))

            audit_result = self.audit_post(post_id, title, content)

            if audit_result:
                stats["links_resolved"] += audit_result.links_resolved

                if self.apply_audit_results(post_id, audit_result):
                    stats["updated"] += 1
                else:
                    stats["failed"] += 1

            stats["processed"] += 1

        # Log summary
        logger.info(f"🧹 Maintenance Complete:")
        logger.info(f"   Total: {stats['total']}")
        logger.info(f"   Processed: {stats['processed']}")
        logger.info(f"   Updated: {stats['updated']}")
        logger.info(f"   Failed: {stats['failed']}")
        logger.info(f"   Links Resolved: {stats['links_resolved']}")

        return stats

    def run_link_fix_only(self, limit: int = 50) -> Dict[str, any]:
        """
        Quick mode: Only fix internal link placeholders without full SEO audit.

        Args:
            limit: Maximum number of posts to process

        Returns:
            Summary dict with stats
        """
        logger.info(f"🔗 Running Link-Fix Only Mode (Limit: {limit})...")

        # Pre-load posts cache
        self.get_existing_posts_for_linking(force_refresh=True)

        posts = self.fetch_all_posts(per_page=limit)

        stats = {
            "total": len(posts),
            "processed": 0,
            "updated": 0,
            "links_resolved": 0
        }

        for post in posts:
            post_id = post['id']
            content = post.get('content', {}).get('raw', post.get('content', {}).get('rendered', ''))

            # Count placeholders before
            placeholders_before = len(re.findall(r'\[INSERT_INTERNAL_LINK:', content))

            if placeholders_before == 0:
                continue

            # Resolve links
            resolved_content = self.resolve_content_links(content)

            # Count after
            placeholders_after = len(re.findall(r'\[INSERT_INTERNAL_LINK:', resolved_content))
            links_resolved = placeholders_before - placeholders_after

            if links_resolved > 0:
                if self.wp_client.update_post(post_id, {"content": resolved_content}):
                    stats["updated"] += 1
                    stats["links_resolved"] += links_resolved
                    logger.info(f"✅ Post {post_id}: Resolved {links_resolved} links")

            stats["processed"] += 1

        logger.info(f"🔗 Link Fix Complete:")
        logger.info(f"   Total: {stats['total']}")
        logger.info(f"   Updated: {stats['updated']}")
        logger.info(f"   Links Resolved: {stats['links_resolved']}")

        return stats
