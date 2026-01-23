"""
Auto-Blogging WordPress - Standardized & Optimized Version.
Key Features: Multilingual (TH/EN), Image Gen (Gemini/HF), Maintenance Mode, Centralized Clients.
"""

import os
import datetime
import logging
import sys
import json
from typing import Dict, List, Optional
from dotenv import load_dotenv

# Import centralized components
from src.trend_sources import get_hot_trend
from src.seo_system import SEOPromptBuilder, SchemaMarkupGenerator
from src.image_generator import ImageGenerator, WordPressMediaUploader
from src.research_agent import ResearchAgent
from src.auditor import ContentAuditor
from src.clients.gemini import GeminiClient
from src.clients.wordpress import WordPressClient
from src.yoast_seo import YoastSEOIntegrator
from src.utils.linking import resolve_internal_links

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load env
load_dotenv()

# --- CONFIGURATION ---
WP_URL = os.environ.get("WP_URL")
WP_USER = os.environ.get("WP_USER")
WP_APP_PASSWORD = os.environ.get("WP_APP_PASSWORD")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
SITE_URL = os.environ.get("SITE_URL", "https://pedpro.online")

# --- INITIALIZATION ---
def initialize_system() -> Dict:
    """Initialize all clients and agents."""
    if not all([WP_URL, WP_USER, WP_APP_PASSWORD, GEMINI_API_KEY]):
        logger.error("Missing critical environment variables.")
        return {}

    # 1. Clients
    wp_client = WordPressClient(WP_URL, WP_USER, WP_APP_PASSWORD)
    gemini_client = GeminiClient(GEMINI_API_KEY)

    # 2. Components
    image_gen = ImageGenerator(gemini_client)
    media_uploader = WordPressMediaUploader(WP_URL, WP_USER, WP_APP_PASSWORD)
    yoast = YoastSEOIntegrator(WP_URL, WP_USER, WP_APP_PASSWORD)
    
    auditor = ContentAuditor(wp_client, gemini_client, image_gen, media_uploader, yoast)
    researcher = ResearchAgent(wp_client, gemini_client)
    
    # 3. SEO
    seo_builder = SEOPromptBuilder()
    schema_gen = SchemaMarkupGenerator()

    return {
        "wp": wp_client,
        "gemini": gemini_client,
        "image": image_gen,
        "uploader": media_uploader,
        "auditor": auditor,
        "researcher": researcher,
        "seo": seo_builder,
        "schema": schema_gen,
        "yoast": yoast
    }

# --- PROCESSES ---

def run_content_generation(components: Dict, mode: str = "daily", manual_topic: str = None):
    """Execution flow for generating a new post."""
    logger.info(f"🚀 Starting Auto-Blogging Session ({mode} mode)...")
    
    # Unpack
    wp = components["wp"]
    gemini = components["gemini"]
    image_gen = components["image"]
    uploader = components["uploader"]
    seo = components["seo"]
    schema = components["schema"]
    yoast = components["yoast"]

    # 1. TOPIC SELECTION
    if manual_topic:
        topic, context, lang = manual_topic, "Manual Topic", "en" # Default to EN for manual
    else:
        topic, context, lang = get_hot_trend()
    
    if not topic:
        logger.error("No topic found. Aborting.")
        return

    logger.info(f"🎯 Selected Topic: {topic} (Lang: {lang})")

    # 2. CONTENT GENERATION
    # Build prompt with language instruction
    base_prompt = seo.build_daily_prompt(topic, context)
    lang_instruction = "Ensure the content is written in THAI language." if lang == "th" else "Ensure the content is written in English."
    full_prompt = f"{base_prompt}\n\nIMPORTANT: {lang_instruction}\nStrictly follow the JSON schema."

    # Need schema definition locally if not importable
    from pydantic import BaseModel, Field
    class SEOArticleMetadata(BaseModel):
        content: str = Field(description="The full HTML content of the article")
        seo_title: str = Field(description="SEO-optimized title (max 60 chars)")
        meta_description: str = Field(description="SEO meta description (max 160 chars)")
        focus_keyword: str = Field(description="The primary focus keyword for the article")
        excerpt: str = Field(description="A short summary of the article")
        suggested_categories: List[str] = Field(description="List of relevant WordPress category names")
        suggested_tags: List[str] = Field(description="List of relevant WordPress tag names")
    
    try:
        response = gemini.generate_structured_output(
            model="gemini-2.0-flash",
            prompt=full_prompt,
            schema=SEOArticleMetadata.model_json_schema()
        )
        
        if not response:
            logger.error("Failed to generate content.")
            return

        result = json.loads(response.text)
        meta = SEOArticleMetadata.model_validate(result) # Type-safe validation
        
        logger.info("✅ Content generated.")
    except Exception as e:
        logger.error(f"Content generation error: {e}")
        return

    # 3. IMAGE GENERATION
    featured_image_id = None
    try:
        img_prompt = f"Featured image for blog post: {topic}. High quality, professional, 16:9 aspect ratio."
        img_bytes = image_gen.generate_image(img_prompt)
        if img_bytes:
            fname = f"post_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}.jpg"
            path = image_gen.save_image(img_bytes, fname)
            if path:
                featured_image_id = uploader.upload_media(path, topic, meta.seo_title)
    except Exception as e:
        logger.warning(f"Image generation step failed: {e}")

    # 4. PUBLISHING
    # Resolve Category
    cat_id = 1 # Default Uncategorized
    # (Optional: Add logic to resolve/create categories based on meta.suggested_categories)

    # Resolve internal links and clean up placeholders
    existing_posts = wp.get_existing_links(limit=100)
    final_content = resolve_internal_links(meta.content, existing_posts)
    # Remove any remaining SUGGEST_EXTERNAL_LINK placeholders (external links should be manually curated)
    import re
    final_content = re.sub(r'\[SUGGEST_EXTERNAL_LINK:[^\]]+\]', '', final_content)

    # Schema
    post_url = f"{SITE_URL}/{topic.lower().replace(' ', '-')}"
    schema_markup = schema.generate_article_schema(meta.seo_title, meta.meta_description, final_content, post_url)
    final_content += f"\n\n<script type='application/ld+json'>{json.dumps(schema_markup)}</script>"

    post_data = {
        "title": meta.seo_title,
        "content": final_content,
        "excerpt": meta.excerpt,
        "status": "publish",
        "categories": [cat_id],
        "tags": [] # (Optional tag logic)
    }
    
    if featured_image_id:
        post_data["featured_media"] = featured_image_id

    post_id = wp.create_post(post_data)
    
    if post_id:
        logger.info(f"🚀 Published Post ID: {post_id}")
        # Update Yoast
        yoast.update_yoast_meta_fields(post_id, {
            "focus_keyword": meta.focus_keyword,
            "seo_title": meta.seo_title,
            "meta_description": meta.meta_description
        })
    else:
        logger.error("Failed to create post in WordPress.")

def run_maintenance(components: Dict, limit: int = 10):
    """Run the maintenance auditor."""
    logger.info(f"🔧 Starting Maintenance Mode (Limit: {limit})...")
    auditor = components["auditor"]
    auditor.run_maintenance(limit=limit)
    logger.info("✅ Maintenance Complete.")


if __name__ == "__main__":
    system = initialize_system()
    if system:
        if len(sys.argv) > 1:
            command = sys.argv[1]
            
            if command == "maintenance":
                limit = int(sys.argv[2]) if len(sys.argv) > 2 else 10
                run_maintenance(system, limit=limit)
            elif command == "weekly":
                run_content_generation(system, mode="weekly")
            elif command == "daily":
                run_content_generation(system, mode="daily")
            else:
                # Assume manual topic
                run_content_generation(system, mode="manual", manual_topic=command)
        else:
            # Default behavior (Daily)
            run_content_generation(system, mode="daily")
