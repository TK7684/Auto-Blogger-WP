"""
Auto-Blogging WordPress - Standardized & Optimized Version.
Key Features: Multilingual (TH/EN), Image Gen (Gemini/HF), Maintenance Mode, Centralized Clients.
Focused on SEO and Content Quality.
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
from src.utils.linking import resolve_internal_links, clean_remaining_placeholders
from src.utils import normalize_dict_keys

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load env
load_dotenv(override=True)

# --- CONFIGURATION ---
WP_URL = os.environ.get("WP_URL")
WP_USER = os.environ.get("WP_USER")
WP_APP_PASSWORD = os.environ.get("WP_APP_PASSWORD")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
SITE_URL = os.environ.get("SITE_URL", "https://pedpro.online")
IMAGE_GENERATION_ENABLED = os.environ.get("IMAGE_GENERATION_ENABLED", "true").lower() == "true"

# --- INITIALIZATION ---
def initialize_system() -> Dict:
    """Initialize all clients and agents."""
    if not all([WP_URL, WP_USER, WP_APP_PASSWORD, GEMINI_API_KEY]):
        logger.error("Missing critical environment variables.")
        return {}

    # 1. Clients
    wp_client = WordPressClient(WP_URL, WP_USER, WP_APP_PASSWORD)
    gemini_client = GeminiClient(GEMINI_API_KEY)

    # 2. Components (image generation optional)
    image_gen = None
    media_uploader = None
    if IMAGE_GENERATION_ENABLED:
        image_gen = ImageGenerator(gemini_client)
        media_uploader = WordPressMediaUploader(WP_URL, WP_USER, WP_APP_PASSWORD)
        logger.info("🖼️ Image generation enabled")
    else:
        logger.info("🖼️ Image generation disabled (focus on content/SEO)")

    yoast = YoastSEOIntegrator(WP_URL, WP_USER, WP_APP_PASSWORD)
    auditor = ContentAuditor(wp_client, gemini_client, yoast_integrator=yoast)
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
        topic, context, lang = manual_topic, "Manual Topic", "en"  # Default to EN for manual
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
        # Use gemini-2.5-flash which has better availability and supports structured output
        response = gemini.generate_structured_output(
            model="gemini-2.5-flash",
            prompt=full_prompt,
            schema=SEOArticleMetadata.model_json_schema()
        )

        if not response:
            logger.error("Failed to generate content.")
            return

        result = json.loads(response.text)
        result = normalize_dict_keys(result)  # Normalize uppercase/MixedCase keys → snake_case
        meta = SEOArticleMetadata.model_validate(result)  # Type-safe validation

        logger.info(f"✅ Content generated successfully. Title: '{meta.seo_title}'")
        logger.info(f"🔑 Focus Keyword: {meta.focus_keyword}")
    except Exception as e:
        logger.error(f"❌ Content generation error: {e}")
        return

    # 3. IMAGE GENERATION (Optional)
    featured_image_id = None
    if image_gen and uploader:
        try:
            img_prompt = f"Featured image for blog post: {topic}. High quality, professional, 16:9 aspect ratio."
            img_bytes = image_gen.generate_image(img_prompt)
            if img_bytes:
                fname = f"post_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}.jpg"
                path = image_gen.save_image(img_bytes, fname)
                if path:
                    featured_image_id = uploader.upload_media(path, topic, meta.seo_title)
                    logger.info(f"🖼️ Featured image uploaded (ID: {featured_image_id})")
        except Exception as e:
            logger.warning(f"Image generation step failed: {e}")
    else:
        logger.info("🖼️ Skipping image generation (disabled)")

    # 4. PUBLISHING
    # Resolve Category
    cat_id = 1  # Default Uncategorized

    # Resolve internal links and clean up placeholders
    logger.info("🔗 Resolving internal links...")
    existing_posts = wp.get_existing_links(limit=100)
    final_content = resolve_internal_links(meta.content, existing_posts)
    final_content = clean_remaining_placeholders(final_content)

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
        "tags": []
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
        logger.info("✅ Yoast SEO fields updated")
    else:
        logger.error("Failed to create post in WordPress.")

def run_link_fix(components: Dict, limit: int = 50):
    """Quick mode: Only fix internal link placeholders."""
    logger.info(f"🔗 Starting Link-Fix Mode (Limit: {limit})...")
    auditor = components["auditor"]
    stats = auditor.run_link_fix_only(limit=limit)
    logger.info("✅ Link Fix Complete.")

def run_maintenance(components: Dict, limit: int = 10):
    """Run maintenance checks on existing posts."""
    logger.info(f"🔧 Starting Maintenance Mode (Limit: {limit})...")
    auditor = components["auditor"]
    stats = auditor.run_maintenance(limit=limit)
    logger.info(f"✅ Maintenance Complete. Updated: {stats.get('updated', 0)}, Failed: {stats.get('failed', 0)}")


def show_help():
    """Display usage information."""
    help_text = """
Auto-Blogging WordPress - Usage Guide

Commands:
  python main.py                    Run daily content generation (default)
  python main.py daily              Run daily content generation
  python main.py weekly             Run weekly pillar content generation
  python main.py "your topic"       Generate content for a specific topic
  python main.py maintenance [N]    Run maintenance on N posts (default: 10)
  python main.py fix-links [N]      Fix internal links in N posts (default: 50)
  python main.py help               Show this help message

Environment Variables (Required):
  WP_URL                  WordPress site URL
  WP_USER                 WordPress username
  WP_APP_PASSWORD         WordPress application password
  GEMINI_API_KEY          Google Gemini API key

Environment Variables (Optional):
  SITE_URL                Site URL for schema markup (default: https://pedpro.online)
  IMAGE_GENERATION_ENABLED  Enable/disable image generation (default: true)

Examples:
  python main.py
  python main.py "The Future of AI in 2026"
  python main.py maintenance 20
  python main.py fix-links 100
"""
    print(help_text)


if __name__ == "__main__":
    system = initialize_system()
    if system:
        if len(sys.argv) > 1:
            command = sys.argv[1].lower()

            if command in ["help", "-h", "--help"]:
                show_help()
            elif command == "maintenance":
                limit = int(sys.argv[2]) if len(sys.argv) > 2 else 10
                run_maintenance(system, limit=limit)
            elif command == "fix-links":
                limit = int(sys.argv[2]) if len(sys.argv) > 2 else 50
                run_link_fix(system, limit=limit)
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
    else:
        logger.error("System initialization failed. Please check your environment variables.")
