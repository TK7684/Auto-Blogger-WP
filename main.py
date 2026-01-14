"""
Auto-Blogging WordPress - Gemini 3 Optimized Version

Automatically generates SEO-optimized blog posts based on trending topics.
Optimized using the Gemini 3 Developer Guide with thinking levels and structured outputs.
"""

import os
import datetime
import base64
import logging
import requests
from google import genai
from google.genai import types
from dotenv import load_dotenv
from typing import Dict, List, Optional, Tuple
from pathlib import Path
from pydantic import BaseModel, Field

# Import custom modules
from trend_sources import get_hot_trend
from seo_system import SEOPromptBuilder, SchemaMarkupGenerator, KeywordExtractor
from yoast_seo import YoastSEOIntegrator, ContentAnalyzer
from image_generator import ImageGenerator, WordPressMediaUploader
from research_agent import ResearchAgent

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# --- CONFIGURATION ---
WP_URL = os.environ.get("WP_URL")
WP_USER = os.environ.get("WP_USER")
WP_APP_PASSWORD = os.environ.get("WP_APP_PASSWORD")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
GEMINI_SERVICE_ACCOUNT_KEY_FILE = os.environ.get("GEMINI_SERVICE_ACCOUNT_KEY_FILE")
SITE_URL = os.environ.get("SITE_URL", "https://pedpro.online")
SITE_NAME = os.environ.get("SITE_NAME", "PedPro")

# Feature flags
SEO_ENABLED = os.environ.get("SEO_ENABLED", "true").lower() == "true"
IMAGE_GENERATION_ENABLED = os.environ.get("IMAGE_GENERATION_ENABLED", "true").lower() == "true"
YOAST_INTEGRATION_ENABLED = os.environ.get("YOAST_INTEGRATION_ENABLED", "true").lower() == "true"
SCHEMA_MARKUP_ENABLED = os.environ.get("SCHEMA_MARKUP_ENABLED", "true").lower() == "true"

# --- STRUCTURED OUTPUT MODELS ---
class SEOArticleMetadata(BaseModel):
    content: str = Field(description="The full HTML content of the article")
    seo_title: str = Field(description="SEO-optimized title (max 60 chars)")
    meta_description: str = Field(description="SEO meta description (max 160 chars)")
    focus_keyword: str = Field(description="The primary focus keyword for the article")
    excerpt: str = Field(description="A short summary of the article")
    suggested_categories: List[str] = Field(description="List of relevant WordPress category names")
    suggested_tags: List[str] = Field(description="List of relevant WordPress tag names")

# --- GENAI CLIENT ---
_genai_client = None

def get_genai_client():
    """Get or create the GenAI client (SDK v1.0)."""
    global _genai_client
    if _genai_client is None:
        try:
            # Prefer service account authentication (has billing, higher quotas)
            if GEMINI_SERVICE_ACCOUNT_KEY_FILE:
                import google.auth.transport.requests as tr_requests
                import google.oauth2.service_account as sa

                resolved_path = str(Path(GEMINI_SERVICE_ACCOUNT_KEY_FILE).resolve())
                if Path(resolved_path).exists():
                    logger.info(f"Using service account authentication: {resolved_path}")

                    # Create service account credentials
                    credentials = sa.Credentials.from_service_account_file(
                        resolved_path,
                        scopes=['https://www.googleapis.com/auth/cloud-platform']
                    )

                    # Create client with credentials (using Vertex AI endpoint)
                    _genai_client = genai.Client(
                        vertexai=True,
                        project=credentials.project_id,
                        location="us-central1",
                        credentials=credentials
                    )
                    return _genai_client
                else:
                    logger.warning(f"Service account file not found: {resolved_path}")

            # Fall back to API key authentication
            if GEMINI_API_KEY:
                _genai_client = genai.Client(api_key=GEMINI_API_KEY)
                logger.info("Using google-genai SDK with API key")
                return _genai_client

            logger.error("No valid Gemini credentials found (need GEMINI_API_KEY or GEMINI_SERVICE_ACCOUNT_KEY_FILE)")
        except Exception as e:
            logger.error(f"Error initializing GenAI client: {e}")
    return _genai_client

# Initialize SEO components
seo_prompt_builder = SEOPromptBuilder()
schema_generator = SchemaMarkupGenerator()
keyword_extractor = KeywordExtractor()

if YOAST_INTEGRATION_ENABLED and all([WP_URL, WP_USER, WP_APP_PASSWORD]):
    yoast_integrator = YoastSEOIntegrator(WP_URL, WP_USER, WP_APP_PASSWORD)
else:
    yoast_integrator = None

content_analyzer = ContentAnalyzer()

if IMAGE_GENERATION_ENABLED:
    image_generator = ImageGenerator()
    media_uploader = WordPressMediaUploader(WP_URL, WP_USER, WP_APP_PASSWORD)
else:
    image_generator = None
    media_uploader = None

# --- WORDPRESS HELPERS ---
def fetch_wordpress_terms(taxonomy: str = "categories") -> List[Dict]:
    """Fetch existing categories or tags from WordPress."""
    if not all([WP_URL, WP_USER, WP_APP_PASSWORD]):
        return []
    
    url = f"{WP_URL.rstrip('/')}/wp-json/wp/v2/{taxonomy}"
    credentials = f"{WP_USER}:{WP_APP_PASSWORD}"
    token = base64.b64encode(credentials.encode()).decode('utf-8')
    headers = {"Authorization": f"Basic {token}"}
    
    try:
        response = requests.get(url, headers=headers, params={"per_page": 100}, timeout=20)
        if response.status_code == 200:
            return response.json()
    except Exception as e:
        logger.warning(f"Failed to fetch WordPress {taxonomy}: {e}")
    return []

def resolve_term_ids(suggested_names: List[str], taxonomy: str = "categories") -> List[int]:
    """Map term names to IDs. Creates tags if they don't exist."""
    existing_terms = fetch_wordpress_terms(taxonomy)
    name_to_id = {term['name'].lower(): term['id'] for term in existing_terms}
    
    resolved_ids = []
    for name in suggested_names:
        name_lower = name.lower()
        if name_lower in name_to_id:
            resolved_ids.append(name_to_id[name_lower])
        elif taxonomy == "tags":
            # Create new tag for WordPress
            new_id = create_wordpress_term(name, taxonomy)
            if new_id:
                resolved_ids.append(new_id)
                
    return resolved_ids

def create_wordpress_term(name: str, taxonomy: str) -> Optional[int]:
    """Create a new category or tag in WordPress."""
    url = f"{WP_URL.rstrip('/')}/wp-json/wp/v2/{taxonomy}"
    credentials = f"{WP_USER}:{WP_APP_PASSWORD}"
    token = base64.b64encode(credentials.encode()).decode('utf-8')
    headers = {
        "Authorization": f"Basic {token}",
        "Content-Type": "application/json"
    }
    try:
        response = requests.post(url, headers=headers, json={"name": name}, timeout=20)
        if response.status_code == 201:
            return response.json().get('id')
    except Exception as e:
        logger.warning(f"Failed to create {taxonomy} '{name}': {e}")
    return None

# --- CONTENT GENERATION ---
def generate_content_gemini(topic: str, context: str, mode: str = "daily") -> Dict:
    """
    Generate SEO-optimized content using Gemini 3 thinking models & structured outputs.
    """
    logger.info(f"🧠 Generating optimized content for: {topic} ({mode} mode)...")

    client = get_genai_client()
    if not client:
        return {'content': "AI Content Generation Failed.", 'seo_title': topic}

    # Optimization: Use Gemini 3 models
    model_name = "gemini-3-pro-preview" if mode == "weekly" else "gemini-3-flash-preview"
    
    # Optimization: Thinking Level
    thinking_level = "high" if mode == "weekly" else "low"

    # Optimization: Use SEOPromptBuilder for detailed instructions
    if mode == "weekly":
        base_prompt = seo_prompt_builder.build_weekly_prompt(topic, context)
    else:
        base_prompt = seo_prompt_builder.build_daily_prompt(topic, context)

    prompt = f"{base_prompt}\n\nIMPORTANT: Return the article and all metadata in the following JSON format strictly."

    try:
        # Optimization: Structured Outputs (Pydantic)
        config_dict = {
            "response_mime_type": "application/json",
            "response_json_schema": SEOArticleMetadata.model_json_schema(),
            "temperature": 1.0  # Recommended for Gemini 3
        }

        response = client.models.generate_content(
            model=model_name,
            contents=prompt,
            config=config_dict
        )

        # Parse JSON output directly into dict
        import json
        result = json.loads(response.text)
        logger.info(f"✅ Content generated successfully")
        return result

    except Exception as e:
        logger.error(f"Error generating content: {e}")
        return {'content': f"Failed: {e}", 'seo_title': topic}

# --- PUBLISHING ---
def post_to_wp(title: str, content: str, seo_data: Dict, 
               categories: List[int], tags: List[int],
               featured_image_id: Optional[int] = None) -> Optional[int]:
    """Publish to WordPress with full SEO, categories, and tags."""
    if not all([WP_URL, WP_USER, WP_APP_PASSWORD]):
        return None

    url = f"{WP_URL.rstrip('/')}/wp-json/wp/v2/posts"
    credentials = f"{WP_USER}:{WP_APP_PASSWORD}"
    token = base64.b64encode(credentials.encode()).decode('utf-8')
    headers = {
        "Authorization": f"Basic {token}",
        "Content-Type": "application/json"
    }

    data = {
        "title": seo_data.get('seo_title', title),
        "content": content,
        "excerpt": seo_data.get('excerpt', ''),
        "status": "publish",
        "categories": categories or [1],
        "tags": tags
    }

    if featured_image_id:
        data["featured_media"] = featured_image_id

    try:
        response = requests.post(url, headers=headers, json=data, timeout=30)
        if response.status_code == 201:
            post_data = response.json()
            post_id = post_data.get('id')
            
            # Yoast SEO Update
            if YOAST_INTEGRATION_ENABLED and yoast_integrator:
                yoast_data = {
                    'focus_keyword': seo_data.get('focus_keyword', ''),
                    'seo_title': seo_data.get('seo_title', title),
                    'meta_description': seo_data.get('meta_description', ''),
                }
                yoast_integrator.update_yoast_meta_fields(post_id, yoast_data)
                
            return post_id
    except Exception as e:
        logger.error(f"WordPress posting error: {e}")
    return None

# --- MAIN EXECUTION ---
def run_autoblogger(force_mode: Optional[str] = None, manual_topic: Optional[str] = None):
    """Main flow."""
    mode = force_mode or ("weekly" if datetime.datetime.today().weekday() == 6 else "daily")
    
    if manual_topic:
        topic, context = manual_topic, "Manual"
    elif mode == "weekly":
        # Weekly optimization: Run research to find the best gap to cover
        logger.info("🕵️ Running Research Agent for weekly pillar content topic selection...")
        agent = ResearchAgent(WP_URL, WP_USER, WP_APP_PASSWORD, GEMINI_API_KEY)
        research = agent.run_research(days_back=7, num_ideas=3)
        
        if research.get('article_ideas'):
            # Pick the top ranked idea
            idea = research['article_ideas'][0]
            topic = idea.title
            context = f"Rationale: {idea.rationale}. Competitive Advantage: {idea.competitive_advantage}"
            logger.info(f"🎯 Research-driven topic selected: {topic}")
        else:
            topic, context = get_hot_trend()
    else:
        # Daily mode: use hot trends for speed
        topic, context = get_hot_trend()

    if not topic:
        logger.error("No topic found. Aborting.")
        return
    
    # Generate content
    result = generate_content_gemini(topic, context, mode=mode)
    
    # Resolve WordPress terms
    cat_ids = resolve_term_ids(result.get('suggested_categories', []), "categories")
    tag_ids = resolve_term_ids(result.get('suggested_tags', []), "tags")

    # Generate image (Grounded optimization in image_generator.py)
    featured_image_id = None
    if IMAGE_GENERATION_ENABLED:
        image_data = image_generator.generate_image(f"Featured image for: {topic}", mode=mode)
        if image_data:
            img_path = image_generator.save_image(image_data, f"post_{datetime.datetime.now().strftime('%Y%H%M%S')}.jpg")
            if img_path:
                featured_image_id = media_uploader.upload_media(img_path, topic, result.get('seo_title'))

    # Add Schema
    post_url = f"{SITE_URL}/{topic.lower().replace(' ', '-')}"
    final_content = add_schema_markup(
        result.get('content', ''),
        result.get('seo_title', topic),
        result.get('meta_description', ''),
        result.get('focus_keyword', topic),
        post_url
    )

    # Publish
    post_id = post_to_wp(topic, final_content, result, cat_ids, tag_ids, featured_image_id)
    if post_id:
        logger.info(f"🚀 SUCCESS! Post {post_id} published.")

def add_schema_markup(content: str, title: str, description: str, fk: str, url: str) -> str:
    """Helper for schema."""
    if not SCHEMA_MARKUP_ENABLED: return content
    schema = schema_generator.generate_article_schema(title, description, content, url)
    return content + "\n\n" + schema_generator.wrap_schema_in_script(schema)

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        cmd = sys.argv[1]
        m_topic = sys.argv[2] if len(sys.argv) > 2 else None
        run_autoblogger(force_mode=cmd if cmd in ['daily', 'weekly'] else None, manual_topic=m_topic)
    else:
        run_autoblogger()
