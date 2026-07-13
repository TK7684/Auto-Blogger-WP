"""
Auto-Blogger WordPress — main orchestrator.

Flow:
  1. pick topic (cadence + article_type)
  2. generate SEO content via Gemini/Z.AI (structured output)
  3. generate hero image + inline images via ComfyUI
  4. splice inline images into content (replacing [IMAGE_PLACEHOLDER_N])
  5. resolve internal links, inject JSON-LD schema
  6. publish to WordPress + update Yoast meta
  7. verify published post (HTTP checks, Discord alert on fail)

CLI:
  python -m src.main                              # daily trending (default)
  python -m src.main --cadence daily --type trending
  python -m src.main --cadence weekly --type research
  python -m src.main --cadence monthly --type research
  python -m src.main --topic "your topic here"
  python -m src.main maintenance 20
  python -m src.main fix-links 100
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import os
import re
import sys
from typing import Dict, List, Optional

from dotenv import load_dotenv
from pydantic import BaseModel, Field

from src.analytics import AnalyticsEngine
from src.auditor import ContentAuditor
from src.clients.gemini import GeminiClient
from src.clients.wordpress import WordPressClient
from src.image_generator import ImageGenerator, WordPressMediaUploader
from src.research_agent import ResearchAgent
from src.seo_system import SchemaMarkupGenerator, SEOPromptBuilder
from src.trend_sources import get_trending_topic
from src.utils import normalize_dict_keys, parse_json_lenient
from src.utils.linking import clean_remaining_placeholders, resolve_internal_links
from src.yoast_seo import YoastSEOIntegrator

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

load_dotenv(override=True)

# --- config ---------------------------------------------------------------

WP_URL = os.environ.get("WP_URL")
WP_USER = os.environ.get("WP_USER")
WP_APP_PASSWORD = os.environ.get("WP_APP_PASSWORD")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
SITE_URL = os.environ.get("SITE_URL", "https://pedpro.online").rstrip("/")
IMAGE_GENERATION_ENABLED = os.environ.get("IMAGE_GENERATION_ENABLED", "true").lower() == "true"
VERIFY_AFTER_PUBLISH = os.environ.get("VERIFY_AFTER_PUBLISH", "true").lower() == "true"

INLINE_IMAGES_BY_TYPE = {
    "trending": int(os.environ.get("INLINE_IMAGES_TRENDING", "2")),
    "research": int(os.environ.get("INLINE_IMAGES_RESEARCH", "4")),
}

IMG_PLACEHOLDER_RE = re.compile(r"\[IMAGE_PLACEHOLDER_(\d+)\]")


# --- schemas --------------------------------------------------------------

class SEOArticleMetadata(BaseModel):
    content: str = Field(description="Full HTML content of the article")
    seo_title: str = Field(description="SEO-optimized title (max 60 chars)")
    meta_description: str = Field(description="SEO meta description (max 160 chars)")
    focus_keyword: str = Field(description="Primary focus keyword")
    excerpt: str = Field(description="Short summary")
    suggested_categories: List[str] = Field(default_factory=list)
    suggested_tags: List[str] = Field(default_factory=list)
    image_prompt: Optional[str] = Field(default=None, description="Prompt for hero/featured image")
    in_article_image_prompts: List[str] = Field(
        default_factory=list,
        description="Ordered prompts for each [IMAGE_PLACEHOLDER_N] token",
    )


# --- initialization -------------------------------------------------------

def initialize_system() -> Dict:
    zai_key = os.environ.get("ZAI_API_KEY")
    openrouter_key = os.environ.get("OPENROUTER_API_KEY")
    has_llm = any([GEMINI_API_KEY, zai_key, openrouter_key])
    if not all([WP_URL, WP_USER, WP_APP_PASSWORD]) or not has_llm:
        logger.error(
            "Missing critical env vars. Need WP creds + one LLM key "
            "(ZAI_API_KEY, GEMINI_API_KEY, or OPENROUTER_API_KEY)."
        )
        return {}

    wp_client = WordPressClient(WP_URL, WP_USER, WP_APP_PASSWORD)
    gemini_client = GeminiClient(GEMINI_API_KEY)

    image_gen = None
    media_uploader = None
    if IMAGE_GENERATION_ENABLED:
        image_gen = ImageGenerator(gemini_client)
        media_uploader = WordPressMediaUploader(WP_URL, WP_USER, WP_APP_PASSWORD)
        logger.info("🖼️  Image generation enabled (ComfyUI primary)")
    else:
        logger.info("🖼️  Image generation disabled")

    yoast = YoastSEOIntegrator(WP_URL, WP_USER, WP_APP_PASSWORD)
    auditor = ContentAuditor(wp_client, gemini_client, yoast_integrator=yoast)
    researcher = ResearchAgent(wp_client, gemini_client)

    return {
        "wp": wp_client,
        "gemini": gemini_client,
        "image": image_gen,
        "uploader": media_uploader,
        "auditor": auditor,
        "researcher": researcher,
        "seo": SEOPromptBuilder(),
        "schema": SchemaMarkupGenerator(),
        "yoast": yoast,
    }


# --- Discord --------------------------------------------------------------

def _notify_discord(title: str, url: str, description: str, topic: str,
                    cadence: str, article_type: str) -> None:
    webhook = os.getenv("DISCORD_WEBHOOK_URL")
    if not webhook:
        return
    try:
        import urllib.request
        color = 0x9b59b6 if article_type == "research" else 0x00b894
        payload = json.dumps({
            "embeds": [{
                "title": f"📝 New {cadence}/{article_type} post",
                "description": f"**{title}**\n\n{(description or '')[:220]}",
                "url": url,
                "color": color,
                "fields": [
                    {"name": "Topic", "value": topic[:1024], "inline": True},
                    {"name": "Type", "value": f"{cadence} · {article_type}", "inline": True},
                    {"name": "Link", "value": f"[Read →]({url})", "inline": False},
                ],
                "footer": {"text": "Auto-Blogger-WP → PedPro"},
            }]
        }).encode("utf-8")
        req = urllib.request.Request(
            webhook, data=payload,
            headers={"Content-Type": "application/json", "User-Agent": "Auto-Blogger-WP/1.1"},
        )
        urllib.request.urlopen(req, timeout=10)
    except Exception as e:
        logger.warning(f"Discord notification failed: {e}")


# --- inline image splicing ------------------------------------------------

def _generate_and_splice_inline_images(
    content: str,
    prompts: List[str],
    image_gen: ImageGenerator,
    uploader: WordPressMediaUploader,
    topic: str,
    max_count: int,
) -> str:
    """Replace each [IMAGE_PLACEHOLDER_N] token with a real <img> tag.

    - If prompts[N] exists, use it; else synthesize from topic + section index.
    - Caps at max_count to keep ComfyUI cost bounded.
    - Any generation failure leaves the placeholder removed (empty string).
    """
    tokens = IMG_PLACEHOLDER_RE.findall(content)
    if not tokens:
        return content
    seen = sorted({int(t) for t in tokens})[:max_count]

    for idx in seen:
        prompt = (prompts[idx] if idx < len(prompts) and prompts[idx] else
                  f"Editorial illustration for section {idx+1} of blog post: {topic}. "
                  f"Professional, high-quality, photorealistic, 16:9.")
        img_tag = ""
        try:
            img_bytes = image_gen.generate_image(prompt)
            if img_bytes:
                fname = f"inline_{dt.datetime.now().strftime('%Y%m%d%H%M%S')}_{idx}.jpg"
                local_path = image_gen.save_image(img_bytes, fname)
                if local_path:
                    media_id = uploader.upload_media(local_path, topic, f"{topic} — figure {idx+1}")
                    source_url = _media_source_url(uploader, media_id) if media_id else None
                    if source_url:
                        img_tag = (
                            f'<figure class="wp-block-image size-large is-resized">'
                            f'<img src="{source_url}" alt="{topic} — figure {idx+1}" loading="lazy" style="max-width:100%;height:auto;" />'
                            f'<figcaption>{topic} — figure {idx+1}</figcaption>'
                            f'</figure>'
                        )
        except Exception as e:
            logger.warning(f"inline image {idx} failed: {e}")

        content = content.replace(f"[IMAGE_PLACEHOLDER_{idx}]", img_tag)

    # strip any leftover placeholders we didn't hit
    content = IMG_PLACEHOLDER_RE.sub("", content)
    return content


def _media_source_url(uploader: WordPressMediaUploader, media_id: int) -> Optional[str]:
    """Fetch source_url for a media id. Uploader doesn't return it, so we re-query."""
    import base64
    try:
        token = base64.b64encode(uploader.credentials.encode()).decode("utf-8")
        url = f"{uploader.wp_url}/wp-json/wp/v2/media/{media_id}"
        r = uploader.session.get(url, headers={"Authorization": f"Basic {token}"}, timeout=15)
        if r.status_code == 200:
            return r.json().get("source_url")
    except Exception as e:
        logger.debug(f"media source_url lookup failed: {e}")
    return None


# --- content generation ---------------------------------------------------

def run_content_generation(components: Dict, cadence: str = "daily",
                           article_type: Optional[str] = None,
                           manual_topic: Optional[str] = None,
                           dry_run: bool = False) -> Optional[int]:
    """Generate and publish one post. Returns the WP post ID or None on failure.

    When ``dry_run`` is True, runs steps 1-2 (topic pick + LLM content generation)
    only — image gen, internal-link resolution, schema injection, WP create_post,
    Yoast meta, Discord notify, and post-publish verify are all skipped. Returns
    ``-1`` as a non-None success sentinel so the caller treats it as success.
    """
    wp = components["wp"]
    gemini = components["gemini"]
    image_gen = components["image"]
    uploader = components["uploader"]
    seo = components["seo"]
    schema = components["schema"]
    yoast = components["yoast"]

    # 1. TOPIC
    if manual_topic:
        # Auto-detect language from topic text
        _thai_chars = len([c for c in manual_topic if '\u0E00' <= c <= '\u0E7F'])
        _detected_lang = "th" if _thai_chars > 3 else "en"
        topic, context, lang = manual_topic, "Manual Topic", _detected_lang
        final_type = article_type or "trending"
    else:
        topic, context, lang, final_type = get_trending_topic(cadence, article_type)
    if not topic:
        logger.error("No topic found. Aborting.")
        return None

    logger.info(f"🎯 cadence={cadence} type={final_type} lang={lang} topic={topic!r}")

    # 1b. ANALYTICS FEEDBACK LOOP — use past performance to optimize new content
    analytics_brief = None
    try:
        from src.analytics import AnalyticsEngine
        wp_url = os.environ.get("WP_URL", "").rstrip("/")
        wp_user = os.environ.get("WP_USER", "")
        wp_pass = os.environ.get("WP_APP_PASSWORD", "")
        if wp_url and wp_user and wp_pass:
            analytics = AnalyticsEngine(wp_url, wp_user, wp_pass)
            analytics_brief = analytics.get_content_brief(days=7)
            if analytics_brief:
                logger.info(f"📊 Analytics brief injected into prompt")
    except Exception as e:
        logger.warning(f"Could not load analytics brief: {e}")

    # 2. CONTENT
    if final_type == "research":
        base_prompt = seo.build_weekly_prompt(topic, context, language=("Thai" if lang == "th" else "English"))
    else:
        base_prompt = seo.build_daily_prompt(topic, context, language=("Thai" if lang == "th" else "English"), analytics_brief=analytics_brief)
    lang_instruction = (
        "Ensure ALL content is written entirely in Thai — titles, headings, paragraphs, FAQs, everything. "
        "Follow the Thai writing style rules above strictly. The article must read like a Thai person wrote it, "
        "not like a translation from English."
    ) if lang == "th" else "Ensure all content is in English."
    full_prompt = f"{base_prompt}\n\nIMPORTANT: {lang_instruction}\nStrictly follow the JSON schema."

    # Retry content generation up to 3 times (JSON parse failures from LLM are common)
    max_content_retries = 3
    meta = None
    for content_attempt in range(1, max_content_retries + 1):
        try:
            response = gemini.generate_structured_output(
                model="gemini-2.5-flash",
                prompt=full_prompt,
                schema=SEOArticleMetadata.model_json_schema(),
            )
            if not response:
                logger.error("Empty response from LLM")
                if content_attempt < max_content_retries:
                    logger.warning(f"Retrying content generation (attempt {content_attempt + 1}/{max_content_retries})...")
                    import time; time.sleep(2)
                    continue
                return None
            result = parse_json_lenient(response.text)
            result = normalize_dict_keys(result)
            meta = SEOArticleMetadata.model_validate(result)
            logger.info(f"✅ Content generated (attempt {content_attempt}). Title: {meta.seo_title!r}  focus={meta.focus_keyword!r}")
            break
        except Exception as e:
            logger.error(f"❌ Content generation failed (attempt {content_attempt}/{max_content_retries}): {e}")
            if content_attempt < max_content_retries:
                logger.warning(f"Retrying content generation (attempt {content_attempt + 1}/{max_content_retries})...")
                import time; time.sleep(3)
                continue
            return None
    if meta is None:
        return None

    # DRY-RUN GATE — bail out after LLM call so we can verify backend changes
    # (e.g. Gemini/Z.AI timeout fixes) without writing to the live site.
    if dry_run:
        slug = re.sub(r"[^a-z0-9]+", "-", topic.lower()).strip("-")[:80]
        logger.info(
            f"🟡 DRY RUN: would publish {meta.seo_title!r} "
            f"(focus={meta.focus_keyword!r}, slug={slug!r}); "
            f"skipping image gen, WP create_post, Yoast, Discord, verify."
        )
        return -1

    # 3. HERO IMAGE
    featured_image_id = None
    featured_image_url = None
    if image_gen and uploader:
        try:
            hero_prompt = meta.image_prompt or (
                f"Featured image for blog post: {topic}. "
                f"Professional, high-quality, photorealistic, 16:9 aspect ratio."
            )
            img_bytes = image_gen.generate_image(hero_prompt)
            if img_bytes:
                fname = f"post_{dt.datetime.now().strftime('%Y%m%d%H%M%S')}.jpg"
                local_path = image_gen.save_image(img_bytes, fname)
                if local_path:
                    featured_image_id = uploader.upload_media(local_path, topic, meta.seo_title)
                    if featured_image_id:
                        featured_image_url = _media_source_url(uploader, featured_image_id)
                        logger.info(f"🖼️  Hero image ID={featured_image_id}")
        except Exception as e:
            logger.warning(f"Hero image failed: {e}")

    # 4. INLINE IMAGES
    final_content = meta.content
    if image_gen and uploader:
        max_inline = INLINE_IMAGES_BY_TYPE.get(final_type, 2)
        before = len(IMG_PLACEHOLDER_RE.findall(final_content))
        final_content = _generate_and_splice_inline_images(
            final_content, meta.in_article_image_prompts or [],
            image_gen, uploader, topic, max_inline,
        )
        logger.info(f"🖼️  Inline images processed: {before} placeholders → capped at {max_inline}")

    # 5. INTERNAL LINKS + SCHEMA
    logger.info("🔗 Resolving internal links...")
    existing = wp.get_existing_links(limit=200)
    final_content = resolve_internal_links(final_content, existing)
    final_content = clean_remaining_placeholders(final_content)

    slug = re.sub(r"[^a-z0-9]+", "-", topic.lower()).strip("-")[:80]
    post_url = f"{SITE_URL}/{slug}"
    schema_markup = schema.generate_article_schema(
        meta.seo_title, meta.meta_description, final_content, post_url,
        image_url=featured_image_url,
    )
    final_content += f"\n\n<script type='application/ld+json'>{json.dumps(schema_markup)}</script>"

    # 6. PUBLISH (first pass — without affiliate cards so we get the post_id)
    category_id = _resolve_category(wp, final_type, cadence)
    tag_ids = _resolve_tags(wp, meta.suggested_tags + [final_type, cadence])

    post_data = {
        "title": meta.seo_title,
        "content": final_content,
        "excerpt": meta.excerpt,
        "status": "publish",
        "slug": slug,
        "categories": [category_id] if category_id else [],
        "tags": tag_ids,
    }
    if featured_image_id:
        post_data["featured_media"] = featured_image_id

    post_id = wp.create_post(post_data)
    if not post_id:
        logger.error("❌ Failed to publish to WordPress.")
        return None
    logger.info(f"🚀 Published post ID: {post_id}  ({post_url})")

    # 6b. AFFILIATE CTA — now that we have post_id, inject affiliate cards
    #     with sub_id tracking (pedpro-{post_id}) so you can see which article
    #     drove clicks/conversions in Shopee Affiliate Dashboard → Sub ID report.
    affiliate_products = []
    try:
        from src.affiliate_inserter import insert_shopee_card, track_product_placements, _try_fetch_products
        content_with_affiliate = insert_shopee_card(
            final_content, topic, article_text=final_content, post_id=post_id,
        )
        if content_with_affiliate != final_content:
            # Affiliate cards were injected — update the published post
            final_content = content_with_affiliate
            wp.update_post(post_id, {"content": final_content})
            logger.info(f"🛒 Updated post {post_id} with affiliate cards (sub_id=pedpro-{post_id})")

            # Fetch the products that were placed for tracking
            affiliate_products = _try_fetch_products(topic, limit=3, article_text=final_content)
    except Exception as e:
        logger.warning(f"Affiliate inserter failed (non-blocking): {e}")

    # 6c. RESPONSIVE IMAGE CSS — append AFTER affiliate cards so it's always last.
    #     Without this, 1024×1024 ComfyUI images overflow on mobile.
    _RESPONSIVE_CSS = (
        "<style>"
        ".entry-content img,.post-content img,.wp-block-image img{max-width:100%;height:auto;}"
        ".wp-block-image{max-width:100%;}"
        "</style>"
    )
    if _RESPONSIVE_CSS not in final_content:
        final_content += "\n" + _RESPONSIVE_CSS
        try:
            wp.update_post(post_id, {"content": final_content})
            logger.info(f"📱 Post {post_id} updated with responsive image CSS")
        except Exception as e:
            logger.warning(f"Failed to inject responsive CSS: {e}")

    # 7a. TRACK affiliate product placements for commission visibility
    if affiliate_products:
        try:
            from src.affiliate_inserter import track_product_placements
            placement_mode = "distributed" if len(affiliate_products) >= 2 else "inline"
            track_product_placements(
                affiliate_products, topic, placement=placement_mode, article_id=post_id,
            )
        except Exception as e:
            logger.debug(f"affiliate tracking skipped: {e}")

    # 7b. YOAST + DISCORD
    yoast.update_yoast_meta_fields(post_id, {
        "focus_keyword": meta.focus_keyword,
        "seo_title": meta.seo_title,
        "meta_description": meta.meta_description,
    })
    logger.info("✅ Yoast fields updated")
    _notify_discord(meta.seo_title, post_url, meta.meta_description, topic, cadence, final_type)

    # 8. VERIFY
    if VERIFY_AFTER_PUBLISH:
        try:
            from src.verify_published import verify_post, write_report, notify_discord as vf_notify
            fresh = wp.get_post(post_id)
            if fresh:
                verdict = verify_post(wp, fresh)
                write_report([verdict])
                logger.info(f"🔍 verify: {verdict.status} — {sum(1 for c in verdict.checks if not c.passed)} issue(s)")
                if verdict.status in ("FAIL", "ERROR"):
                    vf_notify([verdict])
        except Exception as e:
            logger.warning(f"post-publish verify failed: {e}")

    return post_id


# --- taxonomy helpers -----------------------------------------------------

def _resolve_category(wp: WordPressClient, article_type: str, cadence: str) -> Optional[int]:
    name = "Research" if article_type == "research" else "Trending"
    try:
        existing = wp.fetch_terms("categories", {"search": name, "per_page": 10})
        for term in existing:
            if term["name"].lower() == name.lower():
                return term["id"]
        return wp.create_term(name, "categories")
    except Exception as e:
        logger.debug(f"category resolve failed: {e}")
        return None


def _resolve_tags(wp: WordPressClient, names: List[str]) -> List[int]:
    ids: List[int] = []
    for n in {(s or "").strip() for s in names if s}:
        if not n:
            continue
        try:
            term_id = wp.create_term(n, "tags")
            if term_id:
                ids.append(term_id)
        except Exception as e:
            logger.debug(f"tag resolve {n!r} failed: {e}")
    return ids


# --- maintenance modes ----------------------------------------------------

def run_link_fix(components: Dict, limit: int = 50) -> None:
    logger.info(f"🔗 Link-Fix Mode (limit={limit})")
    components["auditor"].run_link_fix_only(limit=limit)
    logger.info("✅ Link fix complete")


def run_maintenance(components: Dict, limit: int = 10) -> None:
    logger.info(f"🔧 Maintenance Mode (limit={limit})")
    stats = components["auditor"].run_maintenance(limit=limit)
    logger.info(f"✅ Maintenance complete. Updated={stats.get('updated', 0)} Failed={stats.get('failed', 0)}")


# --- CLI ------------------------------------------------------------------

def main(argv: Optional[List[str]] = None) -> int:
    # back-compat: `python -m src.main daily` / `weekly` / `maintenance N` / `fix-links N`
    argv = argv if argv is not None else sys.argv[1:]
    if argv and argv[0] in ("daily", "weekly", "monthly"):
        cadence = argv[0]
        argv = ["--cadence", cadence] + argv[1:]
    elif argv and argv[0] in ("maintenance", "fix-links"):
        pass  # handled below
    elif argv and argv[0] in ("--help", "-h", "help"):
        pass

    ap = argparse.ArgumentParser(description="Auto-Blogger WordPress")
    sub = ap.add_subparsers(dest="cmd")

    pub = sub.add_parser("publish", help="Generate and publish one post (default)")
    pub.add_argument("--cadence", choices=["daily", "weekly", "monthly"], default="daily")
    pub.add_argument("--type", choices=["trending", "research"], default=None,
                     help="Force article type; default mixes by cadence")
    pub.add_argument("--topic", default=None, help="Manual topic override")
    pub.add_argument("--dry-run", action="store_true",
                     help="Run topic + LLM only; skip image gen, WP publish, Yoast, Discord, verify")

    mnt = sub.add_parser("maintenance")
    mnt.add_argument("limit", type=int, nargs="?", default=10)

    fix = sub.add_parser("fix-links")
    fix.add_argument("limit", type=int, nargs="?", default=50)

    ana = sub.add_parser("analytics", help="Analytics report: views, patterns, optimization")
    ana.add_argument("--days", type=int, default=7, help="Period in days (default: 7)")
    ana.add_argument("--top", type=int, default=10, help="Top N posts to show (default: 10)")
    ana.add_argument("--no-optimize", action="store_true", help="Skip optimization suggestions")
    ana.add_argument("--json", dest="json_out", action="store_true", help="Output JSON to stdout")

    # Default when no subcommand: "publish --cadence daily"
    if not argv or argv[0].startswith("-"):
        args = ap.parse_args(["publish"] + argv)
    elif argv[0] in (["publish", "maintenance", "fix-links", "analytics"]):
        args = ap.parse_args(argv)
    else:
        # Legacy: first arg is a manual topic. Preserve --dry-run if present in tail.
        legacy = ["publish", "--topic", argv[0]]
        if "--dry-run" in argv[1:]:
            legacy.append("--dry-run")
        args = ap.parse_args(legacy)

    system = initialize_system()
    if not system:
        return 2

    if args.cmd == "maintenance":
        run_maintenance(system, limit=args.limit)
        return 0
    if args.cmd == "fix-links":
        run_link_fix(system, limit=args.limit)
        return 0
    if args.cmd == "analytics":
        engine = AnalyticsEngine(WP_URL, WP_USER, WP_APP_PASSWORD)
        if getattr(args, "json_out", False):
            report = engine.generate_json_report(days=args.days)
            print(json.dumps(report, indent=2, ensure_ascii=False))
        else:
            report = engine.generate_report(
                days=args.days,
                top_n=args.top,
                include_optimize=not getattr(args, "no_optimize", False),
            )
            print(report)
        return 0
    # publish
    post_id = run_content_generation(
        system,
        cadence=getattr(args, "cadence", "daily"),
        article_type=getattr(args, "type", None),
        manual_topic=getattr(args, "topic", None),
        dry_run=getattr(args, "dry_run", False),
    )
    return 0 if post_id else 1


if __name__ == "__main__":
    sys.exit(main())
