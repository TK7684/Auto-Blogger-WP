"""
Affiliate Inserter — appends Shopee affiliate CTA cards to published articles.

Reads SHOPEE_AFFILIATE_ID (and optionally SHOPEE_AFFILIATE_LINKS) from env, and
appends a styled "related products" card to the end of generated content before
it's sent to WordPress. Pure content transformation — no external API calls.

Design:
  - No-op if SHOPEE_AFFILIATE_ID is unset (safe default — pipeline still runs).
  - Uses Shopee Thailand's standard affiliate deep-link format:
    https://shopee.co.th/search?keyword=<url-encoded-keyword>&af_id=<AFFILIATE_ID>
  - Card is semantic HTML (not an iframe), SEO-friendly, mobile-responsive inline CSS.
  - Topic keyword is URL-encoded and used to drive a contextual search link.
  - If SHOPEE_AFFILIATE_LINKS env is set (comma-separated full affiliate URLs), one
    is picked round-robin as a fixed "featured product" link alongside the search CTA.

Integration: called from src/main.py between `clean_remaining_placeholders(...)` and
the slug/schema-generation step, via `insert_shopee_card(content, topic)`.
"""
from __future__ import annotations

import hashlib
import logging
import os
from typing import Optional
from urllib.parse import quote

logger = logging.getLogger(__name__)


def _get_affiliate_id() -> Optional[str]:
    """Return SHOPEE_AFFILIATE_ID or None if unset/blank."""
    aid = os.getenv("SHOPEE_AFFILIATE_ID", "").strip()
    return aid or None


def _get_fixed_links() -> list[str]:
    """Return comma-separated SHOPEE_AFFILIATE_LINKS as a cleaned list."""
    raw = os.getenv("SHOPEE_AFFILIATE_LINKS", "")
    return [u.strip() for u in raw.split(",") if u.strip().startswith("http")]


def _build_search_url(keyword: str, affiliate_id: str) -> str:
    """Build a Shopee Thailand affiliate search URL."""
    return (
        f"https://shopee.co.th/search?keyword={quote(keyword)}"
        f"&af_id={quote(affiliate_id)}"
    )


def _pick_fixed_link(links: list[str], seed: str) -> Optional[str]:
    """Deterministic round-robin by topic hash — same topic gets same featured link."""
    if not links:
        return None
    h = int(hashlib.sha1(seed.encode("utf-8")).hexdigest(), 16)
    return links[h % len(links)]


def _render_card(topic: str, search_url: str, featured_url: Optional[str]) -> str:
    """Render the Shopee CTA card as inline-styled HTML."""
    # Thai-primary copy; keeps Thai content voice consistent with the blog.
    heading = "🛍️ ช้อปสินค้าที่เกี่ยวข้องที่ Shopee"
    subcopy = f"พบสินค้ายอดนิยมสำหรับ “{topic}” พร้อมโปรโมชั่นและส่งฟรี"
    cta_primary = "ดูสินค้าทั้งหมดที่ Shopee →"
    cta_featured = "🔥 สินค้าแนะนำของวันนี้"

    featured_block = ""
    if featured_url:
        featured_block = (
            f'<p style="margin:12px 0 0 0;">'
            f'<a href="{featured_url}" rel="sponsored nofollow noopener" target="_blank" '
            f'style="color:#ee4d2d;font-weight:600;text-decoration:none;">'
            f'{cta_featured}'
            f'</a></p>'
        )

    card = (
        '<div class="shopee-affiliate-card" '
        'style="margin:28px 0;padding:20px 22px;border:1px solid #ffd5c8;'
        'border-radius:10px;background:linear-gradient(135deg,#fff6f2 0%,#ffe8df 100%);'
        'font-family:inherit;">'
        f'<h3 style="margin:0 0 8px 0;font-size:1.1em;color:#ee4d2d;">{heading}</h3>'
        f'<p style="margin:0;color:#333;font-size:0.95em;">{subcopy}</p>'
        f'<p style="margin:14px 0 0 0;">'
        f'<a href="{search_url}" rel="sponsored nofollow noopener" target="_blank" '
        f'style="display:inline-block;padding:10px 22px;background:#ee4d2d;color:#fff;'
        f'border-radius:6px;text-decoration:none;font-weight:600;">'
        f'{cta_primary}'
        f'</a></p>'
        f'{featured_block}'
        '<p style="margin:16px 0 0 0;font-size:0.75em;color:#888;">'
        '*โพสต์นี้มีลิงก์พันธมิตร เราอาจได้รับค่าคอมมิชชั่นเมื่อคุณคลิกและซื้อสินค้า ในราคาที่คุณจ่ายเท่าเดิม'
        '</p>'
        '</div>'
    )
    return card


def insert_shopee_card(content: str, topic: str) -> str:
    """Append a Shopee affiliate CTA card to the article content.

    No-op if SHOPEE_AFFILIATE_ID is unset or topic is empty.
    Safe to call unconditionally from the pipeline.

    Args:
        content: The finalized article HTML about to be published.
        topic: The article topic/keyword used to build the search URL.

    Returns:
        Content with the affiliate card appended, or unchanged content on no-op.
    """
    affiliate_id = _get_affiliate_id()
    if not affiliate_id:
        logger.debug("[affiliate] SHOPEE_AFFILIATE_ID unset — skipping card")
        return content
    if not topic or not topic.strip():
        logger.debug("[affiliate] empty topic — skipping card")
        return content

    try:
        search_url = _build_search_url(topic.strip(), affiliate_id)
        featured_url = _pick_fixed_link(_get_fixed_links(), topic.strip())
        card = _render_card(topic.strip(), search_url, featured_url)
        logger.info(
            f"🛒 Shopee affiliate card inserted (topic='{topic[:40]}', "
            f"featured={'yes' if featured_url else 'no'})"
        )
        return content + "\n\n" + card
    except Exception as e:
        # Never let affiliate-injection break publishing.
        logger.warning(f"[affiliate] insertion failed, content unchanged: {e}")
        return content
