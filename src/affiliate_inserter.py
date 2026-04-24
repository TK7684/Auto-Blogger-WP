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


def _render_product_card(product: dict) -> str:
    """Render one Shopee product as a compact HTML card.

    Uses the offerLink (already affiliate-tracked per Shopee API). Safe-defaults
    any missing fields so a partial product response still renders.
    """
    name = (product.get("productName") or "สินค้า Shopee")[:80]
    # Price can arrive as a number or a string; coerce and format with ฿.
    price = product.get("price")
    try:
        price_fmt = f"฿{float(price):,.0f}" if price is not None else ""
    except (TypeError, ValueError):
        price_fmt = ""
    sales = product.get("sales") or 0
    rating = product.get("ratingStar") or 0.0
    image = product.get("imageUrl") or ""
    offer_link = product.get("offerLink") or product.get("productLink") or ""
    if not offer_link:
        return ""

    # Thai review count + rating strip
    meta_parts = []
    try:
        if rating and float(rating) > 0:
            meta_parts.append(f"⭐ {float(rating):.1f}")
    except (TypeError, ValueError):
        pass
    if sales and int(sales) > 0:
        meta_parts.append(f"ขายแล้ว {int(sales):,}")
    meta_strip = " · ".join(meta_parts)

    return (
        f'<a href="{offer_link}" rel="sponsored nofollow noopener" target="_blank" '
        f'style="display:flex;gap:14px;padding:12px;margin:10px 0;'
        f'border:1px solid #ffd5c8;border-radius:10px;background:#fff;'
        f'text-decoration:none;color:#222;align-items:center;">'
        f'<img src="{image}" alt="{name}" style="width:96px;height:96px;'
        f'object-fit:cover;border-radius:8px;flex-shrink:0;" loading="lazy">'
        f'<div style="flex:1;min-width:0;">'
        f'<div style="font-size:0.95em;font-weight:600;line-height:1.35;'
        f'display:-webkit-box;-webkit-line-clamp:2;-webkit-box-orient:vertical;'
        f'overflow:hidden;">{name}</div>'
        f'<div style="margin-top:6px;color:#ee4d2d;font-weight:700;font-size:1.1em;">{price_fmt}</div>'
        f'<div style="margin-top:4px;color:#888;font-size:0.8em;">{meta_strip}</div>'
        f'</div></a>'
    )


def _render_dynamic_card(topic: str, products: list[dict], search_url: str) -> str:
    """Render a rich card with N real product cards + a fallback search CTA."""
    heading = "🛍️ สินค้าแนะนำที่เกี่ยวข้องที่ Shopee"
    subcopy = f"สินค้ายอดนิยมสำหรับ “{topic}” พร้อมโปรโมชั่น"
    product_cards = "".join(_render_product_card(p) for p in products if p)
    cta_more = "ดูสินค้าเพิ่มเติมที่ Shopee →"

    return (
        '<div class="shopee-affiliate-card" '
        'style="margin:28px 0;padding:20px 22px;border:1px solid #ffd5c8;'
        'border-radius:10px;background:linear-gradient(135deg,#fff6f2 0%,#ffe8df 100%);'
        'font-family:inherit;">'
        f'<h3 style="margin:0 0 6px 0;font-size:1.1em;color:#ee4d2d;">{heading}</h3>'
        f'<p style="margin:0 0 8px 0;color:#333;font-size:0.95em;">{subcopy}</p>'
        f'{product_cards}'
        f'<p style="margin:16px 0 0 0;">'
        f'<a href="{search_url}" rel="sponsored nofollow noopener" target="_blank" '
        f'style="display:inline-block;padding:10px 22px;background:#ee4d2d;color:#fff;'
        f'border-radius:6px;text-decoration:none;font-weight:600;">'
        f'{cta_more}'
        f'</a></p>'
        '<p style="margin:16px 0 0 0;font-size:0.75em;color:#888;">'
        '*โพสต์นี้มีลิงก์พันธมิตร เราอาจได้รับค่าคอมมิชชั่นเมื่อคุณคลิกและซื้อสินค้า ในราคาที่คุณจ่ายเท่าเดิม'
        '</p>'
        '</div>'
    )


def _try_fetch_products(topic: str, limit: int = 3) -> list[dict]:
    """Attempt to pull products from the Shopee API. Return [] on any failure.

    Isolated in a try-everywhere wrapper so a misconfigured API / network issue
    can never block publishing. Import is lazy so the inserter module remains
    importable even if src.clients.shopee has an init-time error.
    """
    try:
        from src.clients import shopee  # lazy import — isolates import errors
        products = shopee.search_products(topic, limit=limit)
        return products or []
    except Exception as e:
        logger.info(f"[affiliate] shopee.search_products unavailable: {e}")
        return []


def insert_shopee_card(content: str, topic: str) -> str:
    """Append a Shopee affiliate CTA card to the article content.

    Resolution order:
      1. Try Shopee API (SHOPEE_APP_ID + SHOPEE_APP_SECRET set, creds work) →
         render rich card with N real product cards + "see more" CTA.
      2. Fall back to static search-only CTA card (requires SHOPEE_AFFILIATE_ID).
      3. No-op if neither auth method is configured.

    Safe to call unconditionally from the pipeline. Never raises.
    """
    if not topic or not topic.strip():
        logger.debug("[affiliate] empty topic — skipping card")
        return content

    topic_clean = topic.strip()

    try:
        affiliate_id = _get_affiliate_id()

        # 1. Dynamic path — real product cards from the Shopee API.
        products = _try_fetch_products(topic_clean, limit=3)
        if products and affiliate_id:
            search_url = _build_search_url(topic_clean, affiliate_id)
            card = _render_dynamic_card(topic_clean, products, search_url)
            logger.info(
                f"🛒 Shopee affiliate card inserted — DYNAMIC "
                f"(topic='{topic_clean[:40]}', products={len(products)})"
            )
            return content + "\n\n" + card

        # 2. Static fallback — search CTA + optional rotating featured link.
        if affiliate_id:
            search_url = _build_search_url(topic_clean, affiliate_id)
            featured_url = _pick_fixed_link(_get_fixed_links(), topic_clean)
            card = _render_card(topic_clean, search_url, featured_url)
            logger.info(
                f"🛒 Shopee affiliate card inserted — STATIC "
                f"(topic='{topic_clean[:40]}', featured={'yes' if featured_url else 'no'})"
            )
            return content + "\n\n" + card

        # 3. No auth configured — no-op.
        logger.debug("[affiliate] no SHOPEE_AFFILIATE_ID and no API creds — skipping")
        return content

    except Exception as e:
        # Never let affiliate-injection break publishing.
        logger.warning(f"[affiliate] insertion failed, content unchanged: {e}")
        return content
