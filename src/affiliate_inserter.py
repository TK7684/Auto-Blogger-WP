"""
Affiliate Inserter — injects Shopee affiliate CTA cards into published articles.

Research-informed design (see ψ/writing/pedpro-affiliate-conversion-research.md):
  - INLINE placement (~50% through article) converts 3-5x better than end-only.
  - 3 product cards is the sweet spot for mobile (not 1, not 5).
  - Always show rating + sales count — social proof drives Thai clicks.
  - Seasonal heading variants (Songkran / 6.6 / 11.11 / 12.12) tune CTA energy.
  - Thai PDPA-friendly disclosure line required per affiliate honesty norms.

Attribution:
  Shopee attribution fires ONLY via `s.shopee.co.th/XXX` short links that Shopee's
  Custom Link tool generates. Query-string `?af_id=` does NOT track. Use
  SHOPEE_AFFILIATE_LINKS (pre-generated short links) for real attribution;
  SHOPEE_AFFILIATE_ID builds fallback UI-only URLs when no short links available.

Integration: called from src/main.py via `insert_shopee_card(content, topic)`.
No-op if neither env var is set. Never raises.
"""
from __future__ import annotations

import datetime as _dt
import hashlib
import logging
import os
import re
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
    """Build a Shopee Thailand search URL.

    NOTE: `?af_id=` query-string does NOT track — only `s.shopee.co.th/XXX`
    short links do. Use SHOPEE_AFFILIATE_LINKS for actual attribution; this
    URL is only a fallback "shop on Shopee" exit when no short link is set.
    """
    return f"https://shopee.co.th/search?keyword={quote(keyword)}"


def _seasonal_variant(today: Optional[_dt.date] = None) -> tuple[str, str]:
    """Return (heading, subcopy_prefix) tuned to the Thai commerce calendar.

    Research: seasonal framing materially lifts CTR on Thai fashion/lifestyle
    blogs. Dates adapted for pedpro's 30-45 female audience.
    """
    d = today or _dt.date.today()
    m, day = d.month, d.day

    # Mega sales — biggest boosts
    if m == 6 and day == 6:
        return ("🔥 6.6 Shopee Mega Sale — ลดกระหน่ำทั้งวัน", "โปรโมชั่นแรง:")
    if m == 11 and day == 11:
        return ("🎉 11.11 Shopee Global Sale", "ดีลใหญ่ปีละครั้ง:")
    if m == 12 and day == 12:
        return ("🎁 12.12 Shopee Year-End Sale", "ปิดท้ายปี:")

    # Songkran window (2026-04-13 to 04-19)
    if m == 4 and 10 <= day <= 20:
        return ("🌊 ช้อปของคู่ใจสงกรานต์ที่ Shopee", "สินค้าสงกรานต์:")

    # Year-end gift window
    if (m == 12 and day >= 15) or (m == 1 and day <= 5):
        return ("🎊 ของขวัญปีใหม่จาก Shopee", "ไอเดียของขวัญ:")

    # Back-to-school windows
    if (m == 5 and 10 <= day <= 25) or (m == 11 and day <= 15):
        return ("🎒 เตรียมของเปิดเทอมที่ Shopee", "ของใช้เปิดเทอม:")

    # Default — evergreen framing
    return ("🛍️ สินค้าแนะนำที่เกี่ยวข้องที่ Shopee", "สินค้ายอดนิยม:")


def _find_inline_injection_index(content: str) -> Optional[int]:
    """Find the character index nearest to 50% through the article content that
    sits on a paragraph boundary (end of a </p> or blank line).

    Returns None if the content is too short to merit inline injection (< 1200
    chars), in which case callers should append at the end instead.
    """
    if not content or len(content) < 1200:
        return None

    target = len(content) // 2
    # Find paragraph endings near the midpoint
    boundary_re = re.compile(r"</p>\s*|\n\s*\n")
    boundaries = [m.end() for m in boundary_re.finditer(content)]
    if not boundaries:
        return None
    # Pick the boundary closest to target (prefer just AFTER midpoint for flow)
    after = [b for b in boundaries if b >= target]
    before = [b for b in boundaries if b < target]
    if after:
        return after[0]
    if before:
        return before[-1]
    return None


def _render_exit_cta(search_url: str) -> str:
    """Lighter, footer-only 'shop more' card — secondary exit after the primary
    inline card. No product images, smaller visual weight."""
    return (
        '<div class="shopee-affiliate-exit-cta" '
        'style="margin:24px 0;padding:14px 20px;border-top:2px solid #ee4d2d;'
        'background:#fff8f5;text-align:center;">'
        '<p style="margin:0 0 10px 0;color:#666;font-size:0.9em;">'
        'ชอบบทความนี้? ค้นพบสินค้าที่เกี่ยวข้องบน Shopee'
        '</p>'
        f'<a href="{search_url}" rel="sponsored nofollow noopener" target="_blank" '
        f'style="display:inline-block;padding:8px 20px;background:#ee4d2d;color:#fff;'
        f'border-radius:4px;text-decoration:none;font-weight:600;font-size:0.9em;">'
        f'ช้อปที่ Shopee →'
        f'</a></div>'
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
    heading, subcopy_prefix = _seasonal_variant()
    subcopy = f"{subcopy_prefix} สำหรับ “{topic}” พร้อมโปรโมชั่นและส่งฟรี"
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
    heading, subcopy_prefix = _seasonal_variant()
    subcopy = f"{subcopy_prefix} สำหรับ “{topic}” พร้อมโปรโมชั่น"
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


def _try_fetch_products(topic: str, limit: int = 5) -> list[dict]:
    """Attempt to pull products from the Shopee API. Return [] on any failure.

    Resolution order:
      1. Live Shopee Affiliate API (requires SHOPEE_APP_ID + SHOPEE_APP_SECRET).
      2. Curated fallback cache (data/shopee_products.json) — 250 vetted Thai
         products with real `s.shopee.co.th/...` attribution short links.

    The curated fallback ensures every published post inserts a card with
    `products>0` even when API credentials are missing/expired/rate-limited.
    Imports are lazy so this module stays importable on init-time errors.
    """
    # 1) Try the live Shopee API first — if creds work, real-time data is best.
    try:
        from src.clients import shopee  # lazy import — isolates import errors
        products = shopee.search_products(topic, limit=limit)
        if products:
            return products
    except Exception as e:
        logger.info(f"[affiliate] shopee.search_products unavailable: {e}")

    # 2) Fall back to the curated cache. Same offerLink format (s.shopee.co.th
    # short links → trackable attribution) so commission still flows.
    try:
        from src.clients import shopee_curated  # lazy import
        curated = shopee_curated.search(topic, limit=limit)
        if curated:
            logger.info(
                f"[affiliate] using curated cache fallback "
                f"({len(curated)}/{shopee_curated.cache_size()} products)"
            )
            return curated
    except Exception as e:
        logger.info(f"[affiliate] curated fallback unavailable: {e}")

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
        fixed_links = _get_fixed_links()

        # No auth configured at all — no-op. (Neither API creds work dynamic,
        # nor SHOPEE_AFFILIATE_LINKS short links, nor SHOPEE_AFFILIATE_ID fallback.)
        if not affiliate_id and not fixed_links:
            products_check = _try_fetch_products(topic_clean, limit=1)
            if not products_check:
                logger.debug("[affiliate] no auth + no products — skipping")
                return content

        # Build the primary card (dynamic if API returns products, static otherwise)
        search_url = _build_search_url(topic_clean, affiliate_id or "")
        products = _try_fetch_products(topic_clean, limit=5)

        # DYNAMIC card is preferred whenever we have products — each product
        # ships its own offerLink (already affiliate-tracked), so attribution
        # works even without SHOPEE_AFFILIATE_ID. The id is only used for the
        # secondary "see more" search CTA.
        if products:
            primary_card = _render_dynamic_card(topic_clean, products, search_url)
            mode = "DYNAMIC"
        elif affiliate_id:
            featured_url = _pick_fixed_link(fixed_links, topic_clean)
            primary_card = _render_card(topic_clean, search_url, featured_url)
            mode = "STATIC"
        elif fixed_links:
            # We have short links but no SHOPEE_AFFILIATE_ID — use the first
            # short link as the universal CTA (no search URL built).
            exit_url = _pick_fixed_link(fixed_links, topic_clean) or fixed_links[0]
            logger.info(f"🛒 Shopee affiliate — EXIT-ONLY (short-link rotation, topic='{topic_clean[:40]}')")
            return content + "\n\n" + _render_exit_cta(exit_url)
        else:
            return content

        # Research-informed placement: INLINE at ~50% through the article beats
        # end-only by 3-5x CTR. Also emit a lighter EXIT CTA at article end.
        inject_idx = _find_inline_injection_index(content)
        if inject_idx is not None:
            exit_url = search_url
            out = (
                content[:inject_idx]
                + "\n\n" + primary_card + "\n\n"
                + content[inject_idx:]
                + "\n\n" + _render_exit_cta(exit_url)
            )
            logger.info(
                f"🛒 Shopee affiliate card inserted — {mode} INLINE@{inject_idx} + exit "
                f"(topic='{topic_clean[:40]}', products={len(products) if products else 0})"
            )
            return out

        # Article too short for inline — append primary card + exit CTA at end
        logger.info(
            f"🛒 Shopee affiliate card inserted — {mode} END-ONLY (short article) "
            f"(topic='{topic_clean[:40]}', products={len(products) if products else 0})"
        )
        return content + "\n\n" + primary_card

    except Exception as e:
        # Never let affiliate-injection break publishing.
        logger.warning(f"[affiliate] insertion failed, content unchanged: {e}")
        return content
