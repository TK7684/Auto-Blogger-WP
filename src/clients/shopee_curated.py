"""
Curated Shopee product fallback.

Loads a vetted snapshot of Shopee products (with real `s.shopee.co.th/...`
attribution short links) bundled in `data/shopee_products.json`. Used as a
fallback when the live Shopee Affiliate API is unauthenticated, rate-limited,
or otherwise returning zero — so that every published article still inserts a
Shopee card with `products>0` and earns commission potential.

Why this exists:
  The live Shopee Affiliate GraphQL API requires APP_ID + APP_SECRET to be
  configured AND valid. When credentials are missing/expired, the live path
  returns []. Without a fallback, the affiliate card renders with `products=0`
  (header + outbound search CTA only — no product images, no rating/sales
  social proof), zeroing out the per-post conversion lift.

Source of truth:
  `data/shopee_products.json` — 250 real Thai-market products fetched from the
  Shopee Affiliate API (productOfferV2). Each entry includes:
    - itemId, productName, price, commission, commissionRate, sales, ratingStar
    - imageUrl, offerLink (s.shopee.co.th short link → trackable attribution)
    - productCatIds, shopId, shopName

Matching strategy:
  1. Tokenize topic into lowercase words (Thai + English).
  2. Score each product by token-overlap with productName.
  3. If best matches have score > 0, return top-N sorted by (overlap, value).
  4. Otherwise return top-N globally sorted by `commission × sales`
     (the same value-score the live client uses).

Refresh: regenerate `data/shopee_products.json` periodically (manually or via
a cron) by calling the live API when credentials are valid. See
tools/refresh_shopee_cache.py (TODO) for an automation hook.
"""
from __future__ import annotations

import json
import logging
import os
import re
import threading
from typing import Optional

logger = logging.getLogger(__name__)

# Resolve the data file path relative to the repo root (this file lives at
# src/clients/shopee_curated.py → repo root is two parents up).
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_DATA_PATH = os.path.join(_REPO_ROOT, "data", "shopee_products.json")

_CACHE_LOCK = threading.Lock()
_CACHE: Optional[list[dict]] = None

# Token splitter: keeps Thai and English word characters, splits on punctuation
# and whitespace. Thai words don't have spaces but the productName typically
# uses spaces between brand/feature/product chunks already.
_TOKEN_RE = re.compile(r"[A-Za-z0-9฀-๿]+")

# English stopwords commonly found in trending topics that shouldn't drive
# product matching.
_STOPWORDS = frozenset(
    {
        "the", "a", "an", "and", "or", "but", "of", "in", "on", "at", "to",
        "for", "with", "by", "from", "is", "are", "was", "were", "be", "been",
        "being", "have", "has", "had", "do", "does", "did", "will", "would",
        "could", "should", "may", "might", "must", "shall", "can", "this",
        "that", "these", "those", "i", "you", "he", "she", "it", "we", "they",
        "what", "which", "who", "whom", "whose", "when", "where", "why", "how",
        "all", "any", "both", "each", "few", "more", "most", "other", "some",
        "such", "no", "nor", "not", "only", "own", "same", "so", "than", "too",
        "very", "just", "as", "if", "viral", "trending", "news", "today",
        # Thai stopwords — common particles/connectors that pollute matching
        "และ", "หรือ", "แต่", "ของ", "ใน", "บน", "ที่", "ไป", "มา", "กับ",
        "โดย", "จาก", "คือ", "มี", "เป็น", "ได้", "ให้", "ไม่", "จะ", "ก็",
        "แล้ว", "นี้", "นั้น", "เหล่า", "ทุก", "เพื่อ", "เกี่ยว", "วิธี",
        "เรื่อง", "เกี่ยวกับ", "สำหรับ", "อย่าง", "มาก", "เรา", "คุณ",
        "เขา", "เธอ", "ฉัน", "พวก", "อะไร", "อย่างไร", "ทำไม", "เมื่อไหร่",
    }
)

# Topic-to-product-category mapping for relevance boosting.
# When the topic contains these keywords, products with matching name tokens
# get a relevance bonus on top of token overlap.
_RELEVANCE_KEYWORDS = {
    # Fashion / clothing
    "แฟชั่น": 2.0, "เสื้อ": 2.0, "กระโปรง": 2.0, "กางเกง": 2.0,
    "ชุด": 2.0, "สไตล์": 1.5, "แต่ง": 1.5, "สวย": 1.5, "แม่": 1.3,
    "ผู้หญิง": 1.5, "ผ้า": 1.3, "นุ่ม": 1.2, "ไหม": 1.3,
    # Lifestyle / home
    "บ้าน": 1.3, "ห้อง": 1.2, "ตกแต่ง": 1.5, "อยู่อาศัย": 1.3,
    "ครัว": 1.3, "จัดเก็บ": 1.5, "ทำความสะอาด": 1.3,
    # Beauty / health
    "ผิว": 1.5, "หน้า": 1.3, "บำรุง": 1.5, "สกินแคร์": 2.0,
    "ความงาม": 1.5, "หน้าใส": 1.3, "มาส์ก": 1.5, "น้ำหอม": 1.5,
    # Pet (pedpro = pet professional)
    "สุนัข": 2.0, "แมว": 2.0, "สัตว์เลี้ยง": 2.0, "น้องหมา": 2.0,
    "น้องแมว": 2.0, "อาหารสัตว์": 2.0, "ที่นอนสุนัข": 2.0,
    # Tech / gadgets
    "ชาร์จ": 1.3, "หูฟัง": 1.3, "แก็ดเจ็ต": 1.3, "มือถือ": 1.3,
}

# English topic keyword → Thai product search terms mapping.
# When an English topic has no direct token overlap with Thai product names,
# these expansions bridge the language gap.
_TOPIC_EXPANSIONS = {
    "ai": ["ai", "artificial intelligence", "ปัญญาประดิษฐ์"],
    "tool": ["เครื่องมือ", "อุปกรณ์", "tool"],
    "software": ["ซอฟต์แวร์", "software", "โปรแกรม"],
    "app": ["แอป", "app", "application"],
    "tech": ["เทคโนโลยี", "technology", "gadget"],
    "phone": ["มือถือ", "โทรศัพท์", "smartphone"],
    "laptop": ["แล็ปท็อป", "laptop", "notebook", "คอมพิวเตอร์"],
    "invest": ["ลงทุน", "investment", "การเงิน", "finance"],
    "money": ["เงิน", "การเงิน", "finance", "budget"],
    "pet": ["สัตว์เลี้ยง", "อาหารสัตว์", "pet", "สุนัข", "แมว"],
    "dog": ["สุนัข", "หมา", "น้องหมา", "อาหารสุนัข"],
    "cat": ["แมว", "น้องแมว", "อาหารแมว"],
    "food": ["อาหาร", "food", "ขนม", "อร่อย"],
    "cook": ["ทำอาหาร", "อุปกรณ์ครัว", "kitchen"],
    "kitchen": ["ครัว", "อุปกรณ์ครัว", "kitchen"],
    "beauty": ["ความงาม", "สกินแคร์", "ผิว", "beauty"],
    "skin": ["ผิว", "สกินแคร์", "ครีม", "ดูแลผิว"],
    "fashion": ["แฟชั่น", "เสื้อ", "กระโปรง", "เสื้อผ้า"],
    "health": ["สุขภาพ", "อาหารเสริม", "health", "วิตามิน"],
    "baby": ["เด็ก", "ทารก", "ของใช้เด็ก", "baby"],
    "home": ["บ้าน", "ของใช้ในบ้าน", "อุปกรณ์บ้าน", "home"],
    "travel": ["ท่องเที่ยว", "กระเป๋าเดินทาง", "travel", "การเดินทาง"],
    "camera": ["กล้อง", "camera", "ถ่ายภาพ"],
    "gaming": ["เกม", "gaming", "game", "คอนโซล"],
    "music": ["เพลง", "music", "ลำโพง", "หูฟัง"],
    "sport": ["กีฬา", "sport", "ออกกำลังกาย", "fitness"],
    "fitness": ["ออกกำลังกาย", "fitness", "กีฬา", "sport"],
    "book": ["หนังสือ", "book", "การศึกษา", "เรียน"],
    "study": ["เรียน", "การศึกษา", "หนังสือ", "อุปกรณ์เรียน"],
}


def _load() -> list[dict]:
    """Load the curated product cache once and memoize."""
    global _CACHE
    if _CACHE is not None:
        return _CACHE
    with _CACHE_LOCK:
        if _CACHE is not None:
            return _CACHE
        try:
            with open(_DATA_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
            if not isinstance(data, list):
                logger.warning(
                    "[shopee_curated] %s did not contain a list — got %s",
                    _DATA_PATH, type(data).__name__,
                )
                _CACHE = []
            else:
                _CACHE = [p for p in data if isinstance(p, dict) and p.get("offerLink")]
                logger.info("[shopee_curated] loaded %d curated products from %s",
                            len(_CACHE), _DATA_PATH)
        except FileNotFoundError:
            logger.info("[shopee_curated] no cache file at %s — fallback disabled",
                        _DATA_PATH)
            _CACHE = []
        except (json.JSONDecodeError, OSError) as e:
            logger.warning("[shopee_curated] failed to load cache: %s", e)
            _CACHE = []
    return _CACHE


def _tokenize(text: str) -> set[str]:
    """Lowercase, split into Thai/English/digit tokens, drop stopwords + 1-char tokens."""
    if not text:
        return set()
    tokens = _TOKEN_RE.findall(text.lower())
    return {t for t in tokens if len(t) > 1 and t not in _STOPWORDS}


def _value_score(product: dict) -> float:
    """commission × sales — same metric the live client uses."""
    try:
        commission = float(product.get("commission") or 0)
    except (TypeError, ValueError):
        commission = 0.0
    try:
        sales = float(product.get("sales") or 0)
    except (TypeError, ValueError):
        sales = 0.0
    return commission * sales


def search(keyword: str, limit: int = 3) -> list[dict]:
    """Return up to `limit` curated products best-matching `keyword`.

    Matching:
      1. Token-overlap between keyword tokens and productName tokens — primary key.
      2. Topic expansion matching — bridges English topics to Thai product names.
      3. Relevance bonus from _RELEVANCE_KEYWORDS matching keyword tokens.
      4. value_score (commission × sales) — tiebreak / fallback when no overlap.

    Always returns at most `limit` products and never raises.
    """
    products = _load()
    if not products or limit <= 0:
        return []

    kw_tokens = _tokenize(keyword)
    kw_lower = keyword.lower()
    
    # Build expanded search terms from English keywords
    expanded_terms = set()
    for eng_word, thai_exps in _TOPIC_EXPANSIONS.items():
        if eng_word in kw_lower or eng_word in kw_tokens:
            for exp in thai_exps:
                expanded_terms.update(_tokenize(exp))
    # Merge expanded tokens with original keyword tokens
    search_tokens = kw_tokens | expanded_terms

    # Calculate relevance bonus from keyword-level matching
    relevance_bonus = 0.0
    for rkw, weight in _RELEVANCE_KEYWORDS.items():
        if rkw in kw_lower or rkw in kw_tokens:
            relevance_bonus += weight

    scored: list[tuple[float, float, dict]] = []
    for p in products:
        name_tokens = _tokenize(p.get("productName") or "")
        name_lower = (p.get("productName") or "").lower()
        
        # Primary: direct token overlap
        overlap = len(search_tokens & name_tokens) if search_tokens else 0
        
        # Expansion bonus: partial matches with Thai product names
        expansion_score = 0.0
        for exp_term in expanded_terms:
            if exp_term in name_lower:
                expansion_score += 0.5  # Moderate boost for expansion matches

        # Product-level relevance: check if product name contains relevance keywords
        product_relevance = 0.0
        if relevance_bonus > 0:
            for rkw, weight in _RELEVANCE_KEYWORDS.items():
                if rkw in name_lower:
                    product_relevance += weight

        # Combined score: overlap * 3 + expansion + category match + relevance
        category_match = min(relevance_bonus, product_relevance) if relevance_bonus > 0 else 0
        combined = overlap * 3.0 + expansion_score + category_match
        scored.append((combined, _value_score(p), p))

    scored.sort(key=lambda t: (t[0], t[1]), reverse=True)

    has_match = any(s[0] > 0 for s in scored[:limit])
    selected = [s[2] for s in scored[:limit]]

    if has_match:
        logger.debug(
            "[shopee_curated] keyword=%r expanded=%d matched %d/%d products (score>0)",
            keyword, len(expanded_terms),
            sum(1 for s in scored[:limit] if s[0] > 0), len(selected),
        )
    else:
        logger.debug(
            "[shopee_curated] keyword=%r no match — using top-by-value",
            keyword,
        )

    return selected


def cache_size() -> int:
    """Return the count of products in the curated cache. Useful for diagnostics."""
    return len(_load())
