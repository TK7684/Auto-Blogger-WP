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
# product matching. Thai stopwords intentionally omitted — Thai search benefits
# from preserving particles since the productName tokenization is sparse.
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
    }
)


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
      2. value_score (commission × sales) — tiebreak / fallback when no overlap.

    Always returns at most `limit` products and never raises. If the cache is
    empty (file missing or corrupt), returns [].
    """
    products = _load()
    if not products:
        return []

    if limit <= 0:
        return []

    kw_tokens = _tokenize(keyword)

    scored: list[tuple[int, float, dict]] = []
    for p in products:
        name_tokens = _tokenize(p.get("productName") or "")
        overlap = len(kw_tokens & name_tokens) if kw_tokens else 0
        scored.append((overlap, _value_score(p), p))

    # Sort by overlap DESC, then value DESC. If any product has overlap > 0 we
    # still backfill from top-by-value so the card always renders `limit` items.
    scored.sort(key=lambda t: (t[0], t[1]), reverse=True)

    has_match = any(s[0] > 0 for s in scored[:limit])
    selected = [s[2] for s in scored[:limit]]

    if has_match:
        logger.debug(
            "[shopee_curated] keyword=%r matched %d/%d products via token overlap",
            keyword, sum(1 for s in scored[:limit] if s[0] > 0), len(selected),
        )
    else:
        logger.debug(
            "[shopee_curated] keyword=%r no token overlap — using top-by-value",
            keyword,
        )

    return selected


def cache_size() -> int:
    """Return the count of products in the curated cache. Useful for diagnostics."""
    return len(_load())
