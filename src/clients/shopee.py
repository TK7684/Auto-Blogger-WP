"""
Shopee Affiliate API Client for Python.

Ported from JavaScript: /home/tk578/ShopeeTH-aff/backup/shopeeApi.js

Env vars:
  SHOPEE_APP_ID: Affiliate dashboard Developer section
  SHOPEE_APP_SECRET: Affiliate dashboard Developer section
  SHOPEE_API_ENDPOINT (optional): defaults to https://open-api.affiliate.shopee.co.th/graphql

Enhanced: dynamic matching extracts key nouns from article text, fuzzy matches
product titles, and prefers 4+ star rated products for higher conversion.
"""

import hashlib
import json
import logging
import os
import re
import time
from difflib import SequenceMatcher
from typing import Optional

import httpx

logger = logging.getLogger(__name__)

APP_ID = os.getenv("SHOPEE_APP_ID", "")
APP_SECRET = os.getenv("SHOPEE_APP_SECRET", "")
ENDPOINT = os.getenv(
    "SHOPEE_API_ENDPOINT",
    "https://open-api.affiliate.shopee.co.th/graphql",
)


def _sha256(s: str) -> str:
    """SHA256 hash function."""
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


# Common Thai → English keyword translations for Shopee product search.
# When a topic contains Thai terms, we try the English equivalent to find
# more products (Shopee TH indexes both languages).
_THAI_EN_MAP = {
    "สุนัข": "dog", "แมว": "cat", "สัตว์เลี้ยง": "pet",
    "อาหารสัตว์": "pet food", "ของเล่นสุนัข": "dog toy",
    "เสื้อ": "shirt", "กางเกง": "pants", "รองเท้า": "shoes",
    "กระเป๋า": "bag", "หมวก": "hat", "แว่นตา": "glasses",
    "สกินแคร์": "skincare", "ครีม": "cream", "มาส์ก": "mask",
    "ผิว": "skin", "หน้าใส": "brightening",
    "มือถือ": "phone", "หูฟัง": "earphone", "ชาร์จ": "charger",
    "คีย์บอร์ด": "keyboard", "เมาส์": "mouse",
    "บ้าน": "home", "ครัว": "kitchen", "ห้องนอน": "bedroom",
    "กล้อง": "camera", "ดิจิทัล": "digital",
    "ออกกำลังกาย": "exercise", "ฟิตเนส": "fitness",
    "หนังสือ": "book", "เรียน": "study",
}


def _build_keyword_variants(keyword: str) -> list[str]:
    """Build a list of keyword search variants for broader product matching.

    Returns the original keyword first, then Thai→English translations,
    then shortened versions (first 2-3 words).
    """
    variants = [keyword]
    kw_lower = keyword.lower()

    # Thai → English translations
    for th, en in _THAI_EN_MAP.items():
        if th in kw_lower and en not in " ".join(variants).lower():
            variants.append(en)

    # Shorten: take first 2-3 significant words
    words = keyword.split()
    if len(words) > 3:
        short = " ".join(words[:3])
        if short.lower() not in kw_lower:
            variants.append(short)
    if len(words) > 2:
        short2 = " ".join(words[:2])
        if short2.lower() not in kw_lower and short2 != variants[-1]:
            variants.append(short2)

    return variants


def _extract_key_nouns(text: str) -> list[str]:
    """Extract key nouns/phrases from article text for product matching.

    Strategy:
    1. Split text into words/tokens
    2. Filter out stop words (English + Thai common words)
    3. Keep substantial nouns (length >= 2 for English, >= 1 for Thai)
    4. Return unique terms sorted by frequency (most common first)
    """
    if not text:
        return []

    # English stop words
    EN_STOPS = {
        "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
        "have", "has", "had", "do", "does", "did", "will", "would", "could",
        "should", "may", "might", "can", "shall", "to", "of", "in", "for",
        "on", "with", "at", "by", "from", "as", "into", "through", "during",
        "before", "after", "above", "below", "between", "out", "off", "over",
        "under", "again", "further", "then", "once", "here", "there", "when",
        "where", "why", "how", "all", "each", "every", "both", "few", "more",
        "most", "other", "some", "such", "no", "nor", "not", "only", "own",
        "same", "so", "than", "too", "very", "just", "because", "but", "and",
        "or", "if", "while", "about", "up", "it", "its", "this", "that",
        "these", "those", "i", "you", "he", "she", "we", "they", "what",
        "which", "who", "whom", "your", "their", "our", "my", "his", "her",
        "also", "even", "still", "already", "yet", "much", "many", "well",
        "back", "get", "got", "like", "new", "one", "two", "way", "use",
        "make", "made", "know", "think", "see", "look", "come", "take",
        "want", "give", "find", "tell", "say", "said", "try", "need",
    }

    # Thai stop words
    TH_STOPS = {
        "และ", "ที่", "ใน", "ของ", "กับ", "เป็น", "มี", "ไม่", "จะ", "ได้",
        "ให้", "ไป", "มา", "อยู่", "แต่", "เพื่อ", "จาก", "นี้", "นั้น",
        "อะไร", "อย่าง", "โดย", "เหมือน", "หรือ", "ถ้า", "เมื่อ", "ก่อน",
        "หลัง", "ระหว่าง", "เพราะ", "ดังนั้น", "ดัง", "ทั้ง", "สำหรับ",
        "แล้ว", "จึง", "ยัง", "อีก", "ทุก", "บาง", "อยาก", "ต้อง",
    }

    # Tokenize: split by whitespace and punctuation, keep Thai chars together
    tokens = re.findall(r'[\u0E00-\u0E7F]+|[a-zA-Z0-9]+(?:[-\'][a-zA-Z0-9]+)*', text.lower())

    # Count word frequencies
    freq: dict[str, int] = {}
    for token in tokens:
        # Skip stop words and very short English tokens
        if token in EN_STOPS or token in TH_STOPS:
            continue
        if len(token) < 3 and re.match(r'^[a-z]+$', token):
            continue
        # Skip pure numbers
        if re.match(r'^\d+$', token):
            continue
        freq[token] = freq.get(token, 0) + 1

    # Sort by frequency descending
    sorted_terms = sorted(freq.items(), key=lambda x: x[1], reverse=True)
    return [term for term, _ in sorted_terms[:20]]  # Top 20 key terms


def _fuzzy_match_score(query: str, title: str) -> float:
    """Calculate fuzzy match score between query terms and product title.

    Uses token overlap + SequenceMatcher for fuzzy matching.
    Returns a score from 0.0 to 1.0.
    """
    if not query or not title:
        return 0.0

    query_lower = query.lower().strip()
    title_lower = title.lower().strip()
    query_tokens = set(re.findall(r'[\u0E00-\u0E7F]+|[a-z0-9]+', query_lower))
    title_tokens = set(re.findall(r'[\u0E00-\u0E7F]+|[a-z0-9]+', title_lower))

    if not query_tokens or not title_tokens:
        return 0.0

    # Token overlap ratio (Jaccard-like)
    overlap = query_tokens & title_tokens
    token_score = len(overlap) / max(len(query_tokens), 1)

    # Fuzzy string similarity (SequenceMatcher)
    fuzzy_score = SequenceMatcher(None, query_lower, title_lower).ratio()

    # Weighted combination: token overlap is more reliable for product matching
    return (token_score * 0.6) + (fuzzy_score * 0.4)


def _score_product_for_article(
    product: dict,
    key_nouns: list[str],
    article_text: str = "",
    min_rating: float = 4.0,
) -> tuple[float, dict]:
    """Score a product for relevance to an article's key nouns.

    Combines:
    - Fuzzy match score against key nouns (0-6 points)
    - Rating bonus: prefer 4+ stars (0-3 points)
    - Sales × commission value score (0-3 points)

    Returns (score, product) tuple.
    """
    title = product.get("productName", "")
    rating = 0.0
    try:
        rating = float(product.get("ratingStar") or 0)
    except (TypeError, ValueError):
        pass

    # 1. Fuzzy match against all key nouns — take best match
    best_match = 0.0
    for noun in key_nouns[:10]:  # Check top 10 nouns
        score = _fuzzy_match_score(noun, title)
        best_match = max(best_match, score)

    # Also try the full article text's first 100 chars as a query
    if article_text:
        article_snippet = re.sub(r'<[^>]+>', '', article_text)[:200].strip()
        snippet_match = _fuzzy_match_score(article_snippet, title)
        best_match = max(best_match, snippet_match * 0.5)  # De-weight full-text match

    match_score = best_match * 6.0  # Max 6 points

    # 2. Rating bonus — prefer 4+ stars
    if rating >= 4.5:
        rating_bonus = 3.0
    elif rating >= 4.0:
        rating_bonus = 2.0
    elif rating >= 3.5:
        rating_bonus = 0.5
    else:
        rating_bonus = 0.0

    # 3. Value score (commission × sales, normalized)
    try:
        commission = float(product.get("commission") or 0)
        sales = int(product.get("sales") or 0)
        value_score = min((commission * sales) / 50000, 3.0)  # Max 3 points
    except (TypeError, ValueError):
        value_score = 0.0

    # 4. Minimum rating gate: products below min_rating get heavy penalty
    if rating < min_rating:
        rating_penalty = 0.3  # Keep them possible but de-prioritized
    else:
        rating_penalty = 1.0

    total = (match_score + rating_bonus + value_score) * rating_penalty
    return (total, product)


def search_products_for_article(
    article_text: str,
    topic: str = "",
    limit: int = 3,
    min_rating: float = 4.0,
) -> list[dict]:
    """Search for products dynamically matched to article content.

    Extracts key nouns from article text, searches Shopee with those terms,
    then ranks results by fuzzy match + rating + value score.

    Args:
        article_text: Full article HTML or plain text
        topic: Article topic/title (used as primary search keyword)
        limit: Max products to return (default 3 — optimal for mobile)
        min_rating: Minimum star rating preference (default 4.0)

    Returns:
        List of product dicts, sorted by relevance score descending.
    """
    if not APP_ID or not APP_SECRET:
        logger.warning("SHOPEE_APP_ID or SHOPEE_APP_SECRET not set")
        return []

    # Extract key nouns from article content
    key_nouns = _extract_key_nouns(article_text) if article_text else []

    # Build search terms: topic first, then key nouns
    search_terms = []
    if topic and topic.strip():
        search_terms.append(topic.strip())
    search_terms.extend(key_nouns[:5])  # Top 5 nouns

    if not search_terms:
        return []

    # Search products using each term
    all_products = []
    seen_ids = set()

    for term in search_terms[:3]:  # Max 3 API calls to avoid rate limits
        try:
            products = search_products(keyword=term, limit=10)
            for p in products:
                pid = p.get("itemId")
                if pid and pid not in seen_ids:
                    seen_ids.add(pid)
                    all_products.append(p)
        except Exception as e:
            logger.error(f"Error searching for term '{term}': {e}")

    if not all_products:
        return []

    # Score and rank by relevance to article
    scored = [
        _score_product_for_article(p, key_nouns, article_text, min_rating)
        for p in all_products
    ]
    scored.sort(key=lambda x: x[0], reverse=True)

    # Log top matches for debugging
    top = scored[:limit]
    for score, p in top:
        rating = p.get("ratingStar", "?")
        name = (p.get("productName") or "")[:50]
        logger.debug(f"[shopee] matched '{name}' score={score:.2f} rating={rating}")

    return [p for _, p in top]


def _generate_auth(payload: str) -> dict:
    """
    Generate authorization headers for Shopee GraphQL API.

    Based on shopeeApi.js:99-111
    Signature algorithm: SHA256(APP_ID + timestamp + payload + SECRET)
    """
    timestamp = str(int(time.time()))
    sign_string = APP_ID + timestamp + payload + APP_SECRET
    signature = _sha256(sign_string)

    return {
        "headers": {
            "Content-Type": "application/json",
            "Authorization": f"SHA256 Credential={APP_ID}, Timestamp={timestamp}, Signature={signature}",
        },
        "timestamp": timestamp,
        "signature": signature,
    }


def _extract_products(response_data: dict) -> list[dict]:
    """Extract product list from GraphQL response."""
    try:
        if not response_data or not isinstance(response_data, dict):
            return []

        data = response_data.get("data")
        if not data or not isinstance(data, dict):
            return []

        product_offer = data.get("productOfferV2")
        if not product_offer or not isinstance(product_offer, dict):
            return []

        nodes = product_offer.get("nodes", [])
        products = []
        for node in nodes:
            products.append(
                {
                    "itemId": node.get("itemId"),
                    "productName": node.get("productName"),
                    "price": node.get("price"),
                    "commissionRate": node.get("commissionRate"),
                    "commission": node.get("commission"),
                    "sales": node.get("sales"),
                    "ratingStar": node.get("ratingStar"),
                    "imageUrl": node.get("imageUrl"),
                    "productLink": node.get("productLink"),
                    "offerLink": node.get("offerLink"),
                    "productCatIds": node.get("productCatIds", []),
                }
            )
        return products
    except (KeyError, TypeError) as e:
        logger.warning(f"Error extracting products from response: {e}")
        return []


def search_products(keyword: str, limit: int = 5) -> list[dict]:
    """
    Search products by keyword and return top-N by value score.

    Value score = commission × sales (higher is better).

    Strategy: for better relevance, we also try shortened keyword variants
    and English translations of common Thai terms to maximize product matches.

    Args:
        keyword: Search keyword
        limit: Max number of results to return

    Returns:
        List of product dicts with fields: itemId, productName, price, etc.
    """
    if not APP_ID or not APP_SECRET:
        logger.warning("SHOPEE_APP_ID or SHOPEE_APP_SECRET not set, returning empty list")
        return []

    # Build keyword variants for broader matching
    keyword_variants = _build_keyword_variants(keyword)
    
    all_products = []
    seen_ids = set()
    
    for kw in keyword_variants[:3]:  # Try up to 3 variants to avoid rate limits
        query = f"""
    {{
      productSearchV3(keyword: "{kw}", limit: 50, page: 1) {{
        nodes {{
          itemId
          productName
          commissionRate
          commission
          price
          sales
          ratingStar
          imageUrl
          productLink
          offerLink
        }}
        pageInfo {{
          page
          limit
          hasNextPage
        }}
      }}
    }}
    """

        payload = json.dumps({"query": query})
        auth = _generate_auth(payload)

        try:
            with httpx.Client() as client:
                response = client.post(
                    ENDPOINT,
                    headers=auth["headers"],
                    content=payload,
                    timeout=30.0,
                )
                response.raise_for_status()
                data = response.json()

            products = _extract_products(data)
            
            # Deduplicate by itemId
            for p in products:
                pid = p.get("itemId")
                if pid and pid not in seen_ids:
                    seen_ids.add(pid)
                    all_products.append(p)
            
            if len(all_products) >= limit * 2:
                break  # Enough candidates
                
        except Exception as e:
            logger.error(f"Error searching products for '{kw}': {e}")

    # Score by commission × sales, sort descending
    scored = [
        (p, (p.get("commission") or 0) * (p.get("sales") or 0))
        for p in all_products
    ]
    scored.sort(key=lambda x: x[1], reverse=True)
    return [p[0] for p in scored[:limit]]


def fetch_products(limit: int = 100, page: int = 1) -> list[dict]:
    """
    Fetch paginated product list.

    Args:
        limit: Products per page (max ~100)
        page: Page number (1-indexed)

    Returns:
        List of product dicts
    """
    if not APP_ID or not APP_SECRET:
        logger.warning("SHOPEE_APP_ID or SHOPEE_APP_SECRET not set, returning empty list")
        return []

    query = f"""
    {{
      productOfferV2(limit: {limit}, page: {page}) {{
        nodes {{
          itemId
          productName
          commissionRate
          commission
          price
          sales
          ratingStar
          imageUrl
          productLink
          offerLink
        }}
        pageInfo {{
          page
          limit
          hasNextPage
        }}
      }}
    }}
    """

    payload = json.dumps({"query": query})
    auth = _generate_auth(payload)

    try:
        with httpx.Client() as client:
            response = client.post(
                ENDPOINT,
                headers=auth["headers"],
                content=payload,
                timeout=30.0,
            )
            response.raise_for_status()
            data = response.json()

        return _extract_products(data)

    except Exception as e:
        logger.error(f"Error fetching products: {e}")
        return []


def fetch_by_category(
    category_id: int,
    limit: int = 100,
    page: int = 1,
) -> list[dict]:
    """
    Fetch products by category ID.

    Args:
        category_id: Shopee category ID to filter by
        limit: Products per page
        page: Page number (1-indexed)

    Returns:
        List of product dicts filtered by category
    """
    if not APP_ID or not APP_SECRET:
        logger.warning("SHOPEE_APP_ID or SHOPEE_APP_SECRET not set, returning empty list")
        return []

    query = f"""
    {{
      productOfferV2(limit: {limit}, page: {page}) {{
        nodes {{
          itemId
          productName
          commissionRate
          commission
          price
          sales
          ratingStar
          imageUrl
          productLink
          offerLink
          productCatIds
        }}
        pageInfo {{
          page
          limit
          hasNextPage
        }}
      }}
    }}
    """

    payload = json.dumps({"query": query})
    auth = _generate_auth(payload)

    try:
        with httpx.Client() as client:
            response = client.post(
                ENDPOINT,
                headers=auth["headers"],
                content=payload,
                timeout=30.0,
            )
            response.raise_for_status()
            data = response.json()

        products = _extract_products(data)

        # Filter by category ID
        filtered = [
            p
            for p in products
            if category_id in (p.get("productCatIds") or [])
        ]
        logger.info(
            f"Filtered {len(products)} products to {len(filtered)} in category {category_id}"
        )
        return filtered

    except Exception as e:
        logger.error(f"Error fetching products by category: {e}")
        return []


# ---------------------------------------------------------------------------
# sub_id tracking via generateBatchShortLink
# ---------------------------------------------------------------------------

def generate_short_links(
    items: list[dict],
    sub_ids: Optional[list[str]] = None,
) -> list[dict]:
    """Generate tracked short links with sub_ids for conversion attribution.

    Shopee's ``generateBatchShortLink`` mutation accepts an offerLink and up
    to 5 sub_ids per item, returning one short link per sub_id.  The sub_id
    appears in your Shopee Affiliate Dashboard → Performance → Sub ID
    report so you can see *which article* drove the click/order.

    Each item dict must have an ``offerLink`` key.  Returns enriched items
    with an additional ``shortLinks`` key: ``[{sub_id, shortLink}, ...]``.
    """
    if not APP_ID or not APP_SECRET:
        logger.warning("[shopee] generate_short_links: no API creds — no sub_id links")
        return items

    if sub_ids is None:
        sub_ids = ["pedpro"]
    if len(sub_ids) > 5:
        sub_ids = sub_ids[:5]

    enriched: list[dict] = []

    for item in items:
        offer_link = item.get("offerLink") or item.get("productLink") or ""
        if not offer_link:
            enriched.append(item)
            continue

        # Expand template tokens per-item
        expanded_sub_ids = []
        for sid in sub_ids:
            sid_expanded = (
                sid.replace("{{itemId}}", str(item.get("itemId", "")))
                .replace("{{keyword}}", (item.get("productName") or "")[:30])
            )
            expanded_sub_ids.append(sid_expanded)

        # Build GraphQL mutation
        sub_ids_str = ", ".join(f'"{s}"' for s in expanded_sub_ids)
        query = (
            '{ mutation { generateBatchShortLink('
            f'offerLink: "{offer_link}", '
            f'subIds: [{sub_ids_str}]) {{'
            "    shortLinks { subId shortLink }"
            "  }"
            "} }"
        )

        payload = json.dumps({"query": query})
        auth = _generate_auth(payload)

        try:
            with httpx.Client() as client:
                resp = client.post(
                    ENDPOINT,
                    headers=auth["headers"],
                    content=payload,
                    timeout=30.0,
                )
                resp.raise_for_status()
                data = resp.json()

            short_links = (
                data.get("data", {})
                .get("generateBatchShortLink", {})
                .get("shortLinks", [])
            )

            enriched_item = {**item, "shortLinks": short_links}
            for sl in short_links:
                logger.debug(
                    f"[shopee] short link: sub_id={sl['subId']} → {sl['shortLink']}"
                )

        except Exception as e:
            logger.warning(
                f"[shopee] generateBatchShortLink failed for item "
                f"{item.get('itemId')}: {e}"
            )
            enriched_item = {**item, "shortLinks": []}

        enriched.append(enriched_item)

    return enriched


def get_conversion_report(days: int = 30, limit: int = 100) -> list[dict]:
    """Pull conversion report from Shopee Affiliate API.

    Returns list of conversion records with conversionId, itemId,
    orderAmount, commission, status, subId, createTime.
    Requires SHOPEE_APP_ID + SHOPEE_APP_SECRET.
    """
    if not APP_ID or not APP_SECRET:
        logger.warning("[shopee] get_conversion_report: no API creds")
        return []

    query = (
        "{ conversionReport("
        f"limit: {limit}, "
        f"offset: 0, "
        f"dateRange: {{days: {days}}}"
        ") {"
        "    nodes {"
        "      conversionId"
        "      itemId"
        "      itemName"
        "      orderAmount"
        "      commission"
        "      status"
        "      subId"
        "      createTime"
        "    }"
        "    pageInfo { page limit hasNextPage }"
        "  }"
        "}"
    )

    payload = json.dumps({"query": query})
    auth = _generate_auth(payload)

    try:
        with httpx.Client() as client:
            resp = client.post(
                ENDPOINT,
                headers=auth["headers"],
                content=payload,
                timeout=30.0,
            )
            resp.raise_for_status()
            data = resp.json()

        nodes = (
            data.get("data", {})
            .get("conversionReport", {})
            .get("nodes", [])
        )
        logger.info(f"[shopee] fetched {len(nodes)} conversions (last {days} days)")
        return nodes

    except Exception as e:
        logger.error(f"[shopee] get_conversion_report error: {e}")
        return []
