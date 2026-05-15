"""
Shopee Affiliate API Client for Python.

Ported from JavaScript: /home/tk578/ShopeeTH-aff/backup/shopeeApi.js

Env vars:
  SHOPEE_APP_ID: Affiliate dashboard Developer section
  SHOPEE_APP_SECRET: Affiliate dashboard Developer section
  SHOPEE_API_ENDPOINT (optional): defaults to https://open-api.affiliate.shopee.co.th/graphql
"""

import hashlib
import json
import logging
import os
import time
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
