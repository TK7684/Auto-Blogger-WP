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

    Args:
        keyword: Search keyword
        limit: Max number of results to return

    Returns:
        List of product dicts with fields: itemId, productName, price, etc.
    """
    if not APP_ID or not APP_SECRET:
        logger.warning("SHOPEE_APP_ID or SHOPEE_APP_SECRET not set, returning empty list")
        return []

    query = f"""
    {{
      productSearchV3(keyword: "{keyword}", limit: 100, page: 1) {{
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

        # Score by commission × sales, sort descending
        scored = [
            (p, (p.get("commission") or 0) * (p.get("sales") or 0))
            for p in products
        ]
        scored.sort(key=lambda x: x[1], reverse=True)
        return [p[0] for p in scored[:limit]]

    except Exception as e:
        logger.error(f"Error searching products: {e}")
        return []


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
