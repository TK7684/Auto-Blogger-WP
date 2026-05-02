"""
Tests for the curated Shopee product fallback.

Verifies that data/shopee_products.json is loadable, search returns N items,
and that affiliate_inserter._try_fetch_products falls back to the cache when
the live Shopee API returns []. Locks in the products>0 guarantee that fixed
the 2026-05-02 zero-products bug.
"""
from unittest.mock import patch

from src.clients import shopee_curated
from src.affiliate_inserter import _try_fetch_products


class TestCuratedCache:
    def test_cache_loads_with_real_data(self):
        """Cache file ships with the repo and contains products."""
        size = shopee_curated.cache_size()
        assert size > 0, "data/shopee_products.json must contain products"
        assert size >= 100, "expected at least 100 curated products"

    def test_search_returns_requested_limit(self):
        """search(limit=N) returns exactly N items when cache is non-empty."""
        results = shopee_curated.search("camping chair", limit=3)
        assert len(results) == 3
        for p in results:
            assert p.get("offerLink", "").startswith("http"), \
                "every product must have a real offerLink"
            assert p.get("productName"), "every product must have a name"

    def test_search_handles_zero_limit(self):
        """limit=0 returns []."""
        assert shopee_curated.search("anything", limit=0) == []

    def test_search_returns_top_by_value_when_no_match(self):
        """A nonsense keyword still returns N products (top-by-value fallback)."""
        results = shopee_curated.search("xyznonsensequery42", limit=3)
        assert len(results) == 3, \
            "fallback must always return `limit` products to keep cards rendering"


class TestAffiliateInserterFallback:
    def test_fallback_kicks_in_when_live_api_empty(self):
        """_try_fetch_products uses curated cache when live API returns []."""
        with patch("src.clients.shopee.search_products", return_value=[]):
            products = _try_fetch_products("camping chair", limit=3)
            assert len(products) == 3, \
                "must return curated products when live API is empty"

    def test_live_api_results_take_priority(self):
        """When live API returns products, curated cache is bypassed."""
        live_products = [{
            "itemId": "live_1",
            "productName": "Live API product",
            "offerLink": "https://s.shopee.co.th/live_1",
            "price": 100,
            "sales": 10,
            "ratingStar": 5.0,
            "imageUrl": "https://example.com/img.jpg",
        }]
        with patch("src.clients.shopee.search_products", return_value=live_products):
            products = _try_fetch_products("anything", limit=3)
            assert products == live_products

    def test_fallback_handles_live_api_exception(self):
        """A live API exception (network/auth/etc) still triggers the fallback."""
        with patch("src.clients.shopee.search_products", side_effect=RuntimeError("boom")):
            products = _try_fetch_products("camping", limit=3)
            assert len(products) == 3, \
                "must fall back to curated cache when live API raises"
