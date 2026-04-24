"""
Tests for Shopee Affiliate API client.
"""

import json
from unittest.mock import MagicMock, patch

import pytest

from src.clients.shopee import (
    _generate_auth,
    _extract_products,
    fetch_products,
    fetch_by_category,
    search_products,
)


class TestSignatureGeneration:
    """Test signature algorithm (based on shopeeApi.js:99-111)."""

    def test_signature_format(self):
        """
        Verify signature header format matches JS implementation.

        JS line 108:
            Authorization: `SHA256 Credential=${APP_ID}, Timestamp=${timestamp}, Signature=${signature}`
        """
        with patch("src.clients.shopee.APP_ID", "test_app_id"):
            with patch("src.clients.shopee.APP_SECRET", "test_secret"):
                with patch("src.clients.shopee.time.time", return_value=1234567890):
                    payload = '{"query": "test"}'
                    auth = _generate_auth(payload)

                    headers = auth["headers"]
                    assert "Authorization" in headers
                    assert headers["Authorization"].startswith(
                        "SHA256 Credential=test_app_id, Timestamp="
                    )
                    assert ", Signature=" in headers["Authorization"]
                    assert headers["Content-Type"] == "application/json"

    def test_signature_deterministic(self):
        """Same inputs should produce same signature."""
        with patch("src.clients.shopee.APP_ID", "app123"):
            with patch("src.clients.shopee.APP_SECRET", "secret456"):
                with patch("src.clients.shopee.time.time", return_value=9999):
                    payload = '{"query": "test"}'
                    auth1 = _generate_auth(payload)
                    auth2 = _generate_auth(payload)

                    assert auth1["signature"] == auth2["signature"]
                    assert auth1["timestamp"] == auth2["timestamp"]


class TestResponseParsing:
    """Test GraphQL response parsing."""

    def test_extract_products_valid_response(self):
        """Mock GraphQL response should extract products correctly."""
        mock_response = {
            "data": {
                "productOfferV2": {
                    "nodes": [
                        {
                            "itemId": "123",
                            "productName": "Test Product",
                            "price": 100.0,
                            "commissionRate": 5.0,
                            "commission": 5.0,
                            "sales": 1000,
                            "ratingStar": 4.5,
                            "imageUrl": "https://example.com/img.jpg",
                            "productLink": "https://shopee.co.th/product/123",
                            "offerLink": "https://shopee.co.th/offer/123",
                        }
                    ]
                }
            }
        }

        products = _extract_products(mock_response)

        assert len(products) == 1
        assert products[0]["itemId"] == "123"
        assert products[0]["productName"] == "Test Product"
        assert products[0]["price"] == 100.0
        assert products[0]["commission"] == 5.0
        assert products[0]["offerLink"] == "https://shopee.co.th/offer/123"

    def test_extract_products_multiple_items(self):
        """Extract multiple products from response."""
        mock_response = {
            "data": {
                "productOfferV2": {
                    "nodes": [
                        {
                            "itemId": f"id_{i}",
                            "productName": f"Product {i}",
                            "price": 100.0 * i,
                            "commissionRate": 5.0,
                            "commission": 5.0 * i,
                            "sales": 1000,
                            "ratingStar": 4.5,
                            "imageUrl": "https://example.com/img.jpg",
                            "productLink": f"https://shopee.co.th/{i}",
                            "offerLink": f"https://shopee.co.th/offer/{i}",
                        }
                        for i in range(1, 4)
                    ]
                }
            }
        }

        products = _extract_products(mock_response)
        assert len(products) == 3
        assert products[0]["itemId"] == "id_1"
        assert products[2]["itemId"] == "id_3"

    def test_extract_products_malformed_response(self):
        """Gracefully handle malformed response."""
        products = _extract_products({})
        assert products == []

        products = _extract_products({"data": None})
        assert products == []

        products = _extract_products({"data": {"productOfferV2": None}})
        assert products == []


class TestEmptyCredentials:
    """Test graceful degradation when credentials unset."""

    def test_fetch_products_no_creds(self):
        """No credentials should return empty list."""
        with patch("src.clients.shopee.APP_ID", ""):
            with patch("src.clients.shopee.APP_SECRET", ""):
                result = fetch_products()
                assert result == []

    def test_fetch_by_category_no_creds(self):
        """No credentials should return empty list."""
        with patch("src.clients.shopee.APP_ID", ""):
            with patch("src.clients.shopee.APP_SECRET", ""):
                result = fetch_by_category(category_id=123)
                assert result == []

    def test_search_products_no_creds(self):
        """No credentials should return empty list."""
        with patch("src.clients.shopee.APP_ID", ""):
            with patch("src.clients.shopee.APP_SECRET", ""):
                result = search_products(keyword="test")
                assert result == []


class TestFunctionalBehavior:
    """Test function logic with mocked HTTP."""

    @patch("src.clients.shopee.httpx.Client.post")
    def test_fetch_products_with_mock_response(self, mock_post):
        """Test fetch_products with mocked HTTP response."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "data": {
                "productOfferV2": {
                    "nodes": [
                        {
                            "itemId": "item_1",
                            "productName": "Mock Product",
                            "price": 500.0,
                            "commissionRate": 10.0,
                            "commission": 50.0,
                            "sales": 5000,
                            "ratingStar": 4.8,
                            "imageUrl": "https://example.com/img.jpg",
                            "productLink": "https://shopee.co.th/product/1",
                            "offerLink": "https://shopee.co.th/offer/1",
                        }
                    ]
                }
            }
        }
        mock_post.return_value = mock_response

        with patch("src.clients.shopee.APP_ID", "app123"):
            with patch("src.clients.shopee.APP_SECRET", "secret123"):
                products = fetch_products(limit=100, page=1)

                assert len(products) == 1
                assert products[0]["productName"] == "Mock Product"
                assert products[0]["commission"] == 50.0

    @patch("src.clients.shopee.httpx.Client.post")
    def test_fetch_by_category_filters_correctly(self, mock_post):
        """Test category filtering logic."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "data": {
                "productOfferV2": {
                    "nodes": [
                        {
                            "itemId": "item_1",
                            "productName": "Product in Cat 10",
                            "price": 100.0,
                            "commissionRate": 5.0,
                            "commission": 5.0,
                            "sales": 1000,
                            "ratingStar": 4.5,
                            "imageUrl": "https://example.com/img.jpg",
                            "productLink": "https://shopee.co.th/1",
                            "offerLink": "https://shopee.co.th/offer/1",
                            "productCatIds": [10, 20],
                        },
                        {
                            "itemId": "item_2",
                            "productName": "Product in Cat 30",
                            "price": 200.0,
                            "commissionRate": 5.0,
                            "commission": 10.0,
                            "sales": 2000,
                            "ratingStar": 4.5,
                            "imageUrl": "https://example.com/img.jpg",
                            "productLink": "https://shopee.co.th/2",
                            "offerLink": "https://shopee.co.th/offer/2",
                            "productCatIds": [30, 40],
                        },
                    ],
                    "pageInfo": {"page": 1, "limit": 100, "hasNextPage": False},
                }
            }
        }
        mock_post.return_value = mock_response
        mock_response.raise_for_status = MagicMock()

        with patch("src.clients.shopee.APP_ID", "app123"):
            with patch("src.clients.shopee.APP_SECRET", "secret123"):
                products = fetch_by_category(category_id=10)

                assert len(products) == 1
                assert products[0]["itemId"] == "item_1"

    @patch("src.clients.shopee.httpx.Client.post")
    def test_search_products_scores_by_commission_sales(self, mock_post):
        """Test that search results are scored by commission × sales."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "data": {
                "productOfferV2": {
                    "nodes": [
                        {
                            "itemId": "item_1",
                            "productName": "Product 1",
                            "commission": 10.0,  # score = 10 × 100 = 1000
                            "sales": 100,
                            "price": 100.0,
                            "commissionRate": 5.0,
                            "ratingStar": 4.5,
                            "imageUrl": "https://example.com/img.jpg",
                            "productLink": "https://shopee.co.th/1",
                            "offerLink": "https://shopee.co.th/offer/1",
                        },
                        {
                            "itemId": "item_2",
                            "productName": "Product 2",
                            "commission": 50.0,  # score = 50 × 1000 = 50000
                            "sales": 1000,
                            "price": 500.0,
                            "commissionRate": 5.0,
                            "ratingStar": 4.8,
                            "imageUrl": "https://example.com/img.jpg",
                            "productLink": "https://shopee.co.th/2",
                            "offerLink": "https://shopee.co.th/offer/2",
                        },
                        {
                            "itemId": "item_3",
                            "productName": "Product 3",
                            "commission": 5.0,  # score = 5 × 50 = 250
                            "sales": 50,
                            "price": 200.0,
                            "commissionRate": 5.0,
                            "ratingStar": 4.2,
                            "imageUrl": "https://example.com/img.jpg",
                            "productLink": "https://shopee.co.th/3",
                            "offerLink": "https://shopee.co.th/offer/3",
                        },
                    ]
                }
            }
        }
        mock_post.return_value = mock_response

        with patch("src.clients.shopee.APP_ID", "app123"):
            with patch("src.clients.shopee.APP_SECRET", "secret123"):
                products = search_products(keyword="test", limit=2)

                assert len(products) == 2
                # Item 2 should be first (highest score)
                assert products[0]["itemId"] == "item_2"
                # Item 1 should be second
                assert products[1]["itemId"] == "item_1"
