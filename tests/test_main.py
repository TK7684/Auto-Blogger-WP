"""
Tests for Auto-Blogging WordPress application.
Updated to match Pydantic structured outputs in Gemini 3.
"""

import unittest
from unittest.mock import patch, MagicMock
import json
import os
from src.main import initialize_system, run_content_generation, get_hot_trend
from src.main_schemas import SEOArticleMetadata


class TestTrendSources(unittest.TestCase):
    """Test trend source fetching."""

    @patch('src.trend_sources.TwitterTrendsFetcher.get_trending_topic')
    def test_get_hot_trend_twitter_success(self, mock_twitter):
        mock_twitter.return_value = ("Trending Topic", "Context for topic")
        # We need to mock the other fetchers or ensure they are skipped if twitter returns
        # The aggregator tries Twitter first.
        # But get_hot_trend uses a singleton. We might need to reset it or patch TrendAggregator directly.
        
        with patch('src.trend_sources._trend_aggregator', None): # Reset singleton
             # We also need to mock requests or the internal fetchers
             with patch('src.trend_sources.TrendAggregator') as MockAggregator:
                 instance = MockAggregator.return_value
                 instance.get_trending_topic.return_value = ("Trending Topic", "Context", "en")
                 
                 topic, context, lang = get_hot_trend()
                 self.assertEqual(topic, "Trending Topic")
                 self.assertEqual(context, "Context")
                 self.assertEqual(lang, "en")


class TestContentGeneration(unittest.TestCase):
    """Test AI content generation with new GenAI SDK and Pydantic models."""

    @patch('src.clients.gemini.genai.Client')
    def test_generate_content_gemini_daily(self, mock_client_class):
        # Create mock response with structured JSON
        mock_data = {
            "content": "<h1>Test Content</h1>",
            "seo_title": "Test SEO Title",
            "meta_description": "Test meta description including keywords",
            "focus_keyword": "keywords",
            "slug": "test-slug",
            "excerpt": "Test excerpt",
            "suggested_categories": ["News"],
            "suggested_tags": ["AI"],
            "in_article_image_prompts": []
        }
        
        mock_response = MagicMock()
        mock_response.text = json.dumps(mock_data)

        mock_client = MagicMock()
        # Mock the models.generate_content call
        mock_client.models.generate_content.return_value = mock_response
        mock_client_class.return_value = mock_client

        # We test the client wrapper directly or the main function?
        # Let's test src.clients.gemini.GeminiClient since main.py uses it.
        from src.clients.gemini import GeminiClient
        client = GeminiClient("test_key")
        result = client.generate_structured_output("model", "prompt", mock_data)
        
        self.assertIsNotNone(result)
        self.assertEqual(result.text, json.dumps(mock_data))


if __name__ == '__main__':
    unittest.main()
