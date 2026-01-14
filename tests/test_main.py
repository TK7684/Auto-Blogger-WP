"""
Tests for Auto-Blogging WordPress application.
Updated to match Pydantic structured outputs in Gemini 3.
"""

import unittest
from unittest.mock import patch, MagicMock
import json
import main
from trend_sources import get_hot_trend


class TestTrendSources(unittest.TestCase):
    """Test trend source fetching."""

    @patch('trend_sources.TwitterTrendsFetcher.get_trending_topic')
    def test_get_hot_trend_twitter_success(self, mock_twitter):
        mock_twitter.return_value = ("Trending Topic", "Context for topic")
        topic, context = get_hot_trend()
        self.assertEqual(topic, "Trending Topic")
        self.assertEqual(context, "Context for topic")


class TestContentGeneration(unittest.TestCase):
    """Test AI content generation with new GenAI SDK and Pydantic models."""

    @patch('main.genai.Client')
    def test_generate_content_gemini_daily(self, mock_client_class):
        # Create mock response with structured JSON
        mock_data = {
            "content": "<h1>Test Content</h1>",
            "seo_title": "Test SEO Title",
            "meta_description": "Test meta description including keywords",
            "focus_keyword": "keywords",
            "excerpt": "Test excerpt",
            "suggested_categories": ["News"],
            "suggested_tags": ["AI"]
        }
        
        mock_response = MagicMock()
        mock_response.text = json.dumps(mock_data)

        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = mock_response
        mock_client_class.return_value = mock_client

        with patch('main.GEMINI_API_KEY', 'test_key'):
            main._genai_client = None
            result = main.generate_content_gemini("Topic", "Context", mode="daily")

            self.assertIsInstance(result, dict)
            self.assertEqual(result['seo_title'], 'Test SEO Title')
            self.assertEqual(result['focus_keyword'], 'keywords')
            self.assertIn('<h1>', result['content'])

    @patch('main.genai.Client')
    def test_generate_content_gemini_failure(self, mock_client_class):
        mock_client = MagicMock()
        mock_client.models.generate_content.side_effect = Exception("API Error")
        mock_client_class.return_value = mock_client

        with patch('main.GEMINI_API_KEY', 'test_key'):
            main._genai_client = None
            result = main.generate_content_gemini("Topic", "Context")

            self.assertIn('Failed', result['content'])


class TestWordPressPublishing(unittest.TestCase):
    """Test WordPress API publishing."""

    @patch('main.requests.post')
    def test_post_to_wp_success(self, mock_post):
        mock_response = MagicMock()
        mock_response.status_code = 201
        mock_response.json.return_value = {'id': 123}
        mock_post.return_value = mock_response

        with patch('main.WP_URL', 'http://test.site'), \
             patch('main.WP_USER', 'user'), \
             patch('main.WP_APP_PASSWORD', 'pass'):

            post_id = main.post_to_wp(
                "Title", "Content",
                {'seo_title': 'Title', 'meta_description': 'Desc', 'focus_keyword': 'keyword'},
                [1], [1]
            )

            self.assertEqual(post_id, 123)


if __name__ == '__main__':
    unittest.main()
