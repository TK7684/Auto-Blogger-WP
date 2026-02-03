"""
Comprehensive tests for the Research Agent module.

Tests cover:
- Newsletter fetching from RSS feeds
- WordPress content analysis
- Article ideation and ranking
- Full research workflow
"""

import unittest
from unittest.mock import patch, MagicMock, Mock
from datetime import datetime, timedelta
import json
import os
import tempfile

# Import classes from research_agent
from src.research_agent import (
    Article,
    ContentGap,
    ArticleIdea,
    NewsletterFetcher,
    ResearchAgent,
)
from src.clients.wordpress import WordPressClient
from src.clients.gemini import GeminiClient


class TestArticleDataclass(unittest.TestCase):
    """Test the Article dataclass."""

    def test_article_creation(self):
        article = Article(
            title="Test Article",
            url="https://example.com/test",
            summary="This is a test article summary",
            source="Test Source",
            tags=["tag1", "tag2"]
        )
        self.assertEqual(article.title, "Test Article")
        self.assertEqual(article.tags, ["tag1", "tag2"])

    def test_article_defaults(self):
        article = Article(
            title="Test",
            url="https://example.com",
            summary="Summary"
        )
        self.assertIsNone(article.published)
        self.assertEqual(article.source, "")
        self.assertEqual(article.tags, [])


class TestContentGapPydantic(unittest.TestCase):
    """Test the ContentGap Pydantic model."""

    def test_content_gap_creation(self):
        gap = ContentGap(
            topic="AI in Healthcare",
            description="Growing trend of AI applications",
            priority="high",
            trend_score=0.85
        )
        self.assertEqual(gap.topic, "AI in Healthcare")
        self.assertEqual(gap.priority, "high")
        self.assertEqual(gap.trend_score, 0.85)


class TestArticleIdeaPydantic(unittest.TestCase):
    """Test the ArticleIdea Pydantic model."""

    def test_article_idea_creation(self):
        idea = ArticleIdea(
            title="Complete Guide to AI",
            outline="Introduction, applications, future",
            rationale="High demand topic",
            target_keywords=["AI", "machine learning"],
            suggested_length=1500,
            priority="high",
            competitive_advantage="Comprehensive coverage",
            estimated_traffic_potential="high",
            content_type="guide"
        )
        self.assertEqual(idea.title, "Complete Guide to AI")
        self.assertEqual(idea.suggested_length, 1500)
        self.assertEqual(idea.content_type, "guide")


class TestNewsletterFetcher(unittest.TestCase):
    """Test newsletter fetching functionality."""

    def setUp(self):
        self.fetcher = NewsletterFetcher()

    def test_fetch_rss_feeds_no_feeds(self):
        """Test that fetcher handles empty sources correctly."""
        self.fetcher.sources = {"rss_feeds": []}
        articles = self.fetcher.fetch_rss_feeds(days_back=7)
        self.assertEqual(len(articles), 0)

    @patch('src.research_agent.feedparser.parse')
    def test_fetch_rss_returns_articles(self, mock_parse):
        """Test that fetcher returns articles from RSS feeds."""
        mock_feed = MagicMock()
        mock_feed.feed.get.return_value = "Test Source"
        mock_entry = MagicMock()
        mock_entry.get.side_effect = lambda k, d=None: {
            'title': 'Test Article',
            'link': 'https://example.com/test',
            'summary': 'Test summary',
            'published': datetime.now().isoformat()
        }.get(k, d)
        mock_feed.entries = [mock_entry]
        mock_parse.return_value = mock_feed

        self.fetcher.sources = {"rss_feeds": ["https://example.com/feed"]}
        articles = self.fetcher.fetch_rss_feeds(days_back=7)

        # Should have returned articles
        self.assertIsInstance(articles, list)


class TestResearchAgent(unittest.TestCase):
    """Test the main research agent orchestrator."""

    def setUp(self):
        """Set up mock clients for testing."""
        self.mock_wp_client = MagicMock(spec=WordPressClient)
        self.mock_gemini_client = MagicMock(spec=GeminiClient)
        
        self.agent = ResearchAgent(
            wp_client=self.mock_wp_client,
            gemini_client=self.mock_gemini_client
        )

    def test_run_research_full_workflow(self):
        """Test full research workflow with mocked components."""
        # Mock WordPress posts response
        self.mock_wp_client.fetch_posts.return_value = [
            {'title': {'rendered': 'Your Article'}, 'link': 'https://yoursite.com/1'}
        ]

        # Mock Gemini responses for gaps and ideas
        mock_gaps_response = MagicMock()
        mock_gaps_response.text = json.dumps({
            'gaps': [
                {
                    'topic': 'AI Healthcare',
                    'description': 'Gap in AI coverage',
                    'priority': 'high',
                    'trend_score': 0.85
                }
            ]
        })
        
        mock_ideas_response = MagicMock()
        mock_ideas_response.text = json.dumps({
            'article_ideas': [
                {
                    'title': 'AI Healthcare Guide',
                    'outline': 'Comprehensive guide',
                    'rationale': 'High demand',
                    'target_keywords': ['AI', 'healthcare'],
                    'suggested_length': 1500,
                    'priority': 'high',
                    'competitive_advantage': 'Comprehensive',
                    'estimated_traffic_potential': 'high',
                    'content_type': 'guide'
                }
            ]
        })
        
        self.mock_gemini_client.generate_structured_output.side_effect = [
            mock_gaps_response,  # First call for gaps
            mock_ideas_response  # Second call for ideas
        ]

        # Mock newsletter fetcher
        with patch.object(self.agent.newsletter_fetcher, 'fetch_rss_feeds', return_value=[]):
            results = self.agent.run_research(days_back=7, num_ideas=5)

        # Verify results structure
        self.assertIn('timestamp', results)
        self.assertIn('competitor_articles', results)
        self.assertIn('your_articles', results)
        self.assertIn('content_gaps', results)
        self.assertIn('article_ideas', results)

    def test_research_agent_handles_empty_data(self):
        """Test that agent handles empty data gracefully."""
        self.mock_wp_client.fetch_posts.return_value = []
        self.mock_gemini_client.generate_structured_output.return_value = None

        with patch.object(self.agent.newsletter_fetcher, 'fetch_rss_feeds', return_value=[]):
            results = self.agent.run_research(days_back=7, num_ideas=5)

        self.assertEqual(len(results['competitor_articles']), 0)
        self.assertEqual(len(results['your_articles']), 0)


if __name__ == '__main__':
    unittest.main()
