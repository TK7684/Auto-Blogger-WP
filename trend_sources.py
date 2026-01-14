"""
Trend Sources Module - Fetches trending topics from multiple sources.

Supports:
- Twitter/X API v2 for trending topics
- NewsAPI as fallback for current news trends
- Google Trends RSS (legacy, may not work)
"""

import os
import logging
import requests
from typing import Tuple, Optional

logger = logging.getLogger(__name__)


class TwitterTrendsFetcher:
    """Fetch trending topics from Twitter/X API v2."""

    def __init__(self):
        self.bearer_token = os.environ.get("TWITTER_BEARER_TOKEN")
        self.woe_id = os.environ.get("TWITTER_WOE_ID", "23424977")  # Default: US

        if not self.bearer_token:
            logger.warning("TWITTER_BEARER_TOKEN not configured. Twitter trends unavailable.")

    def get_trending_topic(self) -> Optional[Tuple[str, str]]:
        """
        Get the top trending topic from Twitter/X.

        Returns:
            Tuple of (title, description) or None if failed
        """
        if not self.bearer_token:
            return None

        try:
            url = f"https://api.twitter.com/1.1/trends/place.json?id={self.woe_id}"
            headers = {
                "Authorization": f"Bearer {self.bearer_token}"
            }

            response = requests.get(url, headers=headers, timeout=30)

            if response.status_code == 200:
                trends_data = response.json()
                if trends_data and len(trends_data) > 0:
                    trends = trends_data[0].get("trends", [])
                    if trends:
                        top_trend = trends[0]
                        title = top_trend.get("name", "")
                        # Use tweet volume if available, otherwise generate description
                        description = f"Trending with {top_trend.get('tweet_volume', 'high')} volume"
                        if not top_trend.get("tweet_volume"):
                            description = top_trend.get("url", "")
                        logger.info(f"Twitter trend found: {title}")
                        return title, description
            else:
                logger.warning(f"Twitter API returned status {response.status_code}")

        except Exception as e:
            logger.error(f"Error fetching Twitter trends: {e}")

        return None


class NewsAPIFetcher:
    """Fetch trending news from NewsAPI as fallback."""

    def __init__(self):
        self.api_key = os.environ.get("NEWSAPI_KEY")

        if not self.api_key:
            logger.warning("NEWSAPI_KEY not configured. NewsAPI unavailable.")

    def get_trending_topic(self) -> Optional[Tuple[str, str]]:
        """
        Get the top trending news from NewsAPI.

        Returns:
            Tuple of (title, description) or None if failed
        """
        if not self.api_key:
            return None

        try:
            # Get top headlines
            url = "https://newsapi.org/v2/top-headlines"
            params = {
                "apiKey": self.api_key,
                "country": os.environ.get("NEWSAPI_COUNTRY", "us"),
                "pageSize": 1
            }

            response = requests.get(url, params=params, timeout=30)

            if response.status_code == 200:
                data = response.json()
                articles = data.get("articles", [])
                if articles:
                    top_article = articles[0]
                    title = top_article.get("title", "")
                    description = top_article.get("description", top_article.get("content", ""))
                    logger.info(f"NewsAPI trend found: {title}")
                    return title, description
            else:
                logger.warning(f"NewsAPI returned status {response.status_code}")

        except Exception as e:
            logger.error(f"Error fetching NewsAPI trends: {e}")

        return None


class GoogleTrendsFetcher:
    """Legacy Google Trends RSS fetcher (may not work)."""

    def __init__(self):
        self.geo = os.environ.get("TRENDS_GEO", "US")

    def get_trending_topic(self) -> Optional[Tuple[str, str]]:
        """
        Get trending topic from Google Trends RSS.

        Returns:
            Tuple of (title, description) or None if failed
        """
        try:
            import feedparser

            rss_url = f"https://trends.google.com/trends/trendingsearches/daily/rss?geo={self.geo}"
            logger.info(f"Trying Google Trends RSS for geo={self.geo}...")

            feed = feedparser.parse(rss_url)

            if feed.entries:
                item = feed.entries[0]
                title = item.title
                description = item.description
                logger.info(f"Google Trends found: {title}")
                return title, description
            else:
                logger.warning("Google Trends RSS returned no entries")

        except Exception as e:
            logger.error(f"Error fetching Google Trends: {e}")

        return None


class TrendAggregator:
    """
    Aggregates multiple trend sources with fallback logic.

    Priority order:
    1. Twitter/X API
    2. NewsAPI (fallback)
    3. Google Trends RSS (legacy fallback)
    """

    def __init__(self):
        self.twitter_fetcher = TwitterTrendsFetcher()
        self.newsapi_fetcher = NewsAPIFetcher()
        self.google_fetcher = GoogleTrendsFetcher()

    def get_trending_topic(self) -> Tuple[Optional[str], Optional[str]]:
        """
        Get trending topic from first available source.

        Returns:
            Tuple of (title, description) or (None, None) if all fail
        """
        logger.info("=" * 70)
        logger.info("FETCHING TRENDING TOPIC")
        logger.info("=" * 70)

        # Try Twitter first
        logger.info("Attempting Twitter/X API...")
        result = self.twitter_fetcher.get_trending_topic()
        if result:
            return result

        # Fallback to NewsAPI
        logger.info("Twitter unavailable. Trying NewsAPI...")
        result = self.newsapi_fetcher.get_trending_topic()
        if result:
            return result

        # Last resort: Google Trends
        logger.info("NewsAPI unavailable. Trying Google Trends RSS...")
        result = self.google_fetcher.get_trending_topic()
        if result:
            return result

        # Final fallback: Evergreen topics
        logger.info("All trend sources failed. Using evergreen fallback.")
        evergreen_topics = [
            ("The Future of Artificial Intelligence", "How AI is shaping our world in 2026 and beyond."),
            ("Sustainable living in 2026", "Practical tips for a greener lifestyle this year."),
            ("Remote work productivity hacks", "Mastering your home office workflow for maximum efficiency."),
            ("The impact of quantum computing", "Exploring the next frontier of computational power.")
        ]
        import random
        return random.choice(evergreen_topics)


# Singleton instance
_trend_aggregator = None


def get_hot_trend() -> Tuple[Optional[str], Optional[str]]:
    """
    Convenience function to get trending topic.

    Returns:
        Tuple of (title, description) or (None, None) if all fail
    """
    global _trend_aggregator

    if _trend_aggregator is None:
        _trend_aggregator = TrendAggregator()

    return _trend_aggregator.get_trending_topic()


if __name__ == "__main__":
    # Test the trend fetcher
    from dotenv import load_dotenv
    load_dotenv()

    logging.basicConfig(level=logging.INFO)

    title, description = get_hot_trend()

    if title:
        print(f"\nTop Trending Topic:")
        print(f"Title: {title}")
        print(f"Description: {description}")
    else:
        print("\nNo trending topic found from any source.")
