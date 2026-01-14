"""
Trend Sources Module - Fetches trending topics from multiple sources.
Standardized version.
"""

import os
import logging
import requests
import random
from typing import Tuple, Optional

logger = logging.getLogger(__name__)

class TwitterTrendsFetcher:
    def __init__(self):
        self.bearer_token = os.environ.get("TWITTER_BEARER_TOKEN")
        self.woe_id = os.environ.get("TWITTER_WOE_ID", "23424977")

    def get_trending_topic(self) -> Optional[Tuple[str, str]]:
        if not self.bearer_token: return None
        try:
            url = f"https://api.twitter.com/1.1/trends/place.json?id={self.woe_id}"
            headers = {"Authorization": f"Bearer {self.bearer_token}"}
            response = requests.get(url, headers=headers, timeout=30)
            if response.status_code == 200:
                trends_data = response.json()
                if trends_data and len(trends_data) > 0:
                    trends = trends_data[0].get("trends", [])
                    if trends:
                        top = trends[0]
                        return top.get("name", ""), f"Trending with {top.get('tweet_volume', 'high')} volume"
        except Exception as e: logger.error(f"Twitter error: {e}")
        return None

class NewsAPIFetcher:
    def __init__(self):
        self.api_key = os.environ.get("NEWSAPI_KEY")

    def get_trending_topic(self) -> Optional[Tuple[str, str]]:
        if not self.api_key: return None
        try:
            url = "https://newsapi.org/v2/top-headlines"
            params = {"apiKey": self.api_key, "country": os.environ.get("NEWSAPI_COUNTRY", "us"), "pageSize": 1}
            response = requests.get(url, params=params, timeout=30)
            if response.status_code == 200:
                articles = response.json().get("articles", [])
                if articles:
                    top = articles[0]
                    return top.get("title", ""), top.get("description", top.get("content", ""))
        except Exception as e: logger.error(f"NewsAPI error: {e}")
        return None

class PromotionalFetcher:
    def __init__(self, json_file: str = "promotional_topics.json"):
        self.json_file = json_file
        
    def get_trending_topic(self) -> Optional[Tuple[str, str]]:
        import json
        try:
            if os.path.exists(self.json_file):
                with open(self.json_file, 'r', encoding='utf-8') as f:
                    topics = json.load(f)
                    if topics:
                        choice = random.choice(topics)
                        return choice['topic'], choice['context']
        except Exception as e: logger.error(f"Promotional error: {e}")
        return None

class TrendAggregator:
    def __init__(self):
        self.twitter_fetcher = TwitterTrendsFetcher()
        self.newsapi_fetcher = NewsAPIFetcher()
        self.promo_fetcher = PromotionalFetcher()

    def get_trending_topic(self) -> Tuple[Optional[str], Optional[str]]:
        logger.info("Fetching trending topic...")
        
        # 20% chance for promotional
        if random.random() < 0.20:
            result = self.promo_fetcher.get_trending_topic()
            if result: return result

        # Priority: Twitter -> NewsAPI -> Promotional Fallback -> Evergreen
        for fetcher in [self.twitter_fetcher, self.newsapi_fetcher, self.promo_fetcher]:
            result = fetcher.get_trending_topic()
            if result: return result

        evergreen = [
            ("The Future of AI in 2026", "How generative models are reshaping industries."),
            ("Sustainable Travel Tips", "How to explore the world with minimal impact.")
        ]
        return random.choice(evergreen)

# Singleton instance
_trend_aggregator = None

def get_hot_trend() -> Tuple[Optional[str], Optional[str]]:
    """
    Convenience function to get trending topic.
    """
    global _trend_aggregator
    if _trend_aggregator is None:
        _trend_aggregator = TrendAggregator()
    return _trend_aggregator.get_trending_topic()

if __name__ == "__main__":
    t, d = get_hot_trend()
    print(f"Trend: {t}")
