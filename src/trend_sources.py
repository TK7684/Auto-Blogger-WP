"""
Trend Sources Module - Fetches trending topics from multiple sources.
Standardized version with Multi-language support.
"""

import os
import logging
import requests
import random
import json
from typing import Tuple, Optional, Dict, List
from pathlib import Path

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
    def __init__(self, json_file: str = "src/topics.json"):
        self.json_file = json_file
        # Fallback to root if src not found (handling different execution contexts)
        if not os.path.exists(self.json_file) and os.path.exists("topics.json"):
             self.json_file = "topics.json"
        
    def get_trending_topic(self, target_lang: str = "en") -> Optional[Tuple[str, str]]:
        try:
            if os.path.exists(self.json_file):
                with open(self.json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    
                    # Find topics for target language
                    lang_group = next((item for item in data if item["language"] == target_lang), None)
                    
                    # Fallback to English if target not found
                    if not lang_group:
                        lang_group = next((item for item in data if item["language"] == "en"), None)
                    
                    if lang_group and lang_group.get("topics"):
                        choice = random.choice(lang_group["topics"])
                        return choice['topic'], choice['context']
        except Exception as e: logger.error(f"Promotional error: {e}")
        return None

class TrendAggregator:
    def __init__(self):
        self.twitter_fetcher = TwitterTrendsFetcher()
        self.newsapi_fetcher = NewsAPIFetcher()
        self.promo_fetcher = PromotionalFetcher()

    def get_trending_topic(self) -> Tuple[Optional[str], Optional[str], str]:
        """
        Returns: (Topic, Context, Language)
        """
        logger.info("Fetching trending topic...")
        
        # Randomly select language (50% Chance Thai vs English if you want mix)
        # Or bias towards one. Let's do 50/50 for now.
        target_gen_lang = "th" if random.random() < 0.5 else "en"
        logger.info(f"Targeting Language: {target_gen_lang}")
        
        # 40% chance for promotional (K9 topics) - increased from 20%
        if random.random() < 0.40:
            result = self.promo_fetcher.get_trending_topic(target_gen_lang)
            if result: 
                return result[0], result[1], target_gen_lang

        # If English, we can try Twitter/NewsAPI (which are mostly English/Global)
        if target_gen_lang == "en":
            for fetcher in [self.twitter_fetcher, self.newsapi_fetcher]:
                result = fetcher.get_trending_topic()
                if result: return result[0], result[1], "en"
        
        # Fallback to promotional if external sources fail even if en
        result = self.promo_fetcher.get_trending_topic(target_gen_lang)
        if result: return result[0], result[1], target_gen_lang

        # Ultimate fallback
        evergreen = [
            ("The Future of AI in 2026", "How generative models are reshaping industries."),
            ("Sustainable Travel Tips", "How to explore the world with minimal impact.")
        ]
        topic, ctx = random.choice(evergreen)
        return topic, ctx, "en"

# Singleton instance
_trend_aggregator = None

def get_hot_trend() -> Tuple[Optional[str], Optional[str], str]:
    """
    Convenience function to get trending topic.
    Returns: (Topic, Context, Language)
    """
    global _trend_aggregator
    if _trend_aggregator is None:
        _trend_aggregator = TrendAggregator()
    return _trend_aggregator.get_trending_topic()

if __name__ == "__main__":
    t, d, l = get_hot_trend()
    print(f"Trend: {t} [{l}]")
