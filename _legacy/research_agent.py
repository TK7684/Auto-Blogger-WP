"""
Research Agent Module - Competitive Intelligence and Content Ideation
Fully restored and optimized for Gemini 3 reasoning models.
"""

import os
import json
import logging
import requests
import feedparser
import base64
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from collections import Counter
from dataclasses import dataclass, field
from dotenv import load_dotenv
from google import genai
from google.genai import types

logger = logging.getLogger(__name__)

@dataclass
class Article:
    title: str
    url: str
    summary: str
    published: Optional[str] = None
    source: str = ""
    tags: List[str] = field(default_factory=list)

@dataclass
class ContentGap:
    topic: str
    description: str
    priority: str
    trend_score: float = 0.0

@dataclass
class ArticleIdea:
    title: str
    outline: str
    rationale: str
    target_keywords: List[str]
    suggested_length: int
    priority: str
    competitive_advantage: str
    estimated_traffic_potential: str
    content_type: str

def clean_json_response(text: str) -> str:
    """Remove markdown code blocks from JSON response."""
    if '```json' in text:
        text = text.split('```json')[1].split('```')[0]
    elif '```' in text:
        text = text.split('```')[1].split('```')[0]
    return text.strip()

class NewsletterFetcher:
    def __init__(self):
        self.sources = self._load_sources()
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })

    def _load_sources(self) -> Dict[str, List[str]]:
        sources_file = os.environ.get("RESEARCH_SOURCES_FILE", "research_sources.json")
        if os.path.exists(sources_file):
            try:
                with open(sources_file, 'r') as f: return json.load(f)
            except Exception: pass
        return {"rss_feeds": [], "newsletter_urls": [], "competitor_blogs": []}

    def fetch_rss_feeds(self, days_back: int = 7) -> List[Article]:
        articles = []
        cutoff = datetime.now() - timedelta(days=days_back)
        for url in self.sources.get("rss_feeds", []):
            try:
                feed = feedparser.parse(url)
                for entry in feed.entries:
                    # feedparser entries support both dict and attribute access
                    parsed = None
                    if hasattr(entry, 'published_parsed'):
                        parsed = entry.published_parsed
                    elif isinstance(entry, dict):
                        parsed = entry.get('published_parsed')
                        
                    if parsed and hasattr(parsed, '__getitem__'):
                        try:
                            if len(parsed) >= 3:
                                dt = datetime(parsed[0], parsed[1], parsed[2], 
                                             parsed[3] if len(parsed) > 3 else 0,
                                             parsed[4] if len(parsed) > 4 else 0,
                                             parsed[5] if len(parsed) > 5 else 0)
                                if dt < cutoff: continue
                        except (ValueError, TypeError, IndexError):
                            pass
                    
                    articles.append(Article(
                        title=entry.get('title', 'Untitled'),
                        url=entry.get('link', ''),
                        summary=entry.get('summary', '')[:500],
                        published=entry.get('published'),
                        source=feed.feed.get('title', url)
                    ))
            except Exception as e: logger.error(f"RSS error {url}: {e}")
        return articles

class WordPressContentAnalyzer:
    def __init__(self, wp_url: str, wp_user: str, wp_app_password: str):
        self.wp_url = wp_url.rstrip('/')
        self.wp_user = wp_user
        self.wp_app_password = wp_app_password
        self.session = requests.Session()

    def fetch_recent_posts(self, limit: int = 50) -> List[Article]:
        url = f"{self.wp_url}/wp-json/wp/v2/posts"
        credentials = f"{self.wp_user}:{self.wp_app_password}"
        token = base64.b64encode(credentials.encode()).decode('utf-8')
        headers = {"Authorization": f"Basic {token}"}
        try:
            resp = self.session.get(url, headers=headers, params={"per_page": limit}, timeout=20)
            if resp.status_code == 200:
                posts = resp.json()
                return [Article(title=p['title']['rendered'], url=p['link'], summary="") for p in posts]
        except Exception as e: logger.error(f"WP fetch error: {e}")
        return []

    def analyze_content_themes(self, articles: List[Article]) -> Dict:
        words = []
        for a in articles:
            words.extend([w.lower() for w in a.title.split() if len(w) > 3])
        counts = Counter(words)
        return {'total_articles': len(articles), 'top_keywords': dict(counts.most_common(10))}

class TrendAnalyzer:
    def __init__(self, api_key: str):
        self.client = genai.Client(api_key=api_key) if api_key else None

    def identify_trending_topics(self, comp: List[Article], own: List[Article]) -> List[ContentGap]:
        if not self.client: return []
        prompt = f"Analyze gaps between competitors and our site. Competitors: " + ", ".join([a.title for a in comp[:20]])
        try:
            resp = self.client.models.generate_content(
                model="gemini-2.0-flash-exp",
                contents=prompt,
                config=types.GenerateContentConfig(
                    thinking_config=types.ThinkingConfig(thinking_level="low"),
                    response_mime_type="application/json"
                )
            )
            data = json.loads(clean_json_response(resp.text))
            if isinstance(data, list):
                return [ContentGap(**g) for g in data]
            return []
        except Exception as e:
            logger.error(f"Trend error: {e}")
            return []

class ArticleIdeationEngine:
    def __init__(self, api_key: str):
        self.client = genai.Client(api_key=api_key) if api_key else None

    def generate_article_ideas(self, gaps: List[ContentGap], comp: List[Article], own: List[Article], num_ideas: int = 5) -> List[ArticleIdea]:
        if not self.client: return []
        prompt = f"Generate {num_ideas} article ideas based on these gaps: " + str(gaps)
        try:
            resp = self.client.models.generate_content(
                model="gemini-2.5-pro",
                contents=prompt,
                config=types.GenerateContentConfig(
                    thinking_config=types.ThinkingConfig(thinking_level="high"),
                    response_mime_type="application/json"
                )
            )
            data = json.loads(clean_json_response(resp.text))
            if isinstance(data, list):
                return [ArticleIdea(**i) for i in data]
            return []
        except Exception as e:
            logger.error(f"Ideation error: {e}")
            return []

    def rank_ideas(self, ideas: List[ArticleIdea], context: List[Article]) -> List[ArticleIdea]:
        priority_map = {"high": 3, "medium": 2, "low": 1}
        return sorted(ideas, key=lambda x: priority_map.get(x.priority.lower(), 0), reverse=True)

class ResearchAgent:
    def __init__(self, wp_url: str, wp_user: str, wp_app_password: str, api_key: str):
        self.newsletter_fetcher = NewsletterFetcher()
        self.wp_analyzer = WordPressContentAnalyzer(wp_url, wp_user, wp_app_password)
        self.trend_analyzer = TrendAnalyzer(api_key)
        self.ideation_engine = ArticleIdeationEngine(api_key)

    def run_research(self, days_back: int = 7, num_ideas: int = 15) -> Dict[str, Any]:
        comp_articles = self.newsletter_fetcher.fetch_rss_feeds(days_back)
        logger.info(f"🔍 Found {len(comp_articles)} competitor articles.")
        
        your_articles = self.wp_analyzer.fetch_recent_posts()
        logger.info(f"🏠 Found {len(your_articles)} of your own articles.")
        
        themes = self.wp_analyzer.analyze_content_themes(your_articles)
        gaps = self.trend_analyzer.identify_trending_topics(comp_articles, your_articles)
        logger.info(f"🕳️ Identified {len(gaps)} content gaps.")
        
        ideas = self.ideation_engine.generate_article_ideas(gaps, comp_articles, your_articles, num_ideas)
        logger.info(f"💡 Generated {len(ideas)} article ideas.")
        
        ranked = self.ideation_engine.rank_ideas(ideas, your_articles)
        
        return {
            'timestamp': datetime.now().isoformat(),
            'competitor_articles': comp_articles,
            'your_articles': your_articles,
            'content_themes': themes,
            'content_gaps': gaps,
            'article_ideas': ranked
        }

    def save_results(self, results: Dict, output_file: str = "research_results.json"):
        def d_to_d(o):
            if isinstance(o, list): return [d_to_d(i) for i in o]
            if hasattr(o, '__dict__'): return o.__dict__
            return str(o)
        
        serializable = {k: d_to_d(v) for k, v in results.items()}
        with open(output_file, 'w') as f:
            json.dump(serializable, f, indent=4)

def create_default_sources_file(path: str):
    with open(path, 'w') as f:
        json.dump({"rss_feeds": [], "newsletter_urls": [], "competitor_blogs": []}, f)
