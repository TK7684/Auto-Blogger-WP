"""
Trend sources — what people want to read right now.

Returns `(topic, context, language, article_type)` tuples where:
    article_type in {"trending", "research"}

Fetchers by cadence:
    daily   → Google Trends RSS (realtime), Reddit /r/popular, Twitter
              + topics.json (promotional / evergreen)
    weekly  → Hacker News (top/week), Dev.to (top/week), NewsAPI (everything sorted)
    monthly → arXiv recent (research), PubMed trending (research),
              HN all-time peaks for the month

Env:
    DAILY_PROMO_PCT       float 0..1 (default 0.30)   share of promotional topics on daily
    WEEKLY_RESEARCH_PCT   float 0..1 (default 0.60)   share classified research on weekly
    MONTHLY_RESEARCH_PCT  float 0..1 (default 0.85)   share classified research on monthly
    TRENDS_GEO            ISO country (default US)
    REDDIT_UA             custom UA (default Auto-Blogger-WP/1.0)
    NEWSAPI_KEY, TWITTER_BEARER_TOKEN   optional
"""

from __future__ import annotations

import json
import logging
import os
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional, Tuple
from xml.etree import ElementTree as ET

import requests

logger = logging.getLogger(__name__)

# ---- Types ----------------------------------------------------------------

Trend = Tuple[str, str, str, str]  # (topic, context, lang, article_type)


@dataclass
class _Item:
    topic: str
    context: str
    lang: str = "en"
    article_type: str = "trending"  # trending | research
    _subreddit: str = ""  # for Reddit source filtering

    def as_tuple(self) -> Trend:
        return (self.topic, self.context, self.lang, self.article_type)


UA = {"User-Agent": os.environ.get("REDDIT_UA", "Auto-Blogger-WP/1.0 (+https://pedpro.online)")}
TIMEOUT = 15


# ---- Daily (hot right now) -----------------------------------------------

def _fetch_google_trends_realtime(geo: str = "US") -> List[_Item]:
    """Google Trends realtime RSS. No API key required."""
    url = f"https://trends.google.com/trends/trendingsearches/daily/rss?geo={geo}"
    try:
        r = requests.get(url, headers=UA, timeout=TIMEOUT)
        if r.status_code != 200:
            return []
        root = ET.fromstring(r.content)
        items: List[_Item] = []
        for item in root.iter("item"):
            title_el = item.find("title")
            desc_el = item.find("description")
            if title_el is None or not title_el.text:
                continue
            desc = (desc_el.text if desc_el is not None else "") or ""
            items.append(_Item(
                topic=title_el.text.strip(),
                context=re.sub(r"<[^>]+>", " ", desc).strip()[:400] or f"Trending now in {geo}",
                article_type="trending",
            ))
        return items
    except Exception as e:
        logger.debug(f"Google Trends RSS failed: {e}")
        return []


def _fetch_reddit_hot(subreddit: str = "popular", limit: int = 10) -> List[_Item]:
    """Reddit JSON endpoint — no auth, rate-limited but fine for low volume."""
    url = f"https://www.reddit.com/r/{subreddit}/hot.json?limit={limit}"
    try:
        r = requests.get(url, headers=UA, timeout=TIMEOUT)
        if r.status_code != 200:
            return []
        items: List[_Item] = []
        for child in (r.json().get("data", {}).get("children") or []):
            d = child.get("data") or {}
            title = d.get("title")
            if not title or d.get("over_18") or d.get("stickied"):
                continue
            items.append(_Item(
                topic=title[:110],
                context=(d.get("selftext") or "")[:400] or f"r/{d.get('subreddit')} · {d.get('ups')} ups",
                article_type="trending",
                _subreddit=d.get("subreddit", ""),
            ))
        return items
    except Exception as e:
        logger.debug(f"Reddit fetch failed: {e}")
        return []


def _fetch_newsapi(cadence: str = "daily") -> List[_Item]:
    key = os.environ.get("NEWSAPI_KEY")
    if not key:
        return []
    try:
        if cadence == "daily":
            r = requests.get(
                "https://newsapi.org/v2/top-headlines",
                params={"apiKey": key, "country": os.environ.get("NEWSAPI_COUNTRY", "us"), "pageSize": 20},
                timeout=TIMEOUT,
            )
        else:
            # weekly: sorted by popularity for the past 7 days
            r = requests.get(
                "https://newsapi.org/v2/everything",
                params={
                    "apiKey": key,
                    "language": "en",
                    "sortBy": "popularity",
                    "pageSize": 20,
                    "q": "AI OR technology OR business OR science",
                },
                timeout=TIMEOUT,
            )
        if r.status_code != 200:
            return []
        return [
            _Item(
                topic=(a.get("title") or "").split(" - ")[0][:110],
                context=(a.get("description") or a.get("content") or "")[:400],
                article_type="trending",
            )
            for a in r.json().get("articles", []) if a.get("title")
        ]
    except Exception as e:
        logger.debug(f"NewsAPI failed: {e}")
        return []


# ---- Weekly (top of the week, mix trending + research) -------------------

def _fetch_hn_top(period: str = "week") -> List[_Item]:
    """Hacker News via Algolia search API. Ranks by points over a window."""
    tags = "story"
    numeric = "points>100" if period == "month" else "points>50"
    period_secs = {"day": 86400, "week": 86400 * 7, "month": 86400 * 30}.get(period, 86400 * 7)
    import time
    created_after = int(time.time()) - period_secs
    try:
        r = requests.get(
            "https://hn.algolia.com/api/v1/search",
            params={
                "tags": tags,
                "numericFilters": f"{numeric},created_at_i>{created_after}",
                "hitsPerPage": 20,
            },
            headers=UA,
            timeout=TIMEOUT,
        )
        if r.status_code != 200:
            return []
        items: List[_Item] = []
        for h in r.json().get("hits", []):
            title = h.get("title")
            if not title:
                continue
            items.append(_Item(
                topic=title[:110],
                context=(h.get("story_text") or f"HN · {h.get('points')} points · {h.get('url', '')}")[:400],
                article_type="research" if _looks_researchy(title) else "trending",
            ))
        return items
    except Exception as e:
        logger.debug(f"HN fetch failed: {e}")
        return []


def _fetch_devto_top(period: str = "week") -> List[_Item]:
    """Dev.to top articles for the week/month. Public API, no auth."""
    top = {"week": 7, "month": 30}.get(period, 7)
    try:
        r = requests.get(
            "https://dev.to/api/articles",
            params={"top": top, "per_page": 20},
            headers=UA,
            timeout=TIMEOUT,
        )
        if r.status_code != 200:
            return []
        items: List[_Item] = []
        for a in r.json():
            title = a.get("title")
            if not title:
                continue
            items.append(_Item(
                topic=title[:110],
                context=(a.get("description") or "")[:400],
                article_type="research" if _looks_researchy(title) else "trending",
            ))
        return items
    except Exception as e:
        logger.debug(f"Dev.to fetch failed: {e}")
        return []


# ---- Monthly (research, deep-dive) ---------------------------------------

def _fetch_arxiv_recent(query: str = "cs.AI", days: int = 30) -> List[_Item]:
    """arXiv recent submissions in a category. RSS-based, no auth."""
    url = f"http://export.arxiv.org/api/query?search_query=cat:{query}&sortBy=submittedDate&sortOrder=descending&max_results=20"
    try:
        r = requests.get(url, headers=UA, timeout=TIMEOUT)
        if r.status_code != 200:
            return []
        ns = {"atom": "http://www.w3.org/2005/Atom"}
        root = ET.fromstring(r.content)
        items: List[_Item] = []
        for e in root.findall("atom:entry", ns):
            title = (e.find("atom:title", ns).text or "").strip() if e.find("atom:title", ns) is not None else ""
            summary = (e.find("atom:summary", ns).text or "").strip() if e.find("atom:summary", ns) is not None else ""
            if not title:
                continue
            items.append(_Item(
                topic=re.sub(r"\s+", " ", title)[:110],
                context=re.sub(r"\s+", " ", summary)[:500],
                article_type="research",
            ))
        return items
    except Exception as e:
        logger.debug(f"arXiv fetch failed: {e}")
        return []


def _fetch_pubmed_trending(term: str = "(medicine OR health OR clinical)") -> List[_Item]:
    """PubMed E-utilities — top cited recent papers. No auth for basic use."""
    esearch = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
    esummary = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"
    try:
        s = requests.get(
            esearch,
            params={"db": "pubmed", "term": term, "retmax": 15, "sort": "relevance", "retmode": "json"},
            headers=UA, timeout=TIMEOUT,
        )
        if s.status_code != 200:
            return []
        ids = s.json().get("esearchresult", {}).get("idlist") or []
        if not ids:
            return []
        r = requests.get(
            esummary,
            params={"db": "pubmed", "id": ",".join(ids), "retmode": "json"},
            headers=UA, timeout=TIMEOUT,
        )
        if r.status_code != 200:
            return []
        result = r.json().get("result", {})
        items: List[_Item] = []
        for pid in ids:
            entry = result.get(pid) or {}
            title = (entry.get("title") or "").strip()
            if not title:
                continue
            src = entry.get("source") or ""
            date = entry.get("pubdate") or ""
            items.append(_Item(
                topic=title[:110],
                context=f"PubMed · {src} · {date}",
                article_type="research",
            ))
        return items
    except Exception as e:
        logger.debug(f"PubMed fetch failed: {e}")
        return []


# ---- Helpers --------------------------------------------------------------

RESEARCHY_RE = re.compile(
    r"\b(?:study|paper|research|analysis|evidence|meta-?analysis|whitepaper|"
    r"trial|benchmark|deep\s*dive|guide|explained|primer|review)\b",
    re.IGNORECASE,
)

# ---- Topic quality filters -------------------------------------------------

# Blacklist: low-value topics that waste publishing slots
_BLACKLIST_RE = re.compile(
    r"\bmeirl\b|\[OC\]|test results|meme|shitpost|upvote|redditor|"
    r"dank|wholesome|nsfw|cursed|looks(?=\s+like)|when\s+you|"
    r"girl|bro\b|guy\b|unpopular\s+opinion|first\s+time|found\s+the|"
    r"can(?=\s+confirm)|what\s+is|who\s+is|does\s+anyone|"
    r" finally|every\s+time|nobody:|am\s+i\s+the\s+only|"
    r"til\s|lmfao|lol\b|ngl|tbh|imo|fwiw|"
    r"showerthought|perfectly\s+cromulent|just\s+neckbeard",
    re.IGNORECASE,
)

# Blacklist subreddits that produce low-quality content
_BLACKLIST_SUBREDDITS = frozenset({
    "me_irl", "memes", "dankmemes", "shitposting", "mildlyinteresting",
    "todayilearned", "interestingasfuck", "oddlysatisfying",
    "aww", "funny", "gaming", "pics", "EarthPorn",
    "thatsinsane", "insaneparents", "IdiotsInCars",
    "Unexpected", "perfectlycutscreams", "MadeMeSmile",
    "antiwork", "TikTokCringe", "teenagers",
})

# Preferred niche keywords — topics matching these get a score boost
_NICHE_BOOST_RE = re.compile(
    r"\b(?:"
    r"AI|artificial intelligence|machine learning|deep learning|LLM|GPT|gemini|"
    r"technology|software|programming|developer|python|javascript|"
    r"business|startup|entrepreneur|ecommerce|marketing|finance|investing|crypto|bitcoin|"
    r"health|wellness|fitness|nutrition|mental health|"
    r"science|research|study|discovery|space|climate|"
    r"how-?to|guide|tutorial|tips|best|top\s+\d+|review|"
    r"productivity|remote work|automation|robot|IoT|"
    r"sustainable|green|renewable|electric|EV\b|"
    r"cybersecurity|data|cloud|server|"
    r"travel|lifestyle|fashion|beauty|"
    r"pet|dog|cat|animal|"
    r"food|recipe|cooking|"
    r"education|learning|course|"
    r"design|UX|UI|creativ)"
    r"\b",
    re.IGNORECASE,
)

# Minimum quality score threshold (0-100). Topics below this are rejected.
_MIN_QUALITY_SCORE = 50.0


def _topic_quality_score(topic: str, context: str = "", subreddit: str = "") -> float:
    """Score a topic on quality from 0-100.

    Factors:
      + Niche relevance (up to +40)
      + Research-like words (up to +20)
      + Reasonable length (up to +15)
      + Context depth (up to +15)
      - Blacklist match (-100 = auto-reject)
      - Blacklist subreddit (-100 = auto-reject)
      - Very short topic (-20)
    """
    score = 10.0  # base score

    # Auto-reject checks
    if _BLACKLIST_RE.search(topic):
        return 0.0
    if subreddit.lower() in _BLACKLIST_SUBREDDITS:
        return 0.0

    # Niche relevance (biggest boost)
    niche_matches = len(_NICHE_BOOST_RE.findall(topic))
    score += min(niche_matches * 15, 40)

    # Also check context for niche signals
    niche_ctx = len(_NICHE_BOOST_RE.findall(context))
    score += min(niche_ctx * 5, 15)

    # Research-like keywords
    if _looks_researchy(topic):
        score += 20

    # Reasonable length (3+ words is ideal, not too long)
    word_count = len(topic.split())
    if 3 <= word_count <= 12:
        score += 15
    elif 2 <= word_count:
        score += 8
    elif word_count > 12:
        score += 5  # long but could be descriptive

    # Context depth (longer context = more substance)
    if len(context) > 200:
        score += 15
    elif len(context) > 100:
        score += 10
    elif len(context) > 30:
        score += 5

    return min(score, 100.0)


def _looks_researchy(title: str) -> bool:
    return bool(RESEARCHY_RE.search(title or ""))


def _load_topics_json() -> List[_Item]:
    for path in ("src/topics.json", "topics.json"):
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                items: List[_Item] = []
                for group in data:
                    lang = group.get("language", "en")
                    for t in group.get("topics", []):
                        items.append(_Item(
                            topic=t.get("topic", ""),
                            context=t.get("context", ""),
                            lang=lang,
                            article_type=t.get("article_type", "trending"),
                        ))
                return [i for i in items if i.topic]
            except Exception as e:
                logger.debug(f"topics.json load failed: {e}")
    return []


def _evergreen() -> List[_Item]:
    """High-quality curated fallback topics when no good trends found.

    Covers preferred niches: tech, business, health, finance, AI, science, how-to.
    """
    return [
        # AI / Technology
        _Item("How AI is Transforming Small Business Operations in 2026",
              "Machine learning tools that help small businesses automate and scale.",
              "en", "research"),
        _Item("The Future of AI Assistants: What to Expect Next",
              "From Gemini to GPT-5 — how AI assistants are evolving and what it means for productivity.",
              "en", "trending"),
        _Item("Best AI Tools for Content Creators and Marketers",
              "A practical guide to AI-powered writing, design, and video tools available today.",
              "en", "trending"),
        _Item("Understanding Machine Learning: A Beginner's Guide",
              "Core concepts of ML explained simply, with real-world examples.",
              "en", "research"),
        # Business / Finance
        _Item("Top 10 Passive Income Ideas for 2026",
              "Proven strategies for building multiple income streams online and offline.",
              "en", "trending"),
        _Item("How to Start an E-Commerce Business with Minimal Budget",
              "Step-by-step guide from product selection to first sale on a shoestring budget.",
              "en", "research"),
        _Item("Investing in 2026: Where to Put Your Money",
              "A balanced overview of stocks, crypto, real estate, and alternative investments.",
              "en", "trending"),
        # Health / Wellness
        _Item("Science-Backed Benefits of Daily Exercise",
              "What research says about the physical and mental health benefits of regular movement.",
              "en", "research"),
        _Item("Nutrition Guide: What to Eat for Better Energy and Focus",
              "Evidence-based dietary tips for sustained energy throughout the workday.",
              "en", "trending"),
        _Item("How Sleep Quality Affects Your Productivity",
              "The science of sleep and practical tips for improving rest quality.",
              "en", "research"),
        # Tech / How-to
        _Item("Complete Guide to Setting Up a Smart Home in 2026",
              "From smart speakers to automated lighting — the essential smart home guide.",
              "en", "research"),
        _Item("Best Productivity Apps and Tools for Remote Workers",
              "Curated list of apps that help remote teams stay organized and efficient.",
              "en", "trending"),
        _Item("Cybersecurity Basics Everyone Should Know",
              "Essential security practices for protecting your digital life.",
              "en", "trending"),
        # Pet / Lifestyle (pedpro niche)
        _Item("How to Choose the Right Pet for Your Lifestyle",
              "Factors to consider when selecting a dog or cat that matches your daily routine.",
              "en", "trending"),
        _Item("Essential Pet Care Tips for First-Time Dog Owners",
              "Everything you need to know about feeding, training, and health for new dog parents.",
              "en", "research"),
        # Thai-language evergreen
        _Item("เทคนิคการดูแลสุนัขขนยาวในช่วงฤดูร้อน",
              "วิธีดูแลขนสุนัขพันธุ์ยาว เพื่อป้องกันปัญหาหมัดและโรคผิวหนัง",
              "th", "trending"),
        _Item("วิธีเริ่มต้นขายของออนไลน์กับ Shopee สำหรับมือใหม่",
              "คู่มือเปิดร้าน Shopee ตั้งแต่การลงทะเบียนจนถึงการส่งสินค้าแรก",
              "th", "research"),
        _Item("แอปพลิเคชัน AI ที่ควรรู้จักในปี 2026",
              "รวมแอป AI ยอดนิยมที่ช่วยเพิ่มประสิทธิภาพการทำงานและการเรียน",
              "th", "trending"),
    ]


# ---- Public API -----------------------------------------------------------

def _pick(items: List[_Item], target_type: Optional[str] = None) -> Optional[_Item]:
    """Pick a random item, filtered by quality score threshold.

    Only items scoring above _MIN_QUALITY_SCORE are considered. If no items
    pass quality, returns None (caller falls through to evergreen).
    """
    if not items:
        return None

    # Filter by quality score
    qualified = []
    for item in items:
        score = _topic_quality_score(item.topic, item.context, item._subreddit)
        if score >= _MIN_QUALITY_SCORE:
            qualified.append((item, score))

    if not qualified:
        logger.debug(
            f"[topic_quality] no items passed quality threshold "
            f"({_MIN_QUALITY_SCORE}) from {len(items)} candidates"
        )
        return None

    # Prefer items of the target type, among qualified items
    if target_type:
        typed = [(i, s) for i, s in qualified if i.article_type == target_type]
        if typed:
            # Weight by quality score — higher-scored items more likely
            pick = _weighted_choice(typed)
            score = next((s for i, s in typed if i is pick), 0)
            logger.debug(f"[topic_quality] picked {pick.topic!r} score={score:.0f}")
            return pick

    # Fallback: weighted random among all qualified
    pick = _weighted_choice(qualified)
    logger.debug(f"[topic_quality] picked {pick.topic!r}")
    return pick


def _weighted_choice(scored_items: list) -> _Item:
    """Weighted random selection — higher score = higher probability."""
    if len(scored_items) == 1:
        return scored_items[0][0]
    total = sum(s for _, s in scored_items)
    if total <= 0:
        return random.choice([i for i, _ in scored_items])
    r = random.uniform(0, total)
    cumulative = 0
    for item, score in scored_items:
        cumulative += score
        if r <= cumulative:
            return item
    return scored_items[-1][0]


def get_trending_topic(cadence: str = "daily", article_type: Optional[str] = None,
                       lang: Optional[str] = None) -> Trend:
    """
    Primary entrypoint.

    Args:
        cadence: "daily" | "weekly" | "monthly"
        article_type: force "trending" or "research", or None to mix by cadence
        lang: force "en" or "th" (promotional only respects this), default mixes

    Returns:
        (topic, context, lang, article_type)
    """
    cadence = cadence.lower()
    if cadence not in ("daily", "weekly", "monthly"):
        raise ValueError(f"unknown cadence: {cadence}")

    # Language pick (50/50 unless forced)
    target_lang = lang or ("th" if random.random() < 0.5 else "en")

    # Share of promo/K9 topics (pedpro-brand-relevant)
    promo_pct = float(os.environ.get("DAILY_PROMO_PCT", "0.30"))
    if cadence == "daily" and random.random() < promo_pct:
        promo = [i for i in _load_topics_json() if i.lang == target_lang]
        if promo:
            pick = _pick(promo, article_type)
            if pick:
                return pick.as_tuple()

    # Cadence-specific research targets
    research_pct = {
        "daily": 0.25,
        "weekly": float(os.environ.get("WEEKLY_RESEARCH_PCT", "0.60")),
        "monthly": float(os.environ.get("MONTHLY_RESEARCH_PCT", "0.85")),
    }[cadence]
    want_research = article_type == "research" or (
        article_type is None and random.random() < research_pct
    )

    pool: List[_Item] = []
    if cadence == "daily":
        pool.extend(_fetch_google_trends_realtime(os.environ.get("TRENDS_GEO", "US")))
        pool.extend(_fetch_reddit_hot("popular", 10))
        pool.extend(_fetch_newsapi("daily"))
    elif cadence == "weekly":
        pool.extend(_fetch_hn_top("week"))
        pool.extend(_fetch_devto_top("week"))
        pool.extend(_fetch_newsapi("weekly"))
    else:  # monthly
        pool.extend(_fetch_arxiv_recent(os.environ.get("ARXIV_CATEGORY", "cs.AI")))
        pool.extend(_fetch_pubmed_trending())
        pool.extend(_fetch_hn_top("month"))

    desired = "research" if want_research else "trending"
    pick = _pick(pool, desired)
    if pick is None:
        # Fallback path: try the *other* type
        pick = _pick(pool, None)
    if pick is None:
        # Last-resort evergreen
        pick = _pick(_evergreen(), desired) or _evergreen()[0]

    # target_lang overrides only if not already Thai
    if target_lang == "th" and pick.lang != "th":
        pick.lang = "th"  # keep topic content EN, translator handles downstream
    return pick.as_tuple()


# ---- Back-compat shim -----------------------------------------------------

def get_hot_trend() -> Tuple[Optional[str], Optional[str], str]:
    """Legacy 3-tuple entrypoint used by older code paths."""
    topic, ctx, lang, _atype = get_trending_topic("daily")
    return topic, ctx, lang


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    for cad in ("daily", "weekly", "monthly"):
        t, c, lang, atype = get_trending_topic(cad)
        print(f"[{cad:7s}] {atype:8s} [{lang}] {t}")
        print(f"            ↳ {c[:100]}")
