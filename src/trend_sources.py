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
    return [
        _Item("The Future of AI in 2026", "How generative models are reshaping industries.", "en", "research"),
        _Item("Sustainable Travel Tips", "Low-impact ways to explore the world.", "en", "trending"),
        _Item("SAR K9 Training Fundamentals", "Foundations of search-and-rescue dog training.", "en", "research"),
    ]


# ---- Public API -----------------------------------------------------------

def _pick(items: List[_Item], target_type: Optional[str] = None) -> Optional[_Item]:
    if not items:
        return None
    if target_type:
        typed = [i for i in items if i.article_type == target_type]
        if typed:
            return random.choice(typed)
    return random.choice(items)


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
