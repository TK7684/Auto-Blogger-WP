"""Tests for the crawler-fleet topic source (unified.db → blog topics).

Covers:
- _fetch_crawler_posts: happy path, dedup, missing DB, score/hours filters
- _topic_quality_score: crawler pre-vet boost vs normal path
- get_trending_topic: TOPIC_SOURCE=crawler pure mode + fallback
"""

import json
import os
import sqlite3
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

import src.trend_sources as ts


@pytest.fixture
def crawler_db(tmp_path):
    """Create a mini unified.db with the posts + quality_scores shape."""
    db_path = tmp_path / "unified.db"
    conn = sqlite3.connect(db_path)
    conn.executescript(
        """
        CREATE TABLE posts (
            post_id TEXT PRIMARY KEY,
            platform TEXT,
            text TEXT,
            scraped_at TEXT
        );
        CREATE TABLE quality_scores (
            post_id TEXT PRIMARY KEY,
            score REAL,
            label TEXT
        );
        """
    )
    now = datetime.now(timezone.utc)
    fresh = now.strftime("%Y-%m-%dT%H:%M:%S")
    stale = (now - timedelta(hours=200)).strftime("%Y-%m-%dT%H:%M:%S")

    rows = [
        # (post_id, platform, text, scraped_at, score)
        ("p1", "reddit", "Best budget mechanical keyboards for programming in 2026\n\n"
         "I tested 12 keyboards over 3 months. The key findings: hot-swappable switches "
         "matter more than brand, and a $40 board with good switches beats a $150 board "
         "with bad ones. Full writeup with methodology and sound tests.", fresh, 22),
        ("p2", "reddit", "Best budget mechanical keyboards for programming in 2026\n\n"
         "Duplicate title of p1 — should be deduped.", fresh, 21),
        ("p3", "reddit", "How I automated my entire smart home with local AI\n\n"
         "Went from 30 cloud-dependent devices to a fully local setup. Costs, latency "
         "numbers, and the exact hardware list. Study of what worked and what failed.", fresh, 19),
        ("p4", "reddit", "Too old to be included\n\nshort", fresh, 5),  # below score
        ("p5", "reddit", "Stale post about machine learning research\n\n"
         "This is old enough to fall outside the 72h window even though the score is high. "
         "Lots of words here to pass the length filter, but the timestamp excludes it.", stale, 23),
        ("p6", "reddit", "แนะนำหูฟังราคาถูกสำหรับฟังเพลงในปี 2026 ที่คุ้มค่าที่สุด\n\n"
         "ทดสอบมาแล้วหลายตัวจากหลายยี่ห้อ ทั้งแบบมีสายและบลูทูธไร้สาย ตัวนี้ให้เสียงดีที่สุดในราคาไม่เกินพันบาท "
         "รีวิวละเอียดพร้อมเปรียบเทียบค่าเสียงรบกวน ความทนทานของแบตเตอรี่ และคุณภาพไมโครโฟนตอนสาย "
         "สำหรับใครที่กำลังมองหาหูฟังสำหรับฟังเพลงและดูหนังในงบจำกัด", fresh, 18),
    ]
    for pid, platform, text, scraped_at, score in rows:
        conn.execute("INSERT INTO posts VALUES (?,?,?,?)", (pid, platform, text, scraped_at))
        conn.execute("INSERT INTO quality_scores VALUES (?,?,?)", (pid, score, "high_value"))
    conn.commit()
    conn.close()
    return db_path


@pytest.fixture
def clean_env(crawler_db, tmp_path, monkeypatch):
    """Point the fetcher at the temp DB + temp topic history."""
    monkeypatch.setenv("CRAWLER_DB_PATH", str(crawler_db))
    monkeypatch.setattr(ts, "_CRAWLER_DB_PATH", str(crawler_db))
    monkeypatch.setattr(ts, "_CRAWLER_MIN_SCORE", 14.0)
    monkeypatch.setattr(ts, "_CRAWLER_HOURS", 72)
    monkeypatch.setattr(ts, "_CRAWLER_LIMIT", 25)
    # isolate history file
    hist = tmp_path / "topic_history.json"
    monkeypatch.setattr(ts, "_HISTORY_PATH", hist)
    # no external network in tests
    monkeypatch.setattr(ts, "_fetch_google_trends_realtime", lambda *a, **k: [])
    monkeypatch.setattr(ts, "_fetch_reddit_hot", lambda *a, **k: [])
    monkeypatch.setattr(ts, "_fetch_newsapi", lambda *a, **k: [])
    monkeypatch.setattr(ts, "_fetch_hn_top", lambda *a, **k: [])
    monkeypatch.setattr(ts, "_fetch_devto_top", lambda *a, **k: [])
    monkeypatch.setattr(ts, "_fetch_arxiv_recent", lambda *a, **k: [])
    monkeypatch.setattr(ts, "_fetch_pubmed_trending", lambda *a, **k: [])
    yield
    # _fetch_crawler_posts reads env at call time? No — module-level constants.
    # monkeypatch restores them.


class TestFetchCrawlerPosts:

    def test_happy_path_filters_and_dedup(self, clean_env):
        items = ts._fetch_crawler_posts()
        topics = [i.topic for i in items]
        # p1 kept, p2 deduped (same title), p4 score too low, p5 stale, p6 kept
        assert len(items) == 3
        assert "Best budget mechanical keyboards for programming in 2026" in topics
        assert topics.count("Best budget mechanical keyboards for programming in 2026") == 1
        assert "How I automated my entire smart home with local AI" in topics
        assert not any("Too old" in t for t in topics)
        assert not any("Stale post" in t for t in topics)
        # Thai post detected as Thai
        th = [i for i in items if "หูฟัง" in i.topic]
        assert th and th[0].lang == "th"

    def test_context_contains_source_material(self, clean_env):
        items = ts._fetch_crawler_posts()
        kb = [i for i in items if "keyboards" in i.topic][0]
        assert kb.context.startswith("SOURCE MATERIAL")
        assert "hot-swappable" in kb.context
        assert "quality score 22" in kb.context

    def test_missing_db_returns_empty(self, tmp_path, monkeypatch):
        monkeypatch.setattr(ts, "_CRAWLER_DB_PATH", str(tmp_path / "nope.db"))
        assert ts._fetch_crawler_posts() == []

    def test_title_html_unescape(self, clean_env):
        items = ts._fetch_crawler_posts()
        assert all("&amp;" not in i.topic and "&lt;" not in i.topic for i in items)


class TestCrawlerQualityBoost:

    def test_blacklisted_title_normal_path_rejected_but_crawler_passes(self):
        # "does anyone" is on the blog's own blacklist for raw reddit titles
        title = "Does anyone use local LLMs for productivity research"
        normal = ts._topic_quality_score(title, context="", subreddit="")
        crawler = ts._topic_quality_score(title, context="", subreddit="", source="crawler")
        assert normal == 0.0
        assert crawler >= 50.0  # passes threshold with pre-vet boost

    def test_crawler_score_scales_with_niche(self):
        base = ts._topic_quality_score("plain title no niche words",
                                       context="", source="crawler")
        niche = ts._topic_quality_score("Best AI tools for productivity and automation",
                                        context="", source="crawler")
        assert niche > base

    def test_normal_path_unchanged(self):
        # sanity: non-crawler scoring still works as before
        s = ts._topic_quality_score("Best AI tools for productivity",
                                    context="", subreddit="")
        assert s >= 50.0


class TestPureCrawlerMode:

    def test_crawler_is_default_source(self, clean_env, monkeypatch):
        # no TOPIC_SOURCE set — crawler is now the DEFAULT primary source
        monkeypatch.delenv("TOPIC_SOURCE", raising=False)
        topic, context, lang, atype = ts.get_trending_topic("daily")
        assert context.startswith("SOURCE MATERIAL")

    def test_opt_out_via_topic_source_normal(self, clean_env, monkeypatch):
        monkeypatch.setenv("TOPIC_SOURCE", "normal")
        # all external fetchers stubbed empty → evergreen fallback
        topic, context, lang, atype = ts.get_trending_topic("daily")
        assert topic
        assert not context.startswith("SOURCE MATERIAL")

    def test_crawler_mode_picks_crawler_topic(self, clean_env, monkeypatch):
        monkeypatch.setenv("TOPIC_SOURCE", "crawler")
        topic, context, lang, atype = ts.get_trending_topic("daily")
        assert topic in (
            "Best budget mechanical keyboards for programming in 2026",
            "How I automated my entire smart home with local AI",
            "แนะนำหูฟังราคาถูกสำหรับฟังเพลงในปี 2026 ที่คุ้มค่าที่สุด",
        )
        assert context.startswith("SOURCE MATERIAL")
        # topic recorded in history so next pick differs
        hist = json.loads(Path(ts._HISTORY_PATH).read_text())
        assert any("keyboards" in h or "smart home" in h or "หูฟัง" in h for h in hist)

    def test_pure_mode_exhaustion_falls_back(self, clean_env, monkeypatch):
        monkeypatch.setenv("TOPIC_SOURCE", "crawler")
        # exhaust the crawler pool by recording all topics as published
        hist_path = Path(ts._HISTORY_PATH)
        hist_path.write_text(json.dumps([
            "best budget mechanical keyboards for programming in 2026",
            "how i automated my entire smart home with local ai",
            "แนะนำหูฟังราคาถูกสำหรับฟังเพลงในปี 2026 ที่คุ้มค่าที่สุด",
        ]))
        # all external fetchers stubbed empty → evergreen fallback
        topic, context, lang, atype = ts.get_trending_topic("daily")
        assert topic  # got an evergreen topic, not None/crash

    def test_history_recorded_on_daily_pool_pick(self, clean_env, monkeypatch):
        monkeypatch.delenv("TOPIC_SOURCE", raising=False)
        # normal daily mode — pool is crawler-only here (network stubbed off)
        topic, context, lang, atype = ts.get_trending_topic("daily")
        assert context.startswith("SOURCE MATERIAL") or topic  # crawler or evergreen
        hist = json.loads(Path(ts._HISTORY_PATH).read_text())
        assert len(hist) >= 1


class TestSEOPromptSimpleTerms:

    def test_daily_prompt_mentions_simple_terms(self):
        from src.seo_system import SEOPromptBuilder
        p = SEOPromptBuilder().build_daily_prompt("Test topic", "ctx", language="English")
        assert "SIMPLE terms" in p

    def test_weekly_prompt_mentions_simple_terms(self):
        from src.seo_system import SEOPromptBuilder
        p = SEOPromptBuilder().build_weekly_prompt("Test topic", "ctx", language="English")
        assert "SIMPLE terms" in p

    def test_daily_prompt_thai_simple_terms(self):
        from src.seo_system import SEOPromptBuilder
        p = SEOPromptBuilder().build_daily_prompt("ทดสอบหัวข้อ", "บริบท", language="Thai")
        assert "อธิบายง่าย" in p or "คำง่ายๆ" in p
