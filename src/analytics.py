"""
Analytics Engine — fetches per-post view data, identifies patterns, generates optimization reports.

Data sources:
  - Post Views Counter REST API: /wp-json/post-views-counter/get-post-views/{id}
  - WordPress REST API: post metadata (date, title, categories, tags)
  - Yoast SEO: focus keyphrase, SEO score
  - Local analytics snapshots: data/analytics_snapshots.json

Usage:
  from src.analytics import AnalyticsEngine
  engine = AnalyticsEngine(wp_client)
  report = engine.generate_report(days=7)

CLI:
  python main.py analytics                  # Full weekly report
  python main.py analytics --days 30        # Custom period
  python main.py analytics --top 10         # Top N posts
  python main.py analytics --optimize       # Include optimization suggestions
"""

import json
import logging
import re
import time
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from html import unescape
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import httpx
import requests
from requests.auth import HTTPBasicAuth

logger = logging.getLogger(__name__)

SNAPSHOT_DIR = Path(__file__).parent / "data"
SNAPSHOT_FILE = SNAPSHOT_DIR / "analytics_snapshots.json"


def _strip_html(text: str) -> str:
    """Remove HTML tags and decode entities."""
    return unescape(re.sub(r"<[^>]+>", "", text)).strip()


class AnalyticsEngine:
    """Fetches, stores, and analyzes post view data from pedpro.online."""

    def __init__(self, wp_url: str, wp_user: str, wp_app_password: str):
        self.wp_url = wp_url.rstrip("/")
        self.auth = HTTPBasicAuth(wp_user, wp_app_password)
        self.session = requests.Session()
        self.session.auth = self.auth
        self.session.headers.update({"Authorization": f"Basic {self._make_token(wp_user, wp_app_password)}"})

    def _make_token(self, user: str, password: str) -> str:
        import base64
        return base64.b64encode(f"{user}:{password}".encode()).decode()

    def _wp_get(self, url: str, params: Optional[Dict] = None, timeout: int = 20) -> Optional[dict]:
        try:
            r = self.session.get(url, params=params or {}, timeout=timeout)
            if r.status_code == 200:
                return r.json()
            logger.warning(f"WP API [{r.status_code}] {url}")
        except Exception as e:
            logger.error(f"WP API error: {e}")
        return None

    def get_all_posts(self, status: str = "publish") -> List[Dict]:
        """Fetch all post IDs, titles, dates, categories, tags."""
        posts = []
        page = 1
        while True:
            data = self._wp_get(
                f"{self.wp_url}/wp-json/wp/v2/posts",
                params={
                    "per_page": 100,
                    "status": status,
                    "_fields": "id,title,link,date,categories,tags,featured_media",
                    "orderby": "date",
                    "order": "desc",
                    "page": page,
                },
            )
            if not data:
                break
            posts.extend(data)
            total = len(posts)
            # Check if we got a full page; if not, we're done
            if len(data) < 100:
                break
            page += 1
        return posts

    def get_post_views(self, post_id: int) -> int:
        """Get view count for a single post via PVC REST API."""
        try:
            r = requests.get(
                f"{self.wp_url}/wp-json/post-views-counter/get-post-views/{post_id}",
                auth=self.auth,
                timeout=10,
            )
            if r.status_code == 200:
                return int(r.text.strip() or "0")
        except Exception:
            pass
        return 0

    def get_bulk_post_views(self, post_ids: List[int], batch_size: int = 20, delay: float = 0.3) -> Dict[int, int]:
        """Fetch view counts for multiple posts. PVC doesn't support bulk, so we batch with delay."""
        views = {}
        total = len(post_ids)
        for i, pid in enumerate(post_ids):
            views[pid] = self.get_post_views(pid)
            if (i + 1) % batch_size == 0 and i < total - 1:
                logger.info(f"  Views fetched: {i + 1}/{total}")
                time.sleep(delay)  # Be gentle on the API
        logger.info(f"  Views fetched: {total}/{total}")
        return views

    def get_yoast_meta(self, post_id: int) -> Dict:
        """Fetch Yoast SEO meta for a post."""
        data = self._wp_get(
            f"{self.wp_url}/wp-json/wp/v2/posts/{post_id}",
            params={"context": "edit", "_fields": "meta,yoast_head_json"},
        )
        if not data:
            return {}
        meta = data.get("meta", {})
        result = {}
        # Yoast stores its data in various meta keys
        for key in meta:
            if key.startswith("_yoast_wpseo_"):
                clean_key = key.replace("_yoast_wpseo_", "")
                result[clean_key] = meta[key]
        return result

    def collect_full_analytics(self, days: int = 7) -> List[Dict]:
        """Collect views + metadata for all posts published in the last N days."""
        all_posts = self.get_all_posts()
        cutoff = (datetime.utcnow() - timedelta(days=days)).isoformat() + "Z"

        # Filter to recent posts
        recent = [p for p in all_posts if p["date"] >= cutoff]
        logger.info(f"Posts published in last {days} days: {len(recent)} / {len(all_posts)} total")

        if not recent:
            logger.info("No posts in period. Returning all posts.")
            recent = all_posts  # Fall back to all posts

        # Fetch views
        post_ids = [p["id"] for p in recent]
        views = self.get_bulk_post_views(post_ids)

        # Build analytics records
        records = []
        for p in recent:
            pid = p["id"]
            title = _strip_html(p["title"]["rendered"])
            records.append({
                "id": pid,
                "title": title,
                "link": p["link"],
                "date": p["date"],
                "views": views.get(pid, 0),
                "categories": p.get("categories", []),
                "tags": p.get("tags", []),
                "has_featured_image": p.get("featured_media", 0) > 0,
                "url": p["link"],
            })

        # Sort by views descending
        records.sort(key=lambda x: x["views"], reverse=True)
        return records

    def save_snapshot(self, records: List[Dict], label: str = "") -> Dict:
        """Save analytics snapshot to disk."""
        SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)

        snapshots = []
        if SNAPSHOT_FILE.exists():
            try:
                snapshots = json.loads(SNAPSHOT_FILE.read_text())
            except (json.JSONDecodeError, Exception):
                snapshots = []

        snapshot = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "label": label or f"snapshot-{datetime.utcnow().strftime('%Y%m%d-%H%M')}",
            "total_posts": len(records),
            "total_views": sum(r["views"] for r in records),
            "posts": records,
        }

        snapshots.append(snapshot)
        SNAPSHOT_FILE.write_text(json.dumps(snapshots, indent=2, ensure_ascii=False))
        logger.info(f"Snapshot saved: {snapshot['label']} ({len(records)} posts, {snapshot['total_views']} views)")
        return snapshot

    def get_previous_snapshot(self, days_back: int = 7) -> Optional[Dict]:
        """Get the most recent snapshot older than days_back."""
        if not SNAPSHOT_FILE.exists():
            return None
        try:
            snapshots = json.loads(SNAPSHOT_FILE.read_text())
        except Exception:
            return None

        cutoff = (datetime.utcnow() - timedelta(days=days_back)).isoformat() + "Z"
        for snap in reversed(snapshots):
            if snap["timestamp"] <= cutoff:
                return snap
        return None

    def compute_growth(self, current: List[Dict], previous: Optional[Dict]) -> List[Dict]:
        """Compute view growth by comparing current vs previous snapshot."""
        if not previous:
            return [{"post_id": r["id"], "title": r["title"], "current_views": r["views"],
                     "previous_views": 0, "growth": r["views"]} for r in current]

        prev_map = {p["id"]: p["views"] for p in previous["posts"]}
        growth = []
        for r in current:
            prev = prev_map.get(r["id"], 0)
            growth.append({
                "post_id": r["id"],
                "title": r["title"],
                "link": r["link"],
                "current_views": r["views"],
                "previous_views": prev,
                "growth": r["views"] - prev,
                "growth_pct": ((r["views"] - prev) / prev * 100) if prev > 0 else (r["views"] * 100 if r["views"] > 0 else 0),
            })

        growth.sort(key=lambda x: x["growth"], reverse=True)
        return growth

    def identify_patterns(self, records: List[Dict]) -> Dict:
        """Identify what patterns correlate with high views."""
        if not records:
            return {}

        views_list = [r["views"] for r in records]
        avg_views = sum(views_list) / len(views_list) if views_list else 0
        median_views = sorted(views_list)[len(views_list) // 2] if views_list else 0

        top_performers = [r for r in records if r["views"] >= avg_views * 1.5]
        low_performers = [r for r in records if r["views"] < avg_views * 0.5 and r["views"] >= 0]

        # Language detection (Thai vs English)
        def is_thai(text: str) -> bool:
            thai_chars = sum(1 for c in text if '\u0E00' <= c <= '\u0E7F')
            return thai_chars > 3

        lang_views = {"thai": {"total": 0, "count": 0}, "english": {"total": 0, "count": 0}}
        for r in records:
            lang = "thai" if is_thai(r["title"]) else "english"
            lang_views[lang]["total"] += r["views"]
            lang_views[lang]["count"] += 1

        # Has featured image correlation
        with_image = [r for r in records if r["has_featured_image"]]
        without_image = [r for r in records if not r["has_featured_image"]]
        img_avg = sum(r["views"] for r in with_image) / len(with_image) if with_image else 0
        no_img_avg = sum(r["views"] for r in without_image) / len(without_image) if without_image else 0

        # Word count in title correlation (shorter titles often perform better)
        title_len_views = defaultdict(list)
        for r in records:
            bucket = "short" if len(r["title"]) < 40 else "medium" if len(r["title"]) < 70 else "long"
            title_len_views[bucket].append(r["views"])

        patterns = {
            "total_posts": len(records),
            "avg_views": round(avg_views, 1),
            "median_views": median_views,
            "max_views": max(views_list) if views_list else 0,
            "top_performers_count": len(top_performers),
            "low_performers_count": len(low_performers),
            "language_performance": {
                lang: {
                    "avg_views": round(d["total"] / d["count"], 1) if d["count"] > 0 else 0,
                    "post_count": d["count"],
                    "total_views": d["total"],
                }
                for lang, d in lang_views.items()
            },
            "featured_image_impact": {
                "with_image_avg": round(img_avg, 1),
                "without_image_avg": round(no_img_avg, 1),
                "lift_pct": round((img_avg - no_img_avg) / no_img_avg * 100, 1) if no_img_avg > 0 else 0,
            },
            "title_length_impact": {
                bucket: round(sum(views) / len(views), 1) if views else 0
                for bucket, views in title_len_views.items()
            },
            "top_performers": [
                {"id": r["id"], "title": r["title"], "views": r["views"], "link": r["link"]}
                for r in top_performers[:5]
            ],
            "low_performers": [
                {"id": r["id"], "title": r["title"], "views": r["views"], "link": r["link"]}
                for r in low_performers[:5]
            ],
        }

        return patterns

    def generate_optimization_suggestions(self, records: List[Dict], patterns: Dict) -> List[str]:
        """Generate actionable suggestions based on analytics patterns."""
        suggestions = []

        lang_perf = patterns.get("language_performance", {})
        if lang_perf:
            thai_avg = lang_perf.get("thai", {}).get("avg_views", 0)
            en_avg = lang_perf.get("english", {}).get("avg_views", 0)
            if thai_avg > en_avg * 1.3 and en_avg > 0:
                suggestions.append(f"🎯 Thai posts avg {thai_avg} views vs English {en_avg} — increase Thai content ratio")
            elif en_avg > thai_avg * 1.3 and thai_avg > 0:
                suggestions.append(f"🎯 English posts avg {en_avg} views vs Thai {thai_avg} — English resonates more")

        img_impact = patterns.get("featured_image_impact", {})
        if img_impact.get("lift_pct", 0) > 20:
            suggestions.append(f"📸 Posts with images get {img_impact['lift_pct']}% more views — ensure all posts have hero images")
        elif img_impact.get("lift_pct", 0) < -20:
            suggestions.append(f"📸 Image posts underperforming by {abs(img_impact['lift_pct'])}% — check image quality/relevance")

        title_impact = patterns.get("title_length_impact", {})
        if title_impact:
            best_bucket = max(title_impact, key=title_impact.get)
            suggestions.append(f"📝 {best_bucket} titles (< 40 chars if short) perform best — optimize title length")

        # Check for zero-view posts
        zero_views = [r for r in records if r["views"] == 0]
        if zero_views:
            suggestions.append(f"⚠️ {len(zero_views)} posts with 0 views — PVC was just installed, data accumulates from now")

        # Suggest content topics based on top performers
        top = patterns.get("top_performers", [])
        if len(top) >= 3:
            topics = [t["title"] for t in top[:3]]
            suggestions.append(f"🔥 Top topics driving views: {', '.join(topics[:3])}")

        if not suggestions:
            suggestions.append("📊 Not enough data yet — PVC just installed, check back in 7 days")

        return suggestions

    def generate_report(self, days: int = 7, top_n: int = 10, include_optimize: bool = True) -> str:
        """Generate a full analytics report as formatted text."""
        records = self.collect_full_analytics(days=days)
        snapshot = self.save_snapshot(records, label=f"report-{days}d")

        previous = self.get_previous_snapshot(days_back=days)
        growth = self.compute_growth(records, previous)
        patterns = self.identify_patterns(records)

        # Build report
        lines = []
        lines.append(f"📊 PEDPRO.ONLINE ANALYTICS REPORT")
        lines.append(f"{'=' * 50}")
        lines.append(f"Period: last {days} days | Posts: {patterns['total_posts']} | "
                      f"Snapshot: {snapshot['timestamp'][:19]}")
        lines.append(f"Total Views: {snapshot['total_views']} | "
                      f"Avg: {patterns['avg_views']} | Median: {patterns['median_views']}")
        lines.append("")

        # Growth section
        if previous:
            lines.append("📈 VIEW GROWTH (vs previous snapshot)")
            lines.append(f"{'-' * 40}")
            for g in growth[:top_n]:
                arrow = "🔥" if g["growth_pct"] > 100 else "📈" if g["growth"] > 0 else "📉" if g["growth"] < 0 else "➡️"
                title = g["title"][:55]
                pct = f"+{g['growth_pct']:.0f}%" if g["growth_pct"] > 0 else f"{g['growth_pct']:.0f}%"
                lines.append(f"  {arrow} {g['growth']:+d} views ({pct}) | {g['current_views']} total | {title}")
            lines.append("")
        else:
            lines.append("📈 VIEW GROWTH: No previous snapshot to compare (first run)")
            lines.append("")

        # Top posts
        lines.append(f"🏆 TOP {top_n} POSTS BY VIEWS")
        lines.append(f"{'-' * 40}")
        for i, r in enumerate(records[:top_n], 1):
            title = r["title"][:55]
            img = "🖼️" if r["has_featured_image"] else "❌"
            lines.append(f"  {i}. [{r['views']} views] {img} {title}")
            lines.append(f"     {r['date'][:10]} | {r['link']}")
        lines.append("")

        # Patterns
        lines.append("🔍 PATTERNS")
        lines.append(f"{'-' * 40}")
        lang = patterns["language_performance"]
        for l, d in lang.items():
            lines.append(f"  {l.upper()}: {d['post_count']} posts, avg {d['avg_views']} views, {d['total_views']} total")
        lines.append(f"  With image: avg {patterns['featured_image_impact']['with_image_avg']} views")
        lines.append(f"  Without image: avg {patterns['featured_image_impact']['without_image_avg']} views")
        lines.append(f"  Image lift: {patterns['featured_image_impact']['lift_pct']}%")
        for bucket, avg in patterns["title_length_impact"].items():
            lines.append(f"  {bucket} titles: avg {avg} views")
        lines.append("")

        # Optimization suggestions
        if include_optimize:
            suggestions = self.generate_optimization_suggestions(records, patterns)
            lines.append("💡 OPTIMIZATION SUGGESTIONS")
            lines.append(f"{'-' * 40}")
            for s in suggestions:
                lines.append(f"  {s}")
            lines.append("")

        return "\n".join(lines)

    def get_content_brief(self, days: int = 7) -> str:
        """Generate a compact performance brief for injection into content generation prompts.

        This is the feedback loop: past performance data → future content decisions.
        Returns a short paragraph with actionable insights from the last N days.
        """
        records = self.collect_full_analytics(days=days)
        if not records:
            return ""

        patterns = self.identify_patterns(records)
        lines = []

        # Language preference
        lang_perf = patterns.get("language_performance", {})
        thai_avg = lang_perf.get("thai", {}).get("avg_views", 0)
        en_avg = lang_perf.get("english", {}).get("avg_views", 0)
        if thai_avg > en_avg * 1.2 and en_avg > 0:
            lines.append(f"Thai content gets {thai_avg:.1f} avg views vs English {en_avg:.1f} — prefer Thai")
        elif en_avg > thai_avg * 1.2 and thai_avg > 0:
            lines.append(f"English content gets {en_avg:.1f} avg views vs Thai {thai_avg:.1f} — prefer English")

        # Title length
        title_impact = patterns.get("title_length_impact", {})
        if title_impact:
            best_bucket = max(title_impact, key=title_impact.get)
            best_avg = title_impact[best_bucket]
            if best_bucket == "short":
                lines.append(f"Short titles (<40 chars) perform best ({best_avg:.1f} avg views) — keep titles concise")

        # Top performing topics (extract keyword themes)
        top = patterns.get("top_performers", [])
        if top:
            top_titles = [t["title"] for t in top[:5]]
            lines.append(f"Top topics right now: {', '.join(top_titles[:3])} — create related content")

        # Image guidance
        img_impact = patterns.get("featured_image_impact", {})
        lift = img_impact.get("lift_pct", 0)
        if lift > 15:
            lines.append("Posts with featured images get significantly more views — always include hero image")
        elif lift < -15:
            lines.append("Images aren't helping engagement — focus on text quality over image quantity")

        if not lines:
            return "Performance data available but no strong patterns yet — write naturally."

        return "PERFORMANCE BRIEF (based on last " + str(days) + " days analytics):\n" + "\n".join(lines)

    def generate_json_report(self, days: int = 7) -> Dict:
        """Generate analytics report as structured JSON."""
        records = self.collect_full_analytics(days=days)
        snapshot = self.save_snapshot(records, label=f"json-{days}d")
        previous = self.get_previous_snapshot(days_back=days)
        growth = self.compute_growth(records, previous)
        patterns = self.identify_patterns(records)
        suggestions = self.generate_optimization_suggestions(records, patterns)

        return {
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "period_days": days,
            "snapshot": {
                "timestamp": snapshot["timestamp"],
                "total_posts": snapshot["total_posts"],
                "total_views": snapshot["total_views"],
            },
            "patterns": patterns,
            "growth": growth[:20],
            "top_posts": records[:10],
            "suggestions": suggestions,
            "has_previous_data": previous is not None,
        }
