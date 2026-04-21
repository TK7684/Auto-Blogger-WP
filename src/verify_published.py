"""
verify_published.py — Post-publish audit for pedpro.online.

Pulls recent posts via WP REST, runs a check battery, writes JSONL verdicts,
and alerts Discord on any failing post. Run as CLI or imported by main.py
right after publish.

CLI:
    python -m src.verify_published --last 10
    python -m src.verify_published --since 2026-04-21
    python -m src.verify_published --post 1806
    python -m src.verify_published --cadence daily    # yesterday's posts
    python -m src.verify_published --cadence weekly   # last 7 days
    python -m src.verify_published --cadence monthly  # last 30 days

Environment:
    WP_URL, WP_USER, WP_APP_PASSWORD (required)
    DISCORD_WEBHOOK_URL (optional — alerts on FAIL)
    VERIFY_TRENDING_MIN_WORDS (default 500)
    VERIFY_RESEARCH_MIN_WORDS (default 1500)
    VERIFY_REPORT_DIR (default generated_images/verify)
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import os
import re
import sys
import urllib.request
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests
from dotenv import load_dotenv

from src.clients.wordpress import WordPressClient

logger = logging.getLogger(__name__)

load_dotenv(override=True)

# ---- Config ---------------------------------------------------------------

WP_URL = os.environ.get("WP_URL", "")
WP_USER = os.environ.get("WP_USER")
WP_APP_PASSWORD = os.environ.get("WP_APP_PASSWORD")
SITE_HOST = (WP_URL.rstrip("/").split("://", 1)[-1]).lower() if WP_URL else "pedpro.online"
DISCORD_WEBHOOK = os.environ.get("DISCORD_WEBHOOK_URL")
REPORT_DIR = Path(os.environ.get("VERIFY_REPORT_DIR", "generated_images/verify"))
TRENDING_MIN_WORDS = int(os.environ.get("VERIFY_TRENDING_MIN_WORDS", "500"))
RESEARCH_MIN_WORDS = int(os.environ.get("VERIFY_RESEARCH_MIN_WORDS", "1500"))

HTTP_TIMEOUT = 10
IMG_TAG_RE = re.compile(r'<img\b[^>]*\bsrc=["\']([^"\']+)["\']', re.IGNORECASE)
A_TAG_RE = re.compile(r'<a\b[^>]*\bhref=["\']([^"\']+)["\']', re.IGNORECASE)
SCHEMA_RE = re.compile(r'<script[^>]*application/ld\+json[^>]*>(.*?)</script>', re.IGNORECASE | re.DOTALL)
PLACEHOLDER_RE = re.compile(r'\{\{[^}]+\}\}')
TAG_RE = re.compile(r'<[^>]+>')


# ---- Verdict types --------------------------------------------------------

@dataclass
class Check:
    name: str
    passed: bool
    detail: str = ""
    severity: str = "fail"  # "fail" | "warn"


@dataclass
class PostVerdict:
    post_id: int
    url: str
    title: str
    published: str
    article_type: str  # "trending" | "research" | "unknown"
    status: str  # "PASS" | "WARN" | "FAIL" | "ERROR"
    checks: List[Check] = field(default_factory=list)
    stats: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "post_id": self.post_id,
            "url": self.url,
            "title": self.title,
            "published": self.published,
            "article_type": self.article_type,
            "status": self.status,
            "checks": [asdict(c) for c in self.checks],
            "stats": self.stats,
        }


# ---- Core checks ----------------------------------------------------------

def _strip_html(html: str) -> str:
    return TAG_RE.sub(" ", html or "")


def _word_count(html: str) -> int:
    text = _strip_html(html)
    return len([w for w in re.split(r"\s+", text) if w])


def _classify(post: Dict[str, Any], content: str) -> str:
    """Classify article as trending|research|unknown from categories/tags/length."""
    cat_tags = []
    for block in ("categories", "tags"):
        val = post.get(block) or []
        if isinstance(val, list):
            cat_tags.extend(str(x).lower() for x in val)
    joined = " ".join(cat_tags)
    if any(k in joined for k in ("research", "study", "analysis", "whitepaper", "deep-dive")):
        return "research"
    if any(k in joined for k in ("trending", "news", "hot", "today")):
        return "trending"
    # Fallback: length signal
    wc = _word_count(content)
    if wc >= RESEARCH_MIN_WORDS:
        return "research"
    if wc >= TRENDING_MIN_WORDS:
        return "trending"
    return "unknown"


def _head_ok(url: str) -> Tuple[bool, int]:
    try:
        r = requests.head(url, timeout=HTTP_TIMEOUT, allow_redirects=True)
        if r.status_code == 405:
            r = requests.get(url, timeout=HTTP_TIMEOUT, stream=True)
        return (200 <= r.status_code < 400, r.status_code)
    except Exception as e:
        logger.debug(f"HEAD {url} failed: {e}")
        return False, 0


def _resolve_featured_image_url(wp: WordPressClient, media_id: int) -> Optional[str]:
    if not media_id or not wp.wp_url:
        return None
    url = f"{wp.wp_url}/wp-json/wp/v2/media/{media_id}"
    try:
        r = wp.session.get(url, headers=wp.headers, timeout=HTTP_TIMEOUT)
        if r.status_code == 200:
            return r.json().get("source_url")
    except Exception as e:
        logger.debug(f"media fetch {media_id} failed: {e}")
    return None


def _get_yoast_focus_keyword(wp: WordPressClient, post_id: int) -> Optional[str]:
    if not wp.wp_url:
        return None
    # Yoast stores in meta, reachable via context=edit
    url = f"{wp.wp_url}/wp-json/wp/v2/posts/{post_id}?context=edit"
    try:
        r = wp.session.get(url, headers=wp.headers, timeout=HTTP_TIMEOUT)
        if r.status_code == 200:
            meta = r.json().get("meta", {}) or {}
            # Yoast SEO standard key + a few common variants
            for key in ("_yoast_wpseo_focuskw", "yoast_focus_kw", "focus_keyword"):
                val = meta.get(key)
                if val:
                    return str(val).strip()
    except Exception as e:
        logger.debug(f"yoast fetch {post_id} failed: {e}")
    return None


def verify_post(wp: WordPressClient, post: Dict[str, Any]) -> PostVerdict:
    """Run the check battery against a single WP post dict."""
    post_id = int(post.get("id") or 0)
    url = post.get("link") or ""
    title = (post.get("title") or {}).get("rendered") or post.get("title") or ""
    content = (post.get("content") or {}).get("rendered") or post.get("content") or ""
    excerpt = (post.get("excerpt") or {}).get("rendered") or ""
    published = post.get("date") or ""
    featured_media = int(post.get("featured_media") or 0)

    article_type = _classify(post, content)
    min_words = RESEARCH_MIN_WORDS if article_type == "research" else TRENDING_MIN_WORDS
    min_inline_imgs = 3 if article_type == "research" else 1

    checks: List[Check] = []
    wc = _word_count(content)

    # 1. Word count
    checks.append(Check(
        "word_count",
        wc >= min_words,
        f"{wc} words (min {min_words} for {article_type})",
    ))

    # 2. Title length (Yoast rec: ≤ 60)
    tstripped = _strip_html(title).strip()
    checks.append(Check(
        "title_length",
        5 <= len(tstripped) <= 65,
        f"{len(tstripped)} chars",
        severity="warn" if len(tstripped) > 65 else "fail",
    ))

    # 3. Excerpt / meta description length
    estripped = _strip_html(excerpt).strip()
    checks.append(Check(
        "excerpt_length",
        20 <= len(estripped) <= 200,
        f"{len(estripped)} chars",
        severity="warn",
    ))

    # 4. No unresolved {{placeholders}}
    ph = PLACEHOLDER_RE.findall(content)
    checks.append(Check(
        "no_unresolved_placeholders",
        len(ph) == 0,
        f"found {len(ph)}: {ph[:3]}" if ph else "clean",
    ))

    # 5. Inline <img> count
    imgs = IMG_TAG_RE.findall(content)
    checks.append(Check(
        "inline_image_count",
        len(imgs) >= min_inline_imgs,
        f"{len(imgs)} imgs (min {min_inline_imgs})",
        severity="warn" if article_type == "unknown" else "fail",
    ))

    # 6. Inline <img> URLs resolve
    img_fail = []
    for src in imgs[:10]:  # cap probes
        ok, code = _head_ok(src)
        if not ok:
            img_fail.append((src, code))
    checks.append(Check(
        "inline_images_resolve",
        len(img_fail) == 0,
        f"{len(img_fail)} broken: {[f[0][-60:] for f in img_fail[:3]]}" if img_fail else "all OK",
    ))

    # 7. Featured image set + resolves
    fi_url: Optional[str] = None
    if featured_media:
        fi_url = _resolve_featured_image_url(wp, featured_media)
    checks.append(Check(
        "featured_image_set",
        featured_media > 0,
        f"media_id={featured_media}",
    ))
    fi_ok = False
    if fi_url:
        fi_ok, code = _head_ok(fi_url)
    checks.append(Check(
        "featured_image_resolves",
        bool(fi_ok),
        f"{fi_url} status={code if fi_url else 'n/a'}" if fi_url else "no URL",
    ))

    # 8. JSON-LD schema markup
    schema_match = SCHEMA_RE.search(content)
    schema_ok = False
    if schema_match:
        try:
            json.loads(schema_match.group(1).strip())
            schema_ok = True
        except Exception as e:
            schema_ok = False
    checks.append(Check(
        "schema_jsonld_valid",
        schema_ok,
        "present & valid" if schema_ok else ("invalid JSON" if schema_match else "missing"),
        severity="warn",
    ))

    # 9. Internal link count
    hrefs = A_TAG_RE.findall(content)
    internal = [h for h in hrefs if SITE_HOST in h.lower()]
    checks.append(Check(
        "internal_links",
        len(internal) >= 1,
        f"{len(internal)} internal / {len(hrefs)} total",
        severity="warn",
    ))

    # 10. Focus keyword in first 200 words (Yoast)
    fk = _get_yoast_focus_keyword(wp, post_id)
    fk_in_lead = False
    if fk:
        lead = _strip_html(content)[:1200].lower()  # ~200 words
        fk_in_lead = fk.lower() in lead
    checks.append(Check(
        "focus_keyword_in_lead",
        fk_in_lead,
        f"kw='{fk}' in first 200 words" if fk else "no focus keyword set",
        severity="warn",
    ))

    # Aggregate status
    failed = [c for c in checks if not c.passed and c.severity == "fail"]
    warned = [c for c in checks if not c.passed and c.severity == "warn"]
    if failed:
        status = "FAIL"
    elif warned:
        status = "WARN"
    else:
        status = "PASS"

    return PostVerdict(
        post_id=post_id,
        url=url,
        title=tstripped,
        published=published,
        article_type=article_type,
        status=status,
        checks=checks,
        stats={
            "word_count": wc,
            "inline_images": len(imgs),
            "internal_links": len(internal),
            "total_links": len(hrefs),
            "featured_media_id": featured_media,
            "has_schema": schema_ok,
            "focus_keyword": fk,
        },
    )


# ---- Fetching --------------------------------------------------------------

def fetch_posts_by_cadence(wp: WordPressClient, cadence: str) -> List[Dict[str, Any]]:
    now = dt.datetime.now(dt.timezone.utc)
    if cadence == "daily":
        after = now - dt.timedelta(days=1)
    elif cadence == "weekly":
        after = now - dt.timedelta(days=7)
    elif cadence == "monthly":
        after = now - dt.timedelta(days=30)
    else:
        raise ValueError(f"unknown cadence: {cadence}")
    return wp.fetch_posts({
        "per_page": 100,
        "status": "publish",
        "after": after.isoformat(),
        "orderby": "date",
        "order": "desc",
    })


def fetch_last_n(wp: WordPressClient, n: int) -> List[Dict[str, Any]]:
    return wp.fetch_posts({
        "per_page": min(n, 100),
        "status": "publish",
        "orderby": "date",
        "order": "desc",
    })


# ---- Output ---------------------------------------------------------------

def write_report(verdicts: List[PostVerdict]) -> Path:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    path = REPORT_DIR / f"{dt.date.today().isoformat()}.jsonl"
    with path.open("a", encoding="utf-8") as f:
        for v in verdicts:
            f.write(json.dumps(v.to_dict(), ensure_ascii=False) + "\n")
    return path


def notify_discord(verdicts: List[PostVerdict]) -> None:
    if not DISCORD_WEBHOOK:
        return
    failing = [v for v in verdicts if v.status in ("FAIL", "ERROR")]
    warning = [v for v in verdicts if v.status == "WARN"]
    passed = [v for v in verdicts if v.status == "PASS"]
    color = 0xe74c3c if failing else (0xf39c12 if warning else 0x2ecc71)
    header = f"✅ {len(passed)} PASS  •  ⚠️ {len(warning)} WARN  •  ❌ {len(failing)} FAIL"
    fields = []
    for v in (failing + warning)[:8]:
        bad = [c.name for c in v.checks if not c.passed]
        fields.append({
            "name": f"{v.status} · {v.title[:70]}",
            "value": f"[#{v.post_id}]({v.url})\nfailed: `{', '.join(bad[:6])}`",
            "inline": False,
        })
    payload = json.dumps({
        "embeds": [{
            "title": "🔍 Auto-Blogger Verify",
            "description": header,
            "color": color,
            "fields": fields,
            "footer": {"text": "verify_published.py → pedpro.online"},
        }]
    }).encode("utf-8")
    try:
        req = urllib.request.Request(
            DISCORD_WEBHOOK, data=payload,
            headers={"Content-Type": "application/json", "User-Agent": "Auto-Blogger-WP/verify"},
        )
        urllib.request.urlopen(req, timeout=10)
    except Exception as e:
        logger.warning(f"Discord verify alert failed: {e}")


# ---- Entrypoints ----------------------------------------------------------

def verify_batch(wp: WordPressClient, posts: List[Dict[str, Any]]) -> List[PostVerdict]:
    out: List[PostVerdict] = []
    for post in posts:
        try:
            v = verify_post(wp, post)
        except Exception as e:
            logger.exception(f"verify_post({post.get('id')}) crashed: {e}")
            v = PostVerdict(
                post_id=int(post.get("id") or 0),
                url=post.get("link") or "",
                title=str((post.get("title") or {}).get("rendered", "")),
                published=post.get("date") or "",
                article_type="unknown",
                status="ERROR",
                checks=[Check("verifier_exception", False, str(e))],
            )
        out.append(v)
        logger.info(f"{v.status:5s}  #{v.post_id}  {v.title[:60]}")
    return out


def run_cli() -> int:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--last", type=int, help="Verify last N published posts")
    g.add_argument("--post", type=int, help="Verify a single post by ID")
    g.add_argument("--cadence", choices=["daily", "weekly", "monthly"], help="Verify posts in a time window")
    ap.add_argument("--no-discord", action="store_true", help="Skip Discord alerting")
    ap.add_argument("--quiet", action="store_true", help="Only print FAIL/ERROR")
    args = ap.parse_args()

    if not WP_URL:
        print("WP_URL not set", file=sys.stderr)
        return 2

    wp = WordPressClient(WP_URL, WP_USER, WP_APP_PASSWORD)

    if args.post:
        post = wp.get_post(args.post)
        if not post:
            print(f"post {args.post} not found", file=sys.stderr)
            return 3
        posts = [post]
    elif args.cadence:
        posts = fetch_posts_by_cadence(wp, args.cadence)
    else:
        posts = fetch_last_n(wp, args.last or 10)

    logger.info(f"Verifying {len(posts)} post(s)")
    verdicts = verify_batch(wp, posts)
    report_path = write_report(verdicts)
    logger.info(f"Report: {report_path}")

    if not args.no_discord:
        notify_discord(verdicts)

    fails = sum(1 for v in verdicts if v.status in ("FAIL", "ERROR"))
    if args.quiet:
        for v in verdicts:
            if v.status in ("FAIL", "ERROR"):
                print(json.dumps(v.to_dict(), ensure_ascii=False))
    else:
        for v in verdicts:
            print(f"{v.status:5s}  #{v.post_id}  {v.title[:70]}")
            for c in v.checks:
                if not c.passed:
                    print(f"   - [{c.severity}] {c.name}: {c.detail}")
    return 0 if fails == 0 else 1


if __name__ == "__main__":
    sys.exit(run_cli())
