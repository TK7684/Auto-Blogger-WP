"""
SEO Keyword Validator — score Thai affiliate keyword candidates by real search demand.

Uses Google's public autocomplete API (suggestqueries.google.com) as a demand proxy.
Google only returns autocomplete suggestions for queries with real search volume —
so the COUNT of returned suggestions (0-10) is a usable demand signal.

Feeds the results back into:
  - tools/shopee_custom_link_gen.py DEFAULT_KEYWORDS
  - ψ/writing/pedpro-shopee-keywords.md priority list
  - any future affiliate / SEO content selection

Usage:
  python tools/seo_keyword_validator.py                                  # validates DEFAULT_KEYWORDS
  python tools/seo_keyword_validator.py KEYWORD1 KEYWORD2 KEYWORD3      # validates ad-hoc
  python tools/seo_keyword_validator.py --file candidates.txt           # one per line
  python tools/seo_keyword_validator.py --json                          # machine-readable output
  python tools/seo_keyword_validator.py --geo th --lang th              # default (Thai market)

Scoring:
  10 = max demand (Google returns 10 autocomplete suggestions)
   0 = no autocomplete fired → almost no search volume in that exact form

Bonus signal: the autocomplete SUGGESTIONS themselves reveal long-tail opportunities
with high transactional intent (e.g. "ยี่ห้อไหนดี" = "which brand is good" = buyer-ready).
The CLI prints the top 3 suggestions so you can spot those.

Known limitation: autocomplete-count is a demand PROXY, not a volume number. For exact
monthly-searches data, Google Keyword Planner (requires Google Ads account) or
paid tools (Ahrefs / Semrush) remain the ground truth. This validator is the
"free + now" layer.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Optional
from urllib.parse import quote

import httpx


# Matches the SEO-validated defaults shipped in shopee_custom_link_gen.py
# (keep in sync when either changes)
CURRENT_DEFAULTS = [
    "ชุดวินเทจ", "เสื้อวินเทจ", "ชุดเดรสออกงาน", "เสื้อยืดผู้หญิง",
    "เซรั่มวิตามินซี", "มาส์กหน้า", "ครีมกันแดด",
    "กระเป๋าสะพาย", "กระเป๋าสตางค์ผู้หญิง", "รองเท้าผ้าใบผู้หญิง ใส่สบาย",
]


def google_autocomplete(query: str, lang: str = "th", geo: str = "th", timeout: float = 10.0) -> list[str]:
    """Fetch Google autocomplete suggestions for a query.

    Returns up to 10 suggestions ordered by popularity. Empty list on error or
    zero autocomplete coverage.
    """
    url = (
        f"https://suggestqueries.google.com/complete/search"
        f"?client=firefox&q={quote(query)}&hl={lang}&gl={geo}"
    )
    try:
        r = httpx.get(url, timeout=timeout, headers={"User-Agent": "Mozilla/5.0"})
        if r.status_code != 200:
            return []
        data = r.json()
        if isinstance(data, list) and len(data) > 1 and isinstance(data[1], list):
            return data[1]
    except Exception:
        pass
    return []


def score_keywords(
    keywords: list[str],
    lang: str = "th",
    geo: str = "th",
    rate_limit_s: float = 0.8,
) -> list[dict]:
    """Score a list of keywords by autocomplete demand.

    Returns a list of {keyword, score, suggestions} dicts, ranked descending by
    score. rate_limit_s adds a sleep between requests to stay polite.
    """
    results = []
    for i, kw in enumerate(keywords):
        sugg = google_autocomplete(kw, lang=lang, geo=geo)
        results.append(
            {
                "keyword": kw,
                "score": len(sugg),
                "suggestions": sugg[:10],
            }
        )
        if i < len(keywords) - 1:
            time.sleep(rate_limit_s)
    return sorted(results, key=lambda r: (-r["score"], r["keyword"]))


def _load_keywords_from_args(args: argparse.Namespace) -> list[str]:
    if args.keywords:
        return list(args.keywords)
    if args.file:
        lines = Path(args.file).read_text(encoding="utf-8").splitlines()
        return [ln.strip() for ln in lines if ln.strip() and not ln.startswith("#")]
    return list(CURRENT_DEFAULTS)


def _print_human(results: list[dict]) -> None:
    print(f"\n{'SCORE':<7} {'KEYWORD':<40} TOP SUGGESTIONS")
    print("-" * 100)
    for r in results:
        top = " | ".join(r["suggestions"][:3])
        print(f"{r['score']:<7} {r['keyword']:<40} {top}")
    print()

    high = [r for r in results if r["score"] >= 8]
    mid = [r for r in results if 3 <= r["score"] < 8]
    low = [r for r in results if r["score"] < 3]
    print(f"Summary: {len(high)} HIGH (≥8) · {len(mid)} MID (3-7) · {len(low)} LOW (<3)")
    if low:
        print(f"  LOW-demand keywords to consider replacing: {[r['keyword'] for r in low]}")


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[1])
    p.add_argument("keywords", nargs="*", help="Keywords to validate (blank = use defaults)")
    p.add_argument("--file", type=str, help="Read keywords from file (one per line)")
    p.add_argument("--lang", default="th", help="Language code (default: th)")
    p.add_argument("--geo", default="th", help="Geo code (default: th)")
    p.add_argument("--json", action="store_true", help="Output JSON instead of table")
    p.add_argument("--rate-limit", type=float, default=0.8, help="Sleep between queries (s)")
    args = p.parse_args(argv)

    keywords = _load_keywords_from_args(args)
    if not keywords:
        print("No keywords to validate", file=sys.stderr)
        return 1

    results = score_keywords(keywords, lang=args.lang, geo=args.geo, rate_limit_s=args.rate_limit)

    if args.json:
        print(json.dumps(results, ensure_ascii=False, indent=2))
    else:
        _print_human(results)
    return 0


if __name__ == "__main__":
    sys.exit(main())
