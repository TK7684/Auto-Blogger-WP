"""
Shopee Custom Link Batch Generator — Playwright automation.

Automates affiliate.shopee.co.th's "ลิงก์ที่กำหนดเอง" (Custom Link) tool:
  input:  a list of raw Shopee URLs (product pages or search URLs)
  output: the `s.shopee.co.th/XXXX` affiliate-tracked short link for each

Tabby logs in ONCE in headed mode; storage_state is saved; subsequent
batch runs reuse the session. Removes the manual bottleneck of generating
short links one-at-a-time in the browser.

Why this exists: Shopee's affiliate attribution fires ONLY via the short-link
wrapper. Regular affiliates (no Partner-tier API access) must generate these
through the dashboard. One-at-a-time is a time sink; this script does ~30-50
in a single headless run after the initial login.

USAGE
=====

First-time login (headed — you sign in visibly):
    python tools/shopee_custom_link_gen.py --login

Batch-generate for default keyword seeds:
    python tools/shopee_custom_link_gen.py --batch

Batch-generate for custom URL list:
    python tools/shopee_custom_link_gen.py --input urls.txt --output links.csv

Dry run (print URLs without clicking):
    python tools/shopee_custom_link_gen.py --dry-run

Write results directly to .env SHOPEE_AFFILIATE_LINKS:
    python tools/shopee_custom_link_gen.py --batch --write-env

KNOWN FRAGILE POINTS
====================
Shopee's affiliate dashboard UI changes occasionally. If the script breaks,
the selectors below in SHOPEE_SELECTORS are the first place to adjust. Run
with --debug to dump page snapshots on each step.

The current selectors are best-guesses based on the public dashboard structure
circa 2026-04; they MAY need tuning on first real run. Headed mode gives you
visual confirmation.

Default keyword seeds come from ψ/writing/pedpro-shopee-keywords.md +
pedpro-affiliate-conversion-research.md (10 priority Thai keywords).
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Optional
from urllib.parse import quote

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("shopee-link-gen")


# Paths
ROOT = Path(__file__).resolve().parent.parent
STATE_FILE = ROOT / ".shopee-storage-state.json"
OUTPUT_DIR = ROOT / "tools" / "output"
DEFAULT_OUTPUT = OUTPUT_DIR / "shopee-links.csv"


# Shopee dashboard URLs + selectors — adjust if UI changes
DASHBOARD = "https://affiliate.shopee.co.th/"
CUSTOM_LINK_URL = "https://affiliate.shopee.co.th/offer/custom_link"  # best-guess

SHOPEE_SELECTORS = {
    # Login page markers
    "login_form": 'form[name="loginForm"], input[name="loginKey"]',
    "logged_in_indicator": '[data-testid="user-menu"], nav >> text=/dashboard|แดชบอร์ด/i',
    # Custom Link tool page
    "url_input": 'input[placeholder*="URL"], input[placeholder*="ลิงก์"], input[type="url"]',
    "generate_btn": 'button:has-text("สร้างลิงก์"), button:has-text("Generate"), button[type="submit"]',
    "result_link": 'input[readonly][value*="s.shopee.co.th"], input[readonly][value*="shopee"]',
    "copy_btn": 'button:has-text("คัดลอก"), button:has-text("Copy")',
}


# Default keyword seeds — from research doc. These become search URLs that
# Shopee Custom Link tool wraps into tracked short links.
DEFAULT_KEYWORDS = [
    ("universal", "https://shopee.co.th/"),
    ("เสื้อยืดผู้หญิง", "https://shopee.co.th/search?keyword=%E0%B9%80%E0%B8%AA%E0%B8%B7%E0%B9%89%E0%B8%AD%E0%B8%A2%E0%B8%B7%E0%B8%94%E0%B8%9C%E0%B8%B9%E0%B9%89%E0%B8%AB%E0%B8%8D%E0%B8%B4%E0%B8%87"),
    ("กระโปรงแฟชั่น", "https://shopee.co.th/search?keyword=%E0%B8%81%E0%B8%A3%E0%B8%B0%E0%B9%82%E0%B8%9B%E0%B8%A3%E0%B8%87%E0%B9%81%E0%B8%9F%E0%B8%8A%E0%B8%B1%E0%B9%88%E0%B8%99"),
    ("แมสค์บำรุงผิว", "https://shopee.co.th/search?keyword=%E0%B9%81%E0%B8%A1%E0%B8%AA%E0%B8%84%E0%B9%8C%E0%B8%9A%E0%B8%B3%E0%B8%A3%E0%B8%B8%E0%B8%87%E0%B8%9C%E0%B8%B4%E0%B8%A7"),
    ("กระเป๋าสะพาย", "https://shopee.co.th/search?keyword=%E0%B8%81%E0%B8%A3%E0%B8%B0%E0%B9%80%E0%B8%9B%E0%B9%8B%E0%B8%B2%E0%B8%AA%E0%B8%B0%E0%B8%9E%E0%B8%B2%E0%B8%A2"),
    ("เดรสสตรี", "https://shopee.co.th/search?keyword=%E0%B9%80%E0%B8%94%E0%B8%A3%E0%B8%AA%E0%B8%AA%E0%B8%95%E0%B8%A3%E0%B8%B5"),
    ("เซรั่มลดริ้วรอย", f"https://shopee.co.th/search?keyword={quote('เซรั่มลดริ้วรอย')}"),
    ("ครีมกันแดด", f"https://shopee.co.th/search?keyword={quote('ครีมกันแดด')}"),
    ("รองเท้าผ้าใบผู้หญิง", f"https://shopee.co.th/search?keyword={quote('รองเท้าผ้าใบผู้หญิง')}"),
    ("กางเกงยีนส์เอวสูง", f"https://shopee.co.th/search?keyword={quote('กางเกงยีนส์เอวสูง')}"),
    ("ชุดเดรสวินเทจ", f"https://shopee.co.th/search?keyword={quote('ชุดเดรสวินเทจ')}"),
]


def cmd_login() -> int:
    """Headed login flow. Tabby signs in once; storage_state saved for reuse."""
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        log.error("playwright not installed. run: .venv/bin/pip install playwright && .venv/bin/python -m playwright install chromium")
        return 1

    log.info("Launching headed browser — sign in to Shopee affiliate dashboard")
    log.info(f"Will save session to {STATE_FILE}")
    log.info("After login completes, press ENTER in this terminal to save state.")

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False)
        context = browser.new_context()
        page = context.new_page()
        page.goto(DASHBOARD, wait_until="domcontentloaded")

        log.info("Complete login in the browser window...")
        input("Press ENTER once logged in (you should see the dashboard, not the login form)...")

        context.storage_state(path=str(STATE_FILE))
        log.info(f"✓ Session saved to {STATE_FILE}")

        browser.close()
    return 0


def _load_urls(args) -> list[tuple[str, str]]:
    """Load (label, url) pairs from args or defaults."""
    if args.batch:
        return list(DEFAULT_KEYWORDS)
    if args.input:
        pairs = []
        for line in Path(args.input).read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "," in line:
                label, url = [s.strip() for s in line.split(",", 1)]
            else:
                label, url = line, line
            pairs.append((label, url))
        return pairs
    # Default to batch seeds
    return list(DEFAULT_KEYWORDS)


def cmd_generate(args) -> int:
    try:
        from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeout
    except ImportError:
        log.error("playwright not installed")
        return 1

    pairs = _load_urls(args)
    log.info(f"Batch-generating {len(pairs)} short links")

    if args.dry_run:
        for label, url in pairs:
            log.info(f"  DRY: {label:30s} → {url}")
        return 0

    if not STATE_FILE.exists():
        log.error(f"No session state at {STATE_FILE}. Run: python {sys.argv[0]} --login")
        return 1

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = Path(args.output) if args.output else DEFAULT_OUTPUT

    results: list[dict] = []

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=not args.debug)
        context = browser.new_context(storage_state=str(STATE_FILE))
        page = context.new_page()

        # Verify session is still valid — navigate to dashboard
        page.goto(DASHBOARD, wait_until="domcontentloaded", timeout=30000)
        time.sleep(2)
        if page.locator(SHOPEE_SELECTORS["login_form"]).count() > 0:
            log.error("Session expired — re-run with --login")
            browser.close()
            return 1

        # Navigate to Custom Link tool
        page.goto(CUSTOM_LINK_URL, wait_until="domcontentloaded", timeout=30000)
        time.sleep(2)

        for i, (label, raw_url) in enumerate(pairs, 1):
            log.info(f"[{i}/{len(pairs)}] {label}")
            try:
                url_input = page.locator(SHOPEE_SELECTORS["url_input"]).first
                url_input.click()
                url_input.fill("")
                url_input.type(raw_url, delay=30)
                time.sleep(0.8)

                generate_btn = page.locator(SHOPEE_SELECTORS["generate_btn"]).first
                generate_btn.click()

                # Wait for the result to render
                result_el = page.locator(SHOPEE_SELECTORS["result_link"]).first
                result_el.wait_for(timeout=20000)
                short_link = result_el.input_value()
                if not short_link or "s.shopee.co.th" not in short_link:
                    log.warning(f"  unexpected result: {short_link!r}")
                    continue

                log.info(f"  → {short_link}")
                results.append({"label": label, "source_url": raw_url, "short_link": short_link})

                # Human-paced — avoid bot detection
                time.sleep(1.2 + (i % 3) * 0.5)

            except PlaywrightTimeout as e:
                log.error(f"  timeout: {e}")
            except Exception as e:
                log.error(f"  error: {e}")
                if args.debug:
                    page.screenshot(path=str(OUTPUT_DIR / f"error-{label.replace(' ', '_')}.png"))

        browser.close()

    # Write CSV
    if results:
        with output_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["label", "source_url", "short_link"])
            w.writeheader()
            w.writerows(results)
        log.info(f"✓ {len(results)} short links written to {output_path}")

        if args.write_env:
            short_links = ",".join(r["short_link"] for r in results)
            env_file = ROOT / ".env"
            env_lines = env_file.read_text(encoding="utf-8").splitlines() if env_file.exists() else []
            env_lines = [l for l in env_lines if not l.startswith("SHOPEE_AFFILIATE_LINKS=")]
            env_lines.append(f"SHOPEE_AFFILIATE_LINKS={short_links}")
            env_file.write_text("\n".join(env_lines) + "\n", encoding="utf-8")
            log.info(f"✓ SHOPEE_AFFILIATE_LINKS written to {env_file}")
            log.info("  Run: pm2 restart auto-blogger --update-env")
    else:
        log.error("No results generated — check debug screenshots in tools/output/")
        return 1

    return 0


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[2])
    p.add_argument("--login", action="store_true", help="Run interactive login (headed, one-time)")
    p.add_argument("--batch", action="store_true", help="Use default keyword seeds")
    p.add_argument("--input", type=str, help="Path to file with 'label,url' lines")
    p.add_argument("--output", type=str, help=f"Output CSV path (default: {DEFAULT_OUTPUT})")
    p.add_argument("--dry-run", action="store_true", help="Print URLs without browsing")
    p.add_argument("--debug", action="store_true", help="Headed mode + screenshot on errors")
    p.add_argument("--write-env", action="store_true", help="Write results to .env SHOPEE_AFFILIATE_LINKS")
    args = p.parse_args(argv)

    if args.login:
        return cmd_login()
    return cmd_generate(args)


if __name__ == "__main__":
    sys.exit(main())
