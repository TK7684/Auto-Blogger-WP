#!/usr/bin/env python3
"""
Shopee Affiliate Revenue Tracker — Logs article-product associations for conversion tracking.

Usage:
    python track_shopee_revenue.py                    # View all logs
    python track_shopee_revenue.py --article 12345    # View by article
    python track_shopee_revenue.py --summary          # Monthly summary
    python track_shopee_revenue.py --top-products 10  # Top products by clicks

From code:
    from src.track_shopee_revenue import ShopeeTracker
    tracker = ShopeeTracker()
    tracker.log(article_id=123, product_id=456, product_name="...", commission=50.0)
"""
import json
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional
from collections import Counter

LOG_FILE = Path(__file__).parent / "data" / "shopee_affiliate_log.json"


class ShopeeTracker:
    """Tracks Shopee affiliate revenue per article and product."""

    def __init__(self, log_file: Optional[Path] = None):
        self.log_file = log_file or LOG_FILE
        self._ensure_dir()

    def _ensure_dir(self):
        self.log_file.parent.mkdir(parents=True, exist_ok=True)

    def _load(self) -> list[dict]:
        if not self.log_file.exists():
            return []
        try:
            return json.loads(self.log_file.read_text())
        except (json.JSONDecodeError, Exception):
            return []

    def _save(self, entries: list[dict]):
        self._ensure_dir()
        self.log_file.write_text(json.dumps(entries, indent=2, ensure_ascii=False))

    def log(
        self,
        article_id: int,
        product_id: str,
        product_name: str = "",
        product_price: float = 0,
        commission_rate: float = 0,
        commission: float = 0,
        rating_star: float = 0,
        sales: int = 0,
        offer_link: str = "",
        topic: str = "",
        placement: str = "inline",  # inline, distributed, end
        sub_id: str = "",  # pedpro-{post_id} for Shopee dashboard tracking
    ) -> dict:
        """Log a product placement for an article.

        Returns the logged entry dict.
        """
        entries = self._load()

        # Check for duplicate (same article_id + product_id)
        for entry in entries:
            if (entry.get("article_id") == article_id and
                    entry.get("product_id") == product_id):
                # Update existing entry
                entry["updated_at"] = datetime.now().isoformat()
                entry["clicks"] = entry.get("clicks", 0) + 1
                if sub_id:
                    entry["sub_id"] = sub_id
                if offer_link:
                    entry["offer_link"] = offer_link
                self._save(entries)
                return entry

        entry = {
            "article_id": article_id,
            "product_id": str(product_id),
            "product_name": product_name[:100],
            "product_price": product_price,
            "commission_rate": commission_rate,
            "commission": commission,
            "rating_star": rating_star,
            "sales": sales,
            "offer_link": offer_link,
            "topic": topic[:80],
            "placement": placement,
            "sub_id": sub_id,
            "clicks": 1,
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
        }
        entries.append(entry)
        self._save(entries)
        return entry

    def log_click(self, article_id: int, product_id: str):
        """Increment click count for a product placement."""
        entries = self._load()
        for entry in entries:
            if (entry.get("article_id") == article_id and
                    entry.get("product_id") == str(product_id)):
                entry["clicks"] = entry.get("clicks", 0) + 1
                entry["updated_at"] = datetime.now().isoformat()
                self._save(entries)
                return

    def log_conversion(
        self,
        article_id: int,
        product_id: str,
        order_amount: float,
        earned_commission: float,
    ):
        """Log a successful conversion (purchase completed)."""
        entries = self._load()
        for entry in entries:
            if (entry.get("article_id") == article_id and
                    entry.get("product_id") == str(product_id)):
                entry["conversions"] = entry.get("conversions", 0) + 1
                entry["total_revenue"] = entry.get("total_revenue", 0) + earned_commission
                entry["last_conversion"] = datetime.now().isoformat()
                entry["updated_at"] = datetime.now().isoformat()
                self._save(entries)
                return

    def get_by_article(self, article_id: int) -> list[dict]:
        """Get all product placements for a specific article."""
        entries = self._load()
        return [e for e in entries if e.get("article_id") == article_id]

    def get_summary(self, days: int = 30) -> dict:
        """Get a revenue summary for the last N days."""
        entries = self._load()
        cutoff = (datetime.now() - timedelta(days=days)).isoformat()
        recent = [e for e in entries if e.get("created_at", "") >= cutoff]

        total_placements = len(recent)
        total_clicks = sum(e.get("clicks", 0) for e in recent)
        total_conversions = sum(e.get("conversions", 0) for e in recent)
        total_revenue = sum(e.get("total_revenue", 0) for e in recent)
        total_potential = sum(e.get("commission", 0) for e in recent)

        # Articles with placements
        articles_with_placements = len(set(e.get("article_id") for e in recent))

        # Unique products
        unique_products = len(set(e.get("product_id") for e in recent))

        # Top products by clicks
        product_clicks: dict[str, int] = Counter()
        product_revenue: dict[str, float] = {}
        for e in recent:
            pid = e.get("product_id", "unknown")
            product_clicks[pid] += e.get("clicks", 0)
            product_revenue[pid] = product_revenue.get(pid, 0.0) + e.get("total_revenue", 0)

        top_products = [
            {"product_id": pid, "clicks": clicks, "revenue": product_revenue.get(pid, 0)}
            for pid, clicks in product_clicks.most_common(10)
        ]

        return {
            "period_days": days,
            "total_placements": total_placements,
            "total_clicks": total_clicks,
            "total_conversions": total_conversions,
            "total_revenue_earned": round(total_revenue, 2),
            "total_potential_commission": round(total_potential, 2),
            "articles_with_placements": articles_with_placements,
            "unique_products": unique_products,
            "click_through_rate": round(total_clicks / max(total_placements, 1), 4),
            "conversion_rate": round(total_conversions / max(total_clicks, 1), 4),
            "top_products": top_products,
        }

    def get_top_products(self, limit: int = 10) -> list[dict]:
        """Get top products by clicks across all time."""
        entries = self._load()
        product_stats: dict[str, dict] = {}
        for e in entries:
            pid = e.get("product_id", "unknown")
            if pid not in product_stats:
                product_stats[pid] = {
                    "product_id": pid,
                    "product_name": e.get("product_name", ""),
                    "total_clicks": 0,
                    "total_conversions": 0,
                    "total_revenue": 0.0,
                    "article_count": 0,
                }
            product_stats[pid]["total_clicks"] += e.get("clicks", 0)
            product_stats[pid]["total_conversions"] += e.get("conversions", 0)
            product_stats[pid]["total_revenue"] += e.get("total_revenue", 0)
            product_stats[pid]["article_count"] += 1

        sorted_products = sorted(
            product_stats.values(),
            key=lambda x: x["total_revenue"],
            reverse=True,
        )
        return sorted_products[:limit]


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Shopee Affiliate Revenue Tracker")
    parser.add_argument("--article", type=int, help="Filter by article ID")
    parser.add_argument("--summary", action="store_true", help="Show monthly summary")
    parser.add_argument("--top-products", type=int, nargs="?", const=10, help="Show top products")
    parser.add_argument("--days", type=int, default=30, help="Summary period in days")
    args = parser.parse_args()

    tracker = ShopeeTracker()

    if args.article:
        entries = tracker.get_by_article(args.article)
        print(f"\n📋 Product placements for article #{args.article}:")
        print(f"{'Product':<50} {'Clicks':>8} {'Conv.':>8} {'Revenue':>10}")
        print("-" * 80)
        for e in entries:
            name = e.get("product_name", "?")[:48]
            clicks = e.get("clicks", 0)
            conv = e.get("conversions", 0)
            rev = e.get("total_revenue", 0)
            print(f"{name:<50} {clicks:>8} {conv:>8} ฿{rev:>9,.2f}")
        print(f"\nTotal: {len(entries)} products placed")

    elif args.summary:
        summary = tracker.get_summary(days=args.days)
        print(f"\n📊 Shopee Affiliate Summary (last {args.days} days):")
        print(f"  Placements:        {summary['total_placements']}")
        print(f"  Total Clicks:      {summary['total_clicks']}")
        print(f"  Conversions:       {summary['total_conversions']}")
        print(f"  Revenue Earned:    ฿{summary['total_revenue_earned']:,.2f}")
        print(f"  Potential Comm.:   ฿{summary['total_potential_commission']:,.2f}")
        print(f"  Articles:          {summary['articles_with_placements']}")
        print(f"  Unique Products:   {summary['unique_products']}")
        print(f"  CTR:               {summary['click_through_rate']:.2%}")
        print(f"  Conv. Rate:        {summary['conversion_rate']:.2%}")
        print(f"\n  Top Products:")
        for p in summary["top_products"][:5]:
            print(f"    {p['product_id']}: {p['clicks']} clicks, ฿{p['revenue']:,.2f} revenue")

    elif args.top_products:
        top = tracker.get_top_products(limit=args.top_products)
        print(f"\n🏆 Top {args.top_products} Products by Revenue:")
        print(f"{'Product':<50} {'Clicks':>8} {'Conv.':>8} {'Revenue':>10} {'Articles':>10}")
        print("-" * 90)
        for p in top:
            name = p.get("product_name", "?")[:48]
            print(f"{name:<50} {p['total_clicks']:>8} {p['total_conversions']:>8} "
                  f"฿{p['total_revenue']:>9,.2f} {p['article_count']:>10}")

    else:
        # Default: show all entries
        entries = tracker._load()
        print(f"\n📚 All Shopee Affiliate Logs ({len(entries)} entries):")
        for e in entries[-20:]:  # Last 20
            aid = e.get("article_id")
            name = (e.get("product_name", "") or "?")[:40]
            clicks = e.get("clicks", 0)
            date = e.get("created_at", "")[:10]
            print(f"  [{date}] Article #{aid} — {name} ({clicks} clicks)")
        if len(entries) > 20:
            print(f"  ... and {len(entries) - 20} more entries")


if __name__ == "__main__":
    main()
