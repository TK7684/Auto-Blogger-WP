"""
Zero-dep scheduler daemon for Auto-Blogger.

One long-lived process replaces the PM2 hot-loop. Rules fire at specific
wall-clock times; each rule calls `run_content_generation` or `verify_batch`
in-process. State is kept in `generated_images/scheduler_state.json` so
restarts can't re-fire a rule that already ran today.

Default schedule (all times in local timezone):

    09:00  daily    trending    (post)      every day
    09:30  daily    trending    (post)      every day
    10:00  weekly   research    (post)      Mondays
    11:00  monthly  research    (post)      1st of month
    14:00  --       verify      (daily)     every day   — audits yesterday

Override via SCHEDULE_CONFIG env or scheduler_config.json if present.

Run:
    python -m src.scheduler
"""

from __future__ import annotations

import datetime as dt
import json
import logging
import os
import signal
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List, Optional

from dotenv import load_dotenv

from src.clients.wordpress import WordPressClient
from src.main import initialize_system, run_content_generation
from src.verify_published import fetch_posts_by_cadence, notify_discord, verify_batch, write_report

logger = logging.getLogger(__name__)
load_dotenv(override=True)

STATE_PATH = Path(os.environ.get("SCHEDULER_STATE", "generated_images/scheduler_state.json"))
CONFIG_PATH = Path(os.environ.get("SCHEDULER_CONFIG", "scheduler_config.json"))
TICK_SECONDS = int(os.environ.get("SCHEDULER_TICK", "60"))


# ---- Rules ---------------------------------------------------------------

@dataclass
class Rule:
    name: str
    hour: int
    minute: int
    action: str  # "publish" | "verify"
    cadence: str  # "daily" | "weekly" | "monthly"
    article_type: Optional[str] = None
    days_of_week: Optional[List[int]] = None  # 0=Mon .. 6=Sun
    day_of_month: Optional[int] = None  # 1..28

    def should_fire(self, now: dt.datetime, last_fired: Optional[str]) -> bool:
        if now.hour != self.hour or now.minute != self.minute:
            return False
        if self.days_of_week is not None and now.weekday() not in self.days_of_week:
            return False
        if self.day_of_month is not None and now.day != self.day_of_month:
            return False
        today = now.date().isoformat()
        if last_fired == today:
            return False
        return True


DEFAULT_RULES: List[Rule] = [
    Rule("daily_trend_1",  9,  0, "publish", "daily",   "trending"),
    Rule("daily_trend_2",  9, 30, "publish", "daily",   "trending"),
    Rule("weekly_research", 10, 0, "publish", "weekly",  "research", days_of_week=[0]),
    Rule("monthly_research", 11, 0, "publish", "monthly", "research", day_of_month=1),
    Rule("verify_yesterday", 14, 0, "verify", "daily"),
]


def _load_rules() -> List[Rule]:
    if CONFIG_PATH.exists():
        try:
            data = json.loads(CONFIG_PATH.read_text())
            rules: List[Rule] = []
            for r in data.get("rules", []):
                rules.append(Rule(
                    name=r["name"], hour=int(r["hour"]), minute=int(r.get("minute", 0)),
                    action=r.get("action", "publish"), cadence=r.get("cadence", "daily"),
                    article_type=r.get("article_type"),
                    days_of_week=r.get("days_of_week"), day_of_month=r.get("day_of_month"),
                ))
            if rules:
                logger.info(f"Loaded {len(rules)} rules from {CONFIG_PATH}")
                return rules
        except Exception as e:
            logger.error(f"Bad {CONFIG_PATH}: {e}. Falling back to defaults.")
    return DEFAULT_RULES


# ---- State ---------------------------------------------------------------

def _load_state() -> dict:
    if STATE_PATH.exists():
        try:
            return json.loads(STATE_PATH.read_text())
        except Exception:
            pass
    return {}


def _save_state(state: dict) -> None:
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATE_PATH.write_text(json.dumps(state, indent=2))


# ---- Actions -------------------------------------------------------------

def _do_publish(system: dict, rule: Rule) -> None:
    logger.info(f"▶ {rule.name}: publish cadence={rule.cadence} type={rule.article_type}")
    try:
        run_content_generation(system, cadence=rule.cadence, article_type=rule.article_type)
    except Exception as e:
        logger.exception(f"{rule.name} publish crashed: {e}")


def _do_verify(system: dict, rule: Rule) -> None:
    logger.info(f"▶ {rule.name}: verify cadence={rule.cadence}")
    try:
        wp: WordPressClient = system["wp"]
        posts = fetch_posts_by_cadence(wp, rule.cadence)
        verdicts = verify_batch(wp, posts)
        write_report(verdicts)
        notify_discord(verdicts)
    except Exception as e:
        logger.exception(f"{rule.name} verify crashed: {e}")


# ---- Loop ----------------------------------------------------------------

_running = True


def _shutdown(signum, _frame):
    global _running
    logger.info(f"Signal {signum} received — shutting down after current tick")
    _running = False


def run_forever() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    signal.signal(signal.SIGTERM, _shutdown)
    signal.signal(signal.SIGINT, _shutdown)

    system = initialize_system()
    if not system:
        logger.error("Init failed — no clients. Exiting.")
        return 2

    rules = _load_rules()
    state = _load_state()
    logger.info(f"🕒 Scheduler started with {len(rules)} rule(s); tick={TICK_SECONDS}s")
    for r in rules:
        logger.info(
            f"  · {r.name:22s} {r.action:7s} {r.cadence:7s} "
            f"{(r.article_type or '-'):8s} @ {r.hour:02d}:{r.minute:02d}"
        )

    while _running:
        now = dt.datetime.now()
        for rule in rules:
            last = state.get(rule.name)
            if not rule.should_fire(now, last):
                continue
            if rule.action == "publish":
                _do_publish(system, rule)
            elif rule.action == "verify":
                _do_verify(system, rule)
            state[rule.name] = now.date().isoformat()
            _save_state(state)

        # Sleep to the next minute boundary (cheap + precise enough)
        time.sleep(max(1, TICK_SECONDS - (dt.datetime.now().second % TICK_SECONDS)))

    logger.info("Scheduler stopped cleanly")
    return 0


if __name__ == "__main__":
    raise SystemExit(run_forever())
