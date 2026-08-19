"""IndexNow integration — notify search engines (Bing, Yandex, Naver, Seznam, Yep)
of new/changed URLs immediately instead of waiting for recrawl.

Spec: https://www.indexnow.org/documentation
- POST JSON {host, key, keyLocation?, urlList} to any IndexNow endpoint.
- Key: 8-128 hex chars. Key file must be reachable at keyLocation and contain
  exactly the key. Root hosting is typical, but keyLocation lets us host the
  file in the WP media library (no SFTP needed) — spec-compliant.
- 200/202 = accepted. 400 = bad request, 403 = key invalid, 422 = some URLs bad.

State (key + uploaded key-file URL) persists in .indexnow_state.json so the key
file is uploaded exactly once. All failures are non-blocking (caller logs).
"""

from __future__ import annotations

import json
import logging
import os
import uuid
from typing import List, Optional, Tuple
from urllib import request as urlrequest
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

STATE_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          ".indexnow_state.json")
ENDPOINTS = [
    "https://api.indexnow.org/indexnow",   # shared endpoint, fans out to all engines
    "https://www.bing.com/indexnow",
    "https://yandex.com/indexnow",
]
_HTTP_TIMEOUT = 10


def _load_state() -> dict:
    try:
        with open(STATE_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _save_state(state: dict) -> None:
    try:
        with open(STATE_PATH, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2)
    except Exception as e:
        logger.warning(f"indexnow: could not save state: {e}")


def _host_of(url: str) -> str:
    return urlparse(url).netloc


def ensure_key() -> str:
    """Get or create the IndexNow key (persisted)."""
    state = _load_state()
    key = state.get("key") or os.environ.get("INDEXNOW_KEY")
    if not key:
        key = uuid.uuid4().hex  # 32 hex chars, within 8-128 spec
        state["key"] = key
        _save_state(state)
        logger.info(f"indexnow: generated new key {key[:8]}…")
    return key


def ensure_key_location(wp_client) -> Optional[str]:
    """Upload {key}.txt to the WP media library once; return its public URL.

    wp_client must expose upload_media(path, alt, title) -> (media_id, source_url)
    (src.clients.wordpress.WordPressClient does). Returns None on any failure.
    """
    state = _load_state()
    loc = state.get("key_location")
    if loc:
        return loc
    key = ensure_key()
    if wp_client is None:
        return None
    try:
        tmp = os.path.join("/tmp", f"{key}.txt")
        with open(tmp, "w", encoding="utf-8") as f:
            f.write(key)
        media_id, source_url = wp_client.upload_media(
            tmp, alt_text="IndexNow key file", title=f"{key}.txt")
        if media_id and source_url:
            state["key_location"] = source_url
            _save_state(state)
            logger.info(f"indexnow: key file hosted at {source_url}")
            return source_url
        logger.warning("indexnow: upload_media returned no source_url")
    except Exception as e:
        logger.warning(f"indexnow: key file upload failed: {e}")
    return None


def ping(urls: List[str], wp_client=None,
         endpoints: Optional[List[str]] = None) -> Tuple[bool, Optional[int], Optional[str]]:
    """Submit URLs to IndexNow. Returns (ok, status_code, endpoint_used).

    Never raises — callers treat failure as non-blocking. Retries across
    endpoints; first 2xx wins. Without a hosted key file, engines will 403
    (they check the key location), so we still send and let the caller log.
    """
    urls = [u for u in urls if u]
    if not urls:
        return False, None, None
    key = ensure_key()
    key_location = ensure_key_location(wp_client)
    host = _host_of(urls[0])
    payload = {"host": host, "key": key, "urlList": urls}
    if key_location:
        payload["keyLocation"] = key_location
    data = json.dumps(payload).encode("utf-8")

    for ep in (endpoints or ENDPOINTS):
        try:
            req = urlrequest.Request(
                ep, data=data, method="POST",
                headers={"Content-Type": "application/json; charset=utf-8",
                         "User-Agent": "Auto-Blogger-WP/1.0 (+IndexNow)"})
            with urlrequest.urlopen(req, timeout=_HTTP_TIMEOUT) as resp:
                code = resp.status
            if 200 <= code < 300:  # 200 ok / 202 accepted
                return True, code, ep
            return False, code, ep
        except Exception as e:  # HTTPError, URLError, timeout
            code = getattr(e, "code", None)
            if code is not None:
                # Engine responded with an HTTP error — try next endpoint
                logger.debug(f"indexnow: {ep} -> {code}")
                continue
            logger.debug(f"indexnow: {ep} unreachable: {e}")
            continue
    return False, None, None
