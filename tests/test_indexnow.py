"""Tests for src.indexnow — real HTTP behavior via a local test server.

Covers: payload shape, key persistence, keyLocation inclusion, endpoint
fallback on non-2xx, and the never-raises guarantee.
"""

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from src import indexnow


class _Recorder:
    def __init__(self):
        self.requests = []      # (path, body_dict)
        self.fail_first = 0     # respond 500 to first N requests

    def handler(self):
        rec = self

        class H(BaseHTTPRequestHandler):
            def do_POST(self):
                length = int(self.headers.get("Content-Length", 0))
                body = json.loads(self.rfile.read(length) or b"{}")
                rec.requests.append((self.path, body))
                if rec.fail_first and len(rec.requests) <= rec.fail_first:
                    self.send_response(500)
                    self.end_headers()
                    return
                self.send_response(202)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(b'{}')

            def log_message(self, format, *args):  # noqa: A002
                pass

        return H


@pytest.fixture()
def server(tmp_path, monkeypatch):
    """Local HTTP server + state isolated to tmp_path + fresh key."""
    rec = _Recorder()
    srv = HTTPServer(("127.0.0.1", 0), rec.handler())
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    port = srv.server_address[1]
    monkeypatch.setattr(indexnow, "STATE_PATH", str(tmp_path / ".indexnow_state.json"))
    yield rec, f"http://127.0.0.1:{port}/indexnow"
    srv.shutdown()
    t.join(timeout=2)


def test_ping_ok_sends_valid_payload(server):
    rec, ep = server
    class FakeWP:
        def upload_media(self, path, alt_text="", title=""):
            return 123, "https://pedpro.online/wp-content/uploads/key.txt"
    ok, code, used = indexnow.ping(
        ["https://pedpro.online/some-post"], wp_client=FakeWP(), endpoints=[ep])
    assert ok is True and code == 202 and used == ep
    path, body = rec.requests[0]
    assert body["host"] == "pedpro.online"
    assert body["urlList"] == ["https://pedpro.online/some-post"]
    assert body["key"] and 8 <= len(body["key"]) <= 128
    assert body.get("keyLocation", "").startswith("https://")


def test_key_is_stable_across_pings(server):
    rec, ep = server
    class FakeWP:
        def upload_media(self, path, alt_text="", title=""):
            return 123, "https://pedpro.online/wp-content/uploads/key.txt"
    indexnow.ping(["https://pedpro.online/a"], wp_client=FakeWP(), endpoints=[ep])
    key1 = indexnow._load_state()["key"]
    indexnow.ping(["https://pedpro.online/b"], wp_client=FakeWP(), endpoints=[ep])
    key2 = indexnow._load_state()["key"]
    assert key1 == key2  # one key forever, no re-upload


def test_fallback_on_http_error(server):
    rec, ep = server
    rec.fail_first = 1
    ok, code, used = indexnow.ping(
        ["https://pedpro.online/x"], endpoints=[ep, ep])
    assert ok is True  # second endpoint accepted
    assert len(rec.requests) == 2


def test_empty_urls_noop(server):
    rec, ep = server
    ok, code, used = indexnow.ping([], endpoints=[ep])
    assert ok is False and code is None
    assert rec.requests == []


def test_ping_never_raises_on_unreachable(server):
    ok, code, used = indexnow.ping(
        ["https://pedpro.online/x"],
        endpoints=["http://127.0.0.1:1/indexnow"])  # nothing listens
    assert ok is False and code is None and used is None
