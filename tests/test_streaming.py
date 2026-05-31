"""Tests for src/streaming.py — StreamManager and StreamHandler."""

import http.client
import threading
import time
import urllib.request

import numpy as np
import pytest


def _pump(handler, stop: threading.Event, shape=(48, 64, 3)) -> threading.Thread:
    """Background thread that keeps publishing frames so a client gets data."""

    def run():
        i = 0
        while not stop.is_set():
            handler.publish(np.full(shape, i % 256, dtype=np.uint8))
            i += 9
            time.sleep(0.02)

    t = threading.Thread(target=run, daemon=True)
    t.start()
    return t


# ── StreamManager: port allocation & bookkeeping ─────────────────────────────
def test_create_stream_allocates_sequential_ports(manager):
    a = manager.create_stream("a")
    b = manager.create_stream("b")
    assert b.port == a.port + 1
    assert set(manager.streams) == {"a", "b"}


def test_explicit_port_is_used(manager, free_port):
    p = free_port()
    h = manager.create_stream("x", port=p)
    assert h.port == p


def test_duplicate_port_raises(manager):
    h = manager.create_stream("a")
    with pytest.raises(ValueError):
        manager.create_stream("b", port=h.port)


def test_default_name_from_port(manager):
    h = manager.create_stream()
    assert h.name == f"stream-{h.port}"


def test_close_all_clears_streams(manager):
    manager.create_stream("a")
    manager.create_stream("b")
    manager.close_all()
    assert manager.streams == {}


def test_url_uses_loopback_when_bound_to_all(free_port):
    from streaming import StreamManager

    m = StreamManager(host="0.0.0.0", base_port=free_port())
    try:
        h = m.create_stream("a")
        assert h.url.startswith("http://localhost:")
    finally:
        m.close_all()


# ── StreamHandler: HTTP behavior ─────────────────────────────────────────────
def test_root_redirects_to_stream(manager):
    h = manager.create_stream("v")
    conn = http.client.HTTPConnection("127.0.0.1", h.port, timeout=3)
    try:
        conn.request("GET", "/")
        resp = conn.getresponse()
        assert resp.status == 301
        assert resp.getheader("Location") == "/stream.mjpg"
    finally:
        conn.close()


def test_unknown_path_404(manager):
    h = manager.create_stream("v")
    conn = http.client.HTTPConnection("127.0.0.1", h.port, timeout=3)
    try:
        conn.request("GET", "/nope")
        resp = conn.getresponse()
        assert resp.status == 404
    finally:
        conn.close()


def test_publish_then_fetch_returns_mjpeg(manager):
    h = manager.create_stream("v")
    stop = threading.Event()
    t = _pump(h, stop)
    try:
        time.sleep(0.2)
        resp = urllib.request.urlopen(h.url + "stream.mjpg", timeout=3)
        ctype = resp.headers.get("Content-Type", "")
        data = resp.read(4096)
        resp.close()
    finally:
        stop.set()
        t.join()

    assert "multipart/x-mixed-replace" in ctype
    assert b"--FRAME" in data
    assert b"\xff\xd8" in data          # JPEG start-of-image marker
    assert b"image/jpeg" in data


def test_publish_jpeg_passthrough(manager):
    """publish_jpeg stores raw bytes verbatim (no re-encode)."""
    h = manager.create_stream("v")
    marker = b"\xff\xd8RAWJPEGBYTES\xff\xd9"
    h.publish_jpeg(marker)
    stop = threading.Event()

    # Re-publish the same bytes continuously so a connecting client receives them.
    def run():
        while not stop.is_set():
            h.publish_jpeg(marker)
            time.sleep(0.02)

    t = threading.Thread(target=run, daemon=True)
    t.start()
    try:
        time.sleep(0.15)
        data = urllib.request.urlopen(h.url + "stream.mjpg", timeout=3).read(4096)
    finally:
        stop.set()
        t.join()
    assert marker in data
