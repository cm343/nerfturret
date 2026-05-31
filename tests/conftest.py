"""Shared test fixtures. Puts src/ on the import path so the modules under test
(streaming.py, vision.py) import as top-level modules, matching how they are run
on the Pi."""

import os
import socket
import sys

import pytest

SRC = os.path.join(os.path.dirname(__file__), "..", "src")
sys.path.insert(0, os.path.abspath(SRC))


@pytest.fixture
def free_port():
    """Return a callable that yields an unused TCP port on the loopback iface."""

    def _pick() -> int:
        s = socket.socket()
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
        s.close()
        return port

    return _pick


@pytest.fixture
def manager(free_port):
    """A StreamManager bound to loopback with a free base port; closed on teardown."""
    from streaming import StreamManager

    m = StreamManager(host="127.0.0.1", base_port=free_port())
    yield m
    m.close_all()
