#!/usr/bin/env python3
"""
StreamManager — serve multiple MJPEG video streams, each on its own port.

A StreamManager hands out StreamHandlers on request. Each handler runs a tiny
MJPEG-over-HTTP server on a dedicated port; you push OpenCV BGR frames into it
with handler.publish(frame) and view them in any browser. This makes it trivial
to stream several intermediary results of a computer-vision pipeline side by
side (one port per stage).

    manager = StreamManager()
    raw = manager.create_stream("raw")        # http://<host>:8000/
    edges = manager.create_stream("edges")    # http://<host>:8001/
    raw.publish(frame)
    edges.publish(processed_frame)
    ...
    manager.close_all()
"""

import threading
from http import server

import cv2

# ── Configuration ────────────────────────────────────────────────────────────
HOST         = "0.0.0.0"   # bind address (0.0.0.0 = reachable from the LAN)
BASE_PORT    = 8000        # first stream uses this; each new one takes the next free port
JPEG_QUALITY = 80          # 0–100, passed to cv2.IMWRITE_JPEG_QUALITY
BOUNDARY     = "FRAME"     # multipart boundary marker
# ─────────────────────────────────────────────────────────────────────────────


class _Output:
    """Holds the most recent JPEG frame and wakes streaming clients on update."""

    def __init__(self) -> None:
        self.frame: bytes | None = None
        self.condition = threading.Condition()

    def set(self, jpeg: bytes) -> None:
        with self.condition:
            self.frame = jpeg
            self.condition.notify_all()


def _make_request_handler(output: "_Output"):
    """Build a BaseHTTPRequestHandler class bound to a specific output buffer."""

    class _Handler(server.BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802 (http.server API)
            if self.path in ("/", "/index.html"):
                self.send_response(301)
                self.send_header("Location", "/stream.mjpg")
                self.end_headers()
            elif self.path == "/stream.mjpg":
                self.send_response(200)
                self.send_header("Age", "0")
                self.send_header("Cache-Control", "no-cache, private")
                self.send_header("Pragma", "no-cache")
                self.send_header(
                    "Content-Type",
                    f"multipart/x-mixed-replace; boundary={BOUNDARY}",
                )
                self.end_headers()
                try:
                    while True:
                        with output.condition:
                            output.condition.wait()
                            frame = output.frame
                        if frame is None:
                            continue
                        self.wfile.write(f"--{BOUNDARY}\r\n".encode())
                        self.send_header("Content-Type", "image/jpeg")
                        self.send_header("Content-Length", str(len(frame)))
                        self.end_headers()
                        self.wfile.write(frame)
                        self.wfile.write(b"\r\n")
                except (BrokenPipeError, ConnectionResetError):
                    pass  # client disconnected — end this stream loop quietly
            else:
                self.send_error(404)
                self.end_headers()

        def log_message(self, *args) -> None:  # silence per-request logging
            pass

    return _Handler


class StreamHandler:
    """A single MJPEG stream on one port. Push frames in with publish()."""

    def __init__(
        self,
        port: int,
        *,
        name: str | None = None,
        host: str = HOST,
        jpeg_quality: int = JPEG_QUALITY,
    ) -> None:
        self.port = port
        self.name = name or f"stream-{port}"
        self.host = host
        self.jpeg_quality = jpeg_quality

        self._output = _Output()
        self._server = server.ThreadingHTTPServer(
            (host, port), _make_request_handler(self._output)
        )
        self._server.daemon_threads = True
        self._thread = threading.Thread(
            target=self._server.serve_forever, name=f"stream-{self.name}", daemon=True
        )

    def start(self) -> None:
        """Begin serving in a background thread."""
        self._thread.start()

    def publish(self, image) -> None:
        """Encode an OpenCV BGR ndarray to JPEG and push it to viewers."""
        ok, jpeg = cv2.imencode(
            ".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality]
        )
        if ok:
            self._output.set(jpeg.tobytes())

    def publish_jpeg(self, jpeg: bytes) -> None:
        """Push already-encoded JPEG bytes to viewers."""
        self._output.set(jpeg)

    def close(self) -> None:
        """Shut the HTTP server down and join its thread."""
        self._server.shutdown()
        self._server.server_close()
        if self._thread.is_alive():
            self._thread.join(timeout=2.0)

    @property
    def url(self) -> str:
        # Report a connectable host when bound to all interfaces.
        host = "localhost" if self.host in ("0.0.0.0", "") else self.host
        return f"http://{host}:{self.port}/"


class StreamManager:
    """Creates and tracks MJPEG StreamHandlers, one per port."""

    def __init__(self, *, host: str = HOST, base_port: int = BASE_PORT) -> None:
        self.host = host
        self.base_port = base_port
        self._streams: dict[int, StreamHandler] = {}
        self._lock = threading.Lock()

    def create_stream(
        self, name: str | None = None, *, port: int | None = None
    ) -> StreamHandler:
        """Start a new MJPEG stream and return its handler.

        :param name: optional label for the stream (defaults to ``stream-<port>``).
        :param port: explicit port; if omitted, the next free port from
                     ``base_port`` upward is used.
        """
        with self._lock:
            chosen = port if port is not None else self._next_free_port()
            if chosen in self._streams:
                raise ValueError(f"port {chosen} already has a stream")
            handler = StreamHandler(chosen, name=name, host=self.host)
            handler.start()
            self._streams[chosen] = handler
            return handler

    def close_all(self) -> None:
        """Shut down every stream."""
        with self._lock:
            for handler in self._streams.values():
                handler.close()
            self._streams.clear()

    @property
    def streams(self) -> dict[str, str]:
        """Mapping of stream name → URL for every active stream."""
        with self._lock:
            return {h.name: h.url for h in self._streams.values()}

    def _next_free_port(self) -> int:
        port = self.base_port
        while port in self._streams:
            port += 1
        return port
