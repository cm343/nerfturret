#!/usr/bin/env python3
"""
inference_server.py — desktop-side YOLOv8 inference server for the turret.

Run this on a machine with a GPU (a laptop/desktop). The Pi's RemoteDetector
(see detection.py) POSTs a JPEG-encoded frame to /detect and gets the
detections back as JSON, offloading the heavy YOLOv8m inference off the Pi.

Because the released weights are ~YOLOv8m, on-Pi inference is only ~3-4 FPS even
with NCNN; a desktop GPU runs them at 30+ FPS, and over a direct Gigabit
Ethernet link the round-trip adds only a few ms — far lower aim-latency overall.

    python inference_server.py                      # refined model, auto device
    MODEL=.../generalized_40_class/best.pt python inference_server.py

The model is loaded once at startup. The JSON schema matches Detection.to_dict()
so RemoteDetector.detect() can parse it directly.
"""

import json
import os
from http import server

import cv2
import numpy as np

try:
    from .detection import (
        DEFAULT_WEIGHTS, CONF_THRESHOLD, IMG_SIZE, DEVICE,
        LocalYoloDetector,
    )
except ImportError:  # allows running directly from within src/
    from detection import (
        DEFAULT_WEIGHTS, CONF_THRESHOLD, IMG_SIZE, DEVICE,
        LocalYoloDetector,
    )

# ── Configuration ────────────────────────────────────────────────────────────
HOST   = "0.0.0.0"   # bind on all interfaces so the Pi can reach it over the LAN
PORT   = 8100
MODEL  = os.environ.get("MODEL", DEFAULT_WEIGHTS)
CONF   = float(os.environ.get("CONF", CONF_THRESHOLD))
IMGSZ  = int(os.environ.get("IMGSZ", IMG_SIZE))
DEV    = os.environ.get("DEVICE", DEVICE)  # None=auto, "0"=GPU, "cpu"=CPU
# ─────────────────────────────────────────────────────────────────────────────


def _make_handler(detector: LocalYoloDetector):
    class _Handler(server.BaseHTTPRequestHandler):
        def do_POST(self) -> None:  # noqa: N802 (http.server API)
            if self.path != "/detect":
                self.send_error(404)
                return
            length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(length)
            frame = cv2.imdecode(np.frombuffer(body, dtype=np.uint8), cv2.IMREAD_COLOR)
            if frame is None:
                self.send_error(400, "could not decode JPEG body")
                return

            detections = detector.detect(frame)
            payload = json.dumps(
                {"detections": [d.to_dict() for d in detections]}
            ).encode("utf-8")

            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, *args) -> None:  # silence per-request logging
            pass

    return _Handler


def main() -> None:
    print(f"[inference] loading model: {MODEL}")
    detector = LocalYoloDetector(MODEL, conf=CONF, imgsz=IMGSZ, device=DEV)
    print(f"[inference] classes: {detector.names}")

    httpd = server.ThreadingHTTPServer((HOST, PORT), _make_handler(detector))
    httpd.daemon_threads = True
    print(f"[inference] serving on http://{HOST}:{PORT}/detect — Ctrl+C to stop")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()
        detector.close()


if __name__ == "__main__":
    main()
