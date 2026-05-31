#!/usr/bin/env python3
"""
ComputerVision — capture frames, run an OpenCV pipeline, stream the stages.

The module pulls frames from a pluggable source (Picamera2 on a Raspberry Pi,
or any cv2.VideoCapture device/file off-Pi), runs each through a pluggable
Detector (see detection.py), and publishes the raw frame and an annotated frame
(boxes drawn) to their own MJPEG streams (one port per stage, managed by
StreamManager). The chosen target is kept in module state (a TargetState) as a
normalized offset from frame center, retrievable via get_state().

The detector backend is selected by detection.DETECTOR_BACKEND: "null" (default,
finds nothing — keeps this importable with no torch), "local" (Ultralytics on
this machine), or "remote" (offload to a desktop running inference_server.py).

    cv = ComputerVision()
    cv.start()
    ...
    state = cv.get_state()   # latest TargetState
    cv.stop()
"""

import threading
import time
from dataclasses import dataclass, replace

import cv2
import numpy as np

try:
    from .streaming import StreamManager
    from .detection import Detector, create_detector
except ImportError:  # allows running directly from within src/
    from streaming import StreamManager
    from detection import Detector, create_detector

# ── Configuration ────────────────────────────────────────────────────────────
CAPTURE_RESOLUTION = (640, 480)   # (width, height)
TARGET_FPS         = 30           # processing-loop rate cap
STAGE_NAMES        = ("input", "annotated")  # each streamed on its own port
CAMERA_SOURCE      = None         # None/"picam" -> Pi cam (fallback webcam); int/str -> cv2.VideoCapture
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class TargetState:
    """Latest information extracted from the frames (for aiming)."""
    found: bool = False
    x: float = 0.0          # normalized horizontal offset from center, [-1, 1]
    y: float = 0.0          # normalized vertical offset from center, [-1, 1]
    timestamp: float = 0.0  # time.time() when this state was produced


# ── Frame sources (pluggable) ────────────────────────────────────────────────
class FrameSource:
    """Base class: read() returns a BGR ndarray (or None), close() releases it."""

    def read(self) -> "np.ndarray | None":
        raise NotImplementedError

    def close(self) -> None:
        pass


class PiCameraSource(FrameSource):
    """Raspberry Pi camera via picamera2 (imported lazily so this file loads off-Pi)."""

    def __init__(self, resolution=CAPTURE_RESOLUTION) -> None:
        from picamera2 import Picamera2  # lazy: only needed on the Pi

        self._picam = Picamera2()
        # picamera2's "RGB888" actually delivers a numpy array in BGR channel
        # order — exactly what OpenCV expects — so no colour conversion is needed.
        config = self._picam.create_video_configuration(
            main={"size": tuple(resolution), "format": "RGB888"}
        )
        self._picam.configure(config)
        self._picam.start()

    def read(self) -> "np.ndarray | None":
        # Already BGR for OpenCV (see note in __init__); return as-is.
        return self._picam.capture_array()

    def close(self) -> None:
        self._picam.stop()


class OpenCVSource(FrameSource):
    """Any cv2.VideoCapture source: webcam index, video file, or stream URL."""

    def __init__(self, device=0, resolution=CAPTURE_RESOLUTION) -> None:
        self._cap = cv2.VideoCapture(device)
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[0])
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])
        if not self._cap.isOpened():
            raise RuntimeError(f"could not open video source: {device!r}")

    def read(self) -> "np.ndarray | None":
        ok, frame = self._cap.read()
        return frame if ok else None

    def close(self) -> None:
        self._cap.release()


def create_frame_source(spec=CAMERA_SOURCE, resolution=CAPTURE_RESOLUTION) -> FrameSource:
    """Build a frame source from a spec.

    None / "picam" → Picamera2 on the Pi, falling back to webcam 0 elsewhere.
    int / str      → cv2.VideoCapture(spec) (webcam index, file path, or URL).
    """
    if spec is None or spec == "picam":
        try:
            return PiCameraSource(resolution)
        except Exception as exc:  # noqa: BLE001 — picamera2 missing / no Pi cam
            print(f"[vision] Pi camera unavailable ({exc}); falling back to webcam 0")
            return OpenCVSource(0, resolution)
    return OpenCVSource(spec, resolution)


# ── Pipeline ─────────────────────────────────────────────────────────────────
class ComputerVision:
    """Threaded capture → pipeline → stream loop with retrievable target state."""

    def __init__(
        self,
        *,
        source: FrameSource | None = None,
        detector: Detector | None = None,
        stream_manager: StreamManager | None = None,
        resolution=CAPTURE_RESOLUTION,
        target_fps: int = TARGET_FPS,
    ) -> None:
        self.resolution = resolution
        self.target_fps = target_fps

        self._source = source if source is not None else create_frame_source(resolution=resolution)
        self._detector = detector if detector is not None else create_detector()
        self._owns_manager = stream_manager is None
        self.streams = stream_manager if stream_manager is not None else StreamManager()

        # One stream (its own port) per pipeline stage.
        self._stage_streams = {name: self.streams.create_stream(name) for name in STAGE_NAMES}

        self._state = TargetState()
        self._state_lock = threading.Lock()
        self._thread = threading.Thread(target=self._loop, name="vision", daemon=True)
        self._running = threading.Event()

    # ── detection → state / annotation ─────────────────────────────────────────
    @staticmethod
    def _stage_annotate(frame: "np.ndarray", detections=()) -> "np.ndarray":
        """Draw detection boxes + labels and a center crosshair on a copy."""
        out = frame.copy()
        h, w = out.shape[:2]

        for det in detections:
            x1, y1, x2, y2 = (int(v) for v in det.bbox)
            cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                out, f"{det.label} {det.conf * 100:.0f}%", (x1, max(y1 - 6, 10)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA,
            )

        cx, cy = w // 2, h // 2
        color = (0, 255, 0)
        cv2.line(out, (cx - 15, cy), (cx + 15, cy), color, 1)
        cv2.line(out, (cx, cy - 15), (cx, cy + 15), color, 1)
        return out

    @staticmethod
    def _select_target(detections, shape) -> TargetState:
        """Pick the highest-confidence detection and express it as a TargetState.

        The offset is normalized to [-1, 1] relative to the frame center: x is
        positive to the right, y positive downward.
        """
        if not detections:
            return TargetState(found=False, x=0.0, y=0.0, timestamp=time.time())

        target = max(detections, key=lambda d: d.conf)
        h, w = shape[:2]
        x = (target.cx - w / 2.0) / (w / 2.0)
        y = (target.cy - h / 2.0) / (h / 2.0)
        return TargetState(found=True, x=x, y=y, timestamp=time.time())

    # ── lifecycle ──────────────────────────────────────────────────────────────
    def start(self) -> None:
        for name, handler in self._stage_streams.items():
            print(f"[vision] stage '{name}' streaming at {handler.url}")
        self._running.set()
        self._thread.start()

    def stop(self) -> None:
        self._running.clear()
        if self._thread.is_alive():
            self._thread.join(timeout=2.0)
        self._source.close()
        self._detector.close()
        if self._owns_manager:
            self.streams.close_all()

    # ── state getter ───────────────────────────────────────────────────────────
    def get_state(self) -> TargetState:
        """Return a copy of the latest extracted target state (thread-safe)."""
        with self._state_lock:
            return replace(self._state)

    # ── worker loop ──────────────────────────────────────────────────────────
    def _loop(self) -> None:
        period = 1.0 / self.target_fps if self.target_fps > 0 else 0.0
        while self._running.is_set():
            tick = time.time()
            frame = self._source.read()
            if frame is None:
                time.sleep(0.01)
                continue

            # Detect once per frame; feed both the annotated stream and the state.
            detections = self._detector.detect(frame)
            self._stage_streams["input"].publish(frame)
            self._stage_streams["annotated"].publish(
                self._stage_annotate(frame, detections)
            )

            state = self._select_target(detections, frame.shape)
            with self._state_lock:
                self._state = state

            elapsed = time.time() - tick
            if period and elapsed < period:
                time.sleep(period - elapsed)


def main() -> None:
    cv_module = ComputerVision()
    cv_module.start()
    print("[vision] running — Ctrl+C to stop")
    try:
        while True:
            time.sleep(1.0)
            print("  state:", cv_module.get_state())
    except KeyboardInterrupt:
        pass
    finally:
        cv_module.stop()


if __name__ == "__main__":
    main()
