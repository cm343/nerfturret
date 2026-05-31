#!/usr/bin/env python3
"""
ComputerVision — capture frames, run an OpenCV pipeline, stream the stages.

The module pulls frames from a pluggable source (Picamera2 on a Raspberry Pi,
or any cv2.VideoCapture device/file off-Pi), runs them through a small pipeline,
and publishes each stage's image to its own MJPEG stream (one port per stage,
managed by StreamManager). Information extracted from the frames is kept in
module state (a TargetState) and retrievable via get_state().

The pipeline stages and the detection step are deliberate stubs for now: stages
just forward/lightly annotate the image, and detection returns a placeholder.
Replace _build_pipeline() and _detect() with real processing later.

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
except ImportError:  # allows running directly from within src/
    from streaming import StreamManager

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
        config = self._picam.create_video_configuration(
            main={"size": tuple(resolution), "format": "RGB888"}
        )
        self._picam.configure(config)
        self._picam.start()

    def read(self) -> "np.ndarray | None":
        rgb = self._picam.capture_array()
        # picamera2 gives RGB; OpenCV works in BGR
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

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
        stream_manager: StreamManager | None = None,
        resolution=CAPTURE_RESOLUTION,
        target_fps: int = TARGET_FPS,
    ) -> None:
        self.resolution = resolution
        self.target_fps = target_fps

        self._source = source if source is not None else create_frame_source(resolution=resolution)
        self._owns_manager = stream_manager is None
        self.streams = stream_manager if stream_manager is not None else StreamManager()

        # One stream (its own port) per pipeline stage.
        self._stage_streams = {name: self.streams.create_stream(name) for name in STAGE_NAMES}
        self._pipeline = self._build_pipeline()

        self._state = TargetState()
        self._state_lock = threading.Lock()
        self._thread = threading.Thread(target=self._loop, name="vision", daemon=True)
        self._running = threading.Event()

    # ── stubs to replace with real processing ─────────────────────────────────
    def _build_pipeline(self):
        """Return an ordered list of (stage_name, stage_fn).

        Each stage_fn takes a BGR frame and returns a BGR frame to publish.
        Stubs for now: 'input' forwards the raw frame; 'annotated' draws a center
        crosshair to demonstrate modifying the image.
        """
        return [
            ("input", lambda frame: frame),
            ("annotated", self._stage_annotate),
        ]

    @staticmethod
    def _stage_annotate(frame: "np.ndarray") -> "np.ndarray":
        out = frame.copy()
        h, w = out.shape[:2]
        cx, cy = w // 2, h // 2
        color = (0, 255, 0)
        cv2.line(out, (cx - 15, cy), (cx + 15, cy), color, 1)
        cv2.line(out, (cx, cy - 15), (cx, cy + 15), color, 1)
        return out

    def _detect(self, frame: "np.ndarray") -> TargetState:
        """Stub detector — returns a placeholder TargetState (no target found)."""
        return TargetState(found=False, x=0.0, y=0.0, timestamp=time.time())

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

            for name, stage_fn in self._pipeline:
                out = stage_fn(frame)
                self._stage_streams[name].publish(out)

            state = self._detect(frame)
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
