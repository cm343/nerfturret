#!/usr/bin/env python3
"""
Detector — pluggable flying-object detection for the CV pipeline.

This mirrors the pluggable FrameSource pattern in vision.py: a small base class
with concrete implementations selected at runtime by create_detector(). Each
detector turns a BGR frame into a list of Detections; the vision loop picks a
target from them and draws boxes.

Three backends:

  NullDetector   — returns nothing. The default; pulls no heavy deps, so this
                   module (and vision.py) import fine off-Pi and the test suite
                   stays torch-free.
  LocalYoloDetector — runs an Ultralytics YOLOv8 model on this machine. Loads a
                   weights path format-agnostically: YOLO() reads .pt/.onnx/.ncnn
                   transparently, so the Pi can point at an exported NCNN model
                   and a desktop at the .pt with no code change.
  RemoteDetector — offloads inference to a desktop GPU running inference_server.py.
                   JPEG-encodes the frame, POSTs it, parses detections back. Uses
                   only the stdlib + cv2 (already a dependency), so the Pi needs
                   no torch for this path.

    det = create_detector()            # backend chosen by DETECTOR_BACKEND
    for d in det.detect(frame):
        ...                            # d.bbox, d.cx, d.cy, d.conf, d.label
    det.close()
"""

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass

import cv2
import numpy as np

# ── Configuration ────────────────────────────────────────────────────────────
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_WEIGHTS_DIR = os.path.join(
    _REPO_ROOT, "Real-Time-Flying-Object-Detection_with_YOLOv8", "weights"
)
WEIGHTS_REFINED     = os.path.join(_WEIGHTS_DIR, "refined_3class", "best.pt")
WEIGHTS_GENERALIZED = os.path.join(_WEIGHTS_DIR, "generalized_40_class", "best.pt")
DEFAULT_WEIGHTS     = WEIGHTS_REFINED   # refined = SOTA flying-object model

DETECTOR_BACKEND = "remote"   # "null" | "local" | "remote"
CONF_THRESHOLD   = 0.5      # minimum detection confidence to keep
IMG_SIZE         = 320      # inference size; smaller = faster (esp. on the Pi)
DEVICE           = None     # None=auto, 0=GPU, "cpu"=CPU (LocalYoloDetector)
CLASS_FILTER     = None     # None=all classes, or an iterable of class ids to keep

REMOTE_URL       = "http://192.168.178.23:8100/detect"  # inference_server.py endpoint
REMOTE_TIMEOUT_S = 1.0      # per-request timeout; on timeout we report no target
REMOTE_JPEG_QUALITY = 80    # quality of the frame uploaded for inference
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class Detection:
    """One detected object in image (pixel) coordinates."""
    cls_id: int
    label: str
    conf: float
    bbox: tuple[float, float, float, float]  # (x1, y1, x2, y2)
    cx: float = 0.0   # box center x (pixels)
    cy: float = 0.0   # box center y (pixels)

    def __post_init__(self) -> None:
        x1, y1, x2, y2 = self.bbox
        # Center may be passed explicitly (e.g. from JSON); otherwise derive it.
        if self.cx == 0.0 and self.cy == 0.0:
            self.cx = (x1 + x2) / 2.0
            self.cy = (y1 + y2) / 2.0

    def to_dict(self) -> dict:
        return {
            "cls_id": self.cls_id,
            "label": self.label,
            "conf": self.conf,
            "bbox": list(self.bbox),
            "cx": self.cx,
            "cy": self.cy,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Detection":
        x1, y1, x2, y2 = d["bbox"]
        return cls(
            cls_id=int(d["cls_id"]),
            label=str(d["label"]),
            conf=float(d["conf"]),
            bbox=(float(x1), float(y1), float(x2), float(y2)),
            cx=float(d.get("cx", 0.0)),
            cy=float(d.get("cy", 0.0)),
        )


# ── Detector backends ────────────────────────────────────────────────────────
class Detector:
    """Base class: detect() returns a list of Detections; close() releases it."""

    def detect(self, frame: "np.ndarray") -> list[Detection]:
        raise NotImplementedError

    def close(self) -> None:
        pass


class NullDetector(Detector):
    """No-op detector — always finds nothing. Default backend, no heavy deps."""

    def detect(self, frame: "np.ndarray") -> list[Detection]:
        return []


class LocalYoloDetector(Detector):
    """Ultralytics YOLOv8 inference on the local machine (lazy import of torch)."""

    def __init__(
        self,
        weights: str = DEFAULT_WEIGHTS,
        *,
        conf: float = CONF_THRESHOLD,
        imgsz: int = IMG_SIZE,
        device=DEVICE,
        class_filter=CLASS_FILTER,
    ) -> None:
        from ultralytics import YOLO  # lazy: only needed where YOLO actually runs

        self.conf = conf
        self.imgsz = imgsz
        self.device = device
        self.class_filter = set(class_filter) if class_filter is not None else None
        self._model = YOLO(weights)  # loads .pt / .onnx / .ncnn transparently
        self.names = self._model.names  # {id: label}

    def detect(self, frame: "np.ndarray") -> list[Detection]:
        results = self._model.predict(
            source=frame,
            conf=self.conf,
            imgsz=self.imgsz,
            device=self.device,
            verbose=False,
        )
        return self._results_to_detections(results, self.names, self.class_filter)

    @staticmethod
    def _results_to_detections(results, names, class_filter) -> list[Detection]:
        """Convert an Ultralytics Results list into our Detection list.

        Kept static and free of any torch/ultralytics types so it can be unit
        tested with a lightweight fake results object.
        """
        detections: list[Detection] = []
        for result in results:
            boxes = getattr(result, "boxes", None)
            if boxes is None:
                continue
            for box in boxes:
                cls_id = int(box.cls[0])
                if class_filter is not None and cls_id not in class_filter:
                    continue
                x1, y1, x2, y2 = (float(v) for v in box.xyxy[0])
                detections.append(
                    Detection(
                        cls_id=cls_id,
                        label=str(names.get(cls_id, cls_id)),
                        conf=float(box.conf[0]),
                        bbox=(x1, y1, x2, y2),
                    )
                )
        return detections


class RemoteDetector(Detector):
    """Offload inference to a desktop running inference_server.py over HTTP."""

    def __init__(
        self,
        url: str = REMOTE_URL,
        *,
        timeout: float = REMOTE_TIMEOUT_S,
        jpeg_quality: int = REMOTE_JPEG_QUALITY,
    ) -> None:
        self.url = url
        self.timeout = timeout
        self.jpeg_quality = jpeg_quality

    def detect(self, frame: "np.ndarray") -> list[Detection]:
        ok, jpeg = cv2.imencode(
            ".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality]
        )
        if not ok:
            return []
        req = urllib.request.Request(
            self.url,
            data=jpeg.tobytes(),
            headers={"Content-Type": "image/jpeg"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
        except (urllib.error.URLError, OSError, ValueError) as exc:
            # Network hiccup / server down / bad JSON: degrade to "no target"
            # rather than crashing the turret's vision loop.
            print(f"[detection] remote inference failed ({exc}); no detections")
            return []
        return [Detection.from_dict(d) for d in payload.get("detections", [])]


def create_detector(
    spec: str | None = DETECTOR_BACKEND,
    *,
    weights: str = DEFAULT_WEIGHTS,
) -> Detector:
    """Build a detector from a backend spec.

    None / "null" → NullDetector (no deps, finds nothing)
    "local"       → LocalYoloDetector (Ultralytics on this machine)
    "remote"      → RemoteDetector (offload to inference_server.py)
    """
    if spec is None or spec == "null":
        return NullDetector()
    if spec == "local":
        return LocalYoloDetector(weights)
    if spec == "remote":
        return RemoteDetector()
    raise ValueError(f"unknown detector backend: {spec!r}")
