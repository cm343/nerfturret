"""Tests for src/detection.py — Detector backends, factory, and result parsing.

These run off-Pi without torch/ultralytics: the local path is exercised through
its static result-parser with a lightweight fake, and the remote path through a
monkeypatched urllib.
"""

import io
import json
import urllib.error

import numpy as np
import pytest


def _frame():
    return np.zeros((48, 64, 3), dtype=np.uint8)


# ── Detection dataclass ──────────────────────────────────────────────────────
def test_detection_derives_center_when_not_given():
    from detection import Detection

    d = Detection(cls_id=1, label="drone", conf=0.9, bbox=(10, 20, 30, 40))
    assert d.cx == 20.0 and d.cy == 30.0


def test_detection_roundtrips_through_dict():
    from detection import Detection

    d = Detection(cls_id=2, label="bird", conf=0.5, bbox=(0, 0, 8, 6))
    back = Detection.from_dict(d.to_dict())
    assert back == d


# ── NullDetector ─────────────────────────────────────────────────────────────
def test_null_detector_finds_nothing():
    from detection import NullDetector

    assert NullDetector().detect(_frame()) == []


# ── factory ──────────────────────────────────────────────────────────────────
def test_create_detector_null_is_default():
    from detection import NullDetector, create_detector

    assert isinstance(create_detector(None), NullDetector)
    assert isinstance(create_detector("null"), NullDetector)


def test_create_detector_selects_remote():
    from detection import RemoteDetector, create_detector

    assert isinstance(create_detector("remote"), RemoteDetector)


def test_create_detector_local_uses_lazy_import(monkeypatch):
    import detection

    sentinel = object()
    monkeypatch.setattr(detection, "LocalYoloDetector", lambda *a, **k: sentinel)
    assert detection.create_detector("local") is sentinel


def test_create_detector_rejects_unknown():
    from detection import create_detector

    with pytest.raises(ValueError):
        create_detector("bogus")


# ── LocalYoloDetector result parsing (no ultralytics needed) ──────────────────
class _FakeBox:
    def __init__(self, cls_id, conf, xyxy):
        self.cls = [cls_id]
        self.conf = [conf]
        self.xyxy = [xyxy]


class _FakeResult:
    def __init__(self, boxes):
        self.boxes = boxes


def test_results_to_detections_converts_boxes():
    from detection import LocalYoloDetector

    results = [_FakeResult([_FakeBox(0, 0.9, (10, 10, 30, 50))])]
    dets = LocalYoloDetector._results_to_detections(results, {0: "drone"}, None)
    assert len(dets) == 1
    d = dets[0]
    assert d.cls_id == 0 and d.label == "drone"
    assert d.conf == pytest.approx(0.9)
    assert d.bbox == (10.0, 10.0, 30.0, 50.0)
    assert (d.cx, d.cy) == (20.0, 30.0)


def test_results_to_detections_applies_class_filter():
    from detection import LocalYoloDetector

    results = [_FakeResult([_FakeBox(0, 0.9, (0, 0, 4, 4)),
                            _FakeBox(2, 0.8, (1, 1, 5, 5))])]
    dets = LocalYoloDetector._results_to_detections(results, {0: "a", 2: "c"}, {2})
    assert [d.cls_id for d in dets] == [2]


# ── RemoteDetector (monkeypatched HTTP) ───────────────────────────────────────
def _fake_response(payload: dict):
    class _Resp:
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def read(self): return json.dumps(payload).encode("utf-8")

    return _Resp()


def test_remote_detector_parses_json(monkeypatch):
    import detection

    payload = {"detections": [
        {"cls_id": 1, "label": "drone", "conf": 0.77,
         "bbox": [10, 20, 30, 40], "cx": 20, "cy": 30},
    ]}
    monkeypatch.setattr(
        detection.urllib.request, "urlopen",
        lambda req, timeout=None: _fake_response(payload),
    )
    dets = detection.RemoteDetector().detect(_frame())
    assert len(dets) == 1 and dets[0].label == "drone"
    assert dets[0].conf == pytest.approx(0.77)


def test_remote_detector_returns_empty_on_network_error(monkeypatch):
    import detection

    def boom(req, timeout=None):
        raise urllib.error.URLError("server down")

    monkeypatch.setattr(detection.urllib.request, "urlopen", boom)
    assert detection.RemoteDetector().detect(_frame()) == []  # no exception raised
