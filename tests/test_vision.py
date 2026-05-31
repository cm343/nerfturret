"""Tests for src/vision.py — ComputerVision pipeline, state, and frame sources."""

import time
import urllib.request

import numpy as np
import pytest


def _fake_source_cls():
    from vision import FrameSource

    class FakeSource(FrameSource):
        """Synthetic source: a moving solid-color frame; no camera needed."""

        def __init__(self):
            self.i = 0
            self.closed = False

        def read(self):
            self.i = (self.i + 7) % 256
            return np.full((48, 64, 3), self.i, dtype=np.uint8)

        def close(self):
            self.closed = True

    return FakeSource


@pytest.fixture
def cv_module(free_port):
    from streaming import StreamManager
    from vision import ComputerVision

    source = _fake_source_cls()()
    mgr = StreamManager(host="127.0.0.1", base_port=free_port())
    cv = ComputerVision(source=source, stream_manager=mgr, target_fps=120)
    cv._test_source = source  # expose for assertions
    yield cv
    cv.stop()
    mgr.close_all()


# ── pipeline wiring ──────────────────────────────────────────────────────────
def test_each_stage_gets_its_own_stream(cv_module):
    names = set(cv_module.streams.streams)
    assert names == {"input", "annotated"}
    ports = [h.port for h in cv_module._stage_streams.values()]
    assert len(set(ports)) == len(ports) == 2  # distinct ports per stage


def test_annotate_stage_modifies_a_copy():
    from vision import ComputerVision

    blank = np.zeros((40, 40, 3), dtype=np.uint8)
    out = ComputerVision._stage_annotate(blank)
    assert out.shape == blank.shape
    assert out.sum() > 0          # crosshair was drawn
    assert blank.sum() == 0       # original frame left untouched


def test_pipeline_publishes_to_annotated_stream(cv_module):
    cv_module.start()
    time.sleep(0.25)
    url = cv_module._stage_streams["annotated"].url + "stream.mjpg"
    data = urllib.request.urlopen(url, timeout=3).read(4096)
    assert b"\xff\xd8" in data     # JPEG frames flowing


# ── state getter ─────────────────────────────────────────────────────────────
def test_get_state_returns_targetstate_after_start(cv_module):
    from vision import TargetState

    cv_module.start()
    time.sleep(0.2)
    st = cv_module.get_state()
    assert isinstance(st, TargetState)
    assert st.found is False
    assert st.timestamp > 0


def test_get_state_returns_independent_copies(cv_module):
    a = cv_module.get_state()
    b = cv_module.get_state()
    assert a is not b              # defensive copy each call


def test_stop_closes_the_source(cv_module):
    cv_module.start()
    time.sleep(0.1)
    cv_module.stop()
    assert cv_module._test_source.closed is True


# ── frame sources ────────────────────────────────────────────────────────────
def test_opencv_source_invalid_device_raises():
    from vision import OpenCVSource

    with pytest.raises(RuntimeError):
        OpenCVSource("definitely_not_a_real_file_98765.mp4")


def test_create_frame_source_falls_back_when_no_picam(monkeypatch):
    import vision

    def boom(*args, **kwargs):
        raise RuntimeError("no picamera2 / no Pi camera")

    sentinel = object()
    monkeypatch.setattr(vision, "PiCameraSource", boom)
    monkeypatch.setattr(vision, "OpenCVSource", lambda *a, **k: sentinel)

    assert vision.create_frame_source("picam") is sentinel


def test_create_frame_source_explicit_device_uses_opencv(monkeypatch):
    import vision

    captured = {}

    def fake_opencv(device, resolution):
        captured["device"] = device
        return "opencv-source"

    monkeypatch.setattr(vision, "OpenCVSource", fake_opencv)
    result = vision.create_frame_source(3)
    assert result == "opencv-source"
    assert captured["device"] == 3


# ── detector integration ─────────────────────────────────────────────────────
def test_select_target_empty_means_not_found():
    from vision import ComputerVision

    st = ComputerVision._select_target([], (48, 64, 3))
    assert st.found is False
    assert (st.x, st.y) == (0.0, 0.0)
    assert st.timestamp > 0


def test_select_target_picks_highest_conf_and_normalizes():
    from detection import Detection
    from vision import ComputerVision

    # Centered low-conf box vs. bottom-right high-conf box in a 64x48 frame.
    centered = Detection(cls_id=0, label="a", conf=0.4, bbox=(28, 20, 36, 28))
    corner = Detection(cls_id=1, label="b", conf=0.9, bbox=(56, 40, 64, 48))
    st = ComputerVision._select_target([centered, corner], (48, 64, 3))

    assert st.found is True
    # corner center is (60, 44): right of and below center -> positive x and y.
    assert st.x == pytest.approx((60 - 32) / 32)
    assert st.y == pytest.approx((44 - 24) / 24)


def test_annotate_draws_boxes_in_addition_to_crosshair():
    from detection import Detection
    from vision import ComputerVision

    blank = np.zeros((48, 64, 3), dtype=np.uint8)
    crosshair_only = ComputerVision._stage_annotate(blank)
    with_box = ComputerVision._stage_annotate(
        blank, [Detection(cls_id=0, label="x", conf=0.9, bbox=(5, 5, 40, 30))]
    )
    assert with_box.sum() > crosshair_only.sum()  # box + label added pixels


def test_pipeline_uses_injected_detector(free_port):
    from detection import Detection, Detector
    from streaming import StreamManager
    from vision import ComputerVision

    class OneTargetDetector(Detector):
        def detect(self, frame):
            h, w = frame.shape[:2]
            return [Detection(cls_id=0, label="drone", conf=0.9,
                              bbox=(w - 8, h - 6, w, h))]  # bottom-right corner

    source = _fake_source_cls()()
    mgr = StreamManager(host="127.0.0.1", base_port=free_port())
    cv = ComputerVision(source=source, detector=OneTargetDetector(),
                        stream_manager=mgr, target_fps=120)
    try:
        cv.start()
        time.sleep(0.2)
        st = cv.get_state()
        assert st.found is True
        assert st.x > 0 and st.y > 0  # target down-and-right of center
    finally:
        cv.stop()
        mgr.close_all()
