"""Tests for src/homing.py — soft limits and calibration persistence."""

import json

import pytest

import homing


@pytest.mark.parametrize("pos, steps, expected", [
    (0, 50, 50),          # inside, unclamped
    (90, 50, 10),         # shortened at max
    (-90, -50, -10),      # shortened at min
    (100, 5, 0),          # at max, refuse further
    (150, 10, 0),         # beyond max, never further out
    (150, -20, -20),      # beyond max, moving back is fine
    (-150, -10, 0),       # beyond min, never further out
    (-150, 10, 10),       # beyond min, small move back is not stretched
])
def test_clamp_steps(pos, steps, expected):
    cal = {"min_steps": -100, "max_steps": 100}
    assert homing.clamp_steps(pos, steps, cal) == expected


def test_clamp_with_one_sided_limit():
    assert homing.clamp_steps(0, 500, {"max_steps": 100}) == 100
    assert homing.clamp_steps(0, -500, {"max_steps": 100}) == -500


def test_clamp_without_limits_is_a_no_op():
    assert homing.clamp_steps(0, 12345, {}) == 12345


def test_calibration_round_trip(tmp_path):
    path = tmp_path / "sub" / "calibration.json"
    c = homing.Calibration(str(path))
    assert c.data == {}
    c.axis("tilt").update(min_steps=-400, max_steps=800)
    c.save()

    assert json.loads(path.read_text())["tilt"] == {"min_steps": -400, "max_steps": 800}
    assert homing.Calibration(str(path)).axis("tilt")["max_steps"] == 800
    assert [p.name for p in path.parent.iterdir()] == ["calibration.json"]  # no temp left
