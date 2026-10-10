"""Tests for src/stepper.py — pin lifecycle, ramp profile, and motion bookkeeping.

stepper.py imports RPi.GPIO at module load, which isn't installed off-Pi (see
requirements-dev.txt). We inject a fake RPi.GPIO into sys.modules before
importing stepper. The fake records every output() call so pulses and pin
levels can be asserted without hardware.
"""

import sys
import types

import pytest

STEP, DIR, EN = 16, 20, 6


@pytest.fixture
def gpio(monkeypatch):
    """A fake RPi.GPIO that logs (pin, level) for setup(initial=) and output()."""
    fake = types.ModuleType("RPi")
    g = types.ModuleType("RPi.GPIO")
    g.BCM, g.OUT, g.HIGH, g.LOW = "BCM", "OUT", 1, 0
    g._mode = None
    g.log = []
    g.level = {}

    def setup(pin, mode, initial=None, **_):
        if initial is not None:
            g.level[pin] = initial
            g.log.append((pin, initial))

    def output(pin, value):
        g.level[pin] = value
        g.log.append((pin, value))

    g.setmode = lambda m: setattr(g, "_mode", m)
    g.getmode = lambda: g._mode
    g.setup = setup
    g.output = output
    fake.GPIO = g
    monkeypatch.setitem(sys.modules, "RPi", fake)
    monkeypatch.setitem(sys.modules, "RPi.GPIO", g)
    monkeypatch.delitem(sys.modules, "stepper", raising=False)
    return g


@pytest.fixture
def stepper_mod(gpio, monkeypatch):
    import stepper  # imported under the fake GPIO

    monkeypatch.setattr(stepper.time, "sleep", lambda s: None)
    return stepper


def make(stepper_mod, **kw):
    # Very fast speeds keep the busy-wait loops short in tests.
    kw.setdefault("speed", 1e6)
    kw.setdefault("start_speed", 1e6)
    kw.setdefault("accel", 0)
    s = stepper_mod.Stepper(STEP, DIR, EN, microsteps=8, **kw)
    s.setup()
    return s


def pulses(gpio):
    return sum(1 for pin, v in gpio.log if pin == STEP and v == gpio.HIGH)


# ── lifecycle ────────────────────────────────────────────────────────────────
def test_setup_leaves_driver_disabled_and_step_low(stepper_mod, gpio):
    s = make(stepper_mod)
    assert gpio._mode == "BCM"
    assert gpio.level[EN] == gpio.HIGH      # EN is active-LOW → disabled
    assert gpio.level[STEP] == gpio.LOW
    assert not s.enabled


def test_enable_disable_drive_en_pin(stepper_mod, gpio):
    s = make(stepper_mod)
    s.enable()
    assert gpio.level[EN] == gpio.LOW and s.enabled
    s.disable()
    assert gpio.level[EN] == gpio.HIGH and not s.enabled


def test_cleanup_disables(stepper_mod, gpio):
    s = make(stepper_mod)
    s.enable()
    s.cleanup()
    assert gpio.level[EN] == gpio.HIGH
    assert gpio.level[STEP] == gpio.LOW


def test_move_before_setup_raises(stepper_mod):
    with pytest.raises(RuntimeError):
        stepper_mod.Stepper(STEP, DIR, EN).move(10)


def test_invalid_speed_rejected(stepper_mod):
    with pytest.raises(ValueError):
        stepper_mod.Stepper(STEP, DIR, EN, speed=0)


# ── motion ───────────────────────────────────────────────────────────────────
def test_move_forward_pulses_and_tracks_position(stepper_mod, gpio):
    s = make(stepper_mod)
    assert s.move(25) == 25
    assert pulses(gpio) == 25
    assert gpio.level[DIR] == gpio.HIGH
    assert gpio.level[STEP] == gpio.LOW     # every pulse ends low
    assert s.position == 25
    assert s.enabled                        # move() auto-enables


def test_move_backward_sets_dir_low(stepper_mod, gpio):
    s = make(stepper_mod)
    assert s.move(-10) == -10
    assert gpio.level[DIR] == gpio.LOW
    assert s.position == -10


def test_move_zero_does_nothing(stepper_mod, gpio):
    s = make(stepper_mod)
    assert s.move(0) == 0
    assert pulses(gpio) == 0
    assert not s.enabled


def test_stop_check_aborts_and_gets_direction(stepper_mod, gpio):
    s = make(stepper_mod)
    seen = []

    def check(direction):
        seen.append(direction)
        return len(seen) > 4                # allow 4 steps, then stop

    assert s.move(-100, stop_check=check) == -4
    assert pulses(gpio) == 4
    assert set(seen) == {-1}
    assert s.position == -4


def test_move_degrees_converts_with_microsteps(stepper_mod, gpio):
    s = make(stepper_mod)                   # 200 × 8 = 1600 µsteps/rev
    assert s.move_degrees(90) == pytest.approx(90.0)
    assert s.position == 400


# ── ramp profile ─────────────────────────────────────────────────────────────
def test_ramp_is_symmetric_and_capped(stepper_mod):
    s = make(stepper_mod, speed=1000, start_speed=100, accel=5000)
    d = s.step_delays(400)
    assert len(d) == 400
    assert d == d[::-1]                     # decel mirrors accel
    assert d[0] == pytest.approx(1 / 100)   # starts at start_speed
    assert min(d) == pytest.approx(1 / 1000)  # never faster than cruise
    assert all(a >= b for a, b in zip(d[:50], d[1:51]))  # speeding up


def test_short_move_never_reaches_cruise(stepper_mod):
    s = make(stepper_mod, speed=1000, start_speed=100, accel=5000)
    assert min(s.step_delays(10)) > 1 / 1000


def test_no_accel_is_constant_speed(stepper_mod):
    s = make(stepper_mod, speed=500, accel=0)
    assert s.step_delays(5) == [pytest.approx(1 / 500)] * 5


def test_speed_override_applies_to_one_move_only(stepper_mod):
    s = make(stepper_mod, speed=1000, start_speed=100, accel=5000)
    assert min(s.step_delays(400, speed=200)) == pytest.approx(1 / 200)
    assert s.speed == 1000


def test_speed_below_start_speed_runs_flat(stepper_mod):
    s = make(stepper_mod, speed=1000, start_speed=100, accel=5000)
    assert s.step_delays(20, speed=40) == [pytest.approx(1 / 40)] * 20
