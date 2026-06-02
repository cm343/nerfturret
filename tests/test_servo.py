"""Tests for src/servo.py — duty/angle mapping, motion bookkeeping, and the ABC.

servo.py imports RPi.GPIO at module load, which isn't installed off-Pi (see
requirements-dev.txt). We inject a fake RPi.GPIO into sys.modules before
importing servo, mirroring how test_detection.py keeps the suite dep-free. The
fake records the duty cycles pushed to PWM so motion can be asserted without
hardware.
"""

import sys
import types

import pytest


class _FakePWM:
    def __init__(self, pin, freq):
        self.pin = pin
        self.freq = freq
        self.duties = []        # every ChangeDutyCycle/start value, in order
        self.running = False

    def start(self, duty):
        self.running = True
        self.duties.append(duty)

    def ChangeDutyCycle(self, duty):  # noqa: N802 (RPi.GPIO API)
        self.duties.append(duty)

    def stop(self):
        self.running = False


@pytest.fixture
def servo_mod(monkeypatch):
    """Import src/servo with a fake RPi.GPIO injected; return the module."""
    fake = types.ModuleType("RPi")
    gpio = types.ModuleType("RPi.GPIO")
    gpio.BCM = "BCM"
    gpio.OUT = "OUT"
    gpio._mode = None
    gpio.setmode = lambda m: setattr(gpio, "_mode", m)
    gpio.getmode = lambda: gpio._mode
    gpio.setup = lambda *a, **k: None
    gpio.cleanup = lambda *a, **k: None
    gpio.PWM = _FakePWM
    fake.GPIO = gpio
    monkeypatch.setitem(sys.modules, "RPi", fake)
    monkeypatch.setitem(sys.modules, "RPi.GPIO", gpio)
    monkeypatch.delitem(sys.modules, "servo", raising=False)

    import servo  # imported under the fake GPIO
    return servo


# ── PositionalServo: angle ↔ duty mapping ─────────────────────────────────────
def test_angle_to_duty_endpoints_and_midpoint(servo_mod):
    s = servo_mod.PositionalServo(22, 150, duty_min=2.5, duty_max=12.5)
    assert s._angle_to_duty(0) == pytest.approx(2.5)
    assert s._angle_to_duty(180) == pytest.approx(12.5)
    assert s._angle_to_duty(90) == pytest.approx(7.5)


def test_angle_to_duty_clamps_out_of_range(servo_mod):
    s = servo_mod.PositionalServo(22, 150, duty_min=2.5, duty_max=12.5)
    assert s._angle_to_duty(-30) == pytest.approx(2.5)   # clamped to 0°
    assert s._angle_to_duty(999) == pytest.approx(12.5)  # clamped to 180°


def test_initial_angle_is_clamped(servo_mod):
    assert servo_mod.PositionalServo(22, 150, initial_angle=-10).angle == 0.0
    assert servo_mod.PositionalServo(22, 150, initial_angle=500).angle == 180.0


# ── PositionalServo: motion bookkeeping ───────────────────────────────────────
def test_move_to_updates_angle_and_lands_exactly(servo_mod):
    s = servo_mod.PositionalServo(22, 1e6)  # huge rate -> ~no sleeping
    s.start()
    s.move_to(45)
    assert s.angle == pytest.approx(45.0)
    # Last duty pushed corresponds to the final (45°) position.
    assert s._pwm.duties[-1] == pytest.approx(s._angle_to_duty(45))


def test_move_to_clamps_target(servo_mod):
    s = servo_mod.PositionalServo(22, 1e6)
    s.start()
    s.move_to(300)
    assert s.angle == pytest.approx(180.0)


def test_move_by_is_relative(servo_mod):
    s = servo_mod.PositionalServo(22, 1e6, initial_angle=20)
    s.start()
    s.move_by(15)
    assert s.angle == pytest.approx(35.0)
    s.move_by(-50)               # 35 - 50 = -15 -> clamps to 0
    assert s.angle == pytest.approx(0.0)


def test_move_before_start_raises(servo_mod):
    with pytest.raises(RuntimeError):
        servo_mod.PositionalServo(22, 150).move_to(45)


def test_start_emits_initial_position_duty(servo_mod):
    s = servo_mod.PositionalServo(22, 150, initial_angle=0)
    s.start()
    assert s._pwm.duties[0] == pytest.approx(s._angle_to_duty(0))


# ── ContinuousServo: speed ↔ duty regression ──────────────────────────────────
def test_speed_to_duty_mapping(servo_mod):
    s = servo_mod.ContinuousServo(
        17, 300, duty_stop=7.1, duty_full_cw=12.5, duty_full_ccw=2.5
    )
    assert s._speed_to_duty(0) == pytest.approx(7.1)
    assert s._speed_to_duty(1.0) == pytest.approx(12.5)
    assert s._speed_to_duty(-1.0) == pytest.approx(2.5)
    assert s._speed_to_duty(0.5) == pytest.approx(7.1 + 0.5 * (12.5 - 7.1))
    assert s._speed_to_duty(5) == pytest.approx(12.5)   # clamped to +1


# ── base class / alias ────────────────────────────────────────────────────────
def test_base_servo_is_abstract(servo_mod):
    with pytest.raises(TypeError):
        servo_mod.BaseServo(17, 300)


def test_servo_alias_is_continuous(servo_mod):
    assert servo_mod.Servo is servo_mod.ContinuousServo


def test_deg_per_sec_must_be_positive(servo_mod):
    with pytest.raises(ValueError):
        servo_mod.PositionalServo(22, 0)
