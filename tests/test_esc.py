"""Tests for src/esc.py — arming lifecycle, throttle mapping/clamping, ramping,
and the sysfs hardware-PWM backend.

ESC only talks to an `output` object with set_pulse_us(pin, us), so a recording
fake stands in for lgpio. The hardware-PWM backend is pointed at a fake
/sys/class/pwm tree under tmp_path.
"""

import os

import pytest

PIN = 12


class FakeOutput:
    def __init__(self):
        self.log = []

    def set_pulse_us(self, pin, us):
        self.log.append((pin, us))

    @property
    def last(self):
        return self.log[-1][1]


@pytest.fixture
def esc_mod(monkeypatch):
    import esc
    monkeypatch.setattr(esc.time, "sleep", lambda s: None)
    return esc


@pytest.fixture
def out():
    return FakeOutput()


def make(esc_mod, out, armed=True, **kw):
    e = esc_mod.ESC(out, PIN, **kw)
    e.setup()
    if armed:
        e.arm()
    return e


# ── lifecycle ────────────────────────────────────────────────────────────────
def test_setup_sends_stop_pulse_first(esc_mod, out):
    make(esc_mod, out, armed=False)
    assert out.log == [(PIN, 1000)]


def test_throttle_requires_setup_and_arm(esc_mod, out):
    e = esc_mod.ESC(out, PIN)
    with pytest.raises(RuntimeError):
        e.set_throttle(0.1)
    e.setup()
    with pytest.raises(RuntimeError):
        e.set_throttle(0.1)
    e.arm()
    e.set_throttle(0.1)
    assert out.last == 1100


def test_stop_and_cleanup(esc_mod, out):
    e = make(esc_mod, out)
    e.set_throttle(0.5)
    e.stop()
    assert out.last == 1000 and e.throttle == 0
    e.set_throttle(0.5)
    e.cleanup()
    assert out.log[-2:] == [(PIN, 1000), (PIN, 0)]
    assert not e.armed
    e.cleanup()  # idempotent
    assert out.log[-1] == (PIN, 0)


@pytest.mark.parametrize("kw", [dict(min_us=2000, max_us=1000), dict(max_throttle=0),
                                dict(max_throttle=1.5), dict(max_us=3000)])
def test_rejects_bad_config(esc_mod, out, kw):
    with pytest.raises(ValueError):
        esc_mod.ESC(out, PIN, **kw)


# ── throttle mapping ─────────────────────────────────────────────────────────
@pytest.mark.parametrize("t,us", [(0, 1000), (0.25, 1250), (1, 2000), (-1, 1000), (2, 2000)])
def test_pulse_mapping_and_clamp(esc_mod, out, t, us):
    assert make(esc_mod, out).pulse_us(t) == us


def test_max_throttle_caps_output(esc_mod, out):
    e = make(esc_mod, out, max_throttle=0.3)
    assert e.set_throttle(1.0) == pytest.approx(0.3)
    assert out.last == 1300


def test_custom_range(esc_mod, out):
    e = make(esc_mod, out, min_us=1100, max_us=1900)
    e.set_throttle(0.5)
    assert out.last == 1500


# ── ramping ──────────────────────────────────────────────────────────────────
def test_ramp_to_is_monotonic_and_bounded(esc_mod, out):
    e = make(esc_mod, out)
    out.log.clear()
    e.ramp_to(0.5, slew=1.0)                    # 0.02 per 20 ms frame → 25 frames
    pulses = [us for _, us in out.log]
    assert len(pulses) == 25
    assert pulses == sorted(pulses)
    assert max(b - a for a, b in zip([1000] + pulses, pulses)) <= 20
    assert e.throttle == pytest.approx(0.5) and pulses[-1] == 1500

    out.log.clear()
    e.ramp_to(0.0, slew=1.0)
    pulses = [us for _, us in out.log]
    assert pulses == sorted(pulses, reverse=True) and pulses[-1] == 1000


def test_ramp_step_reports_arrival(esc_mod, out):
    e = make(esc_mod, out, max_throttle=0.1)
    assert not e.ramp_step(1.0, 0.06)
    assert e.ramp_step(1.0, 0.06)               # target clamped to 0.1
    assert e.throttle == pytest.approx(0.1)


# ── hardware PWM backend ─────────────────────────────────────────────────────
@pytest.fixture
def sysfs(tmp_path):
    chip = tmp_path / "pwmchip0"
    chip.mkdir()

    # Fake export: writing N creates pwmN/ with its attribute files.
    class Chip:
        def export(self, ch):
            d = chip / f"pwm{ch}"
            d.mkdir(exist_ok=True)
            for name in ("period", "duty_cycle", "enable"):
                (d / name).write_text("0")

    return tmp_path, chip, Chip()


def test_hwpwm_missing_chip_explains_overlay(esc_mod, tmp_path):
    with pytest.raises(RuntimeError, match="pwm-2chan"):
        esc_mod.HardwarePwmOutput(root=str(tmp_path))


def test_hwpwm_writes_period_and_duty(esc_mod, sysfs):
    root, chip, fake = sysfs
    fake.export(1)                              # pretend GPIO 13's channel is exported
    hw = esc_mod.HardwarePwmOutput(root=str(root))
    hw.set_pulse_us(13, 1500)
    assert (chip / "pwm1" / "period").read_text() == "20000000"
    assert (chip / "pwm1" / "duty_cycle").read_text() == "1500000"
    assert (chip / "pwm1" / "enable").read_text() == "1"
    hw.close()
    assert (chip / "pwm1" / "enable").read_text() == "0"
    assert (chip / "unexport").read_text() == "1"


def test_hwpwm_rejects_non_pwm_pin(esc_mod, sysfs):
    root, _, _ = sysfs
    with pytest.raises(ValueError):
        esc_mod.HardwarePwmOutput(root=str(root)).set_pulse_us(17, 1500)
