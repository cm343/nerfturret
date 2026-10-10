#!/usr/bin/env python3
"""
ESC class for a flywheel brushless motor behind a hobby ESC (servo-style input).

  * Signal is 50 Hz servo pulses: min_us (default 1000 µs) = stop,
    max_us (default 2000 µs) = full throttle.
  * setup() starts sending the stop pulse right away, so the ESC never sees
    a floating or high signal. A high signal at power-on puts many ESCs into
    calibration mode.
  * arm() holds the stop pulse long enough for the ESC to finish its
    power-up beeps and arm. Throttle commands before arm() raise an error.
  * Throttle is a fraction 0.0–1.0, clamped to max_throttle. ramp_to() slews
    gradually; set_throttle() jumps straight to the new value.

Pulse output is pluggable, since pigpio isn't packaged on Raspberry Pi OS trixie:

  * LgpioServoOutput — lgpio.tx_servo(). Works out of the box (rpi-lgpio, which
    provides RPi.GPIO here, is built on lgpio). Timed by lgpio's own C thread,
    so jitter is much lower than RPi.GPIO software PWM, though not zero.
  * HardwarePwmOutput — kernel PWM via /sys/class/pwm. Pulses are jitter-free,
    but GPIO 12/13 need this line in /boot/firmware/config.txt plus a reboot:
        dtoverlay=pwm-2chan,pin=12,func=4,pin2=13,func2=4

Like Stepper, ESC never closes the shared output. That is left to the owner,
so both ESCs can share one.
"""

import os
import time

DEFAULT_MIN_US       = 1000     # stop / arming pulse
DEFAULT_MAX_US       = 2000     # full throttle
DEFAULT_ARM_S        = 3.0      # hold min_us this long after power-up
DEFAULT_SLEW         = 1.0      # throttle fraction per second for ramp_to()
SERVO_HZ             = 50
_RAMP_TICK_S         = 1.0 / SERVO_HZ


# ── pulse outputs ────────────────────────────────────────────────────────────
class LgpioServoOutput:
    """Servo pulses from lgpio.tx_servo(). set_pulse_us(pin, 0) stops the pulses."""

    def __init__(self, chip: int = 0) -> None:
        import lgpio
        self._lg = lgpio
        self._h = lgpio.gpiochip_open(chip)
        self._claimed: set[int] = set()

    def set_pulse_us(self, pin: int, us: int) -> None:
        if pin not in self._claimed:
            try:
                self._lg.gpio_claim_output(self._h, pin, 0)
            except self._lg.error as exc:
                if 'busy' in str(exc).lower():
                    raise RuntimeError(
                        f'GPIO {pin} is busy: another process holds it '
                        f'(check with: gpioinfo | grep GPIO{pin}; pgrep -af python)') from exc
                raise
            self._claimed.add(pin)
        self._lg.tx_servo(self._h, pin, int(us), SERVO_HZ)

    def close(self) -> None:
        for pin in self._claimed:
            self._lg.tx_servo(self._h, pin, 0)
            self._lg.gpio_free(self._h, pin)
        self._claimed.clear()
        self._lg.gpiochip_close(self._h)


class HardwarePwmOutput:
    """Servo pulses from the kernel PWM driver (needs the pwm-2chan overlay)."""

    # BCM pin → PWM channel with dtoverlay=pwm-2chan,pin=12,func=4,pin2=13,func2=4
    CHANNELS = {12: 0, 18: 0, 13: 1, 19: 1}
    PERIOD_NS = 1_000_000_000 // SERVO_HZ

    def __init__(self, chip: int = 0, root: str = '/sys/class/pwm') -> None:
        self._chip = os.path.join(root, f'pwmchip{chip}')
        if not os.path.isdir(self._chip):
            raise RuntimeError(
                f'{self._chip} not found. Add dtoverlay=pwm-2chan,pin=12,func=4,'
                'pin2=13,func2=4 to /boot/firmware/config.txt and reboot.')
        self._open: set[int] = set()

    def _path(self, ch: int, name: str) -> str:
        return os.path.join(self._chip, f'pwm{ch}', name)

    @staticmethod
    def _write(path: str, value) -> None:
        with open(path, 'w') as f:
            f.write(str(value))

    def _open_channel(self, ch: int) -> None:
        if not os.path.isdir(os.path.join(self._chip, f'pwm{ch}')):
            self._write(os.path.join(self._chip, 'export'), ch)
            # udev fixes the new files' permissions asynchronously
            deadline = time.monotonic() + 1.0
            while not os.access(self._path(ch, 'period'), os.W_OK):
                if time.monotonic() > deadline:
                    raise RuntimeError(f'cannot write {self._path(ch, "period")}')
                time.sleep(0.01)
        self._write(self._path(ch, 'duty_cycle'), 0)    # duty must stay ≤ period
        self._write(self._path(ch, 'period'), self.PERIOD_NS)
        self._write(self._path(ch, 'enable'), 1)
        self._open.add(ch)

    def set_pulse_us(self, pin: int, us: int) -> None:
        if pin not in self.CHANNELS:
            raise ValueError(f'GPIO {pin} has no hardware PWM channel')
        ch = self.CHANNELS[pin]
        if ch not in self._open:
            self._open_channel(ch)
        self._write(self._path(ch, 'duty_cycle'), int(us) * 1000)

    def close(self) -> None:
        for ch in self._open:
            self._write(self._path(ch, 'duty_cycle'), 0)
            self._write(self._path(ch, 'enable'), 0)
            self._write(os.path.join(self._chip, 'unexport'), ch)
        self._open.clear()


# ── ESC ──────────────────────────────────────────────────────────────────────
class ESC:
    """One ESC driven by servo pulses from a shared pulse output."""

    def __init__(
        self,
        output,
        pin: int,
        *,
        min_us: int = DEFAULT_MIN_US,
        max_us: int = DEFAULT_MAX_US,
        max_throttle: float = 1.0,
    ) -> None:
        """
        :param output:       object with set_pulse_us(pin, us); us=0 stops pulses.
        :param pin:          BCM GPIO pin wired to the ESC signal lead.
        :param min_us:       pulse width that means "stop".
        :param max_us:       pulse width that means "full throttle".
        :param max_throttle: safety cap (0–1); throttle requests above it are clamped.
        """
        if not 500 <= min_us < max_us <= 2500:
            raise ValueError("need 500 <= min_us < max_us <= 2500")
        if not 0.0 < max_throttle <= 1.0:
            raise ValueError("max_throttle must be in (0, 1]")

        self.output = output
        self.pin = pin
        self.min_us = int(min_us)
        self.max_us = int(max_us)
        self.max_throttle = float(max_throttle)

        self.throttle = 0.0
        self.armed = False
        self._ready = False

    # ── lifecycle ────────────────────────────────────────────────────────────
    def setup(self) -> None:
        """Start emitting the stop pulse. If possible, call this before powering the ESC."""
        self.output.set_pulse_us(self.pin, self.min_us)
        self.throttle = 0.0
        self._ready = True

    def arm(self, seconds: float = DEFAULT_ARM_S) -> None:
        """Hold the stop pulse for `seconds` so the ESC arms. Blocking."""
        self._require_setup()
        self.output.set_pulse_us(self.pin, self.min_us)
        self.throttle = 0.0
        time.sleep(seconds)
        self.armed = True

    def stop(self) -> None:
        """Cut throttle to zero immediately. The flywheel coasts down."""
        self._require_setup()
        self.output.set_pulse_us(self.pin, self.min_us)
        self.throttle = 0.0

    def cleanup(self) -> None:
        """Send stop, then switch the pulses off. Does NOT close the output."""
        if self._ready:
            self.output.set_pulse_us(self.pin, self.min_us)
            time.sleep(_RAMP_TICK_S * 2)    # let at least one stop frame go out
            self.output.set_pulse_us(self.pin, 0)
            self.throttle = 0.0
            self.armed = False
            self._ready = False

    def _require_setup(self) -> None:
        if not self._ready:
            raise RuntimeError("ESC.setup() must be called before using the ESC")

    def _require_armed(self) -> None:
        self._require_setup()
        if not self.armed:
            raise RuntimeError("ESC.arm() must be called before applying throttle")

    # ── throttle ─────────────────────────────────────────────────────────────
    def clamp(self, throttle: float) -> float:
        return min(max(float(throttle), 0.0), self.max_throttle)

    def pulse_us(self, throttle: float) -> int:
        """Pulse width for a throttle fraction, after clamping to [0, max_throttle]."""
        return round(self.min_us + self.clamp(throttle) * (self.max_us - self.min_us))

    def set_throttle(self, throttle: float) -> float:
        """Jump to `throttle` (0–1). Returns the clamped value actually applied."""
        self._require_armed()
        t = self.clamp(throttle)
        self.output.set_pulse_us(self.pin, self.pulse_us(t))
        self.throttle = t
        return t

    def ramp_step(self, target: float, max_delta: float) -> bool:
        """Move at most `max_delta` toward `target`. Returns True once it is there."""
        target = self.clamp(target)
        delta = max(-max_delta, min(max_delta, target - self.throttle))
        if abs(delta) > 1e-9:
            self.set_throttle(self.throttle + delta)
        return abs(target - self.throttle) <= 1e-9

    def ramp_to(self, throttle: float, slew: float = DEFAULT_SLEW) -> float:
        """Slew to `throttle` at `slew` fraction/second, one 50 Hz frame per tick. Blocking."""
        self._require_armed()
        if slew <= 0:
            raise ValueError("slew must be positive")
        while not self.ramp_step(throttle, slew * _RAMP_TICK_S):
            time.sleep(_RAMP_TICK_S)
        return self.throttle
