#!/usr/bin/env python3
"""
Stepper class for a STEP/DIR/EN driver board (HW-216 carrier + DRV8825).

  * EN is active-LOW: the driver is disabled (coils unpowered) while EN is high.
    setup() leaves the driver disabled; call enable() before moving, or let
    move() auto-enable.
  * move(steps) is relative and blocking. The sign picks the direction
    (positive = DIR high). Speed ramps linearly up and back down so the motor
    doesn't stall on a hard start.
  * Step timing is a busy-wait on time.perf_counter(). That's fine for a test
    rig, but Linux scheduling will add jitter at high step rates; see the
    "Can the Pi generate the step pulses fast enough?" section of
    nerf-turret-wiring.html.

Like the servo classes, Stepper never calls the global GPIO.cleanup() — that is
left to the owner so several steppers can share the GPIO subsystem.
"""

import math
import time
from typing import Callable, Optional

import RPi.GPIO as GPIO

FULL_STEPS_PER_REV     = 200        # 1.8° NEMA 17
DEFAULT_SPEED          = 800.0      # µsteps/s cruise speed
DEFAULT_ACCEL          = 4000.0     # µsteps/s² ramp
DEFAULT_START_SPEED    = 200.0      # µsteps/s at the first and last step

# DRV8825 timing: STEP high ≥ 1.9 µs, DIR setup ≥ 650 ns. Python's own call
# overhead already exceeds both, but keep explicit margins.
_STEP_PULSE_S          = 5e-6
_DIR_SETUP_S           = 5e-6
_ENABLE_SETTLE_S       = 0.002      # let the driver's charge pump wake up


def _busy_wait_until(deadline: float) -> None:
    while time.perf_counter() < deadline:
        pass


class Stepper:
    """One stepper motor behind a STEP/DIR/EN driver."""

    def __init__(
        self,
        step_pin: int,
        dir_pin: int,
        en_pin: int,
        *,
        microsteps: int = 16,
        full_steps_per_rev: int = FULL_STEPS_PER_REV,
        speed: float = DEFAULT_SPEED,
        accel: float = DEFAULT_ACCEL,
        start_speed: float = DEFAULT_START_SPEED,
    ) -> None:
        """
        :param step_pin:   BCM GPIO pin wired to the driver's STEP input.
        :param dir_pin:    BCM GPIO pin wired to DIR.
        :param en_pin:     BCM GPIO pin wired to EN (active-LOW).
        :param microsteps: microstep divisor set on the DIP switches (1–32).
        :param speed:      cruise speed in µsteps/s.
        :param accel:      ramp in µsteps/s²; 0 disables ramping.
        :param start_speed: speed of the first/last step of a ramped move.
        """
        if speed <= 0 or start_speed <= 0:
            raise ValueError("speed and start_speed must be positive")
        if accel < 0:
            raise ValueError("accel must be >= 0")

        self.step_pin = step_pin
        self.dir_pin = dir_pin
        self.en_pin = en_pin
        self.microsteps = microsteps
        self.steps_per_rev = full_steps_per_rev * microsteps
        self.speed = float(speed)
        self.accel = float(accel)
        self.start_speed = float(start_speed)

        self.position = 0           # µsteps since setup()/zero(), signed
        self.enabled = False
        self._ready = False

    # ── lifecycle ────────────────────────────────────────────────────────────
    def setup(self) -> None:
        """Configure the pins with STEP low and the driver disabled.

        Sets GPIO.setmode(GPIO.BCM) if the caller hasn't set a mode yet.
        """
        if GPIO.getmode() is None:
            GPIO.setmode(GPIO.BCM)
        GPIO.setup(self.en_pin, GPIO.OUT, initial=GPIO.HIGH)
        GPIO.setup(self.step_pin, GPIO.OUT, initial=GPIO.LOW)
        GPIO.setup(self.dir_pin, GPIO.OUT, initial=GPIO.LOW)
        self.enabled = False
        self._ready = True

    def enable(self) -> None:
        self._require_setup()
        if not self.enabled:
            GPIO.output(self.en_pin, GPIO.LOW)
            self.enabled = True
            time.sleep(_ENABLE_SETTLE_S)

    def disable(self) -> None:
        """Cut coil current. The motor will no longer hold its position."""
        self._require_setup()
        GPIO.output(self.en_pin, GPIO.HIGH)
        self.enabled = False

    def cleanup(self) -> None:
        """Disable the driver and leave STEP low. Does NOT call GPIO.cleanup()."""
        if self._ready:
            GPIO.output(self.step_pin, GPIO.LOW)
            self.disable()
            self._ready = False

    def zero(self) -> None:
        """Declare the current position to be 0."""
        self.position = 0

    def _require_setup(self) -> None:
        if not self._ready:
            raise RuntimeError("Stepper.setup() must be called before using the stepper")

    # ── motion ───────────────────────────────────────────────────────────────
    def step_delays(self, n: int, speed: Optional[float] = None) -> list[float]:
        """Seconds between consecutive step pulses for an n-step ramped move.

        Symmetric trapezoid: v_i = min(cruise, sqrt(v0² + 2·a·d)) where d is the
        distance in steps to the nearer end of the move. `speed` overrides the
        cruise speed for this move only.
        """
        if n <= 0:
            return []
        cruise = self.speed if speed is None else float(speed)
        if cruise <= 0:
            raise ValueError("speed must be positive")
        if self.accel == 0:
            return [1.0 / cruise] * n
        # Slower than start_speed: run the whole move at that slower speed.
        v0 = min(self.start_speed, cruise)
        v0_sq = v0 ** 2
        delays = []
        for i in range(n):
            d = min(i, n - 1 - i)
            v = min(cruise, math.sqrt(v0_sq + 2.0 * self.accel * d))
            delays.append(1.0 / v)
        return delays

    def move(
        self,
        steps: int,
        stop_check: Optional[Callable[[int], bool]] = None,
        speed: Optional[float] = None,
    ) -> int:
        """Move by `steps` µsteps (sign = direction). Blocking.

        :param stop_check: called with the direction (+1/-1) before every step;
                           returning True aborts the move immediately (no
                           decel ramp — meant for endstops).
        :param speed:      cruise speed in µsteps/s for this move only.
        :returns: µsteps actually moved, signed. Also added to self.position.
        """
        self._require_setup()
        steps = int(steps)
        if steps == 0:
            return 0
        direction = 1 if steps > 0 else -1
        self.enable()

        GPIO.output(self.dir_pin, GPIO.HIGH if direction > 0 else GPIO.LOW)
        _busy_wait_until(time.perf_counter() + _DIR_SETUP_S)

        moved = 0
        next_t = time.perf_counter()
        for delay in self.step_delays(abs(steps), speed):
            if stop_check is not None and stop_check(direction):
                break
            _busy_wait_until(next_t)
            t = time.perf_counter()
            GPIO.output(self.step_pin, GPIO.HIGH)
            _busy_wait_until(t + _STEP_PULSE_S)
            GPIO.output(self.step_pin, GPIO.LOW)
            moved += 1
            # Schedule from the planned time, but never try to catch up a stall
            # by bursting steps.
            next_t = max(next_t + delay, t + delay * 0.5)

        self.position += direction * moved
        return direction * moved

    def move_degrees(self, degrees: float, stop_check=None) -> float:
        """Move by an angle at the output shaft. Returns degrees actually moved."""
        moved = self.move(round(degrees / 360.0 * self.steps_per_rev), stop_check)
        return moved * 360.0 / self.steps_per_rev
