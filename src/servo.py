#!/usr/bin/env python3
"""
Servo classes for a Raspberry Pi turret.

Two servo types share one lifecycle/PWM base (BaseServo) and a common callable
interface (start / stop / cleanup / move_by):

  * ContinuousServo — a 360° continuous-rotation servo. PWM duty sets *speed and
    direction*, not an absolute angle. Offers set_speed(speed) for free-running
    and an open-loop, timing-based move_by(degrees, speed). Exposed as ``Servo``
    too (backward-compatible alias).

  * PositionalServo — a standard 180° servo. PWM duty sets an *absolute angle*.
    Offers move_to(angle, speed) (absolute) and move_by(degrees, speed)
    (relative); both slew in small steps so ``speed`` is honored and the call
    blocks until the move completes. Tracks the current angle in ``self.angle``.

Neither class calls the global GPIO.cleanup() — that is left to the owner so
several servos can share the GPIO subsystem.
"""

import time
from abc import ABC, abstractmethod

import RPi.GPIO as GPIO

# Default duty-cycle calibration (percent of the 20 ms / 50 Hz period).
DEFAULT_PWM_FREQUENCY = 50      # Hz

# Continuous-rotation calibration. DUTY_STOP is the trimmed neutral (no drift).
DEFAULT_DUTY_STOP     = 7.1     # % — neutral / stopped
DEFAULT_DUTY_FULL_CW  = 12.5    # % — full clockwise speed
DEFAULT_DUTY_FULL_CCW = 2.5     # % — full counter-clockwise speed

# Positional (180°) calibration: duty at the two travel extremes.
DEFAULT_DUTY_MIN      = 2.5     # % — angle 0°   (~1 ms pulse)
DEFAULT_DUTY_MAX      = 12.5    # % — angle 180° (~2 ms pulse)
DEFAULT_ANGLE_RANGE   = 180.0   # degrees of travel for a standard servo

# Step size for the positional servo's timed slew (smaller = smoother/slower loop).
_POSITIONAL_STEP_DEG  = 1.0


class BaseServo(ABC):
    """Shared GPIO/PWM lifecycle for a single servo driven by hardware PWM."""

    def __init__(
        self,
        pin: int,
        deg_per_sec: float,
        *,
        pwm_frequency: int = DEFAULT_PWM_FREQUENCY,
    ) -> None:
        """
        :param pin:           BCM GPIO pin the servo signal wire is connected to.
        :param deg_per_sec:   rotation rate at full speed, in degrees/second
                              (e.g. 60° in 0.2 s -> 300.0). Used to time motion.
        :param pwm_frequency: PWM frequency in Hz (50 Hz is standard).
        """
        if deg_per_sec <= 0:
            raise ValueError("deg_per_sec must be positive")

        self.pin = pin
        self.deg_per_sec = float(deg_per_sec)
        self.pwm_frequency = pwm_frequency

        self._pwm = None  # created in start()

    # ── lifecycle ────────────────────────────────────────────────────────────
    def start(self) -> None:
        """Configure the GPIO pin and begin emitting the idle pulse.

        Assumes GPIO.setmode(GPIO.BCM) has already been set by the caller, or
        sets it here if not yet configured.
        """
        if GPIO.getmode() is None:
            GPIO.setmode(GPIO.BCM)
        GPIO.setup(self.pin, GPIO.OUT)
        self._pwm = GPIO.PWM(self.pin, self.pwm_frequency)
        self._pwm.start(self._idle_duty())

    def cleanup(self) -> None:
        """Settle at the idle duty, then stop this servo's PWM. Does NOT call
        GPIO.cleanup() — leave that to the owner so multiple servos can share the
        GPIO subsystem."""
        if self._pwm is not None:
            self._pwm.ChangeDutyCycle(self._idle_duty())
            time.sleep(0.05)
            self._pwm.stop()
            self._pwm = None

    def _require_started(self) -> None:
        if self._pwm is None:
            raise RuntimeError("Servo.start() must be called before moving the servo")

    # ── per-type behavior ──────────────────────────────────────────────────────
    @abstractmethod
    def _idle_duty(self) -> float:
        """Duty cycle (%) to hold at rest (start-up and cleanup)."""

    @abstractmethod
    def stop(self) -> None:
        """Hold the servo at rest (halt a continuous servo / hold a position)."""

    @abstractmethod
    def move_by(self, degrees: float, speed: float = 1.0) -> None:
        """Move by a relative number of degrees, then stop/hold. Blocking."""


class ContinuousServo(BaseServo):
    """A 360° continuous-rotation servo: PWM duty controls speed and direction."""

    def __init__(
        self,
        pin: int,
        deg_per_sec: float,
        *,
        pwm_frequency: int = DEFAULT_PWM_FREQUENCY,
        duty_stop: float = DEFAULT_DUTY_STOP,
        duty_full_cw: float = DEFAULT_DUTY_FULL_CW,
        duty_full_ccw: float = DEFAULT_DUTY_FULL_CCW,
    ) -> None:
        """
        :param duty_stop:     duty cycle (%) at which the servo is stationary.
        :param duty_full_cw:  duty cycle (%) for full clockwise speed.
        :param duty_full_ccw: duty cycle (%) for full counter-clockwise speed.
        """
        super().__init__(pin, deg_per_sec, pwm_frequency=pwm_frequency)
        self.duty_stop = duty_stop
        self.duty_full_cw = duty_full_cw
        self.duty_full_ccw = duty_full_ccw

    def _idle_duty(self) -> float:
        return self.duty_stop

    def stop(self) -> None:
        """Hold the servo stationary at the neutral duty cycle."""
        self._require_started()
        self._pwm.ChangeDutyCycle(self.duty_stop)

    # ── motion ───────────────────────────────────────────────────────────────
    def set_speed(self, speed: float) -> None:
        """Free-run at a signed speed fraction.

        :param speed: in [-1.0, 1.0]. 0 stops, >0 is clockwise, <0 is
                      counter-clockwise. Magnitude scales linearly between the
                      stop duty and the full-speed duty for that direction.
        """
        self._require_started()
        self._pwm.ChangeDutyCycle(self._speed_to_duty(speed))

    def move_by(self, degrees: float, speed: float = 1.0) -> None:
        """Rotate (open-loop) by a relative number of degrees, then stop.

        Blocking: sleeps for the computed duration. Run from a worker thread if
        you need it off the main thread (see Trigger).

        :param degrees: signed amount to rotate. >0 clockwise, <0 CCW.
        :param speed:   speed magnitude fraction in (0, 1].
        """
        self._require_started()
        speed = abs(speed)
        if speed == 0:
            return
        if speed > 1.0:
            speed = 1.0
        if degrees == 0:
            return

        duration = abs(degrees) / (self.deg_per_sec * speed)
        direction = 1.0 if degrees > 0 else -1.0
        self.set_speed(direction * speed)
        time.sleep(duration)
        self.stop()

    # ── internals ─────────────────────────────────────────────────────────────
    def _speed_to_duty(self, speed: float) -> float:
        """Map a signed speed fraction in [-1, 1] to a duty-cycle percentage."""
        if speed > 1.0:
            speed = 1.0
        elif speed < -1.0:
            speed = -1.0

        if speed == 0:
            return self.duty_stop
        if speed > 0:
            return self.duty_stop + speed * (self.duty_full_cw - self.duty_stop)
        return self.duty_stop + speed * (self.duty_stop - self.duty_full_ccw)


class PositionalServo(BaseServo):
    """A standard 180° servo: PWM duty controls an absolute angle.

    move_to()/move_by() slew the servo in small steps (timed from deg_per_sec)
    so ``speed`` is honored and the call blocks until the target is reached. The
    current angle is tracked in ``self.angle`` (open-loop — there is no position
    feedback, so it assumes the servo follows commands).
    """

    def __init__(
        self,
        pin: int,
        deg_per_sec: float,
        *,
        pwm_frequency: int = DEFAULT_PWM_FREQUENCY,
        duty_min: float = DEFAULT_DUTY_MIN,
        duty_max: float = DEFAULT_DUTY_MAX,
        angle_range: float = DEFAULT_ANGLE_RANGE,
        initial_angle: float = 0.0,
    ) -> None:
        """
        :param duty_min:      duty cycle (%) at angle 0°.
        :param duty_max:      duty cycle (%) at angle ``angle_range``.
        :param angle_range:   total travel in degrees (180 for a standard servo).
        :param initial_angle: angle assumed/held at start().
        """
        super().__init__(pin, deg_per_sec, pwm_frequency=pwm_frequency)
        self.duty_min = duty_min
        self.duty_max = duty_max
        self.angle_range = float(angle_range)
        self.angle = self._clamp_angle(initial_angle)

    def _idle_duty(self) -> float:
        return self._angle_to_duty(self.angle)

    def stop(self) -> None:
        """Hold the current angle (a positional servo holds while it gets pulses)."""
        self._require_started()
        self._pwm.ChangeDutyCycle(self._angle_to_duty(self.angle))

    # ── motion ───────────────────────────────────────────────────────────────
    def move_to(self, angle: float, speed: float = 1.0) -> None:
        """Slew to an absolute angle (clamped to [0, angle_range]), then hold.

        Blocking: steps toward the target so ``speed`` (fraction of full rate)
        is honored. Run from a worker thread for off-main-thread motion.
        """
        self._require_started()
        speed = min(abs(speed), 1.0)
        if speed == 0:
            return

        target = self._clamp_angle(angle)
        delta = target - self.angle
        if delta == 0:
            return

        step = _POSITIONAL_STEP_DEG if delta > 0 else -_POSITIONAL_STEP_DEG
        step_pause = abs(step) / (self.deg_per_sec * speed)
        steps = int(abs(delta) / _POSITIONAL_STEP_DEG)

        for _ in range(steps):
            self.angle += step
            self._pwm.ChangeDutyCycle(self._angle_to_duty(self.angle))
            time.sleep(step_pause)
        # Land exactly on the target (covers a non-integer final step).
        if self.angle != target:
            self.angle = target
            self._pwm.ChangeDutyCycle(self._angle_to_duty(self.angle))
            time.sleep(step_pause)

    def move_by(self, degrees: float, speed: float = 1.0) -> None:
        """Move by a relative number of degrees, then hold. Blocking."""
        self.move_to(self.angle + degrees, speed)

    # ── internals ─────────────────────────────────────────────────────────────
    def _clamp_angle(self, angle: float) -> float:
        return max(0.0, min(self.angle_range, float(angle)))

    def _angle_to_duty(self, angle: float) -> float:
        """Map an angle (clamped to [0, angle_range]) to a duty-cycle percentage."""
        angle = self._clamp_angle(angle)
        return self.duty_min + (angle / self.angle_range) * (self.duty_max - self.duty_min)


# Backward-compatible alias: existing code (trigger.py history, helper scripts)
# imports ``Servo`` and expects the continuous-rotation behavior.
Servo = ContinuousServo
